import asyncio
import logging
import os
import threading
import numpy as np
from typing import Optional
from videosdk.agents import EOU, ChatContext, ChatMessage, ChatRole
from transformers import AutoTokenizer, DistilBertTokenizer
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

NAMO_ONNX_FILENAME = "model_quant.onnx"

_tokenizer_cache: dict[Optional[str], object] = {}
_session_cache: dict[Optional[str], object] = {}
_init_lock = threading.Lock()

def _get_hf_model_repo(language: Optional[str] = None) -> str:
    """
    Get the appropriate Hugging Face model repository based on language.
    """
    if language is None:
        return "videosdk-live/Namo-Turn-Detector-v1-Multilingual"
    else:
        language_map = {
    "ar": "Arabic",
    "bn": "Bengali",
    "zh": "Chinese",
    "da": "Danish",
    "nl": "Dutch",
    "de": "German",
    "en": "English",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mr": "Marathi",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese"
}
        lang_name = language_map.get(language.lower(), language.capitalize())
        return f"videosdk-live/Namo-Turn-Detector-v1-{lang_name}"

def pre_download_namo_turn_v1_model(overwrite_existing: bool = False, language: Optional[str] = None):
    """Pre-download tokenizer for the given language. Skips if quantized ONNX is already in cache."""
    from huggingface_hub import try_to_load_from_cache

    hf_repo = _get_hf_model_repo(language)
    if not overwrite_existing and try_to_load_from_cache(repo_id=hf_repo, filename=NAMO_ONNX_FILENAME) is not None:
        return

    if language is None:
        AutoTokenizer.from_pretrained(hf_repo)
    else:
        DistilBertTokenizer.from_pretrained(hf_repo)

class NamoTurnDetectorV1(EOU):
    """
    A lightweight end-of-utterance detection model using VideoSDK's Namo Turn Detection v1 model.
    """
    
    def __init__(self, threshold: float = 0.7, language: Optional[str] = None, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.language = language
        self.session = None
        self.tokenizer = None
        self.max_length = 8192 if language is None else 512

    @classmethod
    async def download_model(cls, language: Optional[str] = None) -> None:
        """Eagerly download the tokenizer + ONNX for ``language`` into the HF cache.

        Idempotent; safe to call concurrently from multiple processes (HF Hub
        uses atomic renames). Skips network entirely when the cache file
        already exists.
        """
        await asyncio.to_thread(pre_download_namo_turn_v1_model, language=language)

    async def prewarm(self) -> None:
        """Populate this instance's session + tokenizer from the module cache.

        Triggers the download if the cache is cold (the underlying
        :func:`pre_download_namo_turn_v1_model` short-circuits when the
        file is already on disk). Then runs the base EOU dummy inference to
        warm the ONNX kernel.
        """
        try:
            await asyncio.to_thread(self._initialize_model)
        except Exception as e:
            logger.debug(f"NamoTurnDetectorV1 prewarm download failed (non-fatal): {e}")
            return
        await super().prewarm()

    def _initialize_model(self):
        """Initialize (or reuse) the ONNX model and tokenizer.

        Caches tokenizer + ORT session per language at module level so
        repeated NamoTurnDetectorV1(language=...) constructions in the
        same process share the heavy resources.
        """
        try:
            import onnxruntime as ort

            cache_key = self.language

            cached_tokenizer = _tokenizer_cache.get(cache_key)
            cached_session = _session_cache.get(cache_key)
            if cached_tokenizer is not None and cached_session is not None:
                self.tokenizer = cached_tokenizer
                self.session = cached_session
                return

            with _init_lock:
                cached_tokenizer = _tokenizer_cache.get(cache_key)
                cached_session = _session_cache.get(cache_key)
                if cached_tokenizer is None or cached_session is None:
                    hf_repo = _get_hf_model_repo(self.language)

                    if cached_tokenizer is None:
                        if self.language is None:
                            cached_tokenizer = AutoTokenizer.from_pretrained(hf_repo)
                        else:
                            cached_tokenizer = DistilBertTokenizer.from_pretrained(hf_repo)
                        _tokenizer_cache[cache_key] = cached_tokenizer

                    if cached_session is None:
                        model_path = hf_hub_download(repo_id=hf_repo, filename=NAMO_ONNX_FILENAME)
                        cached_session = ort.InferenceSession(model_path)
                        _session_cache[cache_key] = cached_session
                        logger.info(f"Namo model loaded from {hf_repo}.")

            self.tokenizer = cached_tokenizer
            self.session = cached_session

        except Exception as e:
            print(f"Error loading model: {e}")
            logger.error(f"Failed to initialize TurnDetection model: {e}")
            self.emit("error", f"Failed to initialize TurnDetection model: {str(e)}")
            raise
    
    def _get_last_user_message(self, chat_context: ChatContext) -> str:
        """
        Extract the last user message from chat context.
        This is what we want to analyze for EOU detection.
        """
        user_messages = [
            item for item in chat_context.items 
            if isinstance(item, ChatMessage) and item.role == ChatRole.USER
        ]
        
        if not user_messages:
            return ""
        
        last_message = user_messages[-1]
        content = last_message.content
        
        if isinstance(content, list):
            text_content = " ".join([c.text if hasattr(c, 'text') else str(c) for c in content])
        else:
            text_content = str(content)
        
        return text_content.strip()
    
    def _chat_context_to_text(self, chat_context: ChatContext) -> str:
        """
        Transform ChatContext to model-compatible format.
        Focus on the last user message for EOU detection.
        """
        last_user_text = self._get_last_user_message(chat_context)
        
        if not last_user_text:
            return ""
        
        return last_user_text

    def detect_turn(self, sentence: str) -> float:
        """
        Detect turn probability for the given sentence.
        """
        try:
            if self.session is None or self.tokenizer is None:
                self._initialize_model()
            inputs = self.tokenizer(sentence.strip(), truncation=True, max_length=self.max_length, return_tensors="np")
            
            input_dict = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
            
            if "token_type_ids" in inputs:
                input_dict["token_type_ids"] = inputs["token_type_ids"]
            
            outputs = self.session.run(None, input_dict)
            
            logits = outputs[0][0]
            
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            eou_probability = float(probabilities[1])
            
            return eou_probability
            
        except Exception as e:
            print(e)
            logger.error(f"Error detecting turn: {e}")
            self.emit("error", f"Error detecting turn: {str(e)}")
            return 0.0

    def get_eou_probability(self, chat_context: ChatContext) -> float:
        """
        Get the probability score for end of utterance detection.
        """
        try:
            sentence = self._chat_context_to_text(chat_context)
            if not sentence:
                return 0.0
            return self.detect_turn(sentence)
        except Exception as e:
            logger.error(f"Error getting EOU probability: {e}")
            self.emit("error", f"Error getting EOU probability: {str(e)}")
            return 0.0

    def detect_end_of_utterance(self, chat_context: ChatContext, threshold: Optional[float] = None) -> bool:
        """
        Detect if the given chat context represents an end of utterance.
        """
        try:
            effective_threshold = threshold if threshold is not None else self.threshold
            
            probability = self.get_eou_probability(chat_context)
            return probability >= effective_threshold
            
        except Exception as e:
            logger.error(f"Error in EOU detection: {e}")
            self.emit("error", f"Error in EOU detection: {str(e)}")
            return False
    
    async def aclose(self) -> None:
        """Cleanup ONNX model and tokenizer from memory"""
        logger.info("Cleaning up NamoTurnDetectorV1 model resources")
        
        if hasattr(self, 'session') and self.session is not None:
            try:
                del self.session
                self.session = None
                logger.info("Namo ONNX session cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up Namo ONNX session: {e}")
        
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            try:
                del self.tokenizer
                self.tokenizer = None
                logger.info("Namo tokenizer cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up Namo tokenizer: {e}")
        self.language = None
        
        try:
            import gc
            gc.collect()
            logger.info("Garbage collection completed")
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
        
        logger.info("NamoTurnDetectorV1 cleanup completed")
        await super().aclose()