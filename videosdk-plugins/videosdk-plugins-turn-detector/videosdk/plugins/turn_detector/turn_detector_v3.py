import logging
import os
import threading
import numpy as np
from typing import Optional
from videosdk.agents import EOU, ChatContext, ChatMessage, ChatRole
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

NAMO_ONNX_FILENAME = "model_quant.onnx"
TOKENIZER_FILENAME = "tokenizer.json"

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

    hf_hub_download(repo_id=hf_repo, filename=TOKENIZER_FILENAME)

class NamoTurnDetectorV1(EOU):
    """
    A lightweight end-of-utterance detection model using VideoSDK's Namo Turn Detection v1 model.
    """
    
    def __init__(self, threshold: float = 0.7, language: Optional[str] = None, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.language = language
        self.session = None
        self.tokenizer = None
        self._input_names: list[str] = []
        self._initialize_model()

    def _initialize_model(self):
        """Initialize (or reuse) the ONNX model and tokenizer.

        Caches tokenizer + ORT session per language at module level so
        repeated NamoTurnDetectorV1(language=...) constructions in the
        same process share the heavy resources.
        """
        try:
            import onnxruntime as ort
            from ._inference import load_tokenizer, make_session_options, model_input_names

            self.max_length = 8192 if self.language is None else 512
            cache_key = self.language

            cached_tokenizer = _tokenizer_cache.get(cache_key)
            cached_session = _session_cache.get(cache_key)
            if cached_tokenizer is not None and cached_session is not None:
                self.tokenizer = cached_tokenizer
                self.session = cached_session
                self._input_names = model_input_names(self.session)
                return

            with _init_lock:
                cached_tokenizer = _tokenizer_cache.get(cache_key)
                cached_session = _session_cache.get(cache_key)
                if cached_tokenizer is None or cached_session is None:
                    hf_repo = _get_hf_model_repo(self.language)

                    if cached_session is None:
                        model_path = hf_hub_download(repo_id=hf_repo, filename=NAMO_ONNX_FILENAME)
                        cached_session = ort.InferenceSession(
                            model_path,
                            sess_options=make_session_options(),
                            providers=["CPUExecutionProvider"],
                        )
                        _session_cache[cache_key] = cached_session
                        logger.info(f"Namo model loaded from {hf_repo}.")

                    if cached_tokenizer is None:
                        tokenizer_path = hf_hub_download(repo_id=hf_repo, filename=TOKENIZER_FILENAME)
                        cached_tokenizer = load_tokenizer(tokenizer_path, max_length=self.max_length)
                        _tokenizer_cache[cache_key] = cached_tokenizer

            self.tokenizer = cached_tokenizer
            self.session = cached_session
            self._input_names = model_input_names(self.session)

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
            from ._inference import encode_for_model

            input_dict = encode_for_model(self.tokenizer, sentence.strip(), self._input_names)

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