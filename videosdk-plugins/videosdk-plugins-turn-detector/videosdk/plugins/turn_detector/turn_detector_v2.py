import logging
import os
import numpy as np
from typing import Optional
from .model import VIDEOSDK_MODEL_URL, VIDEOSDK_MODEL_FILES, MODEL_DIR
from videosdk.agents import EOU, ChatContext, ChatMessage, ChatRole
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

def pre_download_videosdk_model(overwrite_existing: bool = False):
    from .download_model import download_model_files_to_directory
    download_model_files_to_directory(
        base_cdn_url=VIDEOSDK_MODEL_URL,
        file_names=VIDEOSDK_MODEL_FILES,
        local_save_directory=MODEL_DIR,
        overwrite_existing=overwrite_existing,
    )
    BertTokenizer.from_pretrained(MODEL_DIR)

class VideoSDKTurnDetector(EOU):
    """
    A lightweight end-of-utterance detection model using VideoSDK's Turn Detection model.
    Based on BERT, doing binary classification for turn detection.
    """
    
    def __init__(self, threshold: float = 0.7, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.session = None
        self.tokenizer = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ONNX model and tokenizer"""
        try:
            import onnxruntime as ort
            
            if not os.path.exists(MODEL_DIR):
                logger.warning(f"Model directory {MODEL_DIR} does not exist. Running pre_download_model()...")
                pre_download_videosdk_model(overwrite_existing=True)
            
            pre_download_videosdk_model(overwrite_existing=False)
            
            self.tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
            
            model_path = os.path.join(MODEL_DIR, "model.onnx")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            self.session = ort.InferenceSession(model_path)
            print(f"Model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            logger.error(f"Failed to initialize TurnDetection model: {e}")
            self.emit("error", f"Failed to initialize TurnDetection model: {str(e)}")
            raise
    
    def _get_last_user_message(self, chat_context: ChatContext) -> str:
        """
        Extract the last user message from chat context.
        This is what we want to analyze for EOU detection.
        
        Args:
            chat_context: The chat context to analyze
            
        Returns:
            str: The last user message content
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
        
        Args:
            chat_context: The chat context to transform
            
        Returns:
            str: Raw text for the model (without special formatting)
        """
        last_user_text = self._get_last_user_message(chat_context)
        
        if not last_user_text:
            return ""
        
        return last_user_text

    def detect_turn(self, sentence: str):
        """
        Args:
            sentence: Input sentence to analyze
            
        Returns:
            str: "True" if turn detected, "False" otherwise
        """
        try:
            inputs = self.tokenizer(sentence.strip(), truncation=True, max_length=512, return_tensors="np")
            outputs = self.session.run(None, {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "token_type_ids": inputs["token_type_ids"],
            })
            pred = np.argmax(outputs)
            if pred == 0:
                pred = "False"
            else:
                pred = "True"
            if pred == "False":
                self.emit("error", f"Turn detection failed: result was {pred}")
            return pred
        except Exception as e:
            print(e)
            self.emit("error", f"Error detecting turn: {str(e)}")
            return "False"

    def get_eou_probability(self, chat_context: ChatContext) -> float:
        """
        Get the probability score for end of utterance detection.
        For binary classifier, returns 1.0 or 0.0 based on classification.
        
        Args:
            chat_context: Chat context to analyze
            
        Returns:
            float: Probability score (0.0 or 1.0)
        """
        try:
            sentence = self._chat_context_to_text(chat_context)
            result = self.detect_turn(sentence)
            return 1.0 if result == "True" else 0.0
        except Exception as e:
            logger.error(f"Error getting EOU probability: {e}")
            self.emit("error", f"Error getting EOU probability: {str(e)}")
            return 0.0

    def detect_end_of_utterance(self, chat_context: ChatContext, threshold: Optional[float] = None) -> bool:
        """
        Detect if the given chat context represents an end of utterance.
        Uses direct binary classification, ignoring threshold.
        
        Args:
            chat_context: Chat context to analyze
            threshold: Not used in binary classification
            
        Returns:
            bool: True if end of utterance is detected, False otherwise
        """
        try:
            sentence = self._chat_context_to_text(chat_context)
            result = self.detect_turn(sentence)
            return result == "True"
        except Exception as e:
            logger.error(f"Error in EOU detection: {e}")
            self.emit("error", f"Error in EOU detection: {str(e)}")
            return False
