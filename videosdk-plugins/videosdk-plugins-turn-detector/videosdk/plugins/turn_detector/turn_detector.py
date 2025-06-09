import logging
import numpy as np
from typing import Optional
from .model import HG_MODEL, ONNX_FILENAME
from videosdk.agents import EOU, ChatContext, ChatMessage, ChatRole

logger = logging.getLogger(__name__)

def _download_from_hf_hub(repo_id, filename, **kwargs):
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    return local_path

def pre_download_model():
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(HG_MODEL)
    _download_from_hf_hub(
        repo_id=HG_MODEL,
        filename=ONNX_FILENAME,
    )

class TurnDetector(EOU):
    """
    A lightweight end-of-utterance detection model using TurnSense.
    Based on SmolLM2-135M, optimized for edge devices.
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
            from transformers import AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(HG_MODEL)
            
            model_path = _download_from_hf_hub(
                repo_id=HG_MODEL,
                filename=ONNX_FILENAME,
            )
            
            self.session = ort.InferenceSession(
                model_path, 
                providers=["CPUExecutionProvider"]
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize TurnSense model: {e}")
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
            str: Formatted text for the model
        """
        last_user_text = self._get_last_user_message(chat_context)
        
        if not last_user_text:
            return "<|user|>  <|im_end|>"
        
        formatted_text = f"<|user|> {last_user_text} <|im_end|>"
        
        return formatted_text
    
    def get_eou_probability(self, chat_context: ChatContext) -> float:
        """
        Get the probability score for end of utterance detection.
        
        Args:
            chat_context: Chat context to analyze
            
        Returns:
            float: Probability score (0.0 to 1.0)
        """
        if not self.session or not self.tokenizer:
            raise RuntimeError("Model not initialized")
        
        try:
            formatted_text = self._chat_context_to_text(chat_context)
            
            inputs = self.tokenizer(
                formatted_text,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="np"
            )
            
            ort_inputs = {
                'input_ids': inputs['input_ids'].astype(np.int64),
                'attention_mask': inputs['attention_mask'].astype(np.int64)
            }
            
            outputs = self.session.run(None, ort_inputs)
            
            probabilities = outputs[0]
            
            eou_prob = float(probabilities[0][1])
            
            return eou_prob
            
        except Exception as e:
            logger.error(f"Error getting EOU probability: {e}")
            return 0.0

    def detect_end_of_utterance(self, chat_context: ChatContext, threshold: Optional[float] = None) -> bool:
        """
        Detect if the given chat context represents an end of utterance.
        
        Args:
            chat_context: Chat context to analyze
            threshold: Probability threshold for EOU detection (uses instance threshold if None)
            
        Returns:
            bool: True if end of utterance is detected, False otherwise
        """
        if threshold is None:
            threshold = self.threshold
        
        try:
            probability = self.get_eou_probability(chat_context)
            is_eou = probability >= threshold
            
            return is_eou
            
        except Exception as e:
            logger.error(f"Error during EOU detection: {e}")
            return False

