from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Type, Tuple
import logging

logger = logging.getLogger(__name__)


class SIPProvider(ABC):
    """
    Abstract base class for SIP providers.
    
    This class defines the common interface that all SIP providers must implement.
    It is designed to handle major differences between providers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """Validate provider-specific configuration."""
        pass
    
    @abstractmethod
    async def handle_incoming_and_route(
        self, 
        call_data: Dict[str, Any], 
        destination_sip_uri: str,
        username: str,
        password: str,
        **kwargs
    ) -> Tuple[str, int, Dict[str, str]]:
        """
        Handles an incoming call webhook and prepares the appropriate response or
        actions to route the call to a SIP destination.

        This is the core of the abstraction for incoming calls. It hides the
        declarative (TwiML response) vs. imperative (API commands) logic.

        Args:
            call_data: The raw webhook data from the provider.
            destination_sip_uri: The SIP URI to route the call to.
            username: SIP username for authentication.
            password: SIP password for authentication.
            **kwargs: Additional provider-specific options.

        Returns:
            A tuple containing (response_body, status_code, headers).
            - For Twilio (declarative): ("<TwiML>...</TwiML>", 200, {"Content-Type": "application/xml"})
            - For Telnyx (imperative): ("OK", 200, {"Content-Type": "text/plain"})
        """
        pass
    
    @abstractmethod
    async def make_outgoing_call(self, 
                                to_number: str, 
                                webhook_url: str,
                                **kwargs) -> Dict[str, Any]:
        """
        Make an outgoing call using the provider's API.
        
        Returns a standardized dictionary with call details.
        """
        pass
    
    @abstractmethod
    async def hangup_call(self, call_id: str) -> Dict[str, Any]:
        """Hang up an active call."""
        pass
    
    @abstractmethod
    async def transfer_call(self, call_id: str, transfer_to: str) -> Dict[str, Any]:
        """Transfer a call to another number or SIP destination."""
        pass
    
    @abstractmethod
    async def get_call_status(self, call_id: str) -> str:
        """Get the current status of a call."""
        pass
    
    @abstractmethod
    async def send_dtmf(self, call_id: str, digits: str) -> Dict[str, Any]:
        """Send DTMF tones on an active call."""
        pass
    
    def normalize_phone_number(self, number: str) -> str:
        """
        Normalize a phone number to E.164 format.
        
        This is a common utility method used by all providers.
        """
        if not number:
            return number
            
        cleaned = ''.join(c for c in number if c.isdigit() or c == '+')
        
        if not cleaned.startswith('+') and len(cleaned) > 10:
            cleaned = '+' + cleaned
        
        return cleaned


class SIPProviderRegistry:
    """Registry for SIP provider implementations"""
    _providers: Dict[str, Type[SIPProvider]] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[SIPProvider]):
        """Register a SIP provider implementation"""
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered SIP provider: {name}")
    
    @classmethod
    def get_provider(cls, name: str, config: Dict[str, Any]) -> SIPProvider:
        """Get a SIP provider instance by name"""
        provider_name = name.lower()
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown SIP provider: {name}. Available: {list(cls._providers.keys())}")
        
        provider_class = cls._providers[provider_name]
        return provider_class(config)
    
    @classmethod
    def list_providers(cls) -> list:
        """List all available provider names"""
        return list(cls._providers.keys())


from .twilio import TwilioProvider

SIPProviderRegistry.register("twilio", TwilioProvider)

def create_sip_provider(provider_name: str, config: Optional[Dict[str, Any]] = None) -> SIPProvider:
    """
    Create a SIP provider instance
    
    This is the main entry point for creating providers. Just change the provider_name
    and the same code will work with any provider.
    
    Args:
        provider_name: Name of the provider (e.g., "twilio", "telnyx")
        config: Provider-specific configuration
    
    Returns:
        SIPProvider instance
        
    Example:
        # Switch between providers by just changing this line:
        sip = create_sip_provider("twilio", config)  # or "telnyx"
        
        # All other code stays the same:
        result = await sip.make_outgoing_call("+1234567890", webhook_url)
    """
    config = config or {}
    return SIPProviderRegistry.get_provider(provider_name, config)


__all__ = ["SIPProvider", "create_sip_provider", "SIPProviderRegistry", "TwilioProvider"] 