from typing import Dict, Any, Tuple
import logging
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Dial
from . import SIPProvider

logger = logging.getLogger(__name__)


class TwilioProvider(SIPProvider):
    """Twilio implementation of the SIPProvider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = Client(config.get("account_sid"), config.get("auth_token"))
        self.from_number = config.get("phone_number")

    def validate_config(self) -> None:
        if not all(
            k in self.config for k in ["account_sid", "auth_token", "phone_number"]
        ):
            raise ValueError(
                "Twilio config requires 'account_sid', 'auth_token', and 'phone_number'"
            )

    async def make_outgoing_call(
        self, to_number: str, webhook_url: str, **kwargs
    ) -> Dict[str, Any]:
        logger.info(f"Making Twilio call from {self.from_number} to {to_number}")
        call = self.client.calls.create(
            to=to_number, from_=self.from_number, url=webhook_url
        )
        return {"sid": call.sid, "status": call.status}

    async def handle_incoming_and_route(
        self, call_data: Dict[str, Any], destination_sip_uri: str, username: str, password: str, **kwargs
    ) -> Tuple[str, int, Dict[str, str]]:
        response = VoiceResponse()
        response.say("Please wait a moment while we connect you.", voice='alice')
        dial = Dial(answer_on_bridge=True)
        dial.sip(destination_sip_uri, username=username, password=password)
        response.append(dial)
        xml_response = str(response)
        logger.info(f"Responding with TwiML for incoming call: {xml_response}")
        return xml_response, 200, {"Content-Type": "application/xml"}

    async def hangup_call(self, call_id: str) -> Dict[str, Any]:
        logger.info(f"Hanging up Twilio call: {call_id}")
        call = self.client.calls(call_id).update(status="completed")
        return {"sid": call.sid, "status": call.status}

    async def transfer_call(self, call_id: str, transfer_to: str) -> Dict[str, Any]:
        """
        Transfer an active call to another phone number or SIP URI using Twilio's <Dial> verb.
        transfer_to must be provided by the caller (agent/session).
        """
        from twilio.twiml.voice_response import VoiceResponse
        logger.info(f"Transferring Twilio call {call_id} to {transfer_to}")
        response = VoiceResponse()
        dial = response.dial()
        if transfer_to.startswith("+"):
            dial.number(transfer_to)
        else:
            dial.sip(transfer_to)
        twiml_str = str(response)
        call = self.client.calls(call_id).update(twiml=twiml_str)
        logger.info(f"Transfer initiated for call {call_id} to {transfer_to}, status: {call.status}")
        return {"sid": call.sid, "status": call.status}

    async def get_call_status(self, call_id: str) -> str:
        call = self.client.calls(call_id).fetch()
        return call.status

    async def send_dtmf(self, call_id: str, digits: str) -> Dict[str, Any]:
        raise NotImplementedError("Twilio send_dtmf is not yet implemented.")
