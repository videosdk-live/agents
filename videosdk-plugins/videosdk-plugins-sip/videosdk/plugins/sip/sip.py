import asyncio
import logging
import httpx
import os
import functools
from typing import Dict, Any, Optional, Callable, Type, AsyncIterator
from fastapi import HTTPException
from .providers import create_sip_provider, SIPProvider
from videosdk.agents import Agent, AgentSession, JobContext, RoomOptions, WorkerJob, RealTimePipeline, CascadingPipeline, ConversationFlow, ChatRole
from videosdk import PubSubSubscribeConfig

logger = logging.getLogger(__name__)

def on_pubsub_message(message):
    logger.info(f"Pubsub message received: {message}")

class DefaultConversationFlow(ConversationFlow):
    """Default conversation flow for cascading pipelines in SIP calls."""
    
    def __init__(self, agent, stt=None, llm=None, tts=None):
        super().__init__(agent, stt, llm, tts)

    async def run(self, transcript: str) -> AsyncIterator[str]:
        """Main conversation loop: handle a user turn."""
        await self.on_turn_start(transcript)
        
        async for response_chunk in self.process_with_llm():
            yield response_chunk

        await self.on_turn_end()

    async def on_turn_start(self, transcript: str) -> None:
        """Called at the start of a user turn."""
        self.is_turn_active = True

    async def on_turn_end(self) -> None:
        """Called at the end of a user turn."""
        self.is_turn_active = False

async def _agent_entrypoint(ctx: JobContext, agent_class: Type[Agent], pipeline: Callable, agent_config: dict):
    """The generic entrypoint for any agent job."""
    room_id = ctx.room_options.room_id
    session: Optional[AgentSession] = None

    try:
        pipeline_instance = pipeline()
        agent = agent_class(ctx=ctx)
        
        # Check if we need a conversation flow for CascadingPipeline
        conversation_flow = None
        if isinstance(pipeline_instance, CascadingPipeline):
            conversation_flow = DefaultConversationFlow(agent)
            logger.info(f"[{room_id}] Using DefaultConversationFlow for CascadingPipeline")
        
        session = AgentSession(
            agent=agent, 
            pipeline=pipeline_instance,
            conversation_flow=conversation_flow
        )

        await ctx.connect()
        await session.start()
        await ctx.room.wait_for_participant()
        await agent.greet_user()

        if agent_config.get("enable_pubsub", False):
            await ctx.room.subscribe_to_pubsub(
                PubSubSubscribeConfig(topic="CHAT_MESSAGE", cb=on_pubsub_message)
            )
        await asyncio.Event().wait()

    except Exception as e:
        logger.error(f"[{room_id}] EXCEPTION in agent job: {e}", exc_info=True)
    finally:
        logger.info(f"[{room_id}] Agent job ending. Cleaning up...")
        if session:
            await session.close()
        await ctx.shutdown()

def _make_context(room_id: str, room_name: str, call_id: Optional[str] = None) -> JobContext:
    ctx = JobContext(room_options=RoomOptions(room_id=room_id, name=room_name, playground=True))
    if call_id:
        ctx.call_id = call_id
    return ctx


def launch_agent_job(
    room_id: str,
    agent_class: Type[Agent],
    pipeline: Callable,
    agent_config: Optional[Dict[str, Any]] = None,
    call_id: Optional[str] = None,
) -> WorkerJob:
    """Creates and starts a WorkerJob using a pipeline factory."""
    if agent_config is None:
        agent_config = {}

    entrypoint_partial = functools.partial(
        _agent_entrypoint,
        agent_class=agent_class,
        pipeline=pipeline,
        agent_config=agent_config
    )
    context_factory_partial = functools.partial(
        _make_context,
        room_id=room_id,
        room_name=agent_config.get("room_name", "AI Call"),
        call_id=call_id
    )
    job = WorkerJob(entrypoint=entrypoint_partial, jobctx=context_factory_partial)
    job.start()
    return job

class VideoSDKMeeting:
    """Service for managing VideoSDK rooms and operations."""
    def __init__(self, auth_token: str):
        self.auth_token = auth_token
        self.base_url = "https://api.videosdk.live/v2"

    async def create_room(self) -> str:
        url = f"{self.base_url}/rooms"
        headers = {"Authorization": self.auth_token}
        payload = {}
        region = os.getenv("VIDEOSDK_REGION")
        if region:
            payload["geoFence"] = region
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                room_id = response.json().get("roomId")
                if not room_id: raise ValueError("roomId not found")
                return room_id
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error creating room: {e.response.status_code} - {e.response.text}")
                raise HTTPException(status_code=500, detail="Failed to create VideoSDK room.")

    def get_sip_endpoint(self, room_id: str) -> str:
        return f"sip:{room_id}@sip.videosdk.live"

    def get_sip_credentials(self) -> Dict[str, str]:
        username = os.getenv("VIDEOSDK_SIP_USERNAME")
        password = os.getenv("VIDEOSDK_SIP_PASSWORD")
        if not username or not password:
            raise ValueError("VIDEOSDK_SIP_USERNAME and VIDEOSDK_SIP_PASSWORD must be set")
        return {"username": username, "password": password}

class SIPManager:
    """Unified SIP management interface."""
    def __init__(self, provider: SIPProvider, videosdk_token: str):
        self.provider = provider
        self.videosdk_token = videosdk_token
        self.base_url: Optional[str] = None
        self.meeting_service = VideoSDKMeeting(auth_token=videosdk_token)
        self.active_sessions: Dict[str, WorkerJob] = {}
        logger.info(f"SIP Manager initialized with provider: {provider.__class__.__name__}")

    def set_base_url(self, base_url: str):
        if "?" in base_url: 
            base_url = base_url.split("?")[0]
        self.base_url = base_url
        logger.info(f"Base URL set: {self.base_url}")

    async def make_call(
        self,
        to_number: str,
        agent_class: Type[Agent],
        pipeline: Callable,
        agent_config: Optional[Dict[str, Any]] = None
    ):
        try:
            room_id = await self.meeting_service.create_room()
            webhook_url = f"{self.base_url}/sip/answer/{room_id}"
            # Make the outgoing call and get the call_id (sid)
            result = await self.provider.make_outgoing_call(to_number=to_number, webhook_url=webhook_url)
            call_id = result.get("sid")
            agent_job = launch_agent_job(
                room_id=room_id,
                agent_class=agent_class,
                pipeline=pipeline,
                agent_config=agent_config,
                call_id=call_id
            )
            self.active_sessions[room_id] = agent_job
            await asyncio.sleep(1)
            result["room_id"] = room_id
            return result
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def handle_incoming_call(
        self,
        webhook_data: Dict[str, Any],
        agent_class: Type[Agent],
        pipeline: Callable,
        agent_config: Optional[Dict[str, Any]] = None
    ) -> tuple:
        try:
            room_id = await self.meeting_service.create_room()
            call_id = webhook_data.get("CallSid")
            caller_number = webhook_data.get("From")
            called_number = webhook_data.get("To")
            if agent_config is None:
                agent_config = {}
            agent_config['caller_number'] = caller_number
            agent_config['called_number'] = called_number
            agent_config['call_id'] = call_id
            agent_job = launch_agent_job(
                room_id=room_id,
                agent_class=agent_class,
                pipeline=pipeline,
                agent_config=agent_config,
                call_id=call_id
            )
            self.active_sessions[room_id] = agent_job
            sip_endpoint = self.meeting_service.get_sip_endpoint(room_id)
            sip_creds = self.meeting_service.get_sip_credentials()
            return await self.provider.handle_incoming_and_route(
                call_data=webhook_data,
                destination_sip_uri=sip_endpoint,
                username=sip_creds["username"],
                password=sip_creds["password"],
            )
        except Exception as e:
            logger.error(f"Error handling incoming call: {e}", exc_info=True)
            return "An error occurred", 500, {"Content-Type": "text/plain"}

    def get_sip_response_for_room(self, room_id: str) -> tuple:
        try:
            from twilio.twiml.voice_response import VoiceResponse, Dial
            sip_endpoint = self.meeting_service.get_sip_endpoint(room_id)
            sip_creds = self.meeting_service.get_sip_credentials()
            response = VoiceResponse()
            dial = Dial(answer_on_bridge=True)
            dial.sip(sip_endpoint, username=sip_creds["username"], password=sip_creds["password"])
            response.append(dial)
            return str(response), 200, {"Content-Type": "application/xml"}
        except Exception as e:
            logger.error(f"Error generating SIP response for room {room_id}: {e}")
            return "An error occurred", 500, {"Content-Type": "text/plain"}

    def get_active_sessions(self) -> Dict[str, Any]:
        return {"total_sessions": len(self.active_sessions), "room_ids": list(self.active_sessions.keys())}

def create_sip_manager(
    provider: str,
    videosdk_token: Optional[str] = None,
    provider_config: Optional[Dict[str, Any]] = None,
) -> SIPManager:
    videosdk_token = videosdk_token or os.getenv("VIDEOSDK_AUTH_TOKEN")
    if not videosdk_token:
        raise ValueError("videosdk_token must be provided or VIDEOSDK_AUTH_TOKEN environment variable must be set")
    
    sip_provider = create_sip_provider(provider, provider_config)
    return SIPManager(provider=sip_provider, videosdk_token=videosdk_token)
