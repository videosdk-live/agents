from __future__ import annotations

from typing import Any

from .agent import Agent
from .llm.chat_context import ChatMessage, ChatRole
from .conversation_flow import ConversationFlow
from .pipeline import Pipeline
import os
from .telemetry.videosdk_telemetry import initialize_telemetry, get_telemetry, cleanup_telemetry, VideoSDKTelemetry
from opentelemetry.trace import StatusCode
class AgentSession:
    """
    Manages an agent session with its associated conversation flow and pipeline.
    """
    
    def __init__(
        self,
        agent: Agent,
        pipeline: Pipeline,
        conversation_flow: ConversationFlow,
        context: dict | None = None,
    ) -> None:
        """
        Initialize an agent session.
        
        Args:
            agent: Instance of an Agent class that handles the core logic
            flow: ConversationFlow instance to manage conversation state
            pipeline: Pipeline instance to process the agent's operations
            context: Dictionary containing session context (pid, meetingId, name)
        """
        self.agent = agent
        self.pipeline = pipeline
        self.conversation_flow = conversation_flow
        self.context = context or {}
        self.agent.session = self
        self.telemetry: VideoSDKTelemetry | None = None
        
        if hasattr(self.pipeline, 'set_agent'):
            self.pipeline.set_agent(self.agent)
        if hasattr(self.pipeline, 'set_conversation_flow'):
            self.pipeline.set_conversation_flow(self.conversation_flow)

    async def start(self, **kwargs: Any) -> None:
        """
        Start the agent session.
        This will:
        1. Initialize the agent (including MCP tools if configured)
        2. Call the agent's on_enter hook
        3. Start the pipeline processing
        
        Args:
            **kwargs: Additional arguments to pass to the pipeline start method
        Raises:
        ValueError: If meetingId is not provided in the context
        """
        # Validate context
        if "meetingId" not in self.context:
            if self.context.get("join_meeting") == True:
                raise ValueError("meetingId must be provided in the context")
            
        meeting_id = self.context.get("meetingId")
        name = self.context.get("name", "Agent")
        
        # Initialize telemetry
        if meeting_id:
            metadata = {
                "name": name,
                "agent_type": type(self.agent).__name__,
                "version": "1.0.0"
            }
            self.telemetry = initialize_telemetry(meeting_id, name, "videosdk-agents", metadata)
            self.agent.telemetry = self.telemetry
            if hasattr(self.conversation_flow, 'set_telemetry'):
                self.conversation_flow.set_telemetry(self.telemetry)

        # Start session span
        session_span = None
        if self.telemetry:
            session_span = self.telemetry.trace("Agent Session Start", {
                "agent.type": type(self.agent).__name__,
                "agent.name": name,
                "meeting.id": meeting_id
            })

        try:
            join_meeting = self.context.get("join_meeting", True)
            videosdk_auth = self.context.get("videosdk_auth", None)
            if videosdk_auth is None:
                videosdk_auth = os.getenv("VIDEOSDK_AUTH_TOKEN")
                
            # Handle playground mode
            if "playground" in self.context and self.context.get("playground") == True:
                if videosdk_auth:
                    playground_url = f"https://playground.videosdk.live?token={videosdk_auth}&meetingId={meeting_id}"
                    print(f"\033[1;36m" + "Agent started in playground mode" + "\033[0m")
                    print("\033[1;75m" + "Interact with agent here at:" + "\033[0m")
                    print("\033[1;4;94m" + playground_url + "\033[0m")
                    if self.telemetry:
                        self.telemetry.add_span_attribute(session_span, "playground.enabled", True)
                        self.telemetry.add_span_attribute(session_span, "playground.url", playground_url)
                else:
                    raise ValueError("VIDEOSDK_AUTH_TOKEN environment variable not found")
                     
            # Initialize MCP
            if self.telemetry:
                mcp_span = self.telemetry.trace("MCP Initialization", parent_span=session_span)
            
            await self.agent.initialize_mcp()
            
            if self.telemetry:
                self.telemetry.complete_span(mcp_span, StatusCode.OK, "MCP initialized successfully")
            
            # Set up pipeline
            if hasattr(self.pipeline, 'set_agent'):
                self.pipeline.set_agent(self.agent)
            
            # Start pipeline
            if self.telemetry:
                pipeline_span = self.telemetry.trace("Pipeline Start", parent_span=session_span)
            
            await self.pipeline.start(meeting_id=meeting_id, name=name, videosdk_auth=videosdk_auth, join_meeting=join_meeting)
            
            if self.telemetry:
                self.telemetry.complete_span(pipeline_span, StatusCode.OK, "Pipeline started successfully")
            
            # Agent entry
            if self.telemetry:
                entry_span = self.telemetry.trace("Agent Entry", parent_span=session_span)
            
            await self.agent.on_enter()
            
            if self.telemetry:
                self.telemetry.complete_span(entry_span, StatusCode.OK, "Agent entered successfully")
                self.telemetry.complete_span(session_span, StatusCode.OK, "Session started successfully")
                
        except Exception as e:
            if self.telemetry:
                self.telemetry.complete_span(session_span, StatusCode.ERROR, f"Session start failed: {str(e)}")
            raise
        
    async def say(self, message: str) -> None:
        """
        Send an initial message to the agent.
        """
        if self.telemetry:
            with self.telemetry.span_context("Agent Say Message", {
                "message.length": len(message),
                "agent.type": type(self.agent).__name__
            }):
                self.agent.chat_context.add_message(role=ChatRole.ASSISTANT, content=message)
                await self.pipeline.send_message(message)
        else:
            self.agent.chat_context.add_message(role=ChatRole.ASSISTANT, content=message)
            await self.pipeline.send_message(message)
    
    async def close(self) -> None:
        """
        Close the agent session.
        """
        if self.telemetry:
            with self.telemetry.span_context("Agent Session Close", {
                "agent.type": type(self.agent).__name__
            }):
                # await self.agent.on_exit()
                await self.pipeline.cleanup()
                cleanup_telemetry()
        else:
            # await self.agent.on_exit()
            await self.pipeline.cleanup()
    
    async def leave(self) -> None:
        """
        Leave the agent session.
        """
        if self.telemetry:
            self.telemetry.trace_auto_complete("Agent Session Leave", {
                "agent.type": type(self.agent).__name__
            })
        await self.pipeline.leave()
        