from __future__ import annotations

from typing import Any

from .agent import Agent
from .llm.chat_context import ChatMessage, ChatRole
from .conversation_flow import ConversationFlow
from .pipeline import Pipeline
import os
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
        if "meetingId" not in self.context:
            if  self.context.get("join_meeting") == True:
                raise ValueError("meetingId must be provided in the context")
            
        meeting_id = self.context.get("meetingId")
        name = self.context.get("name", "Agent")
        join_meeting = self.context.get("join_meeting",True)
        videosdk_auth = self.context.get("videosdk_auth",None)
        if videosdk_auth is None:
            videosdk_auth = os.getenv("VIDEOSDK_AUTH_TOKEN")
        if "playground" in self.context and self.context.get("playground") == True:
                if videosdk_auth:
                    playground_url = f"https://playground.videosdk.live?token={videosdk_auth}&meetingId={meeting_id}"
                    print(f"\033[1;36m" + "Agent started in playground mode" + "\033[0m")
                    print("\033[1;75m" + "Interact with agent here at:" + "\033[0m")
                    print("\033[1;4;94m" + playground_url + "\033[0m")
                else:
                    raise ValueError("VIDEOSDK_AUTH_TOKEN environment variable not found")
             
        # Initialize the agent (including MCP tools if configured)
        await self.agent.initialize_mcp()
        if hasattr(self.pipeline, 'set_agent'):
            self.pipeline.set_agent(self.agent)
        
        await self.pipeline.start(meeting_id=meeting_id, name=name, videosdk_auth=videosdk_auth, join_meeting=join_meeting)
        await self.agent.on_enter()
        
    async def say(self, message: str) -> None:
        """
        Send an initial message to the agent.
        """
        self.agent.chat_context.add_message(role=ChatRole.ASSISTANT, content=message)
        await self.pipeline.send_message(message)
    
    async def close(self) -> None:
        """
        Close the agent session.
        """
        # await self.agent.on_exit()
        await self.pipeline.cleanup()
    
    async def leave(self) -> None:
        """
        Leave the agent session.
        """
        await self.pipeline.leave()
        