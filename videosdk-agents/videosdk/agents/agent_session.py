from __future__ import annotations

from typing import Any, Optional

from .agent import Agent
from .llm.chat_context import ChatMessage, ChatRole
from .conversation_flow import ConversationFlow, DefaultConversationFlow
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
        conversation_flow: Optional[ConversationFlow] = None,
    ) -> None:
        """
        Initialize an agent session.

        Args:
            agent: Instance of an Agent class that handles the core logic
            conversation_flow: ConversationFlow instance to manage conversation state (optional)
            pipeline: Pipeline instance to process the agent's operations
        """
        self.agent = agent
        self.pipeline = pipeline
        self.conversation_flow = conversation_flow
        self.agent.session = self

        # Auto-create default conversation flow if none provided
        if (
            self.conversation_flow is None
            and hasattr(pipeline, "stt")
            and hasattr(pipeline, "llm")
            and hasattr(pipeline, "tts")
        ):
            self.conversation_flow = DefaultConversationFlow(
                agent=agent,
                stt=getattr(pipeline, "stt", None),
                llm=getattr(pipeline, "llm", None),
                tts=getattr(pipeline, "tts", None),
                vad=getattr(pipeline, "vad", None),
                turn_detector=getattr(pipeline, "turn_detector", None),
            )

        if hasattr(self.pipeline, "set_agent"):
            self.pipeline.set_agent(self.agent)
        if (
            hasattr(self.pipeline, "set_conversation_flow")
            and self.conversation_flow is not None
        ):
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
        await self.agent.initialize_mcp()
        if hasattr(self.pipeline, "set_agent"):
            self.pipeline.set_agent(self.agent)

        await self.pipeline.start()
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
        await self.agent.on_exit()
        await self.pipeline.cleanup()

    async def leave(self) -> None:
        """
        Leave the agent session.
        """
        await self.pipeline.leave()
