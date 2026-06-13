from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_END_STREAM = "__END_STREAM__"

def prepare_strict_schema(schema_dict):
    if isinstance(schema_dict, dict):
        if schema_dict.get("type") == "object":
            schema_dict["additionalProperties"] = False
            if "properties" in schema_dict:
                all_props = list(schema_dict["properties"].keys())
                schema_dict["required"] = all_props
        
        for key, value in schema_dict.items():
            if isinstance(value, dict):
                prepare_strict_schema(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        prepare_strict_schema(item)
    return schema_dict

class ExtractedField(BaseModel):
    key: str = Field(..., description="The name of the field")
    value: Union[str, int, float, bool] = Field(..., description="The value of the field")


class ConversationalGraphResponse(BaseModel):
    """Structured LLM response format used when a conversational graph is active."""

    response_to_user: str = Field(..., description="Response to the user by agent")
    extracted_values: List[ExtractedField] = Field(
        default_factory=list,
        description="List of extracted values from the user input",
    )


class GraphPipelineAdapter:
    """`ConversationalGraph` instance with the agent pipeline."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph
        self._agent: Any = None
        self._callbacks_bound: bool = False
        self._conversational_graph_schema_cache = None
        self._hangup_handler_registered: bool = False
        self._hangup_fired: bool = False
        self._hangup_timer_task: Optional[asyncio.Task] = None

    @property
    def has_agent(self) -> bool:
        """Check whether an agent is already bound."""
        return self._agent is not None

    def set_agent(self, agent: Any) -> None:
        """Bind *agent* to the graph via callbacks. Rebinds if a different agent is provided."""
        if self._callbacks_bound and self._agent is agent:
            return
        self._agent = agent
        self._callbacks_bound = True
        self._hangup_handler_registered = False
        self._hangup_fired = False

        async def _say_cb(message: str, interruptible: bool = True) -> None:
            if hasattr(agent, "session") and agent.session:
                await agent.session.say(message, interruptible=interruptible)

        async def _ask_cb(instruction: str, interruptible: bool = False) -> None:
            if hasattr(agent, "session") and agent.session:
                await agent.session.reply(
                    instruction, interruptible=interruptible, wait_for_playback=True,
                )

        def _hangup_cb(speech_done: bool = False) -> None:
            self._schedule_hangup(agent, speech_done=speech_done)

        self._graph.set_callbacks(say=_say_cb, ask=_ask_cb, hangup=_hangup_cb)

        try:
            from .llm.chat_context import ChatRole
            agent.chat_context.add_message(
                role=ChatRole.SYSTEM,
                content=agent.instructions,
                replace=True,
            )
        except Exception as exc:
            logger.warning("[GRAPH] could not inject graph system instructions: %s", exc)

        try:
            room = agent.session._job_context.room
            sid = getattr(room, "_session_id", None)
            if sid:
                self._graph._session_id = sid
        except AttributeError:
            pass

    def _schedule_hangup(self, agent: Any, speech_done: bool = False) -> None:
        """Hang up exactly when the final TTS stream ends."""
        if speech_done:
            self._start_hangup_timer(agent)
            return

        pipeline = (
            agent.session.pipeline
            if hasattr(agent, "session") and agent.session and hasattr(agent.session, "pipeline")
            else None
        )

        if pipeline and hasattr(pipeline, "on"):
            if self._hangup_handler_registered:
                return
            self._hangup_handler_registered = True

            async def _on_tts_end():
                if self._hangup_fired:
                    return
                self._hangup_fired = True
                if self._hangup_timer_task and not self._hangup_timer_task.done():
                    self._hangup_timer_task.cancel()
                if hasattr(agent, "hangup"):
                    try:
                        await agent.hangup()
                    except Exception as exc:
                        logger.warning("[GRAPH] agent.hangup() failed: %s", exc)

            pipeline.on("agent_turn_end")(_on_tts_end)
            self._start_hangup_timer(agent, delay=30.0, skip_if_speaking=True)
        else:
            self._start_hangup_timer(agent)

    def _start_hangup_timer(
        self, agent: Any, delay: float | None = None, skip_if_speaking: bool = False
    ) -> None:
        if self._hangup_timer_task and not self._hangup_timer_task.done():
            self._hangup_timer_task.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("[GRAPH] no running event loop — cannot schedule hangup timer")
            return
        self._hangup_timer_task = loop.create_task(
            self._hangup_after_timer(agent, delay=delay, skip_if_speaking=skip_if_speaking)
        )

    async def _hangup_after_timer(
        self, agent: Any, delay: float | None = None, skip_if_speaking: bool = False
    ) -> None:
        if delay is None:
            delay = getattr(self._graph.config, "hangup_delay", 4.0)
        await asyncio.sleep(delay)
        if self._hangup_fired:
            return
        if skip_if_speaking:
            session = getattr(agent, "session", None)
            utterance = getattr(session, "current_utterance", None) if session else None
            if utterance is not None and not utterance.done():
                return
        self._hangup_fired = True
        if hasattr(agent, "hangup"):
            try:
                await agent.hangup()
            except Exception as exc:
                logger.warning("[GRAPH] agent.hangup() failed: %s", exc)

    def get_system_instructions(self) -> str:
        """Return graph-specific system instructions to prepend to the agent prompt."""
        return self._graph.get_system_instructions()


    async def handle_input(self, text: str) -> tuple[str | None, bool]:
        """ Pre-process user text through the graph. """
        result = await self._graph.handle_input(text)
        if result == _END_STREAM:
            return None, True
        return result, False

    async def handle_decision(
        self,
        agent: Any,
        graph_response: Union[str, dict, ConversationalGraphResponse, Any],
    ) -> tuple[str | None, bool]:
        """Post-process the LLM's structured response through the graph."""
        if not self._callbacks_bound and agent:
            self.set_agent(agent)

        result = await self._graph.handle_decision(graph_response)
        if result == _END_STREAM:
            return None, True
        return result, False

    def stream_conversational_graph_response(
        self, current_content: str, state: Dict[str, Any]
    ) -> Iterator[str]:
        """Stream parsed ``response_to_user`` characters for TTS."""
        return self._graph.stream_conversational_graph_response(current_content, state)

    @staticmethod
    def get_response_schema() -> dict:
        """Return the JSON schema that LLM providers use as ``response_format``."""
        return ConversationalGraphResponse.model_json_schema()

    def _get_graph_schema(self):
        """Get the prepared strict schema from the graph adapter, cached after first call."""
        if self._conversational_graph_schema_cache is None:
            self._conversational_graph_schema_cache = prepare_strict_schema(
                copy.deepcopy(self.get_response_schema())
            )
        return self._conversational_graph_schema_cache
