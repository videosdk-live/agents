from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Iterator, List, Union

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

    @property
    def has_agent(self) -> bool:
        """Check whether an agent is already bound."""
        return self._agent is not None

    def set_agent(self, agent: Any) -> None:
        """Bind *agent* to the graph via callbacks"""
        if self._callbacks_bound:
            return
        self._agent = agent
        self._callbacks_bound = True

        async def _say_cb(message: str, interruptible: bool = True) -> None:
            if hasattr(agent, "session") and agent.session:
                await agent.session.say(message, interruptible=interruptible)

        async def _ask_cb(instruction: str, interruptible: bool = False) -> None:
            if hasattr(agent, "session") and agent.session:
                await agent.session.reply(
                    instruction, interruptible=interruptible, wait_for_playback=False,
                )

        def _hangup_cb() -> None:
            self._schedule_hangup(agent)

        self._graph.set_callbacks(say=_say_cb, ask=_ask_cb, hangup=_hangup_cb)
        try:
            room = agent.session._job_context.room
            sid = getattr(room, "_session_id", None)
            if sid:
                self._graph._session_id = sid
        except Exception:
            pass

    def _schedule_hangup(self, agent: Any) -> None:
        """Hang up exactly when the final TTS stream ends."""
        pipeline = (
            agent.session.pipeline
            if hasattr(agent, "session") and agent.session and hasattr(agent.session, "pipeline")
            else None
        )

        if pipeline and hasattr(pipeline, "on"):
            fired = False

            async def _on_tts_end():
                nonlocal fired
                if fired:
                    return
                fired = True
                if hasattr(agent, "hangup"):
                    try:
                        await agent.hangup()
                    except Exception as exc:
                        logger.warning("[GRAPH] agent.hangup() failed: %s", exc)

            pipeline.on("agent_turn_end")(_on_tts_end)
        else:
            asyncio.ensure_future(self._hangup_after_timer(agent))

    async def _hangup_after_timer(self, agent: Any) -> None:
        delay = getattr(self._graph.config, "hangup_delay", 4.0)
        await asyncio.sleep(delay)
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

    async def handle_decision(self, agent: Any, graph_response: str) -> tuple[str | None, bool]:
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
            self._conversational_graph_schema_cache = prepare_strict_schema(self.get_response_schema())
        return self._conversational_graph_schema_cache
