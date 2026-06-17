from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..agent import Agent
from ..event_bus import global_event_emitter
from ..job import (
    JobContext,
    RoomOptions,
    _reset_current_job_context,
    _set_current_job_context,
)
from ..llm.chat_context import ChatContext, ChatRole
from ..utils import function_tool, generate_videosdk_token
from ..utterance_handle import UtteranceHandle
from ..voice_mail_detector import VoiceMailDetector

if TYPE_CHECKING:
    from ..agent_session import AgentSession
    from ..llm.llm import LLM
    from ..pipeline import Pipeline

logger = logging.getLogger(__name__)


_DEFAULT_SUMMARY_PROMPT = (
    "You summarize phone calls for a human supervisor who is about to take over. "
    "Produce a concise briefing under 150 words covering: (1) caller identity if "
    "known, (2) their core issue, (3) what the primary agent has already done or "
    "tried, (4) what the supervisor needs to resolve or decide. Do not greet the "
    "supervisor and do not add preamble — output only the briefing."
)
_FALLBACK_SUMMARY = "Caller escalated; no conversation summary is available."
_FAIL_APOLOGY = "I ran into a problem transferring you. Let me keep helping you."

_AGENT_INSTRUCTIONS = (
    "You are an AI assistant joining a private consultation room to brief a human "
    "supervisor about a caller that the primary agent is handing off. When the "
    "supervisor arrives you will narrate the call summary clearly and concisely, "
    "then ask whether they are ready to take the call.\n\n"
    "CRITICAL: while you are delivering the briefing and asking your question, do "
    "NOT call any tool. Call a tool ONLY in response to what the supervisor "
    "actually says back to you — never in the same turn as the narration.\n\n"
    "Once the supervisor responds, call exactly ONE tool:\n"
    "  • Supervisor gives ANY affirmative ('yes', 'ready', 'go ahead', 'ok', "
    "    'I'm here', etc.) → call `done_briefing`.\n"
    "  • Supervisor explicitly declines ('no', 'I can't take this', 'not now', "
    "    etc.) → call `decline_transfer` with their reason.\n"
    "  • You hear an automated voicemail / answering-machine greeting → call "
    "    `voicemail_detected` immediately, do not narrate the summary.\n"
    "Do not chit-chat, ask out-of-scope questions, or act as a generic assistant."
)

_BRIEFING_APOLOGIES = {
    "declined": "The supervisor isn't able to take this call right now; let me keep helping you.",
    "voicemail": "I couldn't reach a live supervisor; let me keep helping you.",
    "supervisor_disconnected": "I lost the supervisor before we could connect you; let me keep helping you.",
}

_DEFAULT_BRIEFING_APOLOGY = "I couldn't complete the transfer; let me keep helping you."


class WarmTransferError(Exception):
    """Raised when a warm transfer cannot be performed or fails mid-flight."""


class WarmTransferPhase(str, Enum):
    """Lifecycle phase of a warm transfer, emitted on the ``warm_transfer`` event."""

    STARTED = "started"
    SUMMARY_GENERATING = "summary_generating"
    SUMMARY_READY = "summary_ready"
    CALLER_ON_HOLD = "caller_on_hold"
    CONSULTATION_ROOM_CREATED = "consultation_room_created"
    SUPERVISOR_DIALED = "supervisor_dialed"
    SUPERVISOR_JOINED = "supervisor_joined"
    BRIEFING_STARTED = "briefing_started"
    BRIEFING_COMPLETE = "briefing_complete"
    CALL_SWITCHED = "call_switched"
    TRANSFER_COMPLETE = "transfer_complete"
    TRANSFER_CANCELLED = "transfer_cancelled"
    TRANSFER_FAILED = "transfer_failed"


@dataclass
class SIPDestination:
    """The human supervisor, reached by dialing out over SIP to ``sip_call_to``.

    The outbound call is placed via ``POST /v2/sip/call`` and terminates at the
    consultation room so the agent and the human arrive together.

    Required:
      * ``routing_rule_id`` — the SIP routing rule to dial through (``routingRuleId``).
      * ``sip_call_to``     — the supervisor's number (``sipCallTo``).
      * ``sip_call_from``   — the caller-ID to present (``sipCallFrom``); must be
        a number the routing rule / trunk is authorised to send, otherwise the
        carrier may reject or drop the call.

    ``extra_options`` is forwarded verbatim to the request body
    (``recordAudio``, ``ringingTimeout``, ``headers``, ``metadata``, …);
    ``routingRuleId``, ``sipCallTo``, ``sipCallFrom`` and ``destinationRoomId``
    are set by the runner — don't put them in ``extra_options``.
    """

    routing_rule_id: str
    sip_call_to: str
    sip_call_from: str
    participant_display_name: str = "Supervisor"
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class WarmTransferConfig:
    """Configuration for a SIP-to-SIP warm transfer.

    Only ``destination`` is required. ``summary_llm`` / ``summary_prompt``
    customize the call summary; the timeouts default to sane values.

    The consultation room always runs the built-in :class:`WarmTransferAgent`.
    By default its pipeline is auto-built by re-instantiating the primary
    session's STT/LLM/TTS/VAD/turn-detector classes with no arguments — this
    works when those providers read their credentials from the environment
    (the common case). It deliberately does **not** reuse the primary session's
    live component instances: the briefing session runs concurrently with the
    primary one, so a shared VAD/STT/TTS would corrupt both sessions and
    tearing the briefing session down would kill the primary's components.
    If your providers need constructor arguments (API keys, model/voice ids,
    …), pass ``briefing_pipeline_factory`` to build the consultation pipeline
    explicitly, e.g. ``lambda: Pipeline(stt=DeepgramSTT(model="nova-3"), …)``.
    """

    destination: SIPDestination
    summary_llm: Optional["LLM"] = None
    summary_prompt: Optional[str] = None
    briefing_pipeline_factory: Optional[Callable[[], "Pipeline"]] = None
    supervisor_join_timeout: float = 120.0
    briefing_timeout: float = 180.0


@dataclass
class WarmTransferResult:
    """Result returned by ``AgentSession.warm_transfer``."""

    success: bool
    phase: WarmTransferPhase
    consultation_room_id: Optional[str]
    supervisor_call_id: Optional[str]
    switched_call_id: Optional[str]
    summary: str
    error: Optional[str] = None

def _mint_participant_token(role_prefix: str, fallback_token: Optional[str] = None) -> tuple[str, str]:
    """Return ``(participant_id, jwt)``; mint from API key/secret, else use ``fallback_token``."""
    pid = f"{role_prefix}_{uuid.uuid4().hex[:8]}"
    try:
        token = generate_videosdk_token()
    except (ValueError, ImportError) as exc:
        if fallback_token:
            logger.info(
                "WarmTransfer: VIDEOSDK_API_KEY/SECRET not set; using the session "
                "auth token for the %s participant token", role_prefix
            )
            return pid, fallback_token
        raise WarmTransferError(str(exc)) from exc
    return pid, token


def _serialize_history(items: list[dict]) -> str:
    """Render the dict-form history from ``get_context_history`` as a transcript."""
    speakers = {"user": "Caller", "assistant": "Agent", "system": "System"}
    lines: list[str] = []
    for item in items:
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, list):
            text = " ".join(p for p in content if isinstance(p, str))
        elif isinstance(content, str):
            text = content
        else:
            text = "" if content is None else str(content)
        if text:
            role = item.get("role", "user")
            lines.append(f"{speakers.get(role, role)}: {text}")
    return "\n".join(lines)


async def _summarize_history(
    llm: "LLM", history_items: list[dict], custom_prompt: Optional[str] = None
) -> str:
    """One-shot LLM summary; retry once, then fall back to a static string."""
    conversation = _serialize_history(history_items)
    if not conversation.strip():
        return _FALLBACK_SUMMARY
    for attempt in (1, 2):
        try:
            ctx = ChatContext.empty()
            ctx.add_message(role=ChatRole.SYSTEM, content=custom_prompt or _DEFAULT_SUMMARY_PROMPT)
            ctx.add_message(role=ChatRole.USER, content=f"Here is the call transcript. Summarize it.\n\n{conversation}")
            chunks: list[str] = []
            async for chunk in llm.chat(ctx):
                if chunk and chunk.content:
                    chunks.append(chunk.content)
            summary = "".join(chunks).strip()
            if summary:
                return summary
            raise RuntimeError("LLM returned an empty summary")
        except Exception as exc:  
            logger.warning(
                f"WarmTransfer: summary attempt {attempt} failed ({exc}); "
                f"{'retrying' if attempt == 1 else 'using fallback'}"
            )
    return _FALLBACK_SUMMARY


def _participant_in_room(participant, room) -> bool:
    return bool(participant and room and participant.id in getattr(room, "participants_data", {}))

def _is_agent_name(participant) -> bool:
    return "agent" in (getattr(participant, "display_name", "") or "").lower()

class WarmTransferAgent(Agent):
    """The built-in consultation agent that briefs the human supervisor.

    Used internally by :class:`WarmTransferRunner`. The runner drives three
    :class:`asyncio.Event` attributes on it:

      * ``supervisor_ready_event`` — set once the supervisor's audio is live;
        ``on_enter`` awaits it so the narration isn't delivered into a ringing
        phone.
      * ``briefing_done_event`` — set by the ``done_briefing`` tool when the
        supervisor confirms they're ready.
      * ``briefing_failed_event`` — set by the ``decline_transfer`` /
        ``voicemail_detected`` tools, the framework's VoiceMailDetector
        classifier, or the supervisor disconnecting mid-call;
        ``_briefing_failure_reason`` carries the reason (``declined`` /
        ``voicemail`` / ``supervisor_disconnected``).
    """

    def __init__(self, summary: str, instructions: Optional[str] = None):
        super().__init__(instructions=instructions or _AGENT_INSTRUCTIONS)
        self._summary = summary
        self.supervisor_ready_event: asyncio.Event = asyncio.Event()
        self.briefing_done_event: asyncio.Event = asyncio.Event()
        self.briefing_failed_event: asyncio.Event = asyncio.Event()
        self._briefing_failure_reason: Optional[str] = None

    async def on_enter(self) -> None:
        self.chat_context.add_message(
            role=ChatRole.SYSTEM, content=f"CALL SUMMARY TO BRIEF SUPERVISOR:\n{self._summary}"
        )
        logger.info("WarmTransferAgent: waiting for supervisor to join the consultation room…")
        try:
            await self.supervisor_ready_event.wait()
        except asyncio.CancelledError:
            return
        logger.info("WarmTransferAgent: supervisor joined — starting briefing narration")
        try:
            await self.session.say("Hi, I'm the AI assistant. Let me bring you up to speed on this call.")
            await self.session.reply(
                instructions=(
                    "Narrate the call summary to the supervisor clearly and concisely (no preamble), "
                    "then ask whether they're ready to take the call. Do NOT call any tool in this "
                    "message — wait for the supervisor to reply first, then apply the decision rules "
                    "from your system prompt."
                ),
                wait_for_playback=False,
            )
        except Exception as exc:
            logger.error(f"WarmTransferAgent: error while briefing supervisor: {exc}")

    async def on_exit(self) -> None:
        for evt in (self.supervisor_ready_event, self.briefing_done_event, self.briefing_failed_event):
            if not evt.is_set():
                evt.set()

    def _fail_briefing(self, reason: str) -> None:
        self._briefing_failure_reason = reason
        self.briefing_failed_event.set()

    @function_tool
    async def done_briefing(self) -> str:
        """Signal that the supervisor confirmed they're ready to take the call."""
        self._briefing_failure_reason = None
        self.briefing_done_event.set()
        return "Connecting the caller now. Stand by."

    @function_tool
    async def decline_transfer(self, reason: str) -> str:
        """Record that the supervisor declined to take the call.
        Args:
            reason: A short explanation of why the supervisor declined.
        """
        logger.info(f"WarmTransferAgent: supervisor declined transfer — reason={reason!r}")
        self._fail_briefing("declined")
        return "Understood. The caller will be kept with the primary agent."

    @function_tool
    async def voicemail_detected(self) -> str:
        """Call this when the consultation line reached voicemail / an answering machine."""
        logger.info("WarmTransferAgent: voicemail detected on supervisor leg")
        self._fail_briefing("voicemail")
        return "Voicemail detected; aborting transfer."

class WarmTransferRunner:
    """Runs the SIP-to-SIP warm-transfer state machine for one :class:`AgentSession`.

    Emits ``warm_transfer`` on the primary session (and ``WARM_TRANSFER`` on the
    global event bus) at every transition.
    """

    def __init__(self, session: "AgentSession", config: WarmTransferConfig) -> None:
        self._session = session
        self._config = config
        self._consultation_room_id: Optional[str] = None
        self._supervisor_call_id: Optional[str] = None
        self._caller_call_id: Optional[str] = None
        self._summary: str = ""
        self._briefing_ctx: Optional[JobContext] = None
        self._briefing_session: Optional["AgentSession"] = None
        self._briefing_start_task: Optional[asyncio.Task] = None
        self._primary_input_was_enabled: Optional[bool] = None
        self._primary_auto_end_was_enabled: Optional[bool] = None
        self._participant_left_handler: Optional[Any] = None
        self._consultation_known_before_switch: Optional[set] = None

    # ── Public entry ───────────────────────────────────────────────────

    async def run(self) -> WarmTransferResult:
        await self._emit(WarmTransferPhase.STARTED, {"start_time": time.time()})
        try:
            self._mute_primary_input()
            self._suspend_primary_auto_end()
            self._validate_caller_and_cache_callid()
            self._summary = await self._generate_summary()
            await self._place_caller_on_hold()

            self._consultation_room_id = await self._create_consultation_room()
            await self._emit(
                WarmTransferPhase.CONSULTATION_ROOM_CREATED,
                {"consultation_room_id": self._consultation_room_id},
            )

            await self._spawn_briefing_session()
            await self._dial_supervisor()

            if not await self._wait_for_supervisor():
                return await self._abort(
                    "supervisor_join_timeout",
                    apology="Sorry, I couldn't reach a supervisor; let me continue helping you.",
                    phase=WarmTransferPhase.TRANSFER_CANCELLED,
                )

            ev = getattr(self._briefing_session.agent, "supervisor_ready_event", None)
            if ev is not None:
                ev.set()
            await self._emit(WarmTransferPhase.SUPERVISOR_JOINED, {})

            outcome = await self._wait_for_briefing_complete()
            if outcome["status"] != "ready":
                return await self._abort(
                    outcome["status"], apology=outcome["apology"], phase=WarmTransferPhase.TRANSFER_CANCELLED
                )

            if not await self._switch_caller_to_consultation_room():
                return await self._abort(
                    "switch_failed",
                    apology="I couldn't complete the transfer; please try again in a moment.",
                    phase=WarmTransferPhase.TRANSFER_FAILED,
                )

            await self._wait_for_caller_in_consultation_room()

            await self._close_briefing()
            with suppress(Exception):
                await self._session.leave()

            await self._emit(WarmTransferPhase.TRANSFER_COMPLETE, {})

            ctx = getattr(self._session, "_job_context", None)
            if ctx is not None and hasattr(ctx, "shutdown"):
                with suppress(Exception):
                    asyncio.create_task(ctx.shutdown())

            return WarmTransferResult(
                success=True,
                phase=WarmTransferPhase.TRANSFER_COMPLETE,
                consultation_room_id=self._consultation_room_id,
                supervisor_call_id=self._supervisor_call_id,
                switched_call_id=self._caller_call_id,
                summary=self._summary,
            )
        except asyncio.CancelledError:
            self._restore_primary_input()
            self._restore_primary_auto_end()
            raise
        except WarmTransferError as exc:
            await self._abort(str(exc), apology=_FAIL_APOLOGY, phase=WarmTransferPhase.TRANSFER_FAILED)
            raise
        except Exception as exc:
            logger.exception("Warm transfer failed with unexpected error")
            await self._abort(str(exc), apology=_FAIL_APOLOGY, phase=WarmTransferPhase.TRANSFER_FAILED)
            raise WarmTransferError(str(exc)) from exc

    def _validate_caller_and_cache_callid(self) -> None:
        room = getattr(getattr(self._session, "_job_context", None), "room", None)
        if room is None:
            raise WarmTransferError("Primary session has no active room; cannot perform warm transfer.")
        sip_manager = getattr(room, "sip_manager", None)
        session_id = getattr(room, "_session_id", None)
        if sip_manager is None or not session_id:
            raise WarmTransferError(
                "Primary room has no SIP manager / session id — warm transfer only works for SIP calls."
            )
        caller_call_id: Optional[str] = None
        try:
            info = sip_manager.fetch_call_info(session_id)
            if info:
                caller_call_id = info.get("callId")
        except Exception as exc:  
            logger.warning(f"WarmTransfer: fetch_call_info failed: {exc}")
        if caller_call_id is None:
            raise WarmTransferError(
                "Caller is not a SIP participant — warm transfer requires an active SIP (telephony) call."
            )
        self._caller_call_id = caller_call_id

    async def _generate_summary(self) -> str:
        await self._emit(WarmTransferPhase.SUMMARY_GENERATING, {})
        llm = self._config.summary_llm
        if llm is None:
            pipeline = self._session.pipeline
            if getattr(getattr(pipeline, "config", None), "is_realtime", False):
                logger.warning(
                    "WarmTransfer: primary pipeline is realtime and no summary_llm "
                    "was provided; a realtime model cannot summarize text. Using the "
                    "fallback summary — pass WarmTransferConfig.summary_llm=<a text LLM> "
                    "(e.g. GoogleLLM) for a real briefing."
                )
                await self._emit(WarmTransferPhase.SUMMARY_READY, {"summary": _FALLBACK_SUMMARY})
                return _FALLBACK_SUMMARY
            llm = getattr(pipeline, "llm", None)
        if llm is None:
            logger.warning("WarmTransfer: no LLM available for summarization; using fallback text")
            await self._emit(WarmTransferPhase.SUMMARY_READY, {"summary": _FALLBACK_SUMMARY})
            return _FALLBACK_SUMMARY
        try:
            history = self._session.get_context_history(include_function_calls=False, include_system_messages=False)
        except Exception as exc:  
            logger.warning(f"WarmTransfer: get_context_history failed: {exc}")
            history = []
        summary = await _summarize_history(llm, history, self._config.summary_prompt)
        await self._emit(WarmTransferPhase.SUMMARY_READY, {"summary": summary})
        return summary

    def _speak_on_primary(self, message: str) -> None:
        """Speak ``message`` to the caller without disturbing the transfer.

        Deliberately *not* ``AgentSession.say``: that flips ``_accept_user_input``
        back on (re-arming interruptions on the caller's leg) and blocks until the
        agent has fully stopped speaking. Here we push the message straight into
        the pipeline as a non-interruptible utterance, fire-and-forget — the
        caller stays muted and the transfer keeps moving.
        """
        with suppress(Exception):
            self._session.agent.chat_context.add_message(role=ChatRole.ASSISTANT, content=message)
        pipeline = getattr(self._session, "pipeline", None)
        if pipeline is None or not hasattr(pipeline, "send_message"):
            return
        handle = UtteranceHandle(utterance_id=f"warm_transfer_{uuid.uuid4().hex[:8]}", interruptible=False)
        with suppress(Exception):
            asyncio.create_task(pipeline.send_message(message, handle=handle))

    async def _place_caller_on_hold(self) -> None:
        # Queues after whatever the agent was already saying when it called the
        # escalation tool, so the caller hears that line in full and then the
        # hold message; doesn't block the transfer on playback.
        self._speak_on_primary("Please hold while I connect you with a supervisor.")
        await self._emit(WarmTransferPhase.CALLER_ON_HOLD, {})

    async def _create_consultation_room(self) -> str:
        ctx = self._session._job_context
        auth_token = getattr(ctx, "videosdk_auth", None)
        if not auth_token:
            raise WarmTransferError("No VideoSDK auth token available to create consultation room.")
        signaling_base_url = getattr(ctx.room_options, "signaling_base_url", "api.videosdk.live")
        logger.info("WarmTransfer: creating consultation room via POST /v2/rooms …")
        room_id = await asyncio.to_thread(JobContext.create_room_static, auth_token, signaling_base_url)
        logger.info(f"WarmTransfer: consultation room created: {room_id}")
        return room_id

    def _build_briefing_pipeline(self) -> "Pipeline":
        """Build the consultation-room pipeline with its **own** component instances.

        The briefing session runs concurrently with the primary one, so it must
        not share live STT/LLM/TTS/VAD/turn-detector instances with it: shared
        streaming state would corrupt both sessions, the briefing TTS would write
        to the primary room's audio track, and closing the briefing session would
        tear down the primary's components (the source of the runaway
        ``'NoneType' object has no attribute 'frame_size'`` VAD errors after a
        transfer). Use the caller-supplied factory if any; otherwise auto-build:
        for a realtime primary, rebuild a fresh realtime model; for a cascade
        primary, re-build each component from its class with no args (works for
        env-configured providers).
        """
        from ..pipeline import Pipeline

        factory = self._config.briefing_pipeline_factory
        if factory is not None:
            pipeline = factory()
            if not isinstance(pipeline, Pipeline):
                raise WarmTransferError(
                    "WarmTransferConfig.briefing_pipeline_factory must return a "
                    "videosdk.agents.Pipeline instance."
                )
            logger.info("WarmTransfer: using briefing_pipeline_factory for the consultation pipeline")
            return pipeline

        primary = self._session.pipeline

        if getattr(getattr(primary, "config", None), "is_realtime", False):
            model = getattr(primary, "_realtime_model", None)
            if model is None:
                raise WarmTransferError(
                    "Primary pipeline is realtime but exposes no realtime model to "
                    "rebuild; pass WarmTransferConfig.briefing_pipeline_factory."
                )
            logger.info("WarmTransfer: auto-building a realtime consultation pipeline from the primary realtime model")
            return Pipeline(llm=self._fresh_realtime_model(model))

        logger.info("WarmTransfer: auto-building the consultation pipeline from the primary component classes")

        def _fresh(component: Any) -> Any:
            if component is None:
                return None
            try:
                return type(component)()
            except Exception as exc:
                raise WarmTransferError(
                    f"Could not auto-build a fresh {type(component).__name__} for the "
                    f"briefing pipeline ({exc}). Pass "
                    f"WarmTransferConfig.briefing_pipeline_factory to construct the "
                    f"consultation pipeline explicitly."
                ) from exc

        return Pipeline(
            stt=_fresh(getattr(primary, "stt", None)),
            llm=_fresh(getattr(primary, "llm", None)),
            tts=_fresh(getattr(primary, "tts", None)),
            vad=_fresh(getattr(primary, "vad", None)),
            turn_detector=_fresh(getattr(primary, "turn_detector", None)),
        )

    @staticmethod
    def _fresh_realtime_model(model: Any) -> Any:
        """Build a fresh instance of the primary's realtime model (own connection/track) for the briefing."""
        cls = type(model)
        kwargs: dict[str, Any] = {}
        if getattr(model, "model", None) is not None:
            kwargs["model"] = model.model
        if getattr(model, "config", None) is not None:
            kwargs["config"] = model.config
        try:
            return cls(**kwargs)
        except Exception as exc_with_args:
            try:
                return cls()
            except Exception as exc_bare:
                raise WarmTransferError(
                    f"Could not auto-build a fresh {cls.__name__} for the briefing "
                    f"pipeline (with args: {exc_with_args}; bare: {exc_bare}). Pass "
                    f"WarmTransferConfig.briefing_pipeline_factory to construct the "
                    f"consultation pipeline explicitly."
                ) from exc_bare

    async def _spawn_briefing_session(self) -> None:
        """Spin up a second JobContext + AgentSession (WarmTransferAgent) for the consultation room."""
        from ..agent_session import AgentSession

        ctx = self._session._job_context
        briefing_ctx = JobContext(
            room_options=RoomOptions(
                room_id=self._consultation_room_id,
                auth_token=getattr(ctx, "videosdk_auth", None),
                name="Warm Transfer Agent",
                agent_participant_id=f"warm_transfer_{int(time.time())}",
                playground=False,
                background_audio=False,
                auto_end_session=False,
                signaling_base_url=getattr(ctx.room_options, "signaling_base_url", "api.videosdk.live"),
            )
        )
        self._briefing_ctx = briefing_ctx
        token = _set_current_job_context(briefing_ctx)
        try:
            briefing_pipeline = self._build_briefing_pipeline()
            briefing_is_realtime = getattr(getattr(briefing_pipeline, "config", None), "is_realtime", False)
            vmd_llm = self._config.summary_llm or (
                None if briefing_is_realtime else getattr(briefing_pipeline, "llm", None)
            )
            vmd = VoiceMailDetector(llm=vmd_llm, callback=self._on_voicemail_detected) if vmd_llm else None
            briefing_session = AgentSession(
                agent=WarmTransferAgent(summary=self._summary),
                pipeline=briefing_pipeline,
                voice_mail_detector=vmd,
            )
        finally:
            _reset_current_job_context(token)

        self._briefing_session = briefing_session
        await briefing_ctx.connect()

        token = _set_current_job_context(briefing_ctx)
        try:
            self._briefing_start_task = asyncio.create_task(briefing_session.start(wait_for_participant=False))
        finally:
            _reset_current_job_context(token)

        self._install_supervisor_disconnect_watcher()

    async def _on_voicemail_detected(self) -> None:
        """Callback for the briefing session's VoiceMailDetector classifier (idempotent)."""
        self._flag_briefing_failed("voicemail", "VoiceMailDetector classifier flagged the supervisor leg as voicemail")

    def _flag_briefing_failed(self, reason: str, log_msg: Optional[str] = None) -> None:
        agent = getattr(self._briefing_session, "agent", None)
        ev = getattr(agent, "briefing_failed_event", None)
        if ev is None or ev.is_set():
            return
        if log_msg:
            logger.warning(f"WarmTransfer: {log_msg}")
        setattr(agent, "_briefing_failure_reason", reason)
        ev.set()
        if reason in ("supervisor_disconnected", "voicemail"):
            with suppress(Exception):
                asyncio.create_task(self._log_supervisor_call_status())

    async def _log_supervisor_call_status(self) -> None:
        """Best-effort: log the supervisor outbound call's final SIP state.

        When the supervisor leg drops, this fetches ``GET /v2/sip/call`` for the
        consultation room and logs each call's ``status`` / ``timelog`` so the
        hangup reason (``no-answer`` / ``busy`` / ``completed`` + reason) shows up
        in the agent logs — useful since the warm-transfer code never ends that
        call itself; a ~timeout-ish drop is always on the SIP/gateway/carrier side.
        """
        room_id = self._consultation_room_id
        auth_token = getattr(getattr(self._session, "_job_context", None), "videosdk_auth", None)
        if not room_id or not auth_token:
            return
        try:
            import requests

            from ..room._sip_manager import FETCH_CALL_INFO_URL

            resp = await asyncio.to_thread(
                requests.get,
                FETCH_CALL_INFO_URL,
                headers={"Authorization": auth_token},
                params={"roomId": room_id},
            )
            calls = (resp.json() or {}).get("data", []) if getattr(resp, "ok", False) else []
            if not calls:
                logger.info(f"WarmTransfer: no SIP call records found for consultation room {room_id}")
                return
            for call in calls:
                logger.info(
                    "WarmTransfer: supervisor SIP call %s — status=%s end=%s timelog=%s",
                    call.get("callId") or call.get("id"),
                    call.get("status"),
                    call.get("end"),
                    call.get("timelog"),
                )
        except Exception as exc:
            logger.debug(f"WarmTransfer: could not fetch supervisor call status: {exc}")

    def _install_supervisor_disconnect_watcher(self) -> None:
        room = self._briefing_ctx.room if self._briefing_ctx else None
        if room is None or getattr(self._briefing_session, "agent", None) is None:
            return

        def _on_participant_left(data):
            p = data.get("participant") if isinstance(data, dict) else None
            if not _participant_in_room(p, room) or _is_agent_name(p):
                return
            self._flag_briefing_failed(
                "supervisor_disconnected", f"supervisor left the briefing room mid-call (participant={p.id})"
            )

        global_event_emitter.on("PARTICIPANT_LEFT", _on_participant_left)
        self._participant_left_handler = _on_participant_left

    async def _dial_supervisor(self) -> None:
        dest = self._config.destination
        if not isinstance(dest, SIPDestination):
            raise WarmTransferError("WarmTransferConfig.destination must be a SIPDestination")
        sip_manager = getattr(self._session._job_context.room, "sip_manager", None)
        if sip_manager is None:
            raise WarmTransferError("Primary room has no SIP manager; cannot place an outbound call.")
        kwargs: dict[str, Any] = {
            "routing_rule_id": dest.routing_rule_id,
            "sip_call_to": dest.sip_call_to,
            "sip_call_from": dest.sip_call_from,
            "destination_room_id": self._consultation_room_id,
        }
        if dest.participant_display_name and dest.participant_display_name != "Supervisor":
            kwargs["participant"] = {"name": dest.participant_display_name}
        kwargs.update(dest.extra_options)
        response = await sip_manager.async_make_outbound_call(**kwargs)
        data = response.get("data", {}) if isinstance(response, dict) else {}
        self._supervisor_call_id = data.get("callId") or data.get("id")
        logger.info(
            f"WarmTransfer: dialed supervisor {dest.sip_call_to} (callId={self._supervisor_call_id}) "
            f"into room {self._consultation_room_id}"
        )
        await self._emit(
            WarmTransferPhase.SUPERVISOR_DIALED,
            {
                "supervisor_call_id": self._supervisor_call_id,
                "consultation_room_id": self._consultation_room_id,
                "response": response,
            },
        )

    async def _wait_for_supervisor(self) -> bool:
        """Wait until the supervisor's audio is live (not just the SIP leg ringing).

        Returns ``False`` if the wait times out, or if the supervisor leg drops /
        hits voicemail before audio ever becomes live (no point waiting out the
        full ``supervisor_join_timeout`` once we know the supervisor is gone).
        """
        room = self._briefing_ctx.room if self._briefing_ctx else None
        if room is None:
            raise WarmTransferError("Briefing room not connected; cannot wait for supervisor.")
        audio_ready = asyncio.Event()

        def _on_audio_enabled(data):
            stream = data.get("stream") if isinstance(data, dict) else None
            p = data.get("participant") if isinstance(data, dict) else None
            if not (stream and getattr(stream, "kind", None) == "audio"):
                return
            if not _participant_in_room(p, room) or _is_agent_name(p):
                return
            logger.info(f"WarmTransfer: supervisor audio stream enabled (participant={p.id})")
            audio_ready.set()

        agent = getattr(self._briefing_session, "agent", None)
        failed_ev = getattr(agent, "briefing_failed_event", None)

        global_event_emitter.on("AUDIO_STREAM_ENABLED", _on_audio_enabled)
        audio_task = asyncio.create_task(audio_ready.wait())
        failed_task = asyncio.create_task(failed_ev.wait() if failed_ev is not None else asyncio.Event().wait())
        try:
            done, _ = await asyncio.wait(
                {audio_task, failed_task},
                timeout=self._config.supervisor_join_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if audio_task in done:
                return True
            if failed_task in done:
                logger.warning(
                    "WarmTransfer: supervisor leg unavailable before audio became live "
                    f"(reason={getattr(agent, '_briefing_failure_reason', 'unknown')})"
                )
                return False
            logger.warning(
                f"WarmTransfer: supervisor audio did not become live within "
                f"{self._config.supervisor_join_timeout}s"
            )
            return False
        finally:
            global_event_emitter.off("AUDIO_STREAM_ENABLED", _on_audio_enabled)
            for t in (audio_task, failed_task):
                if not t.done():
                    t.cancel()
                    with suppress(asyncio.CancelledError):
                        await t

    async def _wait_for_briefing_complete(self) -> dict[str, str]:
        """Race done / failed / timeout. Returns ``{"status": ..., "apology": ...}``."""
        await self._emit(WarmTransferPhase.BRIEFING_STARTED, {})
        agent = self._briefing_session.agent
        done_ev = getattr(agent, "briefing_done_event", None)
        failed_ev = getattr(agent, "briefing_failed_event", None)

        async def _wait(ev):
            if ev is None:
                await asyncio.Event().wait()
            else:
                await ev.wait()

        done_task = asyncio.create_task(_wait(done_ev))
        failed_task = asyncio.create_task(_wait(failed_ev))
        try:
            done, _ = await asyncio.wait(
                {done_task, failed_task}, timeout=self._config.briefing_timeout, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for t in (done_task, failed_task):
                if not t.done():
                    t.cancel()
                    with suppress(asyncio.CancelledError):
                        await t

        if failed_task in done and done_task not in done:
            reason = getattr(agent, "_briefing_failure_reason", "briefing_failed")
            await self._emit(WarmTransferPhase.BRIEFING_COMPLETE, {"resolution": reason})
            return {"status": reason, "apology": _BRIEFING_APOLOGIES.get(reason, _DEFAULT_BRIEFING_APOLOGY)}

        if not done:
            logger.warning(
                f"WarmTransfer: briefing timeout ({self._config.briefing_timeout}s) reached; proceeding."
            )
            await self._emit(WarmTransferPhase.BRIEFING_COMPLETE, {"resolution": "timeout"})
        else:
            await self._emit(WarmTransferPhase.BRIEFING_COMPLETE, {"resolution": "ready"})
        return {"status": "ready", "apology": ""}

    async def _switch_caller_to_consultation_room(self) -> bool:
        sip_manager = getattr(self._session._job_context.room, "sip_manager", None)
        if self._caller_call_id is None or sip_manager is None:
            logger.error("WarmTransfer: no caller SIP callId / SIP manager; cannot switch")
            return False
        auth_token = getattr(self._session._job_context, "videosdk_auth", None)
        pid, tkn = _mint_participant_token("caller", fallback_token=auth_token)
        consult_room = self._briefing_ctx.room if self._briefing_ctx else None
        self._consultation_known_before_switch = (
            set(getattr(consult_room, "participants_data", {}).keys())
            if consult_room is not None
            else set()
        )
        try:
            response = await sip_manager.async_switch_call_room(
                call_id=self._caller_call_id,
                room_id=self._consultation_room_id,
                token=tkn,
                participant_id=pid,
            )
        except Exception as exc:  
            logger.error(f"WarmTransfer: switch_call_room failed: {exc}")
            return False
        await self._emit(
            WarmTransferPhase.CALL_SWITCHED,
            {
                "caller_call_id": self._caller_call_id,
                "consultation_room_id": self._consultation_room_id,
                "caller_participant_id": pid,
                "response": response,
            },
        )
        return True

    async def _wait_for_caller_in_consultation_room(self, timeout: float = 15.0) -> bool:
        """Hold (best-effort) until the switched caller is present in the consultation room."""
        room = self._briefing_ctx.room if self._briefing_ctx else None
        if room is None:
            return False
        known = self._consultation_known_before_switch
        if known is None:
            known = set(getattr(room, "participants_data", {}).keys())
        waited = 0.0
        poll = 0.5
        while waited < timeout:
            for pid, p in list(getattr(room, "participants_data", {}).items()):
                if pid not in known and not _is_agent_name(p):
                    logger.info(f"WarmTransfer: caller present in consultation room (participant={pid})")
                    return True
            await asyncio.sleep(poll)
            waited += poll
        try:
            present = [
                f"{pid}({getattr(p, 'display_name', '') or '?'})"
                for pid, p in getattr(room, "participants_data", {}).items()
            ]
        except Exception:
            present = ["<unavailable>"]
        logger.warning(f"WarmTransfer: caller did not join the consultation room within {timeout}s; tearing down anyway (present: {present})")
        return False

    def _mute_primary_input(self) -> None:
        """Stop the primary session from processing the caller's stray speech while on hold.

        Idempotent: the original ``_accept_user_input`` value is captured only on
        the first call so re-muting after a ``say`` (which re-enables input)
        doesn't clobber it.
        """
        if self._primary_input_was_enabled is None:
            self._primary_input_was_enabled = bool(getattr(self._session, "_accept_user_input", True))
        with suppress(Exception):
            self._session._accept_user_input = False

    def _restore_primary_input(self) -> None:
        if self._primary_input_was_enabled is None:
            return
        with suppress(Exception):
            self._session._accept_user_input = self._primary_input_was_enabled
        self._primary_input_was_enabled = None

    def _primary_room(self) -> Any:
        return getattr(getattr(self._session, "_job_context", None), "room", None)

    def _suspend_primary_auto_end(self) -> None:
        """Disable the primary room's no-participant auto-end during the transfer (idempotent)."""
        room = self._primary_room()
        if room is None or self._primary_auto_end_was_enabled is not None:
            return
        self._primary_auto_end_was_enabled = bool(getattr(room, "auto_end_session", False))
        with suppress(Exception):
            room.auto_end_session = False
        with suppress(Exception):
            room._cancel_session_end_task()
        with suppress(Exception):
            task = getattr(room, "_no_participant_timeout_task", None)
            if task is not None and not task.done():
                task.cancel()
                room._no_participant_timeout_task = None

    def _restore_primary_auto_end(self) -> None:
        if self._primary_auto_end_was_enabled is None:
            return
        room = self._primary_room()
        if room is not None:
            with suppress(Exception):
                room.auto_end_session = self._primary_auto_end_was_enabled
        self._primary_auto_end_was_enabled = None

    async def _close_briefing(self) -> None:
        if self._participant_left_handler is not None:
            with suppress(Exception):
                global_event_emitter.off("PARTICIPANT_LEFT", self._participant_left_handler)
            self._participant_left_handler = None
        if self._briefing_session is not None:
            with suppress(Exception):
                await self._briefing_session.close()
        if self._briefing_start_task is not None and not self._briefing_start_task.done():
            self._briefing_start_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._briefing_start_task
        if self._briefing_ctx is not None:
            with suppress(Exception):
                await self._briefing_ctx.shutdown()

    async def _abort(self, reason: str, *, apology: str, phase: WarmTransferPhase) -> WarmTransferResult:
        """Restore the caller, apologize, tear down the briefing session, emit ``phase``."""
        self._restore_primary_input()
        self._restore_primary_auto_end()
        with suppress(Exception):
            asyncio.create_task(self._session.say(apology, interruptible=True))
        await self._close_briefing()
        await self._emit(phase, {"reason": reason})
        return WarmTransferResult(
            success=False,
            phase=phase,
            consultation_room_id=self._consultation_room_id,
            supervisor_call_id=self._supervisor_call_id,
            switched_call_id=self._caller_call_id,
            summary=self._summary,
            error=reason,
        )
        
    async def _emit(self, phase: WarmTransferPhase, data: dict[str, Any]) -> None:
        payload = {
            "phase": phase,
            "data": data,
            "timestamp": time.time(),
            "consultation_room_id": self._consultation_room_id,
        }
        with suppress(Exception):
            self._session.emit("warm_transfer", payload)
        with suppress(Exception):
            global_event_emitter.emit("WARM_TRANSFER", {**payload, "phase": phase.value})