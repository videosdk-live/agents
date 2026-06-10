"""
VideoSDK Inference Gateway Turn Detection.

Two turn detectors, both served by the VideoSDK Inference Gateway — no model is
loaded locally; the server handles downloads, caching, and inference.

``Turn`` — End-of-Utterance detector returning a float probability via the
``/v1/turn`` endpoint. Three backends:

  * ``Turn.namo()``      — Namo Turn Detector v1 (multilingual, 23 languages).
  * ``Turn.turnsense()`` — TurnSense / SmolLM2-135M (English).
  * ``Turn.videosdk()``  — VideoSDK BERT-based detector (English).

``TurnV2`` — low-latency "echo" detector that classifies each user utterance
into one of four turn states (Complete / Incomplete / Backchannel / Wait),
enabling backchannel suppression and explicit "wait/stop" handling. Two sizes:

  * ``TurnV2.echo_small()`` — faster (default).
  * ``TurnV2.echo_large()`` — higher accuracy.

Example:
    from videosdk.inference import Turn, TurnV2

    turn = Turn.namo(language="en")      # probability-based EOU
    turn = TurnV2.echo_large()           # 4-state turn classification

    pipeline = CascadingPipeline(stt=stt, llm=llm, tts=tts, turn_detector=turn)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests
import threading
import grpc
from ._grpc import turn_detection_pb2, turn_detection_pb2_grpc

from videosdk.agents import EOU, ChatContext, ChatMessage, ChatRole
from videosdk.agents.utils import TurnResult, TurnState

logger = logging.getLogger(__name__)

DEFAULT_TURN_HTTP_URL = "https://inference-gateway.videosdk.live"
DEFAULT_TIMEOUT_SECONDS = 10.0

# FOR TURN V2
DEFAULT_GRPC_HOST = "inference-gateway.videosdk.live:50053"
DEFAULT_GRPC_TIMEOUT_SECONDS = 2.0

_GRPC_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms", 10_000),
    ("grpc.keepalive_timeout_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.http2.min_time_between_pings_ms", 10_000),
    ("grpc.http2.min_ping_interval_without_data_ms", 5_000),
    ("grpc.use_local_subchannel_pool", 1),
]

_RETRYABLE_CODES = frozenset({
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.UNKNOWN,
    grpc.StatusCode.INTERNAL,
})

TURN_STATE_INCOMPLETE = "Incomplete"
TURN_STATE_COMPLETE = "Complete"
TURN_STATE_BACKCHANNEL = "Backchannel"
TURN_STATE_WAIT = "Wait"

_KNOWN_TURN_STATES: frozenset[str] = frozenset(
    {TURN_STATE_INCOMPLETE, TURN_STATE_COMPLETE, TURN_STATE_BACKCHANNEL, TURN_STATE_WAIT}
)

_FINALIZING_TURN_STATES: frozenset[str] = frozenset({TURN_STATE_COMPLETE})

class Turn(EOU):
    """
    VideoSDK Inference Gateway Turn Detection Plugin.

    Delegates EOU scoring to the inference gateway's ``/v1/turn`` endpoint. No
    model is loaded locally — the server caches every Namo-Turn-Detector-v1
    language variant, so first-request latency per language is bounded by
    tokenize + inference time on the server (typically <50ms).

    Args:
        provider: Inference provider identifier (default: ``"videosdk"``).
        model_id: Turn detection model identifier
            (default: ``"namo-turn-detector-v1"``).
        language: Optional language code (e.g. ``"en"``, ``"hi"``, ``"es"``).
            Omit for the multilingual model.
        threshold: EOU probability threshold (default: ``0.7``).
        base_url: Override for the inference gateway URL.
        timeout: Per-request timeout in seconds.
        max_connection_attempts: After this many consecutive failed requests
            (non-200, timeout, invalid payload, etc.) the client stops calling
            the gateway and returns ``fallback_probability`` for the rest of
            the session. The counter resets on the next successful response.
            Default: 5.
        fallback_probability: Probability returned once the breaker has
            tripped. Default ``0.7`` matches the default threshold, so a
            disabled turn detector lets the pipeline assume end-of-utterance
            rather than stalling.
    """

    def __init__(
        self,
        *,
        provider: str = "videosdk",
        model_id: str = "namo-turn-detector-v1",
        language: Optional[str] = None,
        threshold: float = 0.7,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_connection_attempts: int = 5,
        fallback_probability: float = 0.7,
    ) -> None:
        super().__init__(threshold=threshold)

        self._videosdk_token = os.getenv("VIDEOSDK_AUTH_TOKEN")
        if not self._videosdk_token:
            raise ValueError(
                "VIDEOSDK_AUTH_TOKEN environment variable must be set for authentication"
            )

        self.provider = provider
        self.model_id = model_id
        self.language = language
        self.base_url = (base_url or DEFAULT_TURN_HTTP_URL).rstrip("/")
        self.timeout = timeout
        self.max_connection_attempts: int = max(1, int(max_connection_attempts))
        self.fallback_probability: float = float(fallback_probability)

        self._session: Optional[requests.Session] = None
        self._logged_first_success: bool = False

        # Circuit breaker: consecutive failed HTTP attempts. When it reaches
        # max_connection_attempts, every subsequent call returns
        # fallback_probability without hitting the network.
        self._consecutive_failures: int = 0
        self._turn_disabled: bool = False

        logger.info(
            f"[InferenceTurn] Configured (base_url={self.base_url}, "
            f"provider={self.provider}, model={self.model_id}, "
            f"language={self.language or 'multilingual'}, threshold={threshold}, "
            f"max_connection_attempts={self.max_connection_attempts}, "
            f"fallback_probability={self.fallback_probability})"
        )

    # ==================== Factory Methods ====================

    @staticmethod
    def namo(
        *,
        language: Optional[str] = None,
        threshold: float = 0.7,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_connection_attempts: int = 5,
        fallback_probability: float = 0.7,
    ) -> "Turn":
        """
        Create a Turn detector backed by the server-hosted Namo Turn Detector v1.

        Args:
            language: Optional language code (e.g. ``"en"``, ``"hi"``, ``"es"``).
                Omit for the multilingual model.
            threshold: EOU probability threshold (default: ``0.7``).
            base_url: Override for the inference gateway URL.
            timeout: Per-request timeout in seconds.
            max_connection_attempts: Failures before the breaker trips. Default: 5.
            fallback_probability: Probability emitted once disabled. Default: 0.7.
        """
        return Turn(
            provider="videosdk",
            model_id="namo-turn-detector-v1",
            language=language,
            threshold=threshold,
            base_url=base_url,
            timeout=timeout,
            max_connection_attempts=max_connection_attempts,
            fallback_probability=fallback_probability,
        )

    @staticmethod
    def turnsense(
        *,
        threshold: float = 0.7,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_connection_attempts: int = 5,
        fallback_probability: float = 0.7,
    ) -> "Turn":
        """
        Create a Turn detector backed by the server-hosted TurnSense model
        (latishab/turnsense, SmolLM2-135M, English).

        Args:
            threshold: EOU probability threshold (default: ``0.7``).
            base_url: Override for the inference gateway URL.
            timeout: Per-request timeout in seconds.
            max_connection_attempts: Failures before the breaker trips. Default: 5.
            fallback_probability: Probability emitted once disabled. Default: 0.7.
        """
        return Turn(
            provider="turnsense",
            model_id="latishab/turnsense",
            language=None,
            threshold=threshold,
            base_url=base_url,
            timeout=timeout,
            max_connection_attempts=max_connection_attempts,
            fallback_probability=fallback_probability,
        )

    @staticmethod
    def videosdk(
        *,
        threshold: float = 0.7,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_connection_attempts: int = 5,
        fallback_probability: float = 0.7,
    ) -> "Turn":
        """
        Create a Turn detector backed by the server-hosted VideoSDK BERT model
        (cdn.videosdk.live, English, binary classifier with softmax probability).

        Args:
            threshold: EOU probability threshold (default: ``0.7``).
            base_url: Override for the inference gateway URL.
            timeout: Per-request timeout in seconds.
            max_connection_attempts: Failures before the breaker trips. Default: 5.
            fallback_probability: Probability emitted once disabled. Default: 0.7.
        """
        return Turn(
            provider="videosdk",
            model_id="videosdk-turn-detector-v1",
            language=None,
            threshold=threshold,
            base_url=base_url,
            timeout=timeout,
            max_connection_attempts=max_connection_attempts,
            fallback_probability=fallback_probability,
        )

    # ==================== Circuit Breaker ====================

    def _record_attempt_failure(self, reason: str) -> float:
        """
        Increment the consecutive-failure counter and trip the breaker once it
        reaches ``max_connection_attempts``. Returns the probability value the
        caller should emit on this turn — ``0.0`` while pre-trip, or
        ``fallback_probability`` once disabled.
        """
        if self._turn_disabled:
            return self.fallback_probability

        self._consecutive_failures += 1
        logger.warning(
            f"[InferenceTurn] Attempt failure "
            f"{self._consecutive_failures}/{self.max_connection_attempts}: {reason}"
        )

        if self._consecutive_failures >= self.max_connection_attempts:
            self._turn_disabled = True
            logger.error(
                f"[InferenceTurn] Disabled for session after "
                f"{self._consecutive_failures} failed attempts — emitting "
                f"probability={self.fallback_probability} for the rest of the "
                f"turns (provider={self.provider}, model={self.model_id})"
            )
            return self.fallback_probability

        return 0.0

    def _record_attempt_success(self) -> None:
        """Reset the consecutive-failure counter on a successful response."""
        if self._consecutive_failures > 0:
            logger.info(
                f"[InferenceTurn] Recovered after {self._consecutive_failures} "
                f"failed attempt(s) — resetting counter"
            )
            self._consecutive_failures = 0

    # ==================== Core EOU Interface ====================

    def get_eou_probability(self, chat_context: ChatContext) -> float:
        """
        Return the EOU probability for the last user message in ``chat_context``.

        This runs a synchronous HTTP POST against the inference gateway because
        the base :class:`EOU` interface is synchronous and the caller invokes it
        from inside an ``asyncio`` task. Requests are expected to be short
        (<100 ms) and the timeout is capped by ``self.timeout``.

        After ``max_connection_attempts`` consecutive failures the client stops
        calling the gateway and returns ``fallback_probability`` for the rest
        of the session.
        """
        if self._turn_disabled:
            return self.fallback_probability

        payload = self._build_payload(chat_context)
        if payload is None:
            return 0.0

        try:
            session = self._get_session()
            url = f"{self.base_url}/v1/turn"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._videosdk_token}",
            }

            response = session.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )

            if response.status_code != 200:
                logger.error(
                    f"[InferenceTurn] HTTP {response.status_code}: {response.text}"
                )
                self.emit("error", f"HTTP {response.status_code}: {response.text}")
                return self._record_attempt_failure(
                    f"HTTP {response.status_code}"
                )

            data = response.json()
            probability = data.get("probability", 0.0)
            if not isinstance(probability, (int, float)):
                logger.error(f"[InferenceTurn] Invalid probability: {probability!r}")
                return self._record_attempt_failure("invalid probability payload")

            probability_f = float(probability)

            # A zero probability typically means the server omitted the field
            # (default 0.0 from .get) or returned a placeholder payload with
            # no real model signal. Treat it as a failed attempt for breaker
            # purposes, but keep returning 0.0 downstream while pre-trip so
            # EOU logic still sees "no end-of-utterance" as before.
            if probability_f == 0.0:
                return self._record_attempt_failure("probability=0.0")

            self._record_attempt_success()

            if not self._logged_first_success:
                self._logged_first_success = True
                logger.info(
                    f"[InferenceTurn] Connected successfully to {self.base_url}/v1/turn "
                    f"(first probability={probability_f:.4f})"
                )

            return probability_f

        except requests.Timeout:
            logger.error(f"[InferenceTurn] Request timed out after {self.timeout}s")
            self.emit("error", "turn detection request timed out")
            return self._record_attempt_failure("timeout")
        except requests.RequestException as e:
            logger.error(f"[InferenceTurn] Request failed: {e}")
            self.emit("error", f"turn detection request failed: {e}")
            return self._record_attempt_failure(f"request exception: {e}")
        except Exception as e:
            logger.error(f"[InferenceTurn] Unexpected error: {e}")
            self.emit("error", f"turn detection error: {e}")
            return self._record_attempt_failure(f"unexpected error: {e}")

    # ==================== Helpers ====================

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def _build_payload(self, chat_context: ChatContext) -> Optional[Dict[str, Any]]:
        """Build the /v1/turn request body from a ChatContext."""
        items: List[Dict[str, Any]] = []
        for item in chat_context.items:
            if not isinstance(item, ChatMessage):
                continue
            if item.role != ChatRole.USER:
                continue
            text = self._content_to_text(item.content)
            if text:
                items.append({"role": "user", "content": text})

        if not items:
            return None

        # Only the last user message is inspected by the server, but send the
        # full user-turn list to match the documented request shape.
        payload: Dict[str, Any] = {
            "chatContext": {"items": items},
            "provider": self.provider,
            "modelId": self.model_id,
        }
        if self.language:
            payload["language"] = self.language
        return payload

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for c in content:
                if hasattr(c, "text"):
                    parts.append(str(getattr(c, "text") or ""))
                elif isinstance(c, str):
                    parts.append(c)
            return " ".join(p for p in parts if p).strip()
        return str(content or "").strip()

    # ==================== Cleanup ====================

    async def aclose(self) -> None:
        logger.info(f"[InferenceTurn] Closing Turn detector (language={self.language})")
        if self._session is not None:
            try:
                self._session.close()
            except Exception as e:
                logger.error(f"[InferenceTurn] Error closing HTTP session: {e}")
            self._session = None
        await super().aclose()

    # ==================== Properties ====================

    @property
    def label(self) -> str:
        lang = self.language or "multilingual"
        return f"videosdk.inference.Turn.{self.provider}.{self.model_id}.{lang}"


class TurnV2(EOU):
    """Low-latency turn detector with 4-state classification.

    Classifies each user utterance as Complete / Incomplete / Backchannel / Wait
    via the VideoSDK Inference Gateway. Prefer the ``echo_small()`` /
    ``echo_large()`` factory methods.

    Args:
        model_id: ``"echo-small"`` (default, faster) or ``"echo-large"``
            (higher accuracy).
        host: ``host:port`` of the turn server. Falls back to the
            ``VIDEOSDK_TURN_GRPC_HOST`` env var, then to
            ``inference-gateway.videosdk.live:50053``.
        threshold: EOU probability threshold (default ``0.7``).
        timeout: Per-request timeout in seconds (default ``2.0``).
        token: Bearer token for the ``authorization`` metadata header.
            Falls back to the ``VIDEOSDK_AUTH_TOKEN`` env var. If unset,
            requests are sent without auth metadata (compatible with
            ungated dev servers).
    """

    supports_backchannel_classification: bool = True

    def __init__(
        self,
        *,
        model_id: str = "echo-small",
        host: Optional[str] = None,
        threshold: float = 0.7,
        timeout: float = DEFAULT_GRPC_TIMEOUT_SECONDS,
        token: Optional[str] = None,
    ) -> None:
        super().__init__(threshold=threshold)
        self.model_id = model_id
        self.host = host or os.getenv("VIDEOSDK_TURN_GRPC_HOST") or DEFAULT_GRPC_HOST
        self.timeout = timeout

        self._token = token or os.getenv("VIDEOSDK_AUTH_TOKEN")
        self._metadata = (
            (("authorization", f"Bearer {self._token}"),) if self._token else None
        )

        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[turn_detection_pb2_grpc.TurnDetectionStub] = None
        self._stub_lock = threading.Lock()
        self._last_state: Optional[str] = None

        threading.Thread(
            target=self._warmup, name="TurnV2-warmup", daemon=True
        ).start()

    @staticmethod
    def echo_small(
        *,
        host: Optional[str] = None,
        threshold: float = 0.7,
        timeout: float = DEFAULT_GRPC_TIMEOUT_SECONDS,
        token: Optional[str] = None,
    ) -> "TurnV2":
        """ONNX-based echo-small turn detector."""
        return TurnV2(
            model_id="echo-small", host=host, threshold=threshold,
            timeout=timeout, token=token,
        )

    @staticmethod
    def echo_large(
        *,
        host: Optional[str] = None,
        threshold: float = 0.7,
        timeout: float = DEFAULT_GRPC_TIMEOUT_SECONDS,
        token: Optional[str] = None,
    ) -> "TurnV2":
        """Higher-accuracy echo-large turn detector."""
        return TurnV2(
            model_id="echo-large", host=host, threshold=threshold,
            timeout=timeout, token=token,
        )

    def _format_chat_ctx(self, chat_context: ChatContext) -> str:
        """Build the request ``text`` from the current user utterance.

        The echo model is evaluated on a single utterance — the most recent USER
        message (the accumulated current transcript). Conversation history is not
        sent: the model's multi-turn input is not reliable yet.
        """
        for item in reversed(chat_context.items):
            if isinstance(item, ChatMessage) and item.role == ChatRole.USER:
                text = Turn._content_to_text(item.content)
                if text:
                    return text
        return ""

    def _predict(self, text: str) -> Optional["turn_detection_pb2.TurnResponse"]:
        """Run the gRPC Predict with one reconnect retry. Returns None on error."""
        request = turn_detection_pb2.TurnRequest(text=text, model_id=self.model_id)
        for attempt in (1, 2):
            try:
                stub = self._get_stub()
                return stub.Predict(
                    request, timeout=self.timeout, metadata=self._metadata
                )
            except grpc.RpcError as e:
                code = e.code()
                if attempt == 1 and code in _RETRYABLE_CODES:
                    logger.warning(
                        f"[TurnV2] transient gRPC error [{code}], reconnecting"
                    )
                    self._reset_channel()
                    continue
                logger.error(f"[TurnV2] gRPC error [{code}]: {e.details()}")
                self.emit("error", f"turn detection gRPC error: {e.details()}")
                return None
        return None

    def get_turn_result(self, chat_context: ChatContext) -> TurnResult:
        """Classify the current turn into one of the four states.

        Returns a :class:`TurnResult` whose ``state`` is ``None`` when there is
        no text to classify or the gRPC call fails (callers then fall back to
        legacy behavior). ``eou_probability`` stays binary (1.0 for Complete,
        0.0 otherwise) for back-compat with the wait-delay logic.
        """
        self._last_state = None
        text = self._format_chat_ctx(chat_context)
        if not text:
            return TurnResult(state=None, eou_probability=0.0, finalizes_turn=False)

        response = self._predict(text)
        if response is None:
            return TurnResult(state=None, eou_probability=0.0, finalizes_turn=False)

        self._last_state = response.state
        state = TurnState.from_wire(response.state)
        logger.info(f"[TurnV2] state={response.state!r} for text={text!r}")
        if state is None:
            logger.warning(
                f"[TurnV2] Unknown state {response.state!r} — treating as incomplete"
            )
        finalizes = state is TurnState.COMPLETE
        return TurnResult(
            state=state,
            eou_probability=1.0 if finalizes else 0.0,
            finalizes_turn=finalizes,
            raw_state=response.state,
            latency_ms=getattr(response, "latency_ms", None) or None,
            model_id=getattr(response, "model_id", None) or None,
        )

    def get_eou_probability(self, chat_context: ChatContext) -> float:
        return self.get_turn_result(chat_context).eou_probability

    @property
    def last_state(self) -> Optional[str]:
        """Raw state from the last successful Predict, or None.
        One of ``Incomplete | Complete | Backchannel | Wait``."""
        return self._last_state

    @staticmethod
    def _state_to_probability(state: str) -> float:
        """Map state label to the binary EOU probability.

        Only ``Complete`` finalizes the turn (1.0). ``Incomplete``,
        ``Backchannel`` and ``Wait`` all return 0.0
        """
        if state in _FINALIZING_TURN_STATES:
            return 1.0
        if state in _KNOWN_TURN_STATES:
            return 0.0
        logger.warning(f"[TurnV2] Unknown state {state!r} — treating as incomplete")
        return 0.0

    def _get_stub(self) -> turn_detection_pb2_grpc.TurnDetectionStub:
        if self._stub is None:
            with self._stub_lock:
                if self._stub is None:
                    channel = grpc.insecure_channel(
                        self.host, options=_GRPC_CHANNEL_OPTIONS
                    )
                    self._stub = turn_detection_pb2_grpc.TurnDetectionStub(channel)
                    self._channel = channel
        return self._stub

    def _reset_channel(self) -> None:
        with self._stub_lock:
            if self._channel is not None:
                try:
                    self._channel.close()
                except Exception as e:
                    logger.debug(f"[TurnV2] error closing channel: {e}")
            self._channel = None
            self._stub = None

    def _warmup(self) -> None:
        try:
            stub = self._get_stub()
            stub.Health(turn_detection_pb2.HealthRequest(), timeout=self.timeout)
            logger.info(
                f"[TurnV2] warmup OK (host={self.host}, model={self.model_id})"
            )
        except Exception:
            pass

    async def aclose(self) -> None:
        self._reset_channel()
        await super().aclose()

    @property
    def label(self) -> str:
        return f"videosdk.inference.TurnV2.{self.model_id}"