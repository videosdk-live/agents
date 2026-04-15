"""
VideoSDK Inference Gateway Turn Detection Plugin.

HTTP-based End-of-Utterance (EOU) detector that delegates inference to the
VideoSDK Inference Gateway's ``/v1/turn`` endpoint, which runs the
``videosdk-live/Namo-Turn-Detector-v1`` model family.

Unlike the local :class:`NamoTurnDetectorV1` plugin, this implementation
downloads no model weights and loads nothing into the worker process — making
it ideal for low-memory agent workers and keeping all model upgrades server-side.

Example:
    from videosdk.inference import Turn

    # Multilingual (default)
    turn = Turn.namo()

    # Language-specific (dispatches a DistilBert model on the server)
    turn = Turn.namo(language="en")

    pipeline = CascadingPipeline(stt=stt, llm=llm, tts=tts, turn_detector=turn)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from videosdk.agents import EOU, ChatContext, ChatMessage, ChatRole

logger = logging.getLogger(__name__)

DEFAULT_TURN_HTTP_URL = "https://inference-gateway.videosdk.live"
DEFAULT_TIMEOUT_SECONDS = 10.0


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

        self._session: Optional[requests.Session] = None
        self._logged_first_success: bool = False

        logger.info(
            f"[InferenceTurn] Configured (base_url={self.base_url}, "
            f"provider={self.provider}, model={self.model_id}, "
            f"language={self.language or 'multilingual'}, threshold={threshold})"
        )

    # ==================== Factory Methods ====================

    @staticmethod
    def namo(
        *,
        language: Optional[str] = None,
        threshold: float = 0.7,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> "Turn":
        """
        Create a Turn detector backed by the server-hosted Namo Turn Detector v1.

        Args:
            language: Optional language code (e.g. ``"en"``, ``"hi"``, ``"es"``).
                Omit for the multilingual model.
            threshold: EOU probability threshold (default: ``0.7``).
            base_url: Override for the inference gateway URL.
            timeout: Per-request timeout in seconds.
        """
        return Turn(
            provider="videosdk",
            model_id="namo-turn-detector-v1",
            language=language,
            threshold=threshold,
            base_url=base_url,
            timeout=timeout,
        )

    # ==================== Core EOU Interface ====================

    def get_eou_probability(self, chat_context: ChatContext) -> float:
        """
        Return the EOU probability for the last user message in ``chat_context``.

        This runs a synchronous HTTP POST against the inference gateway because
        the base :class:`EOU` interface is synchronous and the caller invokes it
        from inside an ``asyncio`` task. Requests are expected to be short
        (<100 ms) and the timeout is capped by ``self.timeout``.
        """
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
                return 0.0

            data = response.json()
            probability = data.get("probability", 0.0)
            if not isinstance(probability, (int, float)):
                logger.error(f"[InferenceTurn] Invalid probability: {probability!r}")
                return 0.0

            if not self._logged_first_success:
                self._logged_first_success = True
                logger.info(
                    f"[InferenceTurn] Connected successfully to {self.base_url}/v1/turn "
                    f"(first probability={float(probability):.4f})"
                )

            return float(probability)

        except requests.Timeout:
            logger.error(f"[InferenceTurn] Request timed out after {self.timeout}s")
            self.emit("error", "turn detection request timed out")
            return 0.0
        except requests.RequestException as e:
            logger.error(f"[InferenceTurn] Request failed: {e}")
            self.emit("error", f"turn detection request failed: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"[InferenceTurn] Unexpected error: {e}")
            self.emit("error", f"turn detection error: {e}")
            return 0.0

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
