import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import time
import uuid

from videosdk.agent import Agent
from .card import AgentCard
import asyncio
from ..event_bus import global_event_emitter
from ..metrics import metrics_collector

logger = logging.getLogger(__name__)


def _safe_create_task(coro):
    """Create an asyncio task with exception logging."""
    task = asyncio.create_task(coro)
    def _log_exception(t):
        if not t.cancelled() and t.exception():
            logger.error(f"A2A async handler failed: {t.exception()}", exc_info=t.exception())
    task.add_done_callback(_log_exception)
    return task


@dataclass
class A2AMessage:
    """
    Message format for agent-to-agent communication.

    Attributes:
        from_agent (str): ID of the agent sending the message.
        to_agent (str): ID of the agent receiving the message.
        type (str): Type/category of the message (e.g., "specialist_query", "model_response").
        content (Dict[str, Any]): The actual message content and data.
        id (str): Unique identifier for the message. Auto-generated if not provided.
        timestamp (float): Unix timestamp when the message was created.
        metadata (Optional[Dict[str, Any]]): Additional message metadata.
    """
    from_agent: str
    to_agent: str
    type: str
    content: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PendingRequest:
    """
    Tracks an outgoing A2A request awaiting a response.
    Used for correlation-based response routing.
    """
    correlation_id: str
    from_agent: str
    to_agent: str
    created_at: float
    timeout: float = 30.0


@dataclass
class AgentRegistry:
    """
    Singleton registry for managing agent registration and discovery.
    """
    _instance = None
    agents: Dict[str, AgentCard] = field(default_factory=dict)
    agent_instances: Dict[str, 'Agent'] = field(default_factory=dict)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
            if not hasattr(cls._instance, 'agents'):
                cls._instance.agents = {}
            if not hasattr(cls._instance, 'agent_instances'):
                cls._instance.agent_instances = {}
        return cls._instance

    def register_agent(self, card: AgentCard, agent_instance: 'Agent' = None):
        """
        Register an agent with its capabilities in the registry.
        Args:
            card (AgentCard): The agent's capability card containing metadata.
            agent_instance ('Agent', optional): The actual agent instance for direct communication.
        """
        self.agents[card.id] = card
        if agent_instance:
            self.agent_instances[card.id] = agent_instance

            traces_flow_manager = metrics_collector.traces_flow_manager if metrics_collector else None

            if traces_flow_manager:
                total_agents = len(self.agents)
                attributes = {
                    "agent_id": card.id,
                    "agent_name": card.name,
                    "agent_domain": card.domain,
                    "agent_capabilities": ", ".join(card.capabilities),
                    "agent_description": card.description,
                    "registration_order": total_agents,
                    "total_registered_agents": total_agents,
                    "agent_role": "primary" if total_agents == 1 else "secondary",
                }

                if card.metadata:
                    attributes.update(card.metadata)

                try:
                    reg_span = traces_flow_manager.create_a2a_trace(
                        "Agent Registration",
                        attributes
                    )
                    if reg_span:
                        traces_flow_manager.end_a2a_trace(
                            reg_span,
                            f"Agent {card.name} registered successfully"
                        )
                except Exception as e:
                    print(f"Failed to create A2A registration trace: {e}")

    async def unregister_agent(self, agent_id: str):
        """
        Remove an agent from the registry.

        Args:
            agent_id (str): The ID of the agent to unregister.
        """
        self.agents.pop(agent_id, None)
        self.agent_instances.pop(agent_id, None)

    def find_agents_by_domain(self, domain: str) -> List[str]:
        """
        Find all agents that handle a specific domain.

        Args:
            domain (str): The domain to search for (e.g., "customer_service", "technical_support").

        Returns:
            List[str]: List of agent IDs that handle the specified domain.
        """
        return [
            agent_id for agent_id, card in self.agents.items()
            if card.domain == domain
        ]

    def get_all_agents(self) -> Dict[str, AgentCard]:
        """
        Get all registered agents in the system.

        Returns:
            Dict[str, AgentCard]: Dictionary mapping agent IDs to their capability cards.
        """
        return self.agents

    def get_agent_instance(self, agent_id: str) -> Optional['Agent']:
        """
        Get an agent instance by ID for direct communication.

        Args:
            agent_id (str): The ID of the agent to retrieve.

        Returns:
            Optional['Agent']: The agent instance if found, None otherwise.
        """
        return self.agent_instances.get(agent_id)

    async def find_agents_by_capability(self, capability: str) -> List[str]:
        """
        Find all agents that have a specific capability.

        Args:
            capability (str): The capability to search for (e.g., "text_generation", "image_analysis").

        Returns:
            List[str]: List of agent IDs that have the specified capability.
        """
        return [
            agent_id for agent_id, card in self.agents.items()
            if capability in card.capabilities
        ]


class A2AProtocol:
    """
    Handles agent-to-agent communication and message routing.

    Uses correlation-based request-response matching to ensure responses
    are routed to the correct requesting agent, even with concurrent queries.
    """

    def __init__(self, agent: 'Agent'):
        """
        Initialize the A2A protocol for an agent.

        Args:
            agent ('Agent'): The agent instance that will use this protocol.
        """
        self.agent = agent
        self.registry = AgentRegistry()
        self._message_handlers: Dict[str, List[Callable]] = {}
        self._pending_requests: OrderedDict[str, PendingRequest] = OrderedDict()
        self._global_event_handler = None
        self._pipeline_event_handler = None
        self._sweep_task: Optional[asyncio.Task] = None
        self._deferred_model_handlers: List[Callable] = []

    async def register(self, card: AgentCard) -> None:
        """
        Register the agent with the global registry and start background tasks.

        Args:
            card (AgentCard): The agent's capability card.
        """
        self.registry.register_agent(card, self.agent)
        # Start timeout sweep for pending requests
        if self._sweep_task is None:
            self._sweep_task = asyncio.create_task(self._sweep_expired_requests())

    async def unregister(self) -> None:
        """
        Unregister the agent and clean up all resources.
        """
        traces_flow_manager = metrics_collector.traces_flow_manager if metrics_collector else None

        if traces_flow_manager:
            try:
                traces_flow_manager.end_a2a_communication()
            except Exception as e:
                print(f"Failed to end A2A communication trace: {e}")

        # Cancel sweep task
        if self._sweep_task:
            self._sweep_task.cancel()
            self._sweep_task = None

        # Unregister all message handlers
        for message_type in list(self._message_handlers.keys()):
            for handler in list(self._message_handlers.get(message_type, [])):
                self.off_message(message_type, handler)

        await self.registry.unregister_agent(self.agent.id)
        self._pending_requests.clear()

    def on_message(self, message_type: str, handler: Callable[[A2AMessage], None]) -> None:
        """
        Register a message handler for specific message types.

        Args:
            message_type (str): Type of message to handle (e.g., "specialist_query", "model_response").
            handler (Callable[[A2AMessage], None]): Async function to handle the message.

        Raises:
            ValueError: If the handler is not callable or not an async function.
        """
        if not callable(handler):
            raise ValueError("Handler must be a callable function")

        if not asyncio.iscoroutinefunction(handler):
            raise ValueError("Handler must be an async function")

        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []

        if handler not in self._message_handlers[message_type]:
            self._message_handlers[message_type].append(handler)

            if message_type == "model_response":
                self._setup_model_response_listener(handler)

    def _setup_model_response_listener(self, handler: Callable) -> None:
        """
        Set up event listeners for model responses, deferring if pipeline not ready.
        """
        if (hasattr(self.agent, 'session') and self.agent.session
                and hasattr(self.agent.session, 'pipeline') and self.agent.session.pipeline):
            self._attach_response_listeners(handler)
        else:
            self._deferred_model_handlers.append(handler)
            logger.info(f"Deferring model_response listener for agent {self.agent.id} — pipeline not ready yet")

    def _attach_deferred_listeners(self) -> None:
        """
        Attach any deferred model_response listeners.
        Called by AgentSession after the pipeline is started and ready.
        """
        if self._deferred_model_handlers:
            for handler in self._deferred_model_handlers:
                self._attach_response_listeners(handler)
            self._deferred_model_handlers.clear()
            logger.info(f"Attached deferred model_response listeners for agent {self.agent.id}")

    def _attach_response_listeners(self, handler: Callable) -> None:
        """
        Wire up event listeners for capturing LLM responses and routing them
        back to the requesting agent via correlation IDs.
        """
        def on_model_response(data):
            response = data.get('text', '')
            if not response or not self._pending_requests:
                return

            # FIFO: match to the oldest pending request
            correlation_id, pending = next(iter(self._pending_requests.items()))
            del self._pending_requests[correlation_id]

            message = A2AMessage(
                from_agent=self.agent.id,
                to_agent=pending.from_agent,
                type="model_response",
                content={"response": response},
                metadata={"correlation_id": correlation_id}
            )
            _safe_create_task(handler(message))

        pipeline = self.agent.session.pipeline

        # Per-pipeline listener for cascading mode (naturally scoped, no collision)
        self._pipeline_event_handler = on_model_response
        pipeline.on("content_generated", on_model_response)

        # Global listener for realtime mode text responses
        # Collision is handled by the _pending_requests guard above:
        # if no pending requests, the event is ignored
        if getattr(pipeline.config, 'is_realtime', False):
            self._global_event_handler = on_model_response
            global_event_emitter.on("text_response", on_model_response)

    def off_message(self, message_type: str, handler: Callable[[A2AMessage], None]) -> None:
        """
        Unregister a message handler.

        Args:
            message_type (str): Type of message to unregister the handler from.
            handler (Callable[[A2AMessage], None]): The handler function to remove.
        """
        if message_type in self._message_handlers:
            if handler in self._message_handlers[message_type]:
                self._message_handlers[message_type].remove(handler)

                if not self._message_handlers[message_type]:
                    del self._message_handlers[message_type]

                if message_type == "model_response":
                    if self._global_event_handler:
                        global_event_emitter.off(
                            "text_response", self._global_event_handler)
                        self._global_event_handler = None
                    if self._pipeline_event_handler and hasattr(self.agent, 'session') and self.agent.session and hasattr(self.agent.session, 'pipeline'):
                        self.agent.session.pipeline.off(
                            "content_generated", self._pipeline_event_handler)
                        self._pipeline_event_handler = None

    async def send_message(self, to_agent: str, message_type: str, content: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Send a message to another agent.

        Args:
            to_agent (str): ID of the target agent to send the message to.
            message_type (str): Type/category of the message.
            content (Dict[str, Any]): The message content and data.
            metadata (Optional[Dict[str, Any]]): Additional message metadata.
        """
        target_agent = self.registry.get_agent_instance(to_agent)

        if not target_agent:
            logger.error(f"A2A send_message failed: target agent '{to_agent}' not found in registry. "
                         f"Registered agents: {list(self.registry.agents.keys())}")
            return

        # Generate correlation ID for request-response matching
        correlation_id = str(uuid.uuid4())
        if metadata is None:
            metadata = {}
        metadata["correlation_id"] = correlation_id

        message = A2AMessage(
            from_agent=self.agent.id,
            to_agent=to_agent,
            type=message_type,
            content=content,
            metadata=metadata
        )

        traces_flow_manager = metrics_collector.traces_flow_manager if metrics_collector else None

        sender_span = None
        if traces_flow_manager:
            try:
                attributes = {
                    "from_agent": message.from_agent,
                    "to_agent": message.to_agent,
                    "message_type": message.type,
                    "direction": "outgoing",
                    "correlation_id": correlation_id,
                    "content_preview": str(content)[:100] + "..." if len(str(content)) > 100 else str(content)
                }
                sender_span = traces_flow_manager.create_a2a_trace(
                    "Message Sent",
                    attributes
                )
            except Exception as e:
                print(f"Failed to create sender A2A trace: {e}")

        # Store pending request on the TARGET agent ONLY if it has a model_response handler.
        # This means the target will produce an LLM response that needs routing back.
        # Don't store for messages like "specialist_response" where no LLM response is expected.
        if hasattr(target_agent, 'a2a') and "model_response" in target_agent.a2a._message_handlers:
            target_agent.a2a._pending_requests[correlation_id] = PendingRequest(
                correlation_id=correlation_id,
                from_agent=self.agent.id,
                to_agent=to_agent,
                created_at=time.time(),
            )

        if message_type == "specialist_query":
            sender_is_realtime = False
            if hasattr(self.agent, 'session') and hasattr(self.agent.session, 'pipeline'):
                sender_is_realtime = getattr(self.agent.session.pipeline.config, 'is_realtime', False)

            receiver_is_realtime = False
            if hasattr(target_agent, 'session') and hasattr(target_agent.session, 'pipeline'):
                receiver_is_realtime = getattr(target_agent.session.pipeline.config, 'is_realtime', False)

            metrics_collector.set_a2a_handoff()

        receiver_span = None
        if traces_flow_manager:
            try:
                attributes = {
                    "from_agent": message.from_agent,
                    "to_agent": message.to_agent,
                    "message_type": message.type,
                    "direction": "incoming",
                    "correlation_id": correlation_id,
                    "content_preview": str(content)[:100] + "..." if len(str(content)) > 100 else str(content)
                }
                receiver_span = traces_flow_manager.create_a2a_trace(
                    "Message Received",
                    attributes
                )
            except Exception as e:
                print(f"Failed to create receiver A2A trace: {e}")

        # Handle message: invoke registered handlers or fallback to direct pipeline routing
        if hasattr(target_agent, 'a2a') and message_type in target_agent.a2a._message_handlers:
            handlers = target_agent.a2a._message_handlers[message_type]
            for handler_func in handlers:
                try:
                    await handler_func(message)
                except Exception as e:
                    logger.error(
                        f"Error in message handler for {message_type} on agent {to_agent}: {e}",
                        exc_info=True)

        elif message_type == "specialist_query" and hasattr(target_agent, 'session') and hasattr(target_agent.session, 'pipeline'):
            await target_agent.session.pipeline.send_text_message(content.get("query", ""))

        if traces_flow_manager:
            if receiver_span:
                traces_flow_manager.end_a2a_trace(
                    receiver_span,
                    f"Message from {message.from_agent} processed"
                )
            if sender_span:
                traces_flow_manager.end_a2a_trace(
                    sender_span,
                    f"Message to {message.to_agent} delivered"
                )

    async def _sweep_expired_requests(self) -> None:
        """Background task to clean up timed-out pending requests."""
        while True:
            try:
                await asyncio.sleep(5)
                now = time.time()
                expired = [
                    cid for cid, req in self._pending_requests.items()
                    if now - req.created_at > req.timeout
                ]
                for cid in expired:
                    req = self._pending_requests.pop(cid)
                    logger.warning(
                        f"A2A request {cid} from {req.from_agent} to {req.to_agent} "
                        f"timed out after {req.timeout}s"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in A2A sweep task: {e}", exc_info=True)
