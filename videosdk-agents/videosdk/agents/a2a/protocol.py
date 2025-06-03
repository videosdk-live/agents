# videosdk/agents/a2a/protocol.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import time
import uuid
from .card import AgentCard 
import asyncio
from ..event_bus import global_event_emitter

@dataclass
class AgentCapability:
    domain: str
    capabilities: List[str]
    description: str


@dataclass
class A2AMessage:
    """Message format for agent-to-agent communication"""
    from_agent: str
    to_agent: str
    type: str
    content: Dict[str, Any]
    id: str = str(uuid.uuid4())
    timestamp: float = time.time()
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentRegistry:
    _instance = None
    agents: Dict[str, List[AgentCard]] = field(default_factory=dict)
    agent_instances: Dict[str, 'Agent'] = field(default_factory=dict)  # Store agent instances

    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
            cls._instance.agents = {}
            cls._instance.agent_instances = {}
        return cls._instance

    def register_agent(self, card: AgentCard, agent_instance: 'Agent' = None):
        """Register an agent with its capabilities"""
        self.agents[card.id] = card
        if agent_instance:
            self.agent_instances[card.id] = agent_instance

    
    async def unregister_agent(self, agent_id: str):
        """Remove an agent from the registry"""
        self.agents.pop(agent_id, None)
        self.agent_instances.pop(agent_id, None)

    
    def find_agents_by_domain(self, domain: str) -> List[str]:
        """Find all agents that handle a specific domain"""
        print("Finding agents by domain::",domain,self.agents)
        return [
            agent_id for agent_id, card in self.agents.items()
            if card.domain == domain
        ]
    
    def get_all_agents(self) -> Dict[str, AgentCard]:
        """Get all registered agents"""
        print(f"Getting all registered agents: {self.agents}")
        return self.agents
    
    def get_agent_instance(self, agent_id: str) -> Optional['Agent']:
        """Get an agent instance by ID"""
        return self.agent_instances.get(agent_id)
    
    async def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find all agents that have a specific capability"""
        return [
            agent_id for agent_id, card in self.agents.items()
            if capability in card.capabilities
        ]



class A2AProtocol:
    """Handles agent-to-agent communication"""
    def __init__(self, agent: 'Agent'):
        self.agent = agent
        self.registry = AgentRegistry()
        self._message_handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self._last_sender = None
        self._handled_responses = set()  # Track handled responses
        # global_event_emitter.on("text_response", on_model_response)


    async def register(self, card: AgentCard) -> None:
        """Register the agent with the registry"""
        self.registry.register_agent(card,self.agent)
        # if not self._running:
        #     self._running = True
        #     asyncio.create_task(self._process_messages())

    async def unregister(self) -> None:
        """Unregister the agent and clean up event handlers"""
        # Clean up event handlers
        for message_type in list(self._message_handlers.keys()):
            for handler in self._message_handlers[message_type]:
                self.off_message(message_type, handler)
        
        await self.registry.unregister_agent(self.agent.id)
        self._running = False
        self._handled_responses.clear()  # Clear handled responses
    
    # def on_message(self, message_type: str, handler: Callable[[A2AMessage], None]) -> None:
    #     """Register a message handler"""
    #     if message_type not in self._message_handlers:
    #         self._message_handlers[message_type] = []
    #     self._message_handlers[message_type].append(handler)

    def on_message(self, message_type: str, handler: Callable[[A2AMessage], None]) -> None:
        """Register a message handler for specific message types
        
        Args:
            message_type: Type of message to handle (e.g., "model_query", "model_response")
            handler: Async function to handle the message
            
        Example:
            # Register handler for model responses
            self.a2a.on_message("model_response", self.handle_model_response)
            
            # Register handler for specialist queries
            self.a2a.on_message("specialist_query", self.handle_specialist_query)
        """
        if not callable(handler):
            raise ValueError("Handler must be a callable function")
            
        if not asyncio.iscoroutinefunction(handler):
            raise ValueError("Handler must be an async function")
            
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []

        print("!!!!!!!!!!  _message_handlers::",self._message_handlers)
        # Check if handler is already registered
        if handler not in self._message_handlers[message_type]:
            print("!!!!!!!!!!  message_type::",message_type)

            self._message_handlers[message_type].append(handler)
            print(f"Registered handler for message type: {message_type}")

            print(f"Registered self.agent: {self.agent.session}")

            
            # If this is a model_response handler, set up model response handling
            if message_type == "model_response" and hasattr(self.agent, 'session') and hasattr(self.agent.session, 'pipeline'):
                model = self.agent.session.pipeline.model
                if model and hasattr(model, 'on'):
                    def on_model_response(data):
                        response = data.get('text', '')
                        response_id = f"{self.agent.id}_{response}"  # Create unique response ID
                        
                        # Check if we've already handled this response
                        if response_id in self._handled_responses:
                            return
                            
                        self._handled_responses.add(response_id)
                        
                        # Create A2A message for the response
                        message = A2AMessage(
                            from_agent=self.agent.id,
                            to_agent=self._last_sender,
                            type="model_response",
                            content={"response": response}
                        )
                        # Call the handler with the message
                        asyncio.create_task(handler(message))
                    
                    print(">>> Registering the event emitter::",global_event_emitter)
                    # Register the model response handler
                    global_event_emitter.on("text_response", on_model_response)
                    print(f"Registered model response handler for agent: {self.agent.id}")

    def off_message(self, message_type: str, handler: Callable[[A2AMessage], None]) -> None:
        """Unregister a message handler
        
        Args:
            message_type: Type of message to unregister
            handler: Handler function to remove
        """
        if message_type in self._message_handlers:
            if handler in self._message_handlers[message_type]:
                self._message_handlers[message_type].remove(handler)
                print(f"Unregistered handler for message type: {message_type}")
                
                # If this was the last handler for this message type, remove the type
                if not self._message_handlers[message_type]:
                    del self._message_handlers[message_type]
                    
                # If this was a model_response handler, clean up model handler
                if message_type == "model_response" and hasattr(self.agent, 'session') and hasattr(self.agent.session, 'pipeline'):
                    model = self.agent.session.pipeline.model
                    if model and hasattr(model, 'off'):
                        model.off("text_response")
                        print(f"Unregistered model response handler for agent: {self.agent.id}")


    async def send_message(self, to_agent: str, message_type: str, content: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Send a message to another agent"""
        print(f"Sending message to: {to_agent}, type: {message_type}")
        
        # Get target agent from registry
        target_agent = self.registry.get_agent_instance(to_agent)

        print(">>> target_agent::",target_agent)

        if not target_agent:
            print(f"Target agent {to_agent} not found")
            return

        # Create A2A message
        message = A2AMessage(
            from_agent=self.agent.id,
            to_agent=to_agent,
            type=message_type,
            content=content,
            metadata=metadata
        )

        # Store the sender for response handling in target agent
        if hasattr(target_agent, 'a2a'):
            target_agent.a2a._last_sender = self.agent.id

        # Check if target agent has handlers for this message type
        if hasattr(target_agent, 'a2a') and message_type in target_agent.a2a._message_handlers:
            handlers = target_agent.a2a._message_handlers[message_type]
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    print(f"Error in message handler: {e}")

        # If target agent has a model and this is a specialist_query, forward directly to it
        elif message_type == "specialist_query" and hasattr(target_agent, 'session') and hasattr(target_agent.session, 'pipeline') and hasattr(target_agent.session.pipeline, 'model'):
            await target_agent.session.pipeline.send_text_message(content.get("query", ""))
            return

