from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import time
import uuid
from .card import AgentCard 
import asyncio
from ..event_bus import global_event_emitter

@dataclass
class A2AMessage:
    """Message format for agent-to-agent communication"""
    from_agent: str
    to_agent: str
    type: str
    content: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentRegistry:
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
        return [
            agent_id for agent_id, card in self.agents.items()
            if card.domain == domain
        ]
    
    def get_all_agents(self) -> Dict[str, AgentCard]:
        """Get all registered agents"""
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
        self._last_sender = None
        self._handled_responses = set()  
        self._global_event_handler = None  


    async def register(self, card: AgentCard) -> None:
        """Register the agent with the registry"""
        self.registry.register_agent(card,self.agent)

    async def unregister(self) -> None:
        """Unregister the agent and clean up event handlers"""
        for message_type in list(self._message_handlers.keys()):
            for handler in self._message_handlers[message_type]:
                self.off_message(message_type, handler)
        
        await self.registry.unregister_agent(self.agent.id)
        self._handled_responses.clear()

    def on_message(self, message_type: str, handler: Callable[[A2AMessage], None]) -> None:
        """Register a message handler for specific message types
        
        Args:
            message_type: Type of message to handle (e.g., "model_query", "model_response")
            handler: Async function to handle the message
        """
        if not callable(handler):
            raise ValueError("Handler must be a callable function")
            
        if not asyncio.iscoroutinefunction(handler):
            raise ValueError("Handler must be an async function")
            
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []

        if handler not in self._message_handlers[message_type]:
            self._message_handlers[message_type].append(handler)

            if message_type == "model_response" and hasattr(self.agent, 'session') and hasattr(self.agent.session, 'pipeline'):
                
                def on_model_response(data):
                    response = data.get('text', '')
                    
                    if not self._last_sender:
                        return
                        
                    response_id = f"{self.agent.id}_{self._last_sender}_{response}" 
                    
                    if response_id in self._handled_responses:
                        return
                        
                    self._handled_responses.add(response_id)
                    
                    message = A2AMessage(
                        from_agent=self.agent.id,
                        to_agent=self._last_sender,
                        type="model_response",
                        content={"response": response}
                    )
                    asyncio.create_task(handler(message))
                
                self._global_event_handler = on_model_response
                global_event_emitter.on("text_response", on_model_response)

    def off_message(self, message_type: str, handler: Callable[[A2AMessage], None]) -> None:
        """Unregister a message handler"""
        if message_type in self._message_handlers:
            if handler in self._message_handlers[message_type]:
                self._message_handlers[message_type].remove(handler)
                
                if not self._message_handlers[message_type]:
                    del self._message_handlers[message_type]
                    
                if message_type == "model_response" and self._global_event_handler:
                    global_event_emitter.off("text_response", self._global_event_handler)
                    self._global_event_handler = None


    async def send_message(self, to_agent: str, message_type: str, content: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Send a message to another agent"""
        target_agent = self.registry.get_agent_instance(to_agent)

        if not target_agent:
            print(f"Target agent {to_agent} not found in registry.")
            return

        message = A2AMessage(
            from_agent=self.agent.id,
            to_agent=to_agent,
            type=message_type,
            content=content,
            metadata=metadata
        )

        if hasattr(target_agent, 'a2a'):
            target_agent.a2a._last_sender = self.agent.id

        if hasattr(target_agent, 'a2a') and message_type in target_agent.a2a._message_handlers:
            handlers = target_agent.a2a._message_handlers[message_type]
            for handler_func in handlers:
                try:
                    await handler_func(message)
                except Exception as e:
                    print(f"Error in message handler for {message_type} on agent {to_agent}: {e}")

        elif message_type == "specialist_query" and hasattr(target_agent, 'session') and hasattr(target_agent.session, 'pipeline'):
            if hasattr(target_agent.session.pipeline, 'model'):
                await target_agent.session.pipeline.send_text_message(content.get("query", ""))
            elif hasattr(target_agent.session.pipeline, 'send_text_message'):
                await target_agent.session.pipeline.send_text_message(content.get("query", ""))

