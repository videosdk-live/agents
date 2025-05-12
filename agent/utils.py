from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable, Callable, Optional
from functools import wraps
import inspect
from inspect import signature, Parameter
from docstring_parser import parse

@dataclass
class FunctionToolInfo:
    """Metadata for a function tool"""
    name: str
    description: str | None

@runtime_checkable
class FunctionTool(Protocol):
    """Protocol defining what makes a function tool"""
    __tool_info: FunctionToolInfo

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

def is_function_tool(obj: Any) -> bool:
    """Check if an object is a function tool"""
    return hasattr(obj, "__tool_info")

def get_tool_info(tool: FunctionTool) -> FunctionToolInfo:
    """Get the tool info from a function tool"""
    if not is_function_tool(tool):
        raise ValueError("Object is not a function tool")
    return getattr(tool, "__tool_info")

def function_tool(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """Decorator to mark a function as a tool. Can be used with or without parentheses."""
    
    def create_wrapper(fn: Callable) -> FunctionTool:
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            return await fn(*args, **kwargs)
            
        tool_info = FunctionToolInfo(
            name=name or fn.__name__,
            description=fn.__doc__
        )
        
        setattr(wrapper, "__tool_info", tool_info)
        return wrapper

    # Handle both @function_tool and @function_tool() syntax
    if func is None:
        return lambda f: create_wrapper(f)
    
    return create_wrapper(func)

def build_openai_schema(function_tool: FunctionTool) -> dict[str, Any]:
    """Build OpenAI-compatible schema from a function tool"""
    tool_info = get_tool_info(function_tool)
    sig = signature(function_tool)
    docstring = parse(function_tool.__doc__ or "")
    
    param_desc = {param.arg_name: param.description for param in docstring.params}
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for name, param in sig.parameters.items():
        if name == 'self' or name == 'cls':
            continue
            
        parameters["properties"][name] = {
            "type": "string",
            "description": param_desc.get(name, f"Parameter: {name}")
        }
        
        if param.default == Parameter.empty:
            parameters["required"].append(name)

    return {
        "type": "function",
        "name": tool_info.name,
        "description": docstring.short_description or tool_info.description or "",
        "parameters": parameters
    }