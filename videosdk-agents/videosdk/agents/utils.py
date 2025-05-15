from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable, Callable, Optional, get_type_hints, Annotated, get_origin, get_args, Literal
from functools import wraps
import inspect
from docstring_parser import parse_from_object
from google.genai import types
from pydantic import BaseModel, Field, create_model
from pydantic_core import PydanticUndefined
from pydantic.fields import FieldInfo

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

def function_arguments_to_pydantic_model(func: Callable) -> type[BaseModel]:
    """Create a Pydantic model from a function's signature."""
    # Create model name from function name
    fnc_name = func.__name__.split("_")
    fnc_name = "".join(x.capitalize() for x in fnc_name)
    model_name = fnc_name + "Args"

    # Parse docstring for parameter descriptions
    docstring = parse_from_object(func)
    param_docs = {p.arg_name: p.description for p in docstring.params}

    # Get function signature and type hints
    signature = inspect.signature(func)
    type_hints = get_type_hints(func, include_extras=True)

    # Build fields dictionary for model creation
    fields: dict[str, Any] = {}

    for param_name, param in signature.parameters.items():
        if param_name in ('self', 'cls'):
            continue

        if param_name not in type_hints:
            continue

        type_hint = type_hints[param_name]
        default_value = param.default if param.default is not param.empty else ...
        field_info = Field()

        # Handle Annotated types
        if get_origin(type_hint) is Annotated:
            annotated_args = get_args(type_hint)
            type_hint = annotated_args[0]
            # Get field info from annotations if present
            field_info = next(
                (x for x in annotated_args[1:] if isinstance(x, FieldInfo)), 
                field_info
            )

        # Set default value if present
        if default_value is not ... and field_info.default is PydanticUndefined:
            field_info.default = default_value

        # Set description from docstring
        if field_info.description is None:
            field_info.description = param_docs.get(param_name, None)

        fields[param_name] = (type_hint, field_info)

    return create_model(model_name, **fields)

def build_openai_schema(function_tool: FunctionTool) -> dict[str, Any]:
    """Build OpenAI-compatible schema from a function tool"""
    # Convert function to Pydantic model
    model = function_arguments_to_pydantic_model(function_tool)
    tool_info = get_tool_info(function_tool)
    
    # Get JSON schema from model
    schema = model.model_json_schema()

    return {
            "name": tool_info.name,
            "description": tool_info.description or "",
            "parameters": schema,
            "type": "function",
    }
        

class _GeminiJsonSchema:
    """Transforms JSON Schema to be suitable for Gemini."""
    
    TYPE_MAPPING: dict[str, types.Type] = {
        "string": types.Type.STRING,
        "number": types.Type.NUMBER,
        "integer": types.Type.INTEGER,
        "boolean": types.Type.BOOLEAN,
        "array": types.Type.ARRAY,
        "object": types.Type.OBJECT,
    }

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema.copy()
        self.defs = self.schema.pop("$defs", {})

    def simplify(self) -> dict[str, Any] | None:
        """Simplify the schema to Gemini format"""
        self._simplify(self.schema, refs_stack=())
        if self.schema.get("type") == types.Type.OBJECT and not self.schema.get("properties"):
            return None
        return self.schema

    def _simplify(self, schema: dict[str, Any], refs_stack: tuple[str, ...]) -> None:
        """Internal method to simplify schema recursively"""
        # Remove unnecessary fields
        for field in ["title", "default", "additionalProperties"]:
            schema.pop(field, None)

        # Handle type conversion
        if "type" in schema and schema["type"] != "null":
            json_type = schema["type"]
            if json_type in self.TYPE_MAPPING:
                schema["type"] = self.TYPE_MAPPING[json_type]

        # Handle nested objects and arrays
        type_ = schema.get("type")
        if type_ == types.Type.OBJECT:
            if properties := schema.get("properties"):
                for value in properties.values():
                    self._simplify(value, refs_stack)
        elif type_ == types.Type.ARRAY:
            if items := schema.get("items"):
                self._simplify(items, refs_stack)

def build_gemini_schema(function_tool: FunctionTool) -> types.FunctionDeclaration:
    """Build Gemini-compatible schema from a function tool"""
    # Get OpenAI schema first
    openai_schema = build_openai_schema(function_tool)
    
    # Convert to Gemini format
    json_schema = _GeminiJsonSchema(openai_schema["parameters"]).simplify()
    
    # Create FunctionDeclaration
    return types.FunctionDeclaration(
        name=openai_schema["name"],
        description=openai_schema["description"],
        parameters=json_schema
    )
    
ToolChoice = Literal["auto", "required", "none"]
