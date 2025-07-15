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
import json
@dataclass
class FunctionToolInfo:
    """Metadata for a function tool"""
    name: str
    description: str | None
    parameters_schema: Optional[dict] = None

@runtime_checkable
class FunctionTool(Protocol):
    """Protocol defining what makes a function tool"""
    _tool_info: FunctionToolInfo

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

def is_function_tool(obj: Any) -> bool:
    """Check if an object is a function tool"""
    return hasattr(obj, "_tool_info")

def get_tool_info(tool: FunctionTool) -> FunctionToolInfo:
    """Get the tool info from a function tool"""
    if not is_function_tool(tool):
        raise ValueError("Object is not a function tool")
    return getattr(tool, "_tool_info")

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
        
        setattr(wrapper, "_tool_info", tool_info)
        return wrapper

    
    if func is None:
        return lambda f: create_wrapper(f)
    
    return create_wrapper(func)

def build_pydantic_args_model(func: Callable[..., Any]) -> type[BaseModel]:
    """
    Dynamically construct a Pydantic BaseModel class representing all
    valid positional arguments of the given function, complete with types,
    default values, and docstring descriptions.
    """
    name_parts = func.__name__.split("_")
    class_name = "".join(part.title() for part in name_parts) + "Args"

    docs = parse_from_object(func)
    descriptions = {param.arg_name: param.description for param in docs.params}

    sig = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)

    model_fields: dict[str, Any] = {}
    for arg, param in sig.parameters.items():
        if arg in ("self", "cls") or arg not in hints:
            continue

        hint = hints[arg]
        default = param.default if param.default is not inspect.Parameter.empty else ...
        field_info = Field()

        if get_origin(hint) is Annotated:
            base_type, *extras = get_args(hint)
            hint = base_type
            for extra in extras:
                if isinstance(extra, FieldInfo):
                    field_info = extra
                    break

        if default is not ... and field_info.default is PydanticUndefined:
            field_info.default = default

        if field_info.description is None:
            field_info.description = descriptions.get(arg)

        model_fields[arg] = (hint, field_info)

    return create_model(class_name, **model_fields)

def build_openai_schema(function_tool: FunctionTool) -> dict[str, Any]:
    """Build OpenAI-compatible schema from a function tool"""
    tool_info = get_tool_info(function_tool)
    
    params_schema_to_use: Optional[dict] = None

    if tool_info.parameters_schema is not None:
        params_schema_to_use = tool_info.parameters_schema
    else:
        model = build_pydantic_args_model(function_tool)
        params_schema_to_use = model.model_json_schema()

    final_params_schema = params_schema_to_use if params_schema_to_use is not None else {"type": "object", "properties": {}}

    return {
            "name": tool_info.name,
            "description": tool_info.description or "",
            "parameters": final_params_schema,
            "type": "function",
    }
        

class _GeminiJsonSchema:
    """
    Transforms a JSON Schema into a format that is suitable for Gemini models.
    """

    _TYPE_MAPPING: dict[str, types.Type] = {
        "string": types.Type.STRING,
        "number": types.Type.NUMBER,
        "integer": types.Type.INTEGER,
        "boolean": types.Type.BOOLEAN,
        "array": types.Type.ARRAY,
        "object": types.Type.OBJECT,
    }

    _SUPPORTED_TYPES = set(_TYPE_MAPPING.keys())
    _FIELDS_TO_REMOVE = ("title", "default", "additionalProperties", "$defs")

    def __init__(self, schema: dict[str, Any]):
        self._schema = schema.copy()

    def simplify(self) -> dict[str, Any] | None:
        """
        Simplifies the schema to the Gemini format by modifying it in place.
        """
        self._simplify_node(self._schema)
        
        if (
            self._schema.get("type") == types.Type.OBJECT
            and not self._schema.get("properties")
        ):
            return None
            
        return self._schema

    def _simplify_node(self, schema_node: dict[str, Any]) -> None:
        """
        Recursively simplifies a node within the schema.
        """
        for field in self._FIELDS_TO_REMOVE:
            schema_node.pop(field, None)

        json_type = schema_node.get("type")
        if isinstance(json_type, str) and json_type in self._SUPPORTED_TYPES:
            schema_node["type"] = self._TYPE_MAPPING[json_type]

        node_type = schema_node.get("type")
        if node_type == types.Type.OBJECT:
            if properties := schema_node.get("properties"):
                for prop_schema in properties.values():
                    self._simplify_node(prop_schema)
        elif node_type == types.Type.ARRAY:
            if items := schema_node.get("items"):
                self._simplify_node(items)

def build_gemini_schema(function_tool: FunctionTool) -> types.FunctionDeclaration:
    """Build Gemini-compatible schema from a function tool"""
    tool_info = get_tool_info(function_tool)
    
    parameter_json_schema_for_gemini: Optional[dict[str, Any]] = None

    if tool_info.parameters_schema is not None:
         if tool_info.parameters_schema and tool_info.parameters_schema.get("properties", True) is not None:
            simplified_schema = _GeminiJsonSchema(tool_info.parameters_schema).simplify()
            parameter_json_schema_for_gemini = simplified_schema
    else:
        openai_schema = build_openai_schema(function_tool) 

        if openai_schema.get("parameters") and openai_schema["parameters"].get("properties", True) is not None:
             simplified_schema = _GeminiJsonSchema(openai_schema["parameters"]).simplify()
             parameter_json_schema_for_gemini = simplified_schema

    func_declaration = types.FunctionDeclaration(
        name=tool_info.name, 
        description=tool_info.description or "", 
        parameters=parameter_json_schema_for_gemini 
    )
    return func_declaration
    
ToolChoice = Literal["auto", "required", "none"]

def build_mcp_schema(function_tool: FunctionTool) -> dict:
    """Convert function tool to MCP schema"""
    tool_info = get_tool_info(function_tool)
    return {
        "name": tool_info.name,
        "description": tool_info.description,
        "parameters": build_pydantic_args_model(function_tool).model_json_schema()
    }

class ToolError(Exception):
    """Exception raised when a tool execution fails"""
    pass    

class RawFunctionTool(Protocol):
    """Protocol for raw function tool without framework wrapper"""
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

def create_generic_mcp_adapter(
    tool_name: str, 
    tool_description: str | None, 
    input_schema: dict,
    client_call_function: Callable
) -> FunctionTool:
    """
    Create a generic adapter that converts an MCP tool to a framework FunctionTool.
    
    Args:
        tool_name: Name of the MCP tool
        tool_description: Description of the MCP tool (if available)
        input_schema: JSON schema for the tool's input parameters
        client_call_function: Function to call the tool on the MCP server
        
    Returns:
        A function tool that can be registered with the agent
    """
    required_params = input_schema.get('required', [])
    
    param_properties = input_schema.get('properties', {})
    
    docstring = tool_description or f"Call the {tool_name} tool"
    if param_properties and "Args:" not in docstring:
        param_docs = "\n\nArgs:\n"
        for param_name, param_info in param_properties.items():
            required = " (required)" if param_name in required_params else ""
            description = param_info.get('description', f"Parameter for {tool_name}")
            param_docs += f"    {param_name}{required}: {description}\n"
        docstring += param_docs
    
    if not param_properties:
        @function_tool(name=tool_name)
        async def no_param_tool() -> Any:
            return await client_call_function({})
        no_param_tool.__doc__ = docstring
        tool_info_no_param = get_tool_info(no_param_tool)
        tool_info_no_param.parameters_schema = input_schema
        return no_param_tool
    else:
        @function_tool(name=tool_name) 
        async def param_tool(**kwargs) -> Any:
            actual_kwargs = kwargs.copy() # Work with a copy

            if 'instructions' in required_params and 'instructions' not in actual_kwargs:
                other_params_provided = any(p in actual_kwargs for p in param_properties if p != 'instructions')
                if other_params_provided:
                    actual_kwargs['instructions'] = f"Execute tool {tool_name} with the provided parameters."

            
            missing = [p for p in required_params if p not in actual_kwargs]
            if missing:
                missing_str = ", ".join(missing)
                param_details = []
                for param in missing:
                    param_info = param_properties.get(param, {})
                    desc = param_info.get('description', f"Parameter for {tool_name}")
                    param_details.append(f"'{param}': {desc}")
                
                param_help = "; ".join(param_details)
                raise ToolError(
                    f"Missing required parameters for {tool_name}: {missing_str}. "
                    f"Required parameters: {param_help}"
                )
            return await client_call_function(actual_kwargs)
        param_tool.__doc__ = docstring
        tool_info_param = get_tool_info(param_tool)
        tool_info_param.parameters_schema = input_schema
        return param_tool

def build_nova_sonic_schema(function_tool: FunctionTool) -> dict[str, Any]:
    """Build Amazon Nova Sonic-compatible schema from a function tool"""
    tool_info = get_tool_info(function_tool)

    params_schema_to_use: Optional[dict] = None

    if tool_info.parameters_schema is not None:
        params_schema_to_use = tool_info.parameters_schema
    else:
        model = build_pydantic_args_model(function_tool)
        params_schema_to_use = model.model_json_schema()
    

    final_params_schema_for_nova = params_schema_to_use if params_schema_to_use is not None else {"type": "object", "properties": {}}
    input_schema_json_string = json.dumps(final_params_schema_for_nova)

    description = tool_info.description or tool_info.name

    return {
        "toolSpec": {
            "name": tool_info.name,
            "description": description,
            "inputSchema": {
                "json": input_schema_json_string
            }
        }
    }
