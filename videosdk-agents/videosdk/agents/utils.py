from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable, Callable, Optional, get_type_hints, Annotated, get_origin, get_args, Literal, AsyncIterator
from functools import wraps
import inspect
from docstring_parser import parse_from_object
from google.genai import types
from pydantic import BaseModel, Field, create_model
from pydantic_core import PydanticUndefined
from pydantic.fields import FieldInfo
from abc import abstractmethod
import json
import asyncio

@dataclass
class FunctionToolInfo:
    name: str
    description: str | None = None
    parameters_schema: Optional[dict] = None

@runtime_checkable
class FunctionTool(Protocol):
    @property
    @abstractmethod
    def _tool_info(self) -> "FunctionToolInfo": ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

def is_function_tool(obj: Any) -> bool:
    """Check if an object is a function tool"""
    if inspect.ismethod(obj):
        obj = obj.__func__
    return hasattr(obj, "_tool_info")

def get_tool_info(tool: FunctionTool) -> FunctionToolInfo:
    """Get the tool info from a function tool"""
    if not is_function_tool(tool):
        raise ValueError("Object is not a function tool")
    
    if inspect.ismethod(tool):
        tool = tool.__func__
    return getattr(tool, "_tool_info")

def function_tool(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """Decorator to mark a function as a tool. Can be used with or without parentheses."""
    
    def create_wrapper(fn: Callable) -> FunctionTool:
        tool_info = FunctionToolInfo(
            name=name or fn.__name__,
            description=fn.__doc__
        )
        
        if asyncio.iscoroutinefunction(fn):
            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                return await fn(*args, **kwargs)
            
            setattr(async_wrapper, "_tool_info", tool_info)
            return async_wrapper
        else:
            @wraps(fn)
            def sync_wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            
            setattr(sync_wrapper, "_tool_info", tool_info)
            return sync_wrapper
    
    if func is None:
        return create_wrapper
    
    return create_wrapper(func)

def build_pydantic_args_model(func: Callable[..., Any]) -> type[BaseModel]:
    """
    Dynamically construct a Pydantic BaseModel class representing all
    valid positional arguments of the given function, complete with types,
    default values, and docstring descriptions.
    """
    return ModelBuilder(func).construct()

class ModelBuilder:
    def __init__(self, func: Callable[..., Any]):
        self.func = func
        self.sig = inspect.signature(func)
        self.hints = get_type_hints(func, include_extras=True)
        self.docs = parse_from_object(func)
        self._field_registry = {}
    
    def construct(self) -> type[BaseModel]:
        self._register_all_fields()
        return create_model(self._generate_class_name(), **self._field_registry)
    
    def _register_all_fields(self) -> None:
        for name, param in self.sig.parameters.items():
            if self._should_skip_param(name):
                continue
            self._register_field(name, param)
    
    def _should_skip_param(self, name: str) -> bool:
        return name in ("self", "cls") or name not in self.hints
    
    def _register_field(self, name: str, param: inspect.Parameter) -> None:
        type_info = TypeProcessor(self.hints[name])
        field_builder = FieldBuilder(param, type_info, self._find_doc_description(name))
        self._field_registry[name] = field_builder.build()
    
    def _find_doc_description(self, param_name: str) -> str | None:
        return next(
            (p.description for p in self.docs.params if p.arg_name == param_name),
            None
        )
    
    def _generate_class_name(self) -> str:
        return "".join(part.title() for part in self.func.__name__.split("_")) + "Args"

class TypeProcessor:
    def __init__(self, hint: Any):
        self.original_hint = hint
        self.base_type = hint
        self.field_info = Field()
        self._process_annotation()
    
    def _process_annotation(self) -> None:
        if get_origin(self.original_hint) is Annotated:
            args = get_args(self.original_hint)
            self.base_type = args[0]
            self._extract_field_info_from_args(args[1:])
    
    def _extract_field_info_from_args(self, args: tuple) -> None:
        for arg in args:
            if isinstance(arg, FieldInfo):
                self.field_info = arg
                break

class FieldBuilder:
    def __init__(self, param: inspect.Parameter, type_processor: TypeProcessor, description: str | None):
        self.param = param
        self.type_processor = type_processor
        self.description = description
    
    def build(self) -> tuple[Any, FieldInfo]:
        field_info = self.type_processor.field_info
        self._apply_default_if_needed(field_info)
        self._apply_description_if_needed(field_info)
        return (self.type_processor.base_type, field_info)
    
    def _apply_default_if_needed(self, field_info: FieldInfo) -> None:
        if (self.param.default is not inspect.Parameter.empty and 
            field_info.default is PydanticUndefined):
            field_info.default = self.param.default
    
    def _apply_description_if_needed(self, field_info: FieldInfo) -> None:
        if field_info.description is None and self.description is not None:
            field_info.description = self.description

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
        

def simplify_gemini_schema(schema: dict[str, Any]) -> dict[str, Any] | None:
    """
    Transforms a JSON Schema into Gemini compatible format.
    """

    TYPE_MAPPING: dict[str, types.Type] = {
        "string": types.Type.STRING,
        "number": types.Type.NUMBER,
        "integer": types.Type.INTEGER,
        "boolean": types.Type.BOOLEAN,
        "array": types.Type.ARRAY,
        "object": types.Type.OBJECT,
    }

    SUPPORTED_TYPES = set(TYPE_MAPPING.keys())
    FIELDS_TO_REMOVE = ("title", "default", "additionalProperties", "$defs")

    def process_node(node: dict[str, Any]) -> dict[str, Any] | None:
        new_node = node.copy()
        for field in FIELDS_TO_REMOVE:
            if field in new_node:
                del new_node[field]

        json_type = new_node.get("type")
        if json_type in SUPPORTED_TYPES:
            new_node["type"] = TYPE_MAPPING[json_type]

        node_type = new_node.get("type")
        if node_type == types.Type.OBJECT:
            if "properties" in new_node:
                new_props = {}
                for key, prop in new_node["properties"].items():
                    simplified = process_node(prop)
                    if simplified is not None:
                        new_props[key] = simplified
                new_node["properties"] = new_props
                if not new_props:
                    return None
            else:
                return None
        elif node_type == types.Type.ARRAY:
            if "items" in new_node:
                simplified_items = process_node(new_node["items"])
                if simplified_items is not None:
                    new_node["items"] = simplified_items
                else:
                    del new_node["items"]

        return new_node

    result = process_node(schema)
    if result and result.get("type") == types.Type.OBJECT and not result.get("properties"):
        return None
    return result

def build_gemini_schema(function_tool: FunctionTool) -> types.FunctionDeclaration:
    """Build Gemini-compatible schema from a function tool"""
    tool_info = get_tool_info(function_tool)
    
    parameter_json_schema_for_gemini: Optional[dict[str, Any]] = None

    if tool_info.parameters_schema is not None:
         if tool_info.parameters_schema and tool_info.parameters_schema.get("properties", True) is not None:
            simplified_schema = simplify_gemini_schema(tool_info.parameters_schema)
            parameter_json_schema_for_gemini = simplified_schema
    else:
        openai_schema = build_openai_schema(function_tool) 

        if openai_schema.get("parameters") and openai_schema["parameters"].get("properties", True) is not None:
             simplified_schema = simplify_gemini_schema(openai_schema["parameters"])
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

async def segment_text(
    chunks: AsyncIterator[str],
    delimiters: str = ".?!,;:\n",
    keep_delimiter: bool = True,
    min_chars: int = 50,
    min_words: int = 12,
    max_buffer: int = 600,
) -> AsyncIterator[str]:
    """
    Segment an async stream of text on delimiters or soft boundaries to reduce TTS latency.
    Yields segments while keeping the delimiter if requested.
    """
    buffer = ""

    def words_count(s: str) -> int:
        return len(s.split())

    def find_first_delim_index(s: str) -> int:
        indices = [i for d in delimiters if (i := s.find(d)) != -1]
        return min(indices) if indices else -1

    async for chunk in chunks:
        if not chunk:
            continue
        buffer += chunk

        while True:
            di = find_first_delim_index(buffer)
            if di != -1:
                seg = buffer[: di + (1 if keep_delimiter else 0)]
                yield seg
                buffer = buffer[di + 1 :].lstrip()
                continue
            else:
                if len(buffer) >= max_buffer or words_count(buffer) >= (min_words * 2):
                    target = max(min_chars, min(len(buffer), max_buffer))
                    cut_idx = buffer.rfind(" ", 0, target)
                    if cut_idx == -1:
                        cut_idx = target
                    seg = buffer[:cut_idx].rstrip()
                    if seg:
                        yield seg
                    buffer = buffer[cut_idx:].lstrip()
                    continue
                break

    if buffer:
        yield buffer

async def graceful_cancel(*tasks: asyncio.Task) -> None:
    """Simple utility to cancel tasks and wait for them to complete"""
    if not tasks:
        return

    for task in tasks:
        if not task.done():
            task.cancel()
    
    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=0.5
        )
    except asyncio.TimeoutError:
        pass