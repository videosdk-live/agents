"""Helper functions for pipeline component management."""
from typing import Any, Callable
import logging
from .realtime_base_model import RealtimeBaseModel
from .realtime_llm_adapter import RealtimeLLMAdapter

logger = logging.getLogger(__name__)

NO_CHANGE = object()


async def cleanup_pipeline(pipeline, llm_changing: bool = False) -> None:
    """Close existing pipeline execution and all components."""
    if pipeline.orchestrator:
        logger.info("Closing existing orchestrator")
        await pipeline.orchestrator.cleanup()
        pipeline.orchestrator = None
    
    if pipeline.speech_generation:
        logger.info("Closing existing speech generation")
        await pipeline.speech_generation.cleanup()
        pipeline.speech_generation = None
    
    components = ['stt', 'llm', 'tts', 'vad', 'turn_detector', 'denoise']
    for attr in components:
        comp = getattr(pipeline, attr, None)
        if comp:
            logger.info(f"Closing component: {attr}")
            try:
                if hasattr(comp, 'aclose'):
                    await comp.aclose()
                elif hasattr(comp, 'cleanup'):
                    await comp.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up component {attr}: {e}")
            setattr(pipeline, attr, None)

    if llm_changing and pipeline._realtime_model:
        logger.info("Closing previous realtime model")
        try:
            await pipeline._realtime_model.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up realtime model: {e}")
        pipeline._realtime_model = None


def check_mode_shift(pipeline, llm: Any, stt: Any, tts: Any) -> bool:
    """Check if component changes trigger a mode shift."""
    if llm is not NO_CHANGE:
        is_new_llm_realtime = isinstance(llm, RealtimeBaseModel)
        if is_new_llm_realtime != pipeline._is_realtime_mode:
            return True
    if pipeline._is_realtime_mode:
        if stt is not NO_CHANGE and (pipeline.stt is None) != (stt is None):
            return True
        if tts is not NO_CHANGE and (pipeline.tts is None) != (tts is None):
            return True
    
    return False


async def swap_component_in_orchestrator(
    pipeline,
    component_name: str,
    new_value: Any,
    orchestrator_attr: str,
    lock_attr: str = None,
    post_swap_hook: Callable[[Any, Any], None] = None
) -> None:
    """
    Generic component swap with orchestrator lock.
    
    Args:
        pipeline: Pipeline instance
        component_name: Name of the component attribute (e.g., 'stt', 'vad')
        new_value: New component instance
        orchestrator_attr: Orchestrator sub-component name (e.g., 'speech_understanding')
        lock_attr: Lock attribute name (e.g., 'stt_lock')
        post_swap_hook: Optional callback after swap: hook(new_value, container)
    """
    logger.info(f"Changing {component_name} component")
    if new_value is NO_CHANGE:
        return
    old_value = getattr(pipeline, component_name)
    
    if pipeline.orchestrator:
        container = getattr(pipeline.orchestrator, orchestrator_attr, None)
        if container:
            async def _do_swap():
                if old_value:
                    await old_value.aclose()
                setattr(pipeline, component_name, new_value)
                setattr(container, component_name, new_value)
                
                if post_swap_hook:
                    post_swap_hook(new_value, container)

            if lock_attr:
                async with getattr(container, lock_attr):
                    await _do_swap()
            else:
                await _do_swap()
            return
    
    # Fallback
    if old_value:
        await old_value.aclose()
    setattr(pipeline, component_name, new_value)


def register_stt_transcript_listener(stt: Any, container: Any) -> None:
    """Hook to register STT transcript listener after swap."""
    if stt is None:
        return
    if hasattr(stt, 'on_stt_transcript'):
        stt.on_stt_transcript(container._on_stt_transcript)


async def swap_tts(pipeline, new_tts: Any) -> None:
    """
    Swap TTS component (handles multiple possible locations).
    
    TTS can be in:
    - pipeline.speech_generation (Hybrid/Cascading mode)
    - pipeline.orchestrator.speech_generation
    - Direct attribute only
    """
    
    if new_tts is NO_CHANGE:
        return
    swap_done = False

    if pipeline.speech_generation:
        async with pipeline.speech_generation.tts_lock:
            if pipeline.tts:
                await pipeline.tts.aclose()
            pipeline.tts = new_tts
            pipeline.speech_generation.tts = new_tts
            pipeline._configure_components()
            swap_done = True

    if not swap_done and pipeline.orchestrator and pipeline.orchestrator.speech_generation:
        async with pipeline.orchestrator.speech_generation.tts_lock:
            if pipeline.tts:
                await pipeline.tts.aclose()
            pipeline.tts = new_tts
            pipeline.orchestrator.speech_generation.tts = new_tts
            pipeline._configure_components()
            swap_done = True
    
    # Fallback: direct swap only
    if not swap_done:
        if pipeline.tts:
            await pipeline.tts.aclose()
        pipeline.tts = new_tts
        pipeline._configure_components()


async def swap_llm(pipeline, new_llm: Any) -> None:
    """
    Swap LLM component (handles realtime model logic).
    
    If new_llm is RealtimeBaseModel, wraps it in RealtimeLLMAdapter.
    """
    if new_llm is NO_CHANGE:
        return
    
    if pipeline.orchestrator and pipeline.orchestrator.content_generation:
        async with pipeline.orchestrator.content_generation.llm_lock:
            if pipeline.llm:
                await pipeline.llm.aclose()
            
            if isinstance(new_llm, RealtimeBaseModel):
                pipeline._realtime_model = new_llm
                pipeline.llm = RealtimeLLMAdapter(new_llm)
            else:
                pipeline._realtime_model = None
                pipeline.llm = new_llm
            
            pipeline.orchestrator.content_generation.llm = pipeline.llm
    else:
        # Direct swap
        if pipeline.llm:
            await pipeline.llm.aclose()
        
        if isinstance(new_llm, RealtimeBaseModel):
            pipeline._realtime_model = new_llm
            pipeline.llm = RealtimeLLMAdapter(new_llm)
        else:
            pipeline._realtime_model = None
            pipeline.llm = new_llm