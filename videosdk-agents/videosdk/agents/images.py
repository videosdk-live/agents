from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Literal
from PIL import Image as PILImage

import av  
import av.logging  

av.logging.set_level(av.logging.ERROR)

@dataclass
class EncodeOptions:
    """Configuration settings for converting av.VideoFrame into standard image formats."""

    format: Literal["JPEG", "PNG"] = "JPEG"
    """The encoding format for the image."""

    resize_options: ResizeOptions = field(default_factory=lambda: ResizeOptions(
        width=320,  
        height=240
    ))
    """Settings for adjusting the image size."""

    quality: int = 90 
    """Compression level for the image, ranging from 0 to 100. Applicable only to JPEG."""


@dataclass
class ResizeOptions:
    """Configuration for resizing av.VideoFrame during the process of encoding to a standard image format."""

    width: int
    """The target width for resizing"""

    height: int
    """The target height for resizing the image."""

def encode(frame: av.VideoFrame, options: EncodeOptions) -> bytes:
    """Encode with optimized pipeline"""
    img = frame.to_image()
    
    
    if options.resize_options:
        img = img.resize(
            (options.resize_options.width, options.resize_options.height),
            resample=PILImage.Resampling.LANCZOS
        )
    
    
    buffer = io.BytesIO()
    img.save(buffer,
            format=options.format,
            quality=options.quality,
            optimize=True,  
            subsampling=0,  
            qtables="web_high"
    )
    return buffer.getvalue()
