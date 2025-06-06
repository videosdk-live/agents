from __future__ import annotations

import io
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Literal, Optional
from PIL import Image as PILImage

import av  # For av.VideoFrame
import av.logging  # For logging control

# Suppress FFmpeg swscaler warnings about accelerated colorspace conversion
av.logging.set_level(av.logging.ERROR)

@dataclass
class EncodeOptions:
    """Options for encoding av.VideoFrame to portable image formats."""

    format: Literal["JPEG", "PNG"] = "JPEG"
    """The format to encode the image."""

    resize_options: ResizeOptions = field(default_factory=lambda: ResizeOptions(
        width=320,  # Match React scale factor
        height=240,
        strategy="scale_aspect_fit"
    ))
    """Options for resizing the image."""

    quality: int = 90  # Increase quality while maintaining performance
    """Image compression quality, 0-100. Only applies to JPEG."""


@dataclass
class ResizeOptions:
    """Options for resizing av.VideoFrame as part of encoding to a portable image format."""

    width: int
    """The desired resize width"""

    height: int
    """The desired height to resize the image to."""

    strategy: Literal[
        "center_aspect_fit",
        "center_aspect_cover",
        "scale_aspect_fit",
        "scale_aspect_cover",
        "skew",
    ] = "scale_aspect_fit"
    """The strategy to use when resizing the image:
    - center_aspect_fit: Fit the image into the provided dimensions, with letterboxing
    - center_aspect_cover: Fill the provided dimensions, with cropping
    - scale_aspect_fit: Fit the image into the provided dimensions, preserving its original aspect ratio
    - scale_aspect_cover: Fill the provided dimensions, preserving its original aspect ratio (image will be larger than the provided dimensions)
    - skew: Precisely resize the image to the provided dimensions
    """

def encode(frame: av.VideoFrame, options: EncodeOptions) -> bytes:
    """Encode with optimized pipeline"""
    img = frame.to_image()
    
    # Fast resize using LANCZOS filter for better quality
    if options.resize_options:
        img = img.resize(
            (options.resize_options.width, options.resize_options.height),
            resample=PILImage.Resampling.LANCZOS
        )
    
    # Optimized JPEG encoding
    buffer = io.BytesIO()
    img.save(buffer,
            format=options.format,
            quality=options.quality,
            optimize=True,  # Enable optimization
            subsampling=0,  # Keep chroma subsampling
            qtables="web_high"  # Use web-optimized quantization tables
    )
    return buffer.getvalue()
