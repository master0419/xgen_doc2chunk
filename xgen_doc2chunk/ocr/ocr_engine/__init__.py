# xgen_doc2chunk/ocr/ocr_engine/__init__.py
# OCR engine module initialization
"""
OCR Engine Module

Provides OCR engine classes for each LLM provider.
"""

from xgen_doc2chunk.ocr.ocr_engine.openai_ocr import OpenAIOCR
from xgen_doc2chunk.ocr.ocr_engine.anthropic_ocr import AnthropicOCR
from xgen_doc2chunk.ocr.ocr_engine.gemini_ocr import GeminiOCR
from xgen_doc2chunk.ocr.ocr_engine.vllm_ocr import VllmOCR
from xgen_doc2chunk.ocr.ocr_engine.bedrock_ocr import BedrockOCR

__all__ = [
    "OpenAIOCR",
    "AnthropicOCR",
    "GeminiOCR",
    "VllmOCR",
    "BedrockOCR",
]

