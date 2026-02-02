# xgen_doc2chunk/core/processor/doc_helpers/__init__.py
"""
DOC Helper Module

Provides utilities needed for DOC document processing.

RTF-related modules have been moved to rtf_helper.
If RTF processing is needed, use rtf_helper:
    from xgen_doc2chunk.core.processor import rtf_helper
    from xgen_doc2chunk.core.processor.rtf_helper import RTFParser

Module Structure:
- doc_file_converter: DOC file converter
- doc_image_processor: DOC image processor
"""

# DOC-specific components
from xgen_doc2chunk.core.processor.doc_helpers.doc_file_converter import DOCFileConverter
from xgen_doc2chunk.core.processor.doc_helpers.doc_image_processor import DOCImageProcessor

__all__ = [
    'DOCFileConverter',
    'DOCImageProcessor',
]

