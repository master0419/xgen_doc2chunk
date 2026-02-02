# xgen_doc2chunk/core/processor/__init__.py
"""
Processor - Document Type-specific Handler Module

Provides handlers for processing individual document formats.

Handler List:
- pdf_handler: PDF document processing (adaptive complexity-based)
- docx_handler: DOCX document processing
- doc_handler: DOC document processing (OLE, HTML, misnamed DOCX)
- rtf_handler: RTF document processing
- ppt_handler: PPT/PPTX document processing
- excel_handler: Excel (XLSX/XLS) document processing
- hwp_processor: HWP document processing
- hwpx_processor: HWPX document processing
- csv_handler: CSV file processing
- text_handler: Text file processing
- html_reprocessor: HTML reprocessing

Helper Modules (subdirectories):
- csv_helper/: CSV processing helper
- docx_helper/: DOCX processing helper
- doc_helpers/: DOC processing helper
- rtf_helper/: RTF processing helper
- excel_helper/: Excel processing helper
- hwp_helper/: HWP processing helper
- hwpx_helper/: HWPX processing helper
- pdf_helpers/: PDF processing helper
- ppt_helper/: PPT processing helper

Usage Example:
    from xgen_doc2chunk.core.processor import PDFHandler
    from xgen_doc2chunk.core.processor import DOCXHandler
    from xgen_doc2chunk.core.processor import RTFHandler
    from xgen_doc2chunk.core.processor.pdf_helpers import extract_pdf_metadata
"""

# === PDF Handler ===
from xgen_doc2chunk.core.processor.pdf_handler import PDFHandler

# === Document Handlers ===
from xgen_doc2chunk.core.processor.docx_handler import DOCXHandler
from xgen_doc2chunk.core.processor.doc_handler import DOCHandler
from xgen_doc2chunk.core.processor.rtf_handler import RTFHandler
from xgen_doc2chunk.core.processor.ppt_handler import PPTHandler

# === Data Handlers ===
from xgen_doc2chunk.core.processor.excel_handler import ExcelHandler
from xgen_doc2chunk.core.processor.csv_handler import CSVHandler
from xgen_doc2chunk.core.processor.text_handler import TextHandler

# === HWP Handlers ===
from xgen_doc2chunk.core.processor.hwp_handler import HWPHandler
from xgen_doc2chunk.core.processor.hwpx_handler import HWPXHandler

# === Other Processors ===
# from xgen_doc2chunk.core.processor.html_reprocessor import ...  # HTML reprocessing

# === Helper Modules (subpackages) ===
from xgen_doc2chunk.core.processor import csv_helper
from xgen_doc2chunk.core.processor import doc_helpers
from xgen_doc2chunk.core.processor import docx_helper
from xgen_doc2chunk.core.processor import excel_helper
from xgen_doc2chunk.core.processor import hwp_helper
from xgen_doc2chunk.core.processor import hwpx_helper
from xgen_doc2chunk.core.processor import pdf_helpers
from xgen_doc2chunk.core.processor import ppt_helper
from xgen_doc2chunk.core.processor import rtf_helper

__all__ = [
    # PDF Handler
    "PDFHandler",
    # Document Handlers
    "DOCXHandler",
    "DOCHandler",
    "RTFHandler",
    "PPTHandler",
    # Data Handlers
    "ExcelHandler",
    "CSVHandler",
    "TextHandler",
    # HWP Handlers
    "HWPHandler",
    "HWPXHandler",
    # Helper subpackages
    "csv_helper",
    "doc_helpers",
    "docx_helper",
    "excel_helper",
    "hwp_helper",
    "hwpx_helper",
    "pdf_helpers",
    "ppt_helper",
    "rtf_helper",
]

