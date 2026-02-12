"""
Excel Helper Module

Handles extraction of elements (textboxes, charts, images, tables, etc.) from XLSX/XLS files.

Module Structure:
- excel_chart_constants: Chart type mapping constants
- excel_chart_extractor: Chart extraction (ChartExtractor)
- excel_table_xlsx: XLSX table conversion
- excel_table_xls: XLS table conversion
- excel_textbox_xlsx: XLSX textbox extraction
- excel_textbox_xls: XLS textbox extraction
- excel_metadata: Metadata extraction
- excel_image: Image extraction
- excel_layout_detector: Layout detection
"""

# === Textbox ===
from xgen_doc2chunk.core.processor.excel_helper.excel_textbox_xlsx import (
    extract_textboxes_from_xlsx,
)
from xgen_doc2chunk.core.processor.excel_helper.excel_textbox_xls import (
    extract_textboxes_from_xls,
)

# === Metadata ===
from xgen_doc2chunk.core.processor.excel_helper.excel_metadata import (
    ExcelMetadataExtractor,
    XLSXMetadataExtractor,
    XLSMetadataExtractor,
)

# === Chart Extractor ===
from xgen_doc2chunk.core.processor.excel_helper.excel_chart_extractor import (
    ExcelChartExtractor,
    CHART_TYPE_MAP,
)

# === Image Processor (replaces excel_image.py utility functions) ===
from xgen_doc2chunk.core.processor.excel_helper.excel_image_processor_xlsx import (
    ExcelImageProcessor,
)
from xgen_doc2chunk.core.processor.excel_helper.excel_image_processor_xls import (
    XLSImageProcessor,
)

# === Table XLSX ===
from xgen_doc2chunk.core.processor.excel_helper.excel_table_xlsx import (
    has_merged_cells_xlsx,
    convert_xlsx_sheet_to_table,
    convert_xlsx_sheet_to_markdown,
    convert_xlsx_sheet_to_html,
    convert_xlsx_objects_to_tables,
)

# === Table XLS ===
from xgen_doc2chunk.core.processor.excel_helper.excel_table_xls import (
    has_merged_cells_xls,
    convert_xls_sheet_to_table,
    convert_xls_sheet_to_markdown,
    convert_xls_sheet_to_html,
    convert_xls_objects_to_tables,
)

# === Layout Detector ===
from xgen_doc2chunk.core.processor.excel_helper.excel_layout_detector import (
    layout_detect_range_xlsx,
    layout_detect_range_xls,
    object_detect_xlsx,
    object_detect_xls,
    LayoutRange,
)


__all__ = [
    # Textbox
    'extract_textboxes_from_xlsx',
    'extract_textboxes_from_xls',
    # Metadata
    'ExcelMetadataExtractor',
    'XLSXMetadataExtractor',
    'XLSMetadataExtractor',
    # Chart Constants
    'CHART_TYPE_MAP',
    # Chart Extractor
    'ExcelChartExtractor',
    # Image Processor
    'ExcelImageProcessor',
    'XLSImageProcessor',
    # Table XLSX
    'has_merged_cells_xlsx',
    'convert_xlsx_sheet_to_table',
    'convert_xlsx_sheet_to_markdown',
    'convert_xlsx_sheet_to_html',
    'convert_xlsx_objects_to_tables',
    # Table XLS
    'has_merged_cells_xls',
    'convert_xls_sheet_to_table',
    'convert_xls_sheet_to_markdown',
    'convert_xls_sheet_to_html',
    'convert_xls_objects_to_tables',
    # Layout Detector
    'layout_detect_range_xlsx',
    'layout_detect_range_xls',
    'object_detect_xlsx',
    'object_detect_xls',
    'LayoutRange',
]
