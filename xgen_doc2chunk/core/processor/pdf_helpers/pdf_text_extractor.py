# xgen_doc2chunk/core/processor/pdf_helpers/pdf_text_extractor.py
"""
PDF Text Extraction Module

Provides functions for extracting text blocks from PDF pages.
Includes support for:
- Fragmented text reconstruction (Word->PDF conversion issues)
- CJK Compatibility character mapping (broken character fixes)
"""
import logging
from typing import List, Tuple

from xgen_doc2chunk.core.processor.pdf_helpers.types import (
    ElementType,
    PageElement,
    PageBorderInfo,
)
from xgen_doc2chunk.core.processor.pdf_helpers.pdf_utils import is_inside_any_bbox
from xgen_doc2chunk.core.processor.pdf_helpers.pdf_text_quality_analyzer import (
    TextQualityAnalyzer,
    QualityAwareTextExtractor,
    PageOCRFallbackEngine,
    FragmentedTextReconstructor,
    apply_cjk_compat_mapping,
)

logger = logging.getLogger("document-processor")


def extract_text_blocks(
    page,
    page_num: int,
    table_bboxes: List[Tuple[float, float, float, float]],
    border_info: PageBorderInfo,
    use_quality_check: bool = True
) -> List[PageElement]:
    """
    Extract text blocks excluding table regions.

    Improvements:
    1. Text quality analysis (broken text detection)
    2. OCR fallback for low quality text

    Args:
        page: PyMuPDF page object
        page_num: Page number (0-indexed)
        table_bboxes: List of table bounding boxes to exclude
        border_info: Page border information
        use_quality_check: Whether to perform quality checks

    Returns:
        List of PageElement for extracted text
    """
    elements = []

    # Analyze text quality
    if use_quality_check:
        analyzer = TextQualityAnalyzer(page, page_num)
        page_analysis = analyzer.analyze_page()

        # If quality is low, try text reconstruction first (before OCR)
        if page_analysis.quality_result.needs_ocr:
            quality_result = page_analysis.quality_result
            logger.info(
                f"[PDF] Page {page_num + 1}: Low text quality detected - "
                f"score={quality_result.quality_score:.2f}, "
                f"PUA={quality_result.pua_count}, "
                f"CJK_Compat={quality_result.cjk_compat_count}, "
                f"fragmented={quality_result.is_fragmented}"
            )
            
            # Try reconstruction for fragmented text or CJK Compat issues
            if quality_result.is_fragmented or quality_result.cjk_compat_count > 0:
                logger.info(
                    f"[PDF] Page {page_num + 1}: Attempting text reconstruction "
                    f"(excluding {len(table_bboxes)} table regions)"
                )
                
                # Exclude table regions from reconstruction to avoid duplication
                reconstructor = FragmentedTextReconstructor(
                    page, page_num, exclude_bboxes=table_bboxes
                )
                
                # Use section-based reconstruction for proper table positioning
                if table_bboxes:
                    sections = reconstructor.reconstruct_with_sections()
                    
                    if sections:
                        result_elements = []
                        for section in sections:
                            # Apply CJK Compatibility character mapping
                            cleaned_text = apply_cjk_compat_mapping(section['text'])
                            
                            if cleaned_text.strip():
                                # Create element with proper Y position for sorting
                                result_elements.append(PageElement(
                                    element_type=ElementType.TEXT,
                                    content=cleaned_text,
                                    bbox=(0, section['y_start'], page.rect.width, section['y_end']),
                                    page_num=page_num
                                ))
                        
                        if result_elements:
                            logger.info(
                                f"[PDF] Page {page_num + 1}: Text reconstruction successful "
                                f"({len(result_elements)} sections)"
                            )
                            return result_elements
                else:
                    # No tables - use simple reconstruction
                    reconstructed_text = reconstructor.reconstruct()
                    
                    if reconstructed_text:
                        cleaned_text = apply_cjk_compat_mapping(reconstructed_text)
                        
                        logger.info(
                            f"[PDF] Page {page_num + 1}: Text reconstruction successful "
                            f"({len(cleaned_text)} chars)"
                        )
                        
                        return [PageElement(
                            element_type=ElementType.TEXT,
                            content=cleaned_text,
                            bbox=(0, 0, page.rect.width, page.rect.height),
                            page_num=page_num
                        )]
            
            # Fall back to OCR if reconstruction not applicable
            logger.info(
                f"[PDF] Page {page_num + 1}: Using OCR fallback"
            )

            extractor = QualityAwareTextExtractor(page, page_num)
            ocr_text, _ = extractor.extract()

            if ocr_text.strip():
                # Split OCR text into blocks
                # Exclude table regions
                ocr_blocks = split_ocr_text_to_blocks(ocr_text, page, table_bboxes)
                return ocr_blocks

    # Existing logic: regular text extraction
    page_dict = page.get_text("dict", sort=True)

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        block_bbox = block.get("bbox", (0, 0, 0, 0))

        if is_inside_any_bbox(block_bbox, table_bboxes):
            continue

        text_parts = []

        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            if line_text.strip():
                text_parts.append(line_text.strip())

        if text_parts:
            full_text = "\n".join(text_parts)

            # Individual block quality check (when use_quality_check is True)
            if use_quality_check:
                analyzer = TextQualityAnalyzer(page, page_num)
                block_quality = analyzer.analyze_text(full_text)

                if block_quality.needs_ocr:
                    # OCR only this block
                    ocr_engine = PageOCRFallbackEngine(page, page_num)
                    ocr_text = ocr_engine.ocr_region(block_bbox)
                    if ocr_text.strip():
                        full_text = ocr_text
                        logger.debug(f"[PDF] Block OCR: '{ocr_text[:50]}...'")

            elements.append(PageElement(
                element_type=ElementType.TEXT,
                content=full_text,
                bbox=block_bbox,
                page_num=page_num
            ))

    return elements


def split_ocr_text_to_blocks(
    ocr_text: str,
    page,
    table_bboxes: List[Tuple[float, float, float, float]]
) -> List[PageElement]:
    """
    Convert OCR text to page elements.

    Since OCR lacks position info, the entire text is treated as a single block.
    Table regions are excluded.

    Args:
        ocr_text: OCR extracted text
        page: PyMuPDF page object
        table_bboxes: List of table bounding boxes

    Returns:
        List of PageElement
    """
    if not ocr_text.strip():
        return []

    # Calculate page region excluding table areas
    page_width = page.rect.width
    page_height = page.rect.height

    # Return OCR text as a single block (position covers entire page)
    # For actual position info, pytesseract's image_to_data can be used
    return [PageElement(
        element_type=ElementType.TEXT,
        content=ocr_text,
        bbox=(0, 0, page_width, page_height),
        page_num=page.number
    )]

