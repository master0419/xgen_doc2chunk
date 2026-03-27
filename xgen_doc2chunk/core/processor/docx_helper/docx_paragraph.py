# xgen_doc2chunk/core/processor/docx_helper/docx_paragraph.py
"""
DOCX Paragraph Processing Utility

Processes Paragraph elements in DOCX documents.
- process_paragraph_element: Process Paragraph element
- has_page_break_element: Check for page break

Image and drawing extraction is handled by DOCXImageProcessor.
"""
import logging
from lxml import etree
from typing import List, Optional, Set, Tuple, Callable, TYPE_CHECKING

from docx import Document

from xgen_doc2chunk.core.processor.docx_helper.docx_constants import ElementType, NAMESPACES

if TYPE_CHECKING:
    from xgen_doc2chunk.core.processor.docx_helper.docx_image_processor import DOCXImageProcessor

logger = logging.getLogger("document-processor")


def _get_local_tag(elem) -> str:
    """Get the local tag name without namespace."""
    return etree.QName(elem).localname


def _resolve_alternate_content(ac_elem) -> list:
    """
    Resolve mc:AlternateContent by choosing Choice first, Fallback as insurance.

    Word stores shapes in both DrawingML (Choice) and VML (Fallback) formats.
    We prefer Choice (modern format) and only use Fallback if Choice is absent.

    Args:
        ac_elem: mc:AlternateContent XML element

    Returns:
        List of child elements from the chosen branch
    """
    choice = ac_elem.find('mc:Choice', NAMESPACES)
    if choice is not None and len(choice) > 0:
        return list(choice)

    fallback = ac_elem.find('mc:Fallback', NAMESPACES)
    if fallback is not None and len(fallback) > 0:
        return list(fallback)

    return []


def _collect_runs_from_paragraph(para_elem) -> list:
    """
    Collect run elements from a paragraph, properly handling mc:AlternateContent.

    Instead of recursive .//w:r (which finds runs inside both Choice AND Fallback
    of AlternateContent, causing duplicate text), this iterates direct children
    and resolves AlternateContent to only one branch.

    Args:
        para_elem: paragraph XML element

    Returns:
        List of w:r elements to process
    """
    runs = []
    for child in para_elem:
        local_tag = _get_local_tag(child)
        if local_tag == 'r':
            # Direct run — collect it, but also check for AlternateContent inside the run
            _collect_runs_from_element(child, runs)
        elif local_tag == 'AlternateContent':
            # mc:AlternateContent at paragraph level — resolve to one branch
            resolved = _resolve_alternate_content(child)
            for resolved_child in resolved:
                resolved_tag = _get_local_tag(resolved_child)
                if resolved_tag == 'r':
                    _collect_runs_from_element(resolved_child, runs)
        elif local_tag == 'hyperlink':
            # Hyperlinks contain runs
            for sub in child:
                if _get_local_tag(sub) == 'r':
                    _collect_runs_from_element(sub, runs)
    return runs


def _collect_runs_from_element(elem, runs: list):
    """
    Process a single element, resolving any nested AlternateContent inside it.

    A w:r (run) can itself contain mc:AlternateContent (e.g., for shapes).
    This ensures we only take one branch when that happens.

    Args:
        elem: XML element (typically w:r)
        runs: List to append resolved run elements to
    """
    # Check if this run contains AlternateContent
    has_ac = False
    for child in elem:
        if _get_local_tag(child) == 'AlternateContent':
            has_ac = True
            break

    if not has_ac:
        # Normal run with no AlternateContent — use as-is
        runs.append(elem)
    else:
        # Run contains AlternateContent — we still add the run itself
        # (it may have text before/after the AC), but we mark it so
        # the processing knows to handle AC elements inside
        runs.append(elem)


def _process_run_element(
    run_elem,
    doc: Document,
    processed_images: Set[str],
    image_processor: Optional["DOCXImageProcessor"],
    chart_callback: Optional[Callable[[], str]],
    content_parts: List[str],
) -> Tuple[int, int]:
    """
    Process a single w:r element, extracting text, images, and charts.

    Handles mc:AlternateContent inside runs by resolving to one branch.

    Args:
        run_elem: w:r XML element
        doc: python-docx Document object
        processed_images: Set of processed image paths
        image_processor: DOCXImageProcessor instance
        chart_callback: Callback for chart content
        content_parts: List to append extracted content to

    Returns:
        (image_count, chart_count) tuple
    """
    img_count = 0
    ch_count = 0

    for child in run_elem:
        local_tag = _get_local_tag(child)

        if local_tag == 't':
            # Text element
            if child.text:
                content_parts.append(child.text)

        elif local_tag == 'drawing':
            # Drawing (image/chart/diagram)
            if image_processor and hasattr(image_processor, 'process_drawing_element'):
                drawing_content, drawing_type = image_processor.process_drawing_element(
                    child, doc, processed_images, chart_callback=chart_callback
                )
            else:
                drawing_content, drawing_type = "", None
            if drawing_content:
                content_parts.append(drawing_content)
                if drawing_type == ElementType.IMAGE:
                    img_count += 1
                elif drawing_type == ElementType.CHART:
                    ch_count += 1

        elif local_tag == 'pict':
            # Legacy VML element: image or textbox (Word 2007-2009)
            if image_processor and hasattr(image_processor, 'extract_from_pict'):
                pict_content, pict_type = image_processor.extract_from_pict(child, doc, processed_images)
            else:
                pict_content, pict_type = "", None
            if pict_content:
                content_parts.append(pict_content)
                if pict_type == ElementType.IMAGE:
                    img_count += 1

        elif local_tag == 'AlternateContent':
            # mc:AlternateContent inside a run — resolve to one branch and process
            resolved = _resolve_alternate_content(child)
            for resolved_elem in resolved:
                resolved_tag = _get_local_tag(resolved_elem)
                if resolved_tag == 'drawing':
                    if image_processor and hasattr(image_processor, 'process_drawing_element'):
                        drawing_content, drawing_type = image_processor.process_drawing_element(
                            resolved_elem, doc, processed_images, chart_callback=chart_callback
                        )
                    else:
                        drawing_content, drawing_type = "", None
                    if drawing_content:
                        content_parts.append(drawing_content)
                        if drawing_type == ElementType.IMAGE:
                            img_count += 1
                        elif drawing_type == ElementType.CHART:
                            ch_count += 1
                elif resolved_tag == 'pict':
                    # Legacy VML element in AlternateContent Fallback branch
                    if image_processor and hasattr(image_processor, 'extract_from_pict'):
                        pict_content, pict_type = image_processor.extract_from_pict(resolved_elem, doc, processed_images)
                    else:
                        pict_content, pict_type = "", None
                    if pict_content:
                        content_parts.append(pict_content)
                        if pict_type == ElementType.IMAGE:
                            img_count += 1

    return img_count, ch_count


def process_paragraph_element(
    para_elem,
    doc: Document,
    processed_images: Set[str],
    file_path: str = None,
    image_processor: Optional["DOCXImageProcessor"] = None,
    chart_callback: Optional[Callable[[], str]] = None
) -> Tuple[str, bool, int, int]:
    """
    Process Paragraph element.

    Extracts text, images, charts and detects page breaks.
    Handles mc:AlternateContent by choosing Choice first, Fallback as insurance,
    preventing duplicate text extraction from shapes/textboxes.

    Args:
        para_elem: paragraph XML element
        doc: python-docx Document object
        processed_images: Set of processed image paths (deduplication)
        file_path: Original file path
        image_processor: DOCXImageProcessor instance
        chart_callback: Callback function to get next chart content

    Returns:
        (content, has_page_break, image_count, chart_count) tuple
    """
    content_parts = []
    has_page_break = False
    image_count = 0
    chart_count = 0

    try:
        # Check for page break
        has_page_break = has_page_break_element(para_elem)

        # Collect runs with proper AlternateContent resolution
        runs = _collect_runs_from_paragraph(para_elem)

        for run_elem in runs:
            img_count, ch_count = _process_run_element(
                run_elem, doc, processed_images,
                image_processor, chart_callback, content_parts
            )
            image_count += img_count
            chart_count += ch_count

    except Exception as e:
        logger.warning(f"Error processing paragraph: {e}")
        # Fallback: simple text extraction (avoid duplicates by using direct children only)
        try:
            content_parts = []
            for t_elem in para_elem.findall('w:r/w:t', NAMESPACES):
                if t_elem.text:
                    content_parts.append(t_elem.text)
        except:
            pass

    return ''.join(content_parts), has_page_break, image_count, chart_count


def has_page_break_element(element) -> bool:
    """
    Check if element contains a page break.

    Args:
        element: XML element

    Returns:
        Whether page break exists
    """
    try:
        # Explicit page break
        if element.findall('.//w:br[@w:type="page"]', NAMESPACES):
            return True
        # Rendered page break
        if element.findall('.//w:lastRenderedPageBreak', NAMESPACES):
            return True
        return False
    except Exception:
        return False


__all__ = [
    'process_paragraph_element',
    'has_page_break_element',
]
