# xgen_doc2chunk/core/processor/docx_helper/docx_image.py
"""
DOCX Image Extraction Utilities

Extracts images from DOCX documents and saves them locally.
- extract_image_from_drawing: Extract images from Drawing elements
- process_pict_element: Process legacy VML pict elements

Note: These functions are wrappers that call DOCXImageProcessor methods.
      The actual logic is consolidated in DOCXImageProcessor.
"""
import logging
from typing import Optional, Set, Tuple, TYPE_CHECKING

from docx import Document

from xgen_doc2chunk.core.processor.docx_helper.docx_constants import ElementType

if TYPE_CHECKING:
    from xgen_doc2chunk.core.processor.docx_helper.docx_image_processor import DOCXImageProcessor
    from xgen_doc2chunk.core.functions.img_processor import ImageProcessor

logger = logging.getLogger("document-processor")


def extract_image_from_drawing(
    graphic_data,
    doc: Document,
    processed_images: Set[str],
    image_processor: "ImageProcessor"
) -> Tuple[str, Optional[ElementType]]:
    """
    Extract image from Drawing element.

    Args:
        graphic_data: graphicData XML element
        doc: python-docx Document object
        processed_images: Set of processed image paths (for deduplication)
        image_processor: ImageProcessor instance (DOCXImageProcessor recommended)

    Returns:
        (content, element_type) tuple
    """
    # Use integrated method if DOCXImageProcessor
    if hasattr(image_processor, 'extract_from_drawing'):
        content, is_image = image_processor.extract_from_drawing(
            graphic_data, doc, processed_images
        )
        return (content, ElementType.IMAGE) if is_image else ("", None)
    
    # Fallback: Legacy logic (when using base ImageProcessor class)
    from docx.oxml.ns import qn
    from xgen_doc2chunk.core.processor.docx_helper.docx_constants import NAMESPACES
    
    try:
        blip = graphic_data.find('.//a:blip', NAMESPACES)
        if blip is None:
            return "", None

        r_embed = blip.get(qn('r:embed'))
        r_link = blip.get(qn('r:link'))
        rId = r_embed or r_link
        
        if not rId:
            return "", None

        try:
            rel = doc.part.rels.get(rId)
            if rel is None:
                return "", None

            if hasattr(rel, 'target_part') and hasattr(rel.target_part, 'blob'):
                image_data = rel.target_part.blob
                image_tag = image_processor.save_image(image_data, processed_images=processed_images)
                if image_tag:
                    return f"\n{image_tag}\n", ElementType.IMAGE

            return "[Image]", ElementType.IMAGE

        except Exception as e:
            logger.warning(f"Error extracting image from relationship: {e}")
            return "[Image]", ElementType.IMAGE

    except Exception as e:
        logger.warning(f"Error extracting image from drawing: {e}")
        return "", None


def process_pict_element(
    pict_elem,
    doc: Document,
    processed_images: Set[str],
    image_processor: "ImageProcessor"
) -> str:
    """
    Process legacy VML pict element.

    Args:
        pict_elem: pict XML element
        doc: python-docx Document object
        processed_images: Set of processed image paths (for deduplication)
        image_processor: ImageProcessor instance (DOCXImageProcessor recommended)

    Returns:
        Image marker string
    """
    # Use integrated method if DOCXImageProcessor
    if hasattr(image_processor, 'extract_from_pict'):
        return image_processor.extract_from_pict(pict_elem, doc, processed_images)
    
    # Fallback: Legacy logic (when using base ImageProcessor class)
    try:
        ns_v = 'urn:schemas-microsoft-com:vml'
        ns_r = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

        imagedata = pict_elem.find('.//{%s}imagedata' % ns_v)
        if imagedata is None:
            return "[Image]"

        rId = imagedata.get('{%s}id' % ns_r)
        if not rId:
            return "[Image]"

        try:
            rel = doc.part.rels.get(rId)
            if rel and hasattr(rel, 'target_part') and hasattr(rel.target_part, 'blob'):
                image_data = rel.target_part.blob
                image_tag = image_processor.save_image(image_data, processed_images=processed_images)
                if image_tag:
                    return f"\n{image_tag}\n"
        except Exception:
            pass

        return "[Image]"

    except Exception as e:
        logger.warning(f"Error processing pict element: {e}")
        return ""


__all__ = [
    'extract_image_from_drawing',
    'process_pict_element',
]

