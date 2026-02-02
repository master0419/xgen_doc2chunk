# xgen_doc2chunk/core/processor/docx_helper/docx_image.py
"""
DOCX ?´ë?ì§€ ì¶”ì¶œ ? í‹¸ë¦¬í‹°

DOCX ë¬¸ì„œ?ì„œ ?´ë?ì§€ë¥?ì¶”ì¶œ?˜ê³  ë¡œì»¬???€?¥í•©?ˆë‹¤.
- extract_image_from_drawing: Drawing ?”ì†Œ?ì„œ ?´ë?ì§€ ì¶”ì¶œ
- process_pict_element: ?ˆê±°??VML pict ?”ì†Œ ì²˜ë¦¬

Note: ???¨ìˆ˜?¤ì? DOCXImageProcessor??ë©”ì„œ?œë? ?¸ì¶œ?˜ëŠ” wrapper?…ë‹ˆ??
      ?¤ì œ ë¡œì§?€ DOCXImageProcessor???µí•©?˜ì–´ ?ˆìŠµ?ˆë‹¤.
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
    Drawing?ì„œ ?´ë?ì§€ë¥?ì¶”ì¶œ?©ë‹ˆ??

    Args:
        graphic_data: graphicData XML ?”ì†Œ
        doc: python-docx Document ê°ì²´
        processed_images: ì²˜ë¦¬???´ë?ì§€ ê²½ë¡œ ì§‘í•© (ì¤‘ë³µ ë°©ì?)
        image_processor: ImageProcessor ?¸ìŠ¤?´ìŠ¤ (DOCXImageProcessor ê¶Œì¥)

    Returns:
        (content, element_type) ?œí”Œ
    """
    # DOCXImageProcessor??ê²½ìš° ?µí•©??ë©”ì„œ???¬ìš©
    if hasattr(image_processor, 'extract_from_drawing'):
        content, is_image = image_processor.extract_from_drawing(
            graphic_data, doc, processed_images
        )
        return (content, ElementType.IMAGE) if is_image else ("", None)
    
    # Fallback: ê¸°ì¡´ ë¡œì§ (ImageProcessor ê¸°ë³¸ ?´ë˜?¤ì¸ ê²½ìš°)
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

            return "[?´ë?ì§€]", ElementType.IMAGE

        except Exception as e:
            logger.warning(f"Error extracting image from relationship: {e}")
            return "[?´ë?ì§€]", ElementType.IMAGE

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
    ?ˆê±°??VML pict ?”ì†Œë¥?ì²˜ë¦¬?©ë‹ˆ??

    Args:
        pict_elem: pict XML ?”ì†Œ
        doc: python-docx Document ê°ì²´
        processed_images: ì²˜ë¦¬???´ë?ì§€ ê²½ë¡œ ì§‘í•© (ì¤‘ë³µ ë°©ì?)
        image_processor: ImageProcessor ?¸ìŠ¤?´ìŠ¤ (DOCXImageProcessor ê¶Œì¥)

    Returns:
        ?´ë?ì§€ ë§ˆí¬??ë¬¸ì??
    """
    # DOCXImageProcessor??ê²½ìš° ?µí•©??ë©”ì„œ???¬ìš©
    if hasattr(image_processor, 'extract_from_pict'):
        return image_processor.extract_from_pict(pict_elem, doc, processed_images)
    
    # Fallback: ê¸°ì¡´ ë¡œì§ (ImageProcessor ê¸°ë³¸ ?´ë˜?¤ì¸ ê²½ìš°)
    try:
        ns_v = 'urn:schemas-microsoft-com:vml'
        ns_r = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

        imagedata = pict_elem.find('.//{%s}imagedata' % ns_v)
        if imagedata is None:
            return "[?´ë?ì§€]"

        rId = imagedata.get('{%s}id' % ns_r)
        if not rId:
            return "[?´ë?ì§€]"

        try:
            rel = doc.part.rels.get(rId)
            if rel and hasattr(rel, 'target_part') and hasattr(rel.target_part, 'blob'):
                image_data = rel.target_part.blob
                image_tag = image_processor.save_image(image_data, processed_images=processed_images)
                if image_tag:
                    return f"\n{image_tag}\n"
        except Exception:
            pass

        return "[?´ë?ì§€]"

    except Exception as e:
        logger.warning(f"Error processing pict element: {e}")
        return ""


__all__ = [
    'extract_image_from_drawing',
    'process_pict_element',
]

