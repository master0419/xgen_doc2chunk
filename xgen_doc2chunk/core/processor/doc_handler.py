# xgen_doc2chunk/core/processor/doc_handler.py
"""
DOC Handler - Legacy Microsoft Word Document Processor

Class-based handler for DOC files inheriting from BaseHandler.
Automatically detects file format (RTF, OLE, HTML, DOCX) and processes accordingly.
RTF processing is delegated to RTFHandler.
"""
import io
import logging
import os
import re
import struct
import base64
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum
import zipfile

import olefile
from bs4 import BeautifulSoup

from xgen_doc2chunk.core.processor.base_handler import BaseHandler
from xgen_doc2chunk.core.functions.img_processor import ImageProcessor
from xgen_doc2chunk.core.functions.chart_extractor import BaseChartExtractor, NullChartExtractor
from xgen_doc2chunk.core.processor.doc_helpers.doc_image_processor import DOCImageProcessor
from xgen_doc2chunk.core.functions.table_extractor import TableData, TableCell
from xgen_doc2chunk.core.functions.table_processor import TableProcessor, TableProcessorConfig, TableOutputFormat

if TYPE_CHECKING:
    from xgen_doc2chunk.core.document_processor import CurrentFile

logger = logging.getLogger("document-processor")


class DocFormat(Enum):
    """Actual format types for DOC files."""
    RTF = "rtf"
    OLE = "ole"
    HTML = "html"
    DOCX = "docx"
    UNKNOWN = "unknown"


MAGIC_NUMBERS = {
    'RTF': b'{\\rtf',
    'OLE': b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1',
    'ZIP': b'PK\x03\x04',
}


class DOCHandler(BaseHandler):
    """DOC file processing handler class."""

    def _create_file_converter(self):
        """Create DOC-specific file converter."""
        from xgen_doc2chunk.core.processor.doc_helpers.doc_file_converter import DOCFileConverter
        return DOCFileConverter()

    def _create_preprocessor(self):
        """Create DOC-specific preprocessor."""
        from xgen_doc2chunk.core.processor.doc_helpers.doc_preprocessor import DOCPreprocessor
        return DOCPreprocessor()

    def _create_chart_extractor(self) -> BaseChartExtractor:
        """DOC files chart extraction not yet implemented. Return NullChartExtractor."""
        return NullChartExtractor(self._chart_processor)

    def _create_metadata_extractor(self):
        """DOC metadata extraction not yet implemented. Return None to use NullMetadataExtractor."""
        return None

    def _create_format_image_processor(self) -> ImageProcessor:
        """Create DOC-specific image processor."""
        return DOCImageProcessor()

    def extract_text(
        self,
        current_file: "CurrentFile",
        extract_metadata: bool = True,
        **kwargs
    ) -> str:
        """Extract text from DOC file."""
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")

        self.logger.info(f"DOC processing: {file_path}")

        if not file_data:
            self.logger.error(f"Empty file data: {file_path}")
            return f"[DOC file is empty: {file_path}]"

        try:
            # Step 1: Use file_converter to detect format and convert
            converted_obj, doc_format = self.file_converter.convert(file_data)

            # Step 2: Preprocess - may transform converted_obj in the future
            preprocessed = self.preprocess(converted_obj)
            converted_obj = preprocessed.clean_content  # TRUE SOURCE

            if doc_format == DocFormat.RTF:
                # Delegate to RTFHandler for RTF processing
                return self._delegate_to_rtf_handler(converted_obj, current_file, extract_metadata)
            elif doc_format == DocFormat.OLE:
                return self._extract_from_ole_obj(converted_obj, current_file, extract_metadata)
            elif doc_format == DocFormat.HTML:
                return self._extract_from_html_obj(converted_obj, current_file, extract_metadata)
            elif doc_format == DocFormat.DOCX:
                return self._extract_from_docx_obj(converted_obj, current_file, extract_metadata)
            else:
                self.logger.warning(f"Unknown DOC format, trying OLE fallback: {file_path}")
                return self._extract_from_ole(current_file, extract_metadata)
        except Exception as e:
            self.logger.error(f"Error in DOC processing: {e}")
            return f"[DOC file processing failed: {str(e)}]"

    def _delegate_to_rtf_handler(self, rtf_doc, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """
        Delegate RTF processing to RTFHandler.

        When a DOC file is actually in RTF format, delegate to RTFHandler.
        RTFHandler.extract_text() receives raw bytes, so pass current_file as is.

        Args:
            rtf_doc: Pre-converted RTFDocument object (unused, for consistency)
            current_file: CurrentFile dict containing original file_data
            extract_metadata: Whether to extract metadata

        Returns:
            Extracted text
        """
        from xgen_doc2chunk.core.processor.rtf_handler import RTFHandler

        rtf_handler = RTFHandler(
            config=self.config,
            image_processor=self._image_processor,
            page_tag_processor=self._page_tag_processor,
            chart_processor=self._chart_processor
        )

        # RTFHandler.extract_text() reads file_data directly from current_file
        return rtf_handler.extract_text(current_file, extract_metadata=extract_metadata)

    def _extract_from_ole_obj(self, ole, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """OLE Compound Document processing using pre-converted OLE object."""
        file_path = current_file.get("file_path", "unknown")

        self.logger.info(f"Processing OLE: {file_path}")

        result_parts = []
        processed_images: Set[str] = set()

        try:
            # Metadata extraction
            if extract_metadata:
                metadata = self._extract_ole_metadata(ole)
                metadata_str = self.extract_and_format_metadata(metadata)
                if metadata_str:
                    result_parts.append(metadata_str + "\n\n")

            # Parse FIB and Piece Table once — shared by all extraction steps
            fib = self._parse_fib(ole)
            if fib is None:
                self.logger.warning("Failed to parse FIB, falling back to heuristic")
                text = self._extract_ole_text_heuristic(ole)
                if text:
                    result_parts.append(text)
                return "\n".join(result_parts)

            word_data = fib['word_data']
            pieces = fib['pieces']

            # Extract header/footer via Piece Table
            header_text, footer_text = self._extract_headers_footers_via_pieces(
                word_data, pieces, fib
            )
            if header_text:
                result_parts.append(f"[Header]\n{header_text}\n\n")

            page_tag = self.create_page_tag(1)
            result_parts.append(f"{page_tag}\n")

            # Extract body text via Piece Table (CP 0 ~ ccpText)
            raw_body = self._read_cp_range(word_data, pieces, 0, fib['ccpText'])

            # Process body: handle fields, tables, page breaks
            body_text = self._process_doc_body_text(
                raw_body, word_data, fib['table_data'], fib, pieces
            )
            if body_text:
                result_parts.append(body_text)

            # Extract images
            images = self._extract_ole_images(ole, processed_images)
            for img_tag in images:
                result_parts.append(img_tag)

            if footer_text:
                result_parts.append(f"\n[Footer]\n{footer_text}\n")

        except Exception as e:
            self.logger.error(f"OLE processing error: {e}")
            return f"[DOC file processing failed: {str(e)}]"
        finally:
            # Close the OLE object
            self.file_converter.close(ole)

        return "\n".join(result_parts)

    def _extract_from_ole(self, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """OLE Compound Document processing - extract text directly from WordDocument stream."""
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")

        self.logger.info(f"Processing OLE: {file_path}")

        result_parts = []
        processed_images: Set[str] = set()

        try:
            file_stream = io.BytesIO(file_data)
            with olefile.OleFileIO(file_stream) as ole:
                # Metadata extraction
                if extract_metadata:
                    metadata = self._extract_ole_metadata(ole)
                    metadata_str = self.extract_and_format_metadata(metadata)
                    if metadata_str:
                        result_parts.append(metadata_str + "\n\n")

                # Parse FIB and Piece Table once
                fib = self._parse_fib(ole)
                if fib is None:
                    self.logger.warning("Failed to parse FIB, falling back to heuristic")
                    text = self._extract_ole_text_heuristic(ole)
                    if text:
                        result_parts.append(text)
                    return "\n".join(result_parts)

                word_data = fib['word_data']
                pieces = fib['pieces']

                # Extract header/footer via Piece Table
                header_text, footer_text = self._extract_headers_footers_via_pieces(
                    word_data, pieces, fib
                )
                if header_text:
                    result_parts.append(f"[Header]\n{header_text}\n\n")

                page_tag = self.create_page_tag(1)
                result_parts.append(f"{page_tag}\n")

                # Extract body text via Piece Table
                raw_body = self._read_cp_range(word_data, pieces, 0, fib['ccpText'])
                body_text = self._process_doc_body_text(
                    raw_body, word_data, fib['table_data'], fib, pieces
                )
                if body_text:
                    result_parts.append(body_text)

                # Extract textbox text (shapes, drawing textboxes)
                txbx_text = self._extract_textbox_text(word_data, pieces, fib)
                if txbx_text:
                    result_parts.append(f"\n{txbx_text}")

                # Extract images
                images = self._extract_ole_images(ole, processed_images)
                for img_tag in images:
                    result_parts.append(img_tag)

                if footer_text:
                    result_parts.append(f"\n[Footer]\n{footer_text}\n")

        except Exception as e:
            self.logger.error(f"OLE processing error: {e}")
            return f"[DOC file processing failed: {str(e)}]"

        return "\n".join(result_parts)

    def _extract_ole_metadata(self, ole: olefile.OleFileIO) -> Dict[str, Any]:
        """Extract OLE metadata."""
        metadata = {}
        try:
            ole_meta = ole.get_metadata()
            if ole_meta:
                if ole_meta.title:
                    metadata['title'] = self._decode_ole_string(ole_meta.title)
                if ole_meta.subject:
                    metadata['subject'] = self._decode_ole_string(ole_meta.subject)
                if ole_meta.author:
                    metadata['author'] = self._decode_ole_string(ole_meta.author)
                if ole_meta.keywords:
                    metadata['keywords'] = self._decode_ole_string(ole_meta.keywords)
                if ole_meta.comments:
                    metadata['comments'] = self._decode_ole_string(ole_meta.comments)
                if ole_meta.last_saved_by:
                    metadata['last_saved_by'] = self._decode_ole_string(ole_meta.last_saved_by)
                if ole_meta.create_time:
                    metadata['create_time'] = ole_meta.create_time
                if ole_meta.last_saved_time:
                    metadata['last_saved_time'] = ole_meta.last_saved_time
        except Exception as e:
            self.logger.warning(f"Error extracting OLE metadata: {e}")
        return metadata

    def _decode_ole_string(self, value) -> str:
        """Decode OLE string."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, bytes):
            for encoding in ['utf-8', 'cp949', 'euc-kr', 'cp1252', 'latin-1']:
                try:
                    return value.decode(encoding).strip()
                except (UnicodeDecodeError, UnicodeError):
                    continue
            return value.decode('utf-8', errors='replace').strip()
        return str(value).strip()

    def _extract_ole_images(self, ole: olefile.OleFileIO, processed_images: Set[str]) -> List[str]:
        """Extract images from OLE container."""
        images = []
        try:
            for entry in ole.listdir():
                if any(x.lower() in ['pictures', 'data', 'object', 'oleobject'] for x in entry):
                    try:
                        stream = ole.openstream(entry)
                        data = stream.read()

                        if data[:8] == b'\x89PNG\r\n\x1a\n' or data[:2] == b'\xff\xd8' or \
                           data[:6] in (b'GIF87a', b'GIF89a') or data[:2] == b'BM':
                            image_tag = self.format_image_processor.save_image(data)
                            if image_tag:
                                images.append(f"\n{image_tag}\n")
                    except:
                        continue
        except Exception as e:
            self.logger.warning(f"Error extracting OLE images: {e}")
        return images

    def _extract_from_html_obj(self, soup, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """HTML DOC processing using pre-converted BeautifulSoup object."""
        file_path = current_file.get("file_path", "unknown")

        self.logger.info(f"Processing HTML DOC: {file_path}")

        result_parts = []

        if extract_metadata:
            metadata = self._extract_html_metadata(soup)
            metadata_str = self.extract_and_format_metadata(metadata)
            if metadata_str:
                result_parts.append(metadata_str + "\n\n")

        page_tag = self.create_page_tag(1)
        result_parts.append(f"{page_tag}\n")

        # Copy soup to avoid modifying the original
        soup_copy = BeautifulSoup(str(soup), 'html.parser')

        for tag in soup_copy(['script', 'style', 'meta', 'link', 'head']):
            tag.decompose()

        text = soup_copy.get_text(separator='\n', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)

        if text:
            result_parts.append(text)

        for table in soup_copy.find_all('table'):
            table_html = str(table)
            table_html = re.sub(r'\s+style="[^"]*"', '', table_html)
            table_html = re.sub(r'\s+class="[^"]*"', '', table_html)
            result_parts.append("\n" + table_html + "\n")

        for img in soup_copy.find_all('img'):
            src = img.get('src', '')
            if src and src.startswith('data:image'):
                try:
                    match = re.match(r'data:image/(\w+);base64,(.+)', src)
                    if match:
                        image_data = base64.b64decode(match.group(2))
                        image_tag = self.format_image_processor.save_image(image_data)
                        if image_tag:
                            result_parts.append(f"\n{image_tag}\n")
                except:
                    pass

        return "\n".join(result_parts)

    def _extract_from_docx_obj(self, doc, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """Extract from misnamed DOCX using pre-converted Document object."""
        file_path = current_file.get("file_path", "unknown")

        self.logger.info(f"Processing misnamed DOCX: {file_path}")

        try:
            result_parts = []

            if extract_metadata:
                # Basic metadata from docx Document
                if hasattr(doc, 'core_properties'):
                    metadata = {
                        'title': doc.core_properties.title or '',
                        'author': doc.core_properties.author or '',
                        'subject': doc.core_properties.subject or '',
                        'keywords': doc.core_properties.keywords or '',
                    }
                    metadata = {k: v for k, v in metadata.items() if v}
                    metadata_str = self.extract_and_format_metadata(metadata)
                    if metadata_str:
                        result_parts.append(metadata_str + "\n\n")

            page_tag = self.create_page_tag(1)
            result_parts.append(f"{page_tag}\n")

            for para in doc.paragraphs:
                if para.text.strip():
                    result_parts.append(para.text)

            for table in doc.tables:
                for row in table.rows:
                    row_texts = []
                    for cell in row.cells:
                        row_texts.append(cell.text.strip())
                    if any(t for t in row_texts):
                        result_parts.append(" | ".join(row_texts))

            return "\n".join(result_parts)

        except Exception as e:
            self.logger.error(f"Error processing misnamed DOCX: {e}")
            return f"[DOCX processing failed: {str(e)}]"

    def _extract_from_html(self, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """HTML DOC processing."""
        file_path = current_file.get("file_path", "unknown")
        file_data = current_file.get("file_data", b"")

        self.logger.info(f"Processing HTML DOC: {file_path}")

        content = None
        for encoding in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'cp1252', 'latin-1']:
            try:
                content = file_data.decode(encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            content = file_data.decode('utf-8', errors='replace')

        result_parts = []
        soup = BeautifulSoup(content, 'html.parser')

        if extract_metadata:
            metadata = self._extract_html_metadata(soup)
            metadata_str = self.extract_and_format_metadata(metadata)
            if metadata_str:
                result_parts.append(metadata_str + "\n\n")

        page_tag = self.create_page_tag(1)
        result_parts.append(f"{page_tag}\n")

        for tag in soup(['script', 'style', 'meta', 'link', 'head']):
            tag.decompose()

        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)

        if text:
            result_parts.append(text)

        for table in soup.find_all('table'):
            table_html = str(table)
            table_html = re.sub(r'\s+style="[^"]*"', '', table_html)
            table_html = re.sub(r'\s+class="[^"]*"', '', table_html)
            result_parts.append("\n" + table_html + "\n")

        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src and src.startswith('data:image'):
                try:
                    match = re.match(r'data:image/(\w+);base64,(.+)', src)
                    if match:
                        image_data = base64.b64decode(match.group(2))
                        image_tag = self.format_image_processor.save_image(image_data)
                        if image_tag:
                            result_parts.append(f"\n{image_tag}\n")
                except:
                    pass

        return "\n".join(result_parts)

    def _extract_html_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """HTML metadata extraction."""
        metadata = {}
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            metadata['title'] = title_tag.string.strip()

        meta_mappings = {
            'author': 'author', 'description': 'comments', 'keywords': 'keywords',
            'subject': 'subject', 'creator': 'author', 'producer': 'last_saved_by',
        }

        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            content = meta.get('content', '')
            if name in meta_mappings and content:
                metadata[meta_mappings[name]] = content.strip()

        return metadata

    def _extract_from_docx_misnamed(self, current_file: "CurrentFile", extract_metadata: bool) -> str:
        """Process misnamed DOCX file."""
        file_path = current_file.get("file_path", "unknown")

        self.logger.info(f"Processing misnamed DOCX: {file_path}")

        try:
            from xgen_doc2chunk.core.processor.docx_handler import DOCXHandler

            # Pass current_file directly - DOCXHandler now accepts CurrentFile
            docx_handler = DOCXHandler(config=self.config, image_processor=self.format_image_processor)
            return docx_handler.extract_text(current_file, extract_metadata=extract_metadata)
        except Exception as e:
            self.logger.error(f"Error processing misnamed DOCX: {e}")
            return f"[DOC file processing failed: {str(e)}]"

    # ─── FIB / Piece Table parsing ───────────────────────────────────────

    def _parse_fib(self, ole: olefile.OleFileIO) -> Optional[Dict[str, Any]]:
        """
        Parse the FIB (File Information Block) and extract the Piece Table.

        Returns a dict with:
            word_data, pieces (list of tuples), ccpText, ccpFtn, ccpHdd,
            ccpAtn, ccpEdn, ccpTxbx, ccpTxbxHdr
        or None if parsing fails.
        """
        try:
            if not ole.exists('WordDocument'):
                return None

            word_data = ole.openstream('WordDocument').read()
            if len(word_data) < 0x6C:
                return None

            magic = struct.unpack('<H', word_data[0:2])[0]
            if magic not in (0xA5EC, 0xA5DC):
                return None

            # --- ccp values (character counts per section) ---
            ccpText = struct.unpack('<I', word_data[0x4C:0x50])[0]
            ccpFtn  = struct.unpack('<I', word_data[0x50:0x54])[0]
            ccpHdd  = struct.unpack('<I', word_data[0x54:0x58])[0]
            ccpMcr  = struct.unpack('<I', word_data[0x58:0x5C])[0]
            ccpAtn  = struct.unpack('<I', word_data[0x5C:0x60])[0]
            ccpEdn  = struct.unpack('<I', word_data[0x60:0x64])[0]
            ccpTxbx = struct.unpack('<I', word_data[0x64:0x68])[0]
            ccpTxbxHdr = struct.unpack('<I', word_data[0x68:0x6C])[0]

            # --- Locate the Table stream ---
            flags = struct.unpack('<H', word_data[0x0A:0x0C])[0]
            table_name = '1Table' if (flags & 0x0200) else '0Table'
            if not ole.exists(table_name):
                # Try the other one
                table_name = '0Table' if table_name == '1Table' else '1Table'
                if not ole.exists(table_name):
                    return None
            table_data = ole.openstream(table_name).read()

            # --- Locate CLX via FibRgFcLcb ---
            pieces = self._find_and_parse_piece_table(word_data, table_data)
            if not pieces:
                return None

            # --- Extract PlcfHdd location (pair index 11 in FibRgFcLcb) ---
            fc_plcf_hdd = 0
            lcb_plcf_hdd = 0
            fc_plcf_bte_papx = 0
            lcb_plcf_bte_papx = 0
            if len(word_data) >= 0x24:
                csw = struct.unpack('<H', word_data[0x20:0x22])[0]
                fibRgW_end = 0x22 + csw * 2
                if fibRgW_end + 2 <= len(word_data):
                    cslw = struct.unpack('<H', word_data[fibRgW_end:fibRgW_end + 2])[0]
                    fibRgLw_end = fibRgW_end + 2 + cslw * 4
                    if fibRgLw_end + 2 <= len(word_data):
                        cbRgFcLcb = struct.unpack('<H', word_data[fibRgLw_end:fibRgLw_end + 2])[0]
                        fcLcb_start = fibRgLw_end + 2
                        if cbRgFcLcb > 11:
                            off11 = fcLcb_start + 11 * 8
                            if off11 + 8 <= len(word_data):
                                fc_plcf_hdd = struct.unpack('<I', word_data[off11:off11 + 4])[0]
                                lcb_plcf_hdd = struct.unpack('<I', word_data[off11 + 4:off11 + 8])[0]
                        # pair[13] = fcPlcfBtePapx / lcbPlcfBtePapx
                        # (Binary Table of Paragraph Properties - used for TC merge flags)
                        if cbRgFcLcb > 13:
                            off13 = fcLcb_start + 13 * 8
                            if off13 + 8 <= len(word_data):
                                fc_plcf_bte_papx = struct.unpack('<I', word_data[off13:off13 + 4])[0]
                                lcb_plcf_bte_papx = struct.unpack('<I', word_data[off13 + 4:off13 + 8])[0]

            return {
                'word_data': word_data,
                'table_data': table_data,
                'pieces': pieces,
                'ccpText': ccpText,
                'ccpFtn': ccpFtn,
                'ccpHdd': ccpHdd,
                'ccpMcr': ccpMcr,
                'ccpAtn': ccpAtn,
                'ccpEdn': ccpEdn,
                'ccpTxbx': ccpTxbx,
                'ccpTxbxHdr': ccpTxbxHdr,
                'fcPlcfHdd': fc_plcf_hdd,
                'lcbPlcfHdd': lcb_plcf_hdd,
                'fcPlcfBtePapx': fc_plcf_bte_papx,
                'lcbPlcfBtePapx': lcb_plcf_bte_papx,
            }
        except Exception as e:
            self.logger.debug(f"FIB parsing error: {e}")
            return None

    def _find_and_parse_piece_table(
        self, word_data: bytes, table_data: bytes
    ) -> List[Tuple[int, int, int, bool]]:
        """
        Find the CLX structure in the Table stream via FIB offsets,
        then parse the Piece Table (Pcdt) from it.

        Returns list of (cp_start, cp_end, fc, is_compressed) tuples,
        or empty list on failure.
        """
        # --- Walk FIB variable-length sections to reach FibRgFcLcb ---
        if len(word_data) < 0x24:
            return []

        csw = struct.unpack('<H', word_data[0x20:0x22])[0]
        fibRgW_end = 0x22 + csw * 2

        if fibRgW_end + 2 > len(word_data):
            return []
        cslw = struct.unpack('<H', word_data[fibRgW_end:fibRgW_end + 2])[0]
        fibRgLw_end = fibRgW_end + 2 + cslw * 4

        if fibRgLw_end + 2 > len(word_data):
            return []
        cbRgFcLcb = struct.unpack('<H', word_data[fibRgLw_end:fibRgLw_end + 2])[0]
        fcLcb_start = fibRgLw_end + 2

        # Scan every fc/lcb pair for one that points to valid CLX/Pcdt
        for pair_idx in range(cbRgFcLcb):
            off = fcLcb_start + pair_idx * 8
            if off + 8 > len(word_data):
                break
            fc_val = struct.unpack('<I', word_data[off:off + 4])[0]
            lcb_val = struct.unpack('<I', word_data[off + 4:off + 8])[0]

            if lcb_val < 5 or fc_val + lcb_val > len(table_data):
                continue

            pieces = self._try_parse_clx(table_data, fc_val, lcb_val)
            if pieces:
                return pieces

        # Fallback: brute-force search for Pcdt (0x02) in the table stream
        return self._brute_force_find_pcdt(table_data)

    def _try_parse_clx(
        self, table_data: bytes, offset: int, length: int
    ) -> List[Tuple[int, int, int, bool]]:
        """Try to parse a CLX structure at the given offset in the Table stream."""
        end = offset + length
        i = offset
        while i < end - 4:
            tag = table_data[i]
            if tag == 0x01:
                # Grpprl — skip
                if i + 3 > end:
                    break
                cb = struct.unpack('<H', table_data[i + 1:i + 3])[0]
                i += 3 + cb
            elif tag == 0x02:
                return self._parse_pcdt(table_data, i)
            else:
                break
        return []

    def _brute_force_find_pcdt(
        self, table_data: bytes
    ) -> List[Tuple[int, int, int, bool]]:
        """Scan the entire Table stream for a valid Pcdt marker."""
        for si in range(len(table_data) - 5):
            if table_data[si] == 0x02:
                pieces = self._parse_pcdt(table_data, si)
                if pieces:
                    return pieces
        return []

    @staticmethod
    def _parse_pcdt(
        table_data: bytes, offset: int
    ) -> List[Tuple[int, int, int, bool]]:
        """
        Parse a Pcdt (Piece Table Data) at *offset* inside *table_data*.

        Returns list of (cp_start, cp_end, fc, is_compressed) or [] on failure.
        """
        if offset + 5 > len(table_data):
            return []

        pcdt_size = struct.unpack('<I', table_data[offset + 1:offset + 5])[0]
        if pcdt_size < 16:
            return []
        if (pcdt_size - 4) % 12 != 0:
            return []

        n_pieces = (pcdt_size - 4) // 12
        if n_pieces < 1 or n_pieces > 10000:
            return []

        data_start = offset + 5
        if data_start + pcdt_size > len(table_data):
            return []

        pcdt = table_data[data_start:data_start + pcdt_size]

        # Read CPs
        cps = []
        for j in range(n_pieces + 1):
            cp = struct.unpack('<I', pcdt[j * 4:(j + 1) * 4])[0]
            cps.append(cp)

        # CPs must be monotonically increasing
        for j in range(len(cps) - 1):
            if cps[j] >= cps[j + 1]:
                return []

        # Read PCDs
        pcd_offset = (n_pieces + 1) * 4
        pieces = []
        for j in range(n_pieces):
            pcd = pcdt[pcd_offset + j * 8:pcd_offset + (j + 1) * 8]
            if len(pcd) < 8:
                return []
            fc_raw = struct.unpack('<I', pcd[2:6])[0]
            is_compressed = bool(fc_raw & 0x40000000)
            fc = fc_raw & 0x3FFFFFFF
            pieces.append((cps[j], cps[j + 1], fc, is_compressed))

        return pieces

    # ─── CP-range reader ──────────────────────────────────────────────────

    @staticmethod
    def _read_cp_range(
        word_data: bytes,
        pieces: List[Tuple[int, int, int, bool]],
        char_start: int,
        char_count: int,
    ) -> str:
        """
        Read *char_count* characters starting at character position *char_start*
        using the piece table to map CP → byte offset.

        Returns the raw string (may contain control characters).
        """
        char_end = char_start + char_count
        result_parts: List[str] = []

        for cp_start, cp_end, fc, is_compressed in pieces:
            overlap_start = max(cp_start, char_start)
            overlap_end = min(cp_end, char_end)
            if overlap_start >= overlap_end:
                continue

            piece_char_offset = overlap_start - cp_start
            piece_char_count = overlap_end - overlap_start

            if is_compressed:
                byte_offset = fc // 2 + piece_char_offset
                raw = word_data[byte_offset:byte_offset + piece_char_count]
                result_parts.append(raw.decode('cp1252', errors='replace'))
            else:
                byte_offset = fc + piece_char_offset * 2
                raw = word_data[byte_offset:byte_offset + piece_char_count * 2]
                result_parts.append(raw.decode('utf-16-le', errors='replace'))

        return ''.join(result_parts)

    # ─── PAPX / TC helpers for table merge detection ──────────────────────

    @staticmethod
    def _cp_to_fc(
        pieces: List[Tuple[int, int, int, bool]], cp: int
    ) -> int:
        """
        Convert a character position (CP) to its byte offset (FC) in the
        WordDocument stream, using the piece table.

        Returns -1 if the CP is not covered by any piece.
        """
        for (cp_start, cp_end, fc, is_compressed) in pieces:
            if cp_start <= cp < cp_end:
                offset = cp - cp_start
                return (fc // 2 + offset) if is_compressed else (fc + offset * 2)
        return -1

    @staticmethod
    def _get_papx_grpprl(
        word_data: bytes,
        table_data: bytes,
        fc_bte: int,
        lcb_bte: int,
        target_fc: int,
    ) -> bytes:
        """
        Look up the PAPX grpprl (sprm chain) for the paragraph whose FC range
        covers *target_fc*.

        PlcBtePapx (in table_data at fc_bte):
          n+1 4-byte FC entries  +  n 4-byte FKP page numbers
          where n = (lcbPlcfBtePapx - 4) / 8

        Each FKP page is 512 bytes at word_data[page_num * 512].

        FKP(PAP) layout:
          [0 .. (cpara+1)*4)  : rgfc  — FC of start of each paragraph
          ...content...
          [511-cpara .. 510]  : rgbx  — bx[j] * 2 = offset to PapxFkpGrpprl
          [511]               : cpara — paragraph count

        PapxFkpGrpprl layout:
          [0]    = cb (byte count of istd + grpprl; 0 = no data)
          [1..2] = istd (2-byte style index)
          [3..]  = grpprl (sprm bytes, length = cb - 2)
        """
        if not fc_bte or lcb_bte < 8:
            return b''
        try:
            bte = table_data[fc_bte: fc_bte + lcb_bte]
            n = (len(bte) - 4) // 8
            if n < 1:
                return b''

            for i in range(n):
                fc0 = struct.unpack('<I', bte[i * 4: i * 4 + 4])[0]
                fc1 = struct.unpack('<I', bte[(i + 1) * 4: (i + 1) * 4 + 4])[0]
                if fc0 <= target_fc < fc1:
                    pn = struct.unpack('<I', bte[(n + 1) * 4 + i * 4: (n + 1) * 4 + i * 4 + 4])[0]
                    fkp_off = pn * 512
                    if fkp_off + 512 > len(word_data):
                        return b''
                    fkp = word_data[fkp_off: fkp_off + 512]
                    cpara = fkp[511]
                    if cpara == 0:
                        return b''
                    for j in range(cpara):
                        fc_j  = struct.unpack('<I', fkp[j * 4: j * 4 + 4])[0]
                        fc_j1 = struct.unpack('<I', fkp[(j + 1) * 4: (j + 1) * 4 + 4])[0]
                        if fc_j <= target_fc < fc_j1:
                            bx = fkp[511 - cpara + j]
                            if bx == 0:
                                return b''
                            papx_off = bx * 2
                            if papx_off + 3 > 512:
                                return b''
                            cb = fkp[papx_off]
                            if cb == 0:
                                # cb=0: read cb2 which replaces cb (MS-DOC 2.9.167)
                                if papx_off + 1 >= 512:
                                    return b''
                                cb2 = fkp[papx_off + 1]
                                if cb2 < 2:
                                    return b''
                                grpprl_end = papx_off + 4 + (cb2 - 2)
                                if grpprl_end > 512:
                                    return b''
                                return bytes(fkp[papx_off + 4: grpprl_end])
                            if cb < 2 or papx_off + 1 + cb > 512:
                                return b''
                            # grpprl starts after cb (1 byte) + istd (2 bytes)
                            return bytes(fkp[papx_off + 3: papx_off + 1 + cb])
                    return b''
        except Exception:
            pass
        return b''

    @staticmethod
    def _parse_tdef_table_operand(op: bytes) -> List[Dict[str, bool]]:
        """
        Parse a sprmTDefTable operand and extract TC80 merge flags.

        The operand layout may have an extra prefix byte before itcMac
        (observed in some Word versions).  Both layouts are tried:
          Layout A: [itcMac(1)] [rgdxaCenter] [rgtc]
          Layout B: [prefix(1)] [itcMac(1)] [rgdxaCenter] [rgtc]

        TC80 flags:
          bit0=fFirstMerged, bit1=fMerged,
          bit5=fVertMerge, bit6=fVertRestart

        Returns [] if parsing fails.
        """
        if not op or len(op) < 3:
            return []

        # Try both layouts: itcMac at op[0] and at op[1]
        for skip in (0, 1):
            if skip + 1 > len(op):
                continue
            itcMac = op[skip]
            if itcMac < 1 or itcMac > 63:
                continue
            rg_start = skip + 1
            rg_size = (itcMac + 1) * 2
            rg_tc_off = rg_start + rg_size
            if rg_tc_off > len(op):
                continue
            # Validate: rgdxaCenter positions must be monotonically increasing
            positions = []
            valid = True
            for ci in range(itcMac + 1):
                off = rg_start + ci * 2
                if off + 2 > len(op):
                    valid = False
                    break
                pos_val = struct.unpack('<h', op[off: off + 2])[0]
                if ci > 0 and pos_val <= positions[-1]:
                    valid = False
                    break
                positions.append(pos_val)
            if not valid:
                continue
            remaining = len(op) - rg_tc_off
            if remaining < itcMac * 2:
                continue
            tc_size = remaining // itcMac
            if tc_size < 2:
                continue
            cells: List[Dict[str, bool]] = []
            for ci in range(itcMac):
                off = rg_tc_off + ci * tc_size
                if off + 2 > len(op):
                    break
                flags = struct.unpack('<H', op[off: off + 2])[0]
                cells.append({
                    'fFirstMerged': bool(flags & 0x0001),
                    'fMerged':      bool(flags & 0x0002),
                    'fVertMerge':   bool(flags & 0x0020),
                    'fVertRestart': bool(flags & 0x0040),
                })
            if cells:
                return cells
        return []

    @staticmethod
    def _extract_tc_list_from_grpprl(grpprl: bytes) -> List[Dict[str, bool]]:
        """
        Parse a PAPX grpprl and extract TC80 merge flags from
        sprmTDefTable (0xD608) — the row-end paragraph property.

        Returns [] if sprmTDefTable is not found or parsing fails.
        """
        i = 0
        while i + 2 <= len(grpprl):
            try:
                sprm = struct.unpack('<H', grpprl[i:i + 2])[0]
                spra = (sprm >> 13) & 0x7

                if sprm == 0xD608:  # sprmTDefTable
                    if i + 3 > len(grpprl):
                        break
                    cb = grpprl[i + 2]
                    if i + 3 + cb > len(grpprl):
                        break
                    op = grpprl[i + 3: i + 3 + cb]
                    result = DOCHandler._parse_tdef_table_operand(op)
                    if result:
                        return result
                    break

                # Advance past unrelated sprm
                if spra in (0, 1):
                    i += 3
                elif spra in (2, 4):
                    i += 4
                elif spra == 3:
                    i += 6
                elif spra == 5:
                    if i + 4 > len(grpprl):
                        break
                    cb2 = struct.unpack('<H', grpprl[i + 2: i + 4])[0]
                    i += 4 + cb2
                elif spra == 6:
                    if i + 3 > len(grpprl):
                        break
                    cb2 = grpprl[i + 2]
                    i += 3 + cb2
                elif spra == 7:
                    i += 5
                else:
                    break
            except Exception:
                break
        return []

    # ─── Body text post-processing ────────────────────────────────────────

    def _collect_table_tc_data(
        self,
        raw_text: str,
        word_data: bytes,
        table_data: bytes,
        fib: Dict[str, Any],
        pieces: List[Tuple[int, int, int, bool]],
    ) -> List[List[Dict[str, bool]]]:
        """
        Collect TC merge flags for each table row in the raw body text.

        Uses a two-phase approach:
        1. Primary: look up PAPX grpprl for each \\x07 mark via BTE/FKP.
        2. Fallback: if the primary method finds nothing, scan the FKP
           pages that cover row-end CPs for raw sprmTDefTable (0xD608)
           byte patterns.  Some Word versions store TDefTable in a
           separate area of the FKP page not referenced by the
           paragraph's BxPap pointer.

        Returns an ordered list of TC flag lists (one list per table row).
        Each inner list has one dict per column with keys:
          fFirstMerged, fMerged, fVertMerge, fVertRestart
        """
        fc_bte  = fib.get('fcPlcfBtePapx', 0)
        lcb_bte = fib.get('lcbPlcfBtePapx', 0)
        if not fc_bte or not lcb_bte:
            return []

        # ── Phase 1: standard PAPX grpprl lookup ──
        row_tc_data: List[List[Dict[str, bool]]] = []
        idx = 0
        while True:
            idx = raw_text.find('\x07', idx)
            if idx < 0:
                break
            try:
                fc = self._cp_to_fc(pieces, idx)
                if fc >= 0:
                    grpprl = self._get_papx_grpprl(word_data, table_data, fc_bte, lcb_bte, fc)
                    tc_list = self._extract_tc_list_from_grpprl(grpprl)
                    if tc_list:
                        row_tc_data.append(tc_list)
            except Exception:
                pass
            idx += 1

        if row_tc_data:
            return row_tc_data

        # ── Phase 2: FKP page scan fallback ──
        return self._collect_tc_via_fkp_scan(
            raw_text, word_data, table_data, fib, pieces
        )

    def _collect_tc_via_fkp_scan(
        self,
        raw_text: str,
        word_data: bytes,
        table_data: bytes,
        fib: Dict[str, Any],
        pieces: List[Tuple[int, int, int, bool]],
    ) -> List[List[Dict[str, bool]]]:
        """
        Fallback: scan FKP pages directly for sprmTDefTable (0xD608)
        to extract TC merge flags when standard PAPX lookup fails.

        For each table row-end \\x07, finds the FKP page covering that
        CP's FC and scans the page for 0xD608 byte patterns.
        """
        fc_bte = fib.get('fcPlcfBtePapx', 0)
        lcb_bte = fib.get('lcbPlcfBtePapx', 0)
        if not fc_bte or not lcb_bte:
            return []

        bte = table_data[fc_bte: fc_bte + lcb_bte]
        n = (len(bte) - 4) // 8
        if n < 1:
            return []

        # Collect row-end CPs from table paragraphs
        row_end_cps: List[int] = []
        paragraphs = raw_text.split('\r')
        cp_offset = 0
        for para in paragraphs:
            if '\x07' in para:
                first_double = para.find('\x07\x07')
                if first_double >= 0:
                    first_row_text = para[:first_double]
                    n_cols = len(first_row_text.split('\x07'))
                    if n_cols < 1:
                        cp_offset += len(para) + 1
                        continue
                    # First row-end
                    row_end_cps.append(cp_offset + first_double + 1)
                    # Subsequent rows: count n_cols+1 \x07 marks each
                    remaining = para[first_double + 2:]
                    pos_in_para = first_double + 2
                    while remaining:
                        x07_count = 0
                        row_end_pos = -1
                        for ri, ch in enumerate(remaining):
                            if ch == '\x07':
                                x07_count += 1
                                if x07_count == n_cols + 1:
                                    row_end_pos = ri
                                    break
                        if row_end_pos < 0:
                            last_x07 = remaining.rfind('\x07')
                            if last_x07 >= 0:
                                row_end_cps.append(cp_offset + pos_in_para + last_x07)
                            break
                        else:
                            row_end_cps.append(cp_offset + pos_in_para + row_end_pos)
                            remaining = remaining[row_end_pos + 1:]
                            pos_in_para += row_end_pos + 1
            cp_offset += len(para) + 1  # +1 for the \r separator

        if not row_end_cps:
            return []

        # For each row-end CP, find its FKP page and scan for sprmTDefTable
        row_tc_data: List[List[Dict[str, bool]]] = []
        for cp in row_end_cps:
            fc = self._cp_to_fc(pieces, cp)
            if fc < 0:
                row_tc_data.append([])
                continue

            fkp_page_num = -1
            for bi in range(n):
                fc0 = struct.unpack('<I', bte[bi * 4: bi * 4 + 4])[0]
                fc1 = struct.unpack('<I', bte[(bi + 1) * 4: (bi + 1) * 4 + 4])[0]
                if fc0 <= fc < fc1:
                    fkp_page_num = struct.unpack('<I', bte[(n + 1) * 4 + bi * 4: (n + 1) * 4 + bi * 4 + 4])[0]
                    break

            if fkp_page_num < 0:
                row_tc_data.append([])
                continue

            tc_list = self._scan_fkp_for_tdef(word_data, fkp_page_num)
            row_tc_data.append(tc_list)

        return row_tc_data

    def _scan_fkp_for_tdef(
        self,
        word_data: bytes,
        fkp_page_num: int,
    ) -> List[Dict[str, bool]]:
        """
        Scan a single FKP page for sprmTDefTable (0xD608) and parse
        TC80 merge flags from the first valid instance found.
        """
        fkp_start = fkp_page_num * 512
        if fkp_start + 512 > len(word_data):
            return []
        fkp = word_data[fkp_start: fkp_start + 512]

        for si in range(len(fkp) - 4):
            if fkp[si] == 0x08 and fkp[si + 1] == 0xD6:
                cb = fkp[si + 2]
                if cb < 5 or si + 3 + cb > 512:
                    continue
                op = bytes(fkp[si + 3: si + 3 + cb])
                result = self._parse_tdef_table_operand(op)
                if result:
                    return result
        return []

    def _process_doc_body_text(
        self,
        raw: str,
        word_data: Optional[bytes] = None,
        table_data: Optional[bytes] = None,
        fib: Optional[Dict[str, Any]] = None,
        pieces: Optional[List[Tuple[int, int, int, bool]]] = None,
    ) -> str:
        """
        Process raw body text extracted via Piece Table.

        Handles:
        - Field codes (\x13 ... \x14 ... \x15): keep display text only
        - Table cells (\x07): reconstruct as HTML tables (with colspan/rowspan)
        - Page breaks (\x0C): insert [Page Number: N] tags
        - Paragraph marks (\r): convert to newlines
        - Other control characters: strip
        """
        # Step 1 — Collect TC merge data BEFORE field-code stripping
        # (row-end \x07 CPs are needed to look up PAPX in the binary stream)
        row_tc_data: List[List[Dict[str, bool]]] = []
        if word_data and table_data and fib and pieces and '\x07' in raw:
            row_tc_data = self._collect_table_tc_data(raw, word_data, table_data, fib, pieces)

        # Step 2 — Strip field codes, keep display text only
        text = self._strip_field_codes(raw)

        # Step 3 — Reconstruct tables as HTML (with colspan/rowspan when merges exist)
        text = self._reconstruct_tables_html(text, row_tc_data)

        # Step 4 — Convert \r to \n (Word uses \r as paragraph separator)
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Step 5 — Handle page breaks (\x0C)
        lines = text.split('\n')
        result_lines: List[str] = []
        page_number = 1  # Page 1 tag is already emitted by the caller

        for line in lines:
            if '\x0c' in line:
                parts = line.split('\x0c')
                for pi, part in enumerate(parts):
                    part_stripped = part.strip()
                    if part_stripped:
                        result_lines.append(part_stripped)
                    if pi < len(parts) - 1:
                        page_number += 1
                        result_lines.append(self.create_page_tag(page_number))
                continue

            stripped = line.strip()
            if not stripped:
                result_lines.append('')
                continue

            result_lines.append(stripped)

        result = '\n'.join(result_lines)
        # Remove residual control characters (but keep printable + newline + tab)
        result = re.sub(r'[\x00-\x08\x0b\x0e-\x1f]', '', result)
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()

    @staticmethod
    def _strip_field_codes(text: str) -> str:
        """
        Strip Word field codes (\x13 instruction \x14 display \x15).

        Field structure:  \x13 <instruction> \x14 <display_value> \x15
        We discard the instruction and keep only the display value.
        Fields can be nested, so we process innermost first.
        """
        # Process innermost fields first (handles nesting)
        max_iterations = 50
        for _ in range(max_iterations):
            # Find an innermost field: \x13 ... (no \x13 inside) ... \x15
            m = re.search(r'\x13([^\x13\x15]*?)\x14([^\x13\x15]*?)\x15', text)
            if m:
                text = text[:m.start()] + m.group(2) + text[m.end():]
                continue
            # Field with no separator (no display value)
            m = re.search(r'\x13([^\x13\x15]*?)\x15', text)
            if m:
                text = text[:m.start()] + text[m.end():]
                continue
            break
        # Clean up any orphaned field markers
        text = text.replace('\x13', '').replace('\x14', '').replace('\x15', '')
        return text

    @staticmethod
    def _reconstruct_tables_html(
        text: str,
        row_tc_data: List[List[Dict[str, bool]]],
    ) -> str:
        """
        Detect DOC table paragraphs (containing \\x07 cell marks) and rebuild
        them as HTML tables, using TC merge flags (colspan / rowspan) when
        available.

        In DOC binary every table is stored as ONE paragraph (between \\r marks).
        Within that paragraph:
          - cell text ends with \\x07  (cell mark)
          - each row ends with an extra \\x07  (row-end mark), so a row with
            N columns contains N+1 \\x07 marks total
          - the first row's row-end is identified by the first \\x07\\x07 pair
            (the first row cannot have vertical-merge continuation cells)

        *row_tc_data* is the ordered list of TC flag lists collected by
        _collect_table_tc_data() before field-code stripping.  Each entry
        corresponds to one table row in document order, with one TC dict per
        column:
          fFirstMerged  — first cell in a horizontal merge (has colspan >= 2)
          fMerged       — continuation cell in a horizontal merge (skip/hide)
          fVertRestart  — first cell in a vertical merge (has rowspan >= 2)
          fVertMerge    — continuation cell in a vertical merge (skip/hide)

        When *row_tc_data* is empty (e.g., PAPX lookup failed), tables are
        still rendered as plain HTML without merge attributes.
        """
        paragraphs = text.split('\r')
        result_parts: List[str] = []
        tc_row_idx = 0  # sequential index into row_tc_data

        for para in paragraphs:
            if '\x07' not in para:
                result_parts.append(para + '\r')
                continue

            # ── Step A: split the paragraph into raw cell-text rows ──────────

            first_double = para.find('\x07\x07')
            if first_double < 0:
                # Only one row (no row-end double-mark found)
                cells_text = para.split('\x07')
                while cells_text and not cells_text[-1].strip():
                    cells_text.pop()
                raw_rows: List[List[str]] = [cells_text] if cells_text else []
                orig_cell_counts: List[int] = [len(cells_text)] if cells_text else []
                n_cols = len(cells_text)
            else:
                first_row_text = para[:first_double]
                first_cells = first_row_text.split('\x07')
                n_cols = len(first_cells)
                raw_rows = [first_cells]
                orig_cell_counts = [n_cols]

                remaining = para[first_double + 2:]   # skip the \x07\x07
                while remaining:
                    x07_count = 0
                    row_end_pos = -1
                    for pos, ch in enumerate(remaining):
                        if ch == '\x07':
                            x07_count += 1
                            if x07_count == n_cols + 1:
                                row_end_pos = pos
                                break

                    if row_end_pos < 0:
                        row_text  = remaining
                        remaining = ''
                    else:
                        row_text  = remaining[:row_end_pos + 1]
                        remaining = remaining[row_end_pos + 1:]

                    cells_text = row_text.split('\x07')
                    while cells_text and not cells_text[-1].strip():
                        cells_text.pop()
                    if not cells_text:
                        continue
                    # Record original cell count before padding (for colspan)
                    orig_count = len(cells_text)
                    while len(cells_text) < n_cols:
                        cells_text.append('')
                    raw_rows.append(cells_text)
                    orig_cell_counts.append(orig_count)

            if not raw_rows:
                result_parts.append('\r')
                continue

            # ── Step B: fetch TC flag data for these rows ────────────────────
            num_rows = len(raw_rows)
            table_tc: List[List[Dict[str, bool]]] = []
            for _ in range(num_rows):
                if tc_row_idx < len(row_tc_data):
                    tc_row = list(row_tc_data[tc_row_idx])
                    tc_row_idx += 1
                else:
                    tc_row = []
                # Pad to at least n_cols entries
                while len(tc_row) < n_cols:
                    tc_row.append({})
                table_tc.append(tc_row)

            # ── Step C: compute rowspan for cells with fVertRestart ───────────
            rowspan_map: Dict[Tuple[int, int], int] = {}
            for ci in range(n_cols):
                ri = 0
                while ri < num_rows:
                    tc = table_tc[ri][ci] if ci < len(table_tc[ri]) else {}
                    if tc.get('fVertRestart'):
                        span = 1
                        for ri2 in range(ri + 1, num_rows):
                            tc2 = table_tc[ri2][ci] if ci < len(table_tc[ri2]) else {}
                            if tc2.get('fVertMerge') and not tc2.get('fVertRestart'):
                                span += 1
                            else:
                                break
                        rowspan_map[(ri, ci)] = span
                    ri += 1

            # ── Step D: build TableData with proper spans ────────────────────
            table_rows: List[List[TableCell]] = []
            for ri, cells_text in enumerate(raw_rows):
                tc_row = table_tc[ri]
                row_cells: List[TableCell] = []
                ci = 0          # logical column index (honours colspan)
                cell_idx = 0    # index into cells_text

                # Detect structural horizontal merge: when a row has fewer
                # actual cells than n_cols, the last real cell spans the
                # remaining columns.
                orig_count = orig_cell_counts[ri] if ri < len(orig_cell_counts) else n_cols
                structural_colspan_at = orig_count - 1 if orig_count < n_cols else -1
                structural_colspan_size = n_cols - orig_count + 1 if orig_count < n_cols else 1

                while cell_idx < len(cells_text):
                    tc = tc_row[cell_idx] if cell_idx < len(tc_row) else {}

                    # Skip horizontal-merge continuation cells
                    if tc.get('fMerged') and not tc.get('fFirstMerged'):
                        cell_idx += 1
                        ci += 1
                        continue

                    # Skip vertical-merge continuation cells
                    if tc.get('fVertMerge') and not tc.get('fVertRestart'):
                        cell_idx += 1
                        ci += 1
                        continue

                    # Compute colspan from TC flags
                    col_span = 1
                    if tc.get('fFirstMerged'):
                        j = cell_idx + 1
                        while j < len(tc_row):
                            tc_j = tc_row[j]
                            if tc_j.get('fMerged') and not tc_j.get('fFirstMerged'):
                                col_span += 1
                                j += 1
                            else:
                                break

                    # Apply structural colspan for rows with fewer cells
                    if cell_idx == structural_colspan_at and col_span == 1:
                        col_span = structural_colspan_size
                        # Skip the padding empty cells
                        cell_idx += 1
                        ci += col_span
                        row_cells.append(TableCell(
                            content=cells_text[structural_colspan_at].strip(),
                            row_span=rowspan_map.get((ri, ci - col_span), 1),
                            col_span=col_span,
                            is_header=False,
                            row_index=ri,
                            col_index=ci - col_span,
                        ))
                        break

                    # Compute rowspan
                    row_span = rowspan_map.get((ri, ci), 1)

                    row_cells.append(TableCell(
                        content=cells_text[cell_idx].strip(),
                        row_span=row_span,
                        col_span=col_span,
                        is_header=False,
                        row_index=ri,
                        col_index=ci,
                    ))
                    ci       += col_span
                    cell_idx += 1

                if row_cells:
                    table_rows.append(row_cells)

            # ── Step E: render as HTML ───────────────────────────────────────
            if table_rows:
                td = TableData(
                    rows=table_rows,
                    num_rows=len(table_rows),
                    num_cols=n_cols,
                )
                processor = TableProcessor(
                    TableProcessorConfig(output_format=TableOutputFormat.HTML)
                )
                html = processor.format_table_as_html(td)
                result_parts.append(html + '\r')
            else:
                result_parts.append('\r')

        return ''.join(result_parts)



    def _extract_textbox_text(
        self,
        word_data: bytes,
        pieces: List[Tuple[int, int, int, bool]],
        fib: Dict[str, Any],
    ) -> str:
        """
        Extract text from the textbox story (ccpTxbx region).

        Textbox text is stored after body + footnotes + headers + macros +
        annotations + endnotes in the DOC text stream. Each non-zero ccp
        section is followed by a 1-char separator.

        Returns cleaned textbox text with each entry on a new line.
        """
        ccp_txbx = fib.get('ccpTxbx', 0)
        if ccp_txbx == 0:
            return ''

        # Calculate the CP start of the textbox region
        txbx_cp_start = fib['ccpText'] + 1  # body + separator
        for key in ('ccpFtn', 'ccpHdd', 'ccpMcr', 'ccpAtn', 'ccpEdn'):
            if fib.get(key, 0) > 0:
                txbx_cp_start += fib[key] + 1  # section + separator

        raw_txbx = self._read_cp_range(word_data, pieces, txbx_cp_start, ccp_txbx)
        if not raw_txbx:
            return ''

        # Strip field codes and control characters
        text = self._strip_field_codes(raw_txbx)
        text = re.sub(r'[\x00-\x0c\x0e-\x1f]', '', text)

        # Split by \r, collect non-empty entries
        parts = [p.strip() for p in text.split('\r') if p.strip()]
        if not parts:
            return ''

        return '\n'.join(parts)

    # ─── Header / Footer extraction ──────────────────────────────────────

    def _extract_headers_footers_via_pieces(
        self,
        word_data: bytes,
        pieces: List[Tuple[int, int, int, bool]],
        fib: Dict[str, Any],
    ) -> Tuple[str, str]:
        """
        Extract header and footer text using the Piece Table and PlcfHdd.

        PlcfHdd defines CP boundaries for each header/footer slot.
        CPs in PlcfHdd are relative to ccpText (absolute CP base = ccpText).
        Slot layout: [separator] + [even-hdr, odd-hdr, even-ftr, odd-ftr,
                                     first-hdr, first-ftr] × N_sections

        Returns:
            (header_text, footer_text)
        """
        ccp_hdd = fib['ccpHdd']
        if ccp_hdd == 0:
            return '', ''

        # Try PlcfHdd-based extraction first
        table_data = fib.get('table_data', b'')
        fc_plcf_hdd = fib.get('fcPlcfHdd', 0)
        lcb_plcf_hdd = fib.get('lcbPlcfHdd', 0)

        if lcb_plcf_hdd >= 8 and fc_plcf_hdd + lcb_plcf_hdd <= len(table_data):
            result = self._extract_hdd_via_plcfhdd(
                word_data, pieces, fib, table_data, fc_plcf_hdd, lcb_plcf_hdd
            )
            if result and (result[0] or result[1]):
                return result

        # Fallback: extract all hdd text and find non-empty blocks
        return self._extract_hdd_fallback(word_data, pieces, fib)

    def _extract_hdd_via_plcfhdd(
        self,
        word_data: bytes,
        pieces: List[Tuple[int, int, int, bool]],
        fib: Dict[str, Any],
        table_data: bytes,
        fc_plcf_hdd: int,
        lcb_plcf_hdd: int,
    ) -> Tuple[str, str]:
        """Extract headers/footers using PlcfHdd slot boundaries."""
        plcf_data = table_data[fc_plcf_hdd:fc_plcf_hdd + lcb_plcf_hdd]
        n_cps = lcb_plcf_hdd // 4
        if n_cps < 2:
            return '', ''

        cps = [struct.unpack('<I', plcf_data[j * 4:(j + 1) * 4])[0] for j in range(n_cps)]

        # PlcfHdd CPs are relative to ccpText (the body-terminator CP)
        plcf_base_cp = fib['ccpText']

        # Use PlcfHdd's last CP as the true extent of the hdd region
        # (ccpHdd may undercount by 1-2 due to trailing paragraph marks)
        hdd_cp_end = plcf_base_cp + cps[-1]

        headers: List[str] = []
        footers: List[str] = []

        for i in range(len(cps) - 1):
            cp_start = cps[i]
            cp_end = cps[i + 1]
            if cp_start >= cp_end:
                continue

            # Skip separator slot (index 0)
            if i == 0:
                continue

            # Clamp to hdd boundary to avoid reading into the next story
            abs_start = plcf_base_cp + cp_start
            abs_end = min(plcf_base_cp + cp_end, hdd_cp_end)
            if abs_start >= abs_end:
                continue

            char_count = abs_end - abs_start
            slot_text = self._read_cp_range(word_data, pieces, abs_start, char_count)
            if not slot_text:
                continue

            # Strip field codes and control chars, keep only readable text
            slot_text = self._strip_field_codes(slot_text)
            slot_text = re.sub(r'[\x00-\x0c\x0e-\x1f]', '', slot_text)
            slot_text = slot_text.strip()
            if not slot_text:
                continue

            # Slot type: (index - 1) % 6
            # 0=even-hdr, 1=odd-hdr, 2=even-ftr, 3=odd-ftr, 4=first-hdr, 5=first-ftr
            type_idx = (i - 1) % 6
            if type_idx in (0, 1, 4):  # header types
                if slot_text not in headers:
                    headers.append(slot_text)
            elif type_idx in (2, 3, 5):  # footer types
                if slot_text not in footers:
                    footers.append(slot_text)

        return '\n'.join(headers), '\n'.join(footers)

    def _extract_hdd_fallback(
        self,
        word_data: bytes,
        pieces: List[Tuple[int, int, int, bool]],
        fib: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Fallback: extract hdd text and split into header/footer by text blocks."""
        ccp_hdd = fib['ccpHdd']
        hdd_cp_start = fib['ccpText'] + 1
        if fib['ccpFtn']:
            hdd_cp_start += fib['ccpFtn'] + 1

        hdd_text = self._read_cp_range(word_data, pieces, hdd_cp_start, ccp_hdd)
        if not hdd_text:
            return '', ''

        hdd_text = self._strip_field_codes(hdd_text)
        hdd_text = re.sub(r'[\x00-\x0c\x0e-\x1f]', '', hdd_text)

        # Collect non-empty text blocks separated by \r
        parts = [p.strip() for p in hdd_text.split('\r') if p.strip()]
        if not parts:
            return '', ''

        if len(parts) >= 2:
            return parts[0], parts[-1]
        return parts[0], ''

    # ─── Heuristic fallback (kept for corrupted files) ───────────────────

    def _extract_ole_text_heuristic(self, ole: olefile.OleFileIO) -> str:
        """Fallback: extract text via heuristic byte scanning (legacy method)."""
        try:
            if not ole.exists('WordDocument'):
                return ''
            word_data = ole.openstream('WordDocument').read()
            if len(word_data) < 12:
                return ''
            magic = struct.unpack('<H', word_data[0:2])[0]
            if magic not in (0xA5EC, 0xA5DC):
                return ''
            return self._extract_text_from_word_stream(word_data)
        except Exception as e:
            self.logger.warning(f"Heuristic text extraction error: {e}")
            return ''

    # ─── Legacy methods (kept for backward compatibility) ─────────────────
    # _extract_ole_headers_footers and _extract_ole_text_range are no longer
    # used in the primary pipeline but retained for any external callers.

    def _extract_ole_headers_footers(self, ole: olefile.OleFileIO) -> tuple:
        """Legacy header/footer extraction. New code uses _extract_headers_footers_via_pieces."""
        try:
            fib = self._parse_fib(ole)
            if fib is None:
                return '', ''
            return self._extract_headers_footers_via_pieces(
                fib['word_data'], fib['pieces'], fib
            )
        except Exception:
            return '', ''

    def _extract_text_from_word_stream(self, data: bytes) -> str:
        """Extract text from Word stream (heuristic method)."""
        text_parts = []

        # Method 1: UTF-16LE Unicode text extraction
        try:
            # Find consecutive Unicode characters
            i = 0
            while i < len(data) - 1:
                # Find start of Unicode text (printable characters)
                if 0x20 <= data[i] <= 0x7E and data[i+1] == 0x00:
                    # Collect Unicode characters
                    unicode_bytes = []
                    j = i
                    while j < len(data) - 1:
                        char = data[j]
                        next_byte = data[j+1]

                        # ASCII range Unicode character or newline
                        if next_byte == 0x00 and (0x20 <= char <= 0x7E or char in (0x0D, 0x0A, 0x09)):
                            unicode_bytes.extend([char, next_byte])
                            j += 2
                        elif 0xAC <= next_byte <= 0xD7:  # Korean Unicode range (AC00-D7AF)
                            unicode_bytes.extend([char, next_byte])
                            j += 2
                        elif next_byte in range(0x30, 0x4E):  # CJK range partial
                            unicode_bytes.extend([char, next_byte])
                            j += 2
                        else:
                            break

                    if len(unicode_bytes) >= 8:  # Minimum 4 characters
                        try:
                            text = bytes(unicode_bytes).decode('utf-16-le', errors='ignore')
                            text = text.strip()
                            if len(text) >= 4 and not text.startswith('\\'):
                                text = text.replace('\r\n', '\n').replace('\r', '\n')
                                text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)
                                if text:
                                    text_parts.append(text)
                        except:
                            pass
                    i = j
                else:
                    i += 1
        except Exception as e:
            self.logger.debug(f"Unicode extraction error: {e}")

        # Process result
        if text_parts:
            # Remove duplicates and merge
            seen = set()
            unique_parts = []
            for part in text_parts:
                if part not in seen and len(part) > 3:
                    seen.add(part)
                    unique_parts.append(part)

            result = '\n'.join(unique_parts)
            # Handle excessive line breaks
            result = re.sub(r'\n{3,}', '\n\n', result)
            return result.strip()

        return ""

