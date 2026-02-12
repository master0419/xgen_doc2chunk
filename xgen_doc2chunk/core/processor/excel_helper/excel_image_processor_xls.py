# xgen_doc2chunk/core/processor/excel_helper/excel_image_processor_xls.py
"""
Excel XLS Image Processor

Provides XLS-specific image processing that inherits from ImageProcessor.
Handles embedded images from XLS files (OLE compound document format).

XLS files store images in BLIP (Binary Large Image or Picture) format
within the Workbook stream as part of MsoDrawing records.

This class consolidates all XLS image extraction logic including:
- OLE compound document stream extraction
- BIFF8 MsoDrawing record parsing with sheet boundary detection
- Image signature-based extraction (PNG, JPEG, GIF, BMP)
- Per-sheet image extraction using BOF/EOF record parsing
"""
import io
import logging
import struct
from typing import Dict, List, Optional, Tuple

import olefile

from xgen_doc2chunk.core.functions.img_processor import ImageProcessor
from xgen_doc2chunk.core.functions.storage_backend import BaseStorageBackend

logger = logging.getLogger("xgen_doc2chunk.image_processor.excel_xls")

# Image signatures for detection
IMAGE_SIGNATURES = {
    b'\x89PNG\r\n\x1a\n': 'png',
    b'\xff\xd8\xff': 'jpg',
    b'GIF87a': 'gif',
    b'GIF89a': 'gif',
    b'BM': 'bmp',
}

# BIFF8 Record Types
BIFF_BOF = 0x0809       # Beginning of File
BIFF_EOF = 0x000A       # End of File
BIFF_BOUNDSHEET = 0x0085  # Sheet information
BIFF_MSODRAWING = 0x00EC  # MsoDrawing record


class XLSImageProcessor(ImageProcessor):
    """
    XLS-specific image processor.
    
    Inherits from ImageProcessor and provides XLS-specific processing.
    
    XLS files use OLE compound document format, where images are embedded
    within the Workbook stream as BLIP records inside MsoDrawing records.
    
    Handles:
    - Embedded worksheet images (BLIP format)
    - Drawing images from MsoDrawing records
    
    Example:
        processor = XLSImageProcessor()
        
        # Extract all images from XLS file
        images = processor.extract_images_from_xls("file.xls")
        
        # Process and save each image
        for img_data in images:
            tag = processor.save_image(img_data)
    """
    
    def __init__(
        self,
        directory_path: str = "temp/images",
        tag_prefix: str = "[Image:",
        tag_suffix: str = "]",
        storage_backend: Optional[BaseStorageBackend] = None,
    ):
        """
        Initialize XLSImageProcessor.
        
        Args:
            directory_path: Image save directory
            tag_prefix: Tag prefix for image references
            tag_suffix: Tag suffix for image references
            storage_backend: Storage backend for saving images
        """
        super().__init__(
            directory_path=directory_path,
            tag_prefix=tag_prefix,
            tag_suffix=tag_suffix,
            storage_backend=storage_backend,
        )
    
    def process_image(
        self,
        image_data: bytes,
        sheet_name: Optional[str] = None,
        image_index: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Process and save XLS image data.
        
        Args:
            image_data: Raw image binary data
            sheet_name: Source sheet name (for naming)
            image_index: Image index in sheet (for naming)
            **kwargs: Additional options
            
        Returns:
            Image tag string, or None on failure
        """
        custom_name = None
        if sheet_name is not None:
            safe_sheet = sheet_name.replace(' ', '_').replace('/', '_')
            if image_index is not None:
                custom_name = f"xls_{safe_sheet}_{image_index}"
            else:
                custom_name = f"xls_{safe_sheet}"
        
        return self.save_image(image_data, custom_name=custom_name)
    
    def extract_images_from_xls(
        self,
        file_path: str,
    ) -> List[bytes]:
        """
        Extract images from XLS file (OLE compound document).
        XLS files store images in BLIP (Binary Large Image or Picture) format
        within the Workbook stream.

        Args:
            file_path: Path to XLS file

        Returns:
            List of image bytes
        """
        images = []
        
        try:
            # First try: Extract from OLE compound document streams
            images = self._extract_images_from_ole(file_path)
            
            # Second try: Parse BIFF8 Workbook stream for embedded images
            if not images:
                images = self._extract_images_from_biff(file_path)
            
            logger.debug(f"Extracted {len(images)} images from XLS file")
            return images
            
        except Exception as e:
            logger.warning(f"Error extracting images from XLS: {e}")
            return []
    
    def extract_images_by_sheet(
        self,
        file_path: str,
        sheet_names: List[str],
    ) -> Dict[int, List[bytes]]:
        """
        Extract images from XLS file grouped by sheet index.
        
        XLS stores image binary data in the global MSODRAWINGGROUP,
        but each sheet has OBJ records (Picture type) that reference images.
        This method counts Picture objects per sheet and assigns images accordingly.
        
        Args:
            file_path: Path to XLS file
            sheet_names: List of sheet names (from xlrd workbook)
            
        Returns:
            Dictionary mapping sheet index to list of image bytes
            {0: [img1, img2], 1: [img3], ...}
        """
        images_by_sheet: Dict[int, List[bytes]] = {i: [] for i in range(len(sheet_names))}
        
        try:
            if not olefile.isOleFile(file_path):
                return images_by_sheet
            
            with olefile.OleFileIO(file_path) as ole:
                # Find and read the Workbook stream
                workbook_data = None
                for stream_name in ['Workbook', 'Book']:
                    if ole.exists(stream_name):
                        workbook_data = ole.openstream(stream_name).read()
                        break
                
                if workbook_data is None:
                    return images_by_sheet
                
                # Step 1: Extract all images from global drawing group
                all_images = self._extract_global_images(workbook_data)
                logger.debug(f"Found {len(all_images)} images in global drawing group")
                
                if not all_images:
                    return images_by_sheet
                
                # Step 2: Find BOUNDSHEET records to get sheet offsets
                boundsheets = self._parse_boundsheet_records(workbook_data)
                
                # Step 3: Count Picture OBJ records per sheet
                picture_counts = self._count_picture_objects_per_sheet(workbook_data, boundsheets)
                
                # Step 4: Assign images to sheets based on picture counts
                # Images are stored in order, so we assign them sequentially
                image_idx = 0
                for sheet_idx in range(len(sheet_names)):
                    count = picture_counts.get(sheet_idx, 0)
                    for _ in range(count):
                        if image_idx < len(all_images):
                            images_by_sheet[sheet_idx].append(all_images[image_idx])
                            image_idx += 1
                
                # Log results
                for sheet_idx, imgs in images_by_sheet.items():
                    if imgs:
                        sheet_name = sheet_names[sheet_idx] if sheet_idx < len(sheet_names) else f"Sheet{sheet_idx}"
                        logger.debug(f"Sheet '{sheet_name}': {len(imgs)} images")
                
        except Exception as e:
            logger.warning(f"Error extracting images by sheet from XLS: {e}")
        
        return images_by_sheet
    
    def _extract_global_images(self, workbook_data: bytes) -> List[bytes]:
        """
        Extract all images from global MSODRAWINGGROUP and CONTINUE records.
        
        Args:
            workbook_data: Workbook stream data
            
        Returns:
            List of image bytes in order
        """
        images = []
        drawing_group_data = b''
        
        offset = 0
        continue_for_group = False
        
        while offset < len(workbook_data) - 4:
            record_type = struct.unpack('<H', workbook_data[offset:offset+2])[0]
            record_len = struct.unpack('<H', workbook_data[offset+2:offset+4])[0]
            record_data = workbook_data[offset+4:offset+4+record_len]
            
            if record_type == 0x00EB:  # MSODRAWINGGROUP
                drawing_group_data += record_data
                continue_for_group = True
            elif record_type == 0x003C and continue_for_group:  # CONTINUE
                drawing_group_data += record_data
            elif record_type != 0x003C:
                continue_for_group = False
            
            offset += 4 + record_len
        
        # Extract images from drawing group data
        images = self._find_images_by_signature(drawing_group_data)
        
        return images
    
    def _parse_boundsheet_records(self, workbook_data: bytes) -> List[Tuple[int, str]]:
        """
        Parse BOUNDSHEET records to get sheet offsets and names.
        
        Args:
            workbook_data: Workbook stream data
            
        Returns:
            List of (offset, name) tuples
        """
        boundsheets = []
        offset = 0
        
        while offset < len(workbook_data) - 4:
            record_type = struct.unpack('<H', workbook_data[offset:offset+2])[0]
            record_len = struct.unpack('<H', workbook_data[offset+2:offset+4])[0]
            record_data = workbook_data[offset+4:offset+4+record_len]
            
            if record_type == 0x0085:  # BOUNDSHEET
                if len(record_data) >= 8:
                    sheet_offset = struct.unpack('<I', record_data[0:4])[0]
                    name_len = record_data[6]
                    name = "unknown"
                    if len(record_data) > 8:
                        encoding = record_data[7]
                        try:
                            if encoding == 0:
                                name = record_data[8:8+name_len].decode('latin-1', errors='replace')
                            else:
                                name = record_data[8:8+name_len*2].decode('utf-16-le', errors='replace')
                        except Exception:
                            pass
                    boundsheets.append((sheet_offset, name))
            
            offset += 4 + record_len
        
        return boundsheets
    
    def _count_picture_objects_per_sheet(
        self,
        workbook_data: bytes,
        boundsheets: List[Tuple[int, str]]
    ) -> Dict[int, int]:
        """
        Count Picture OBJ records in each sheet.
        
        OBJ records with object type 0x08 are Picture objects.
        
        Args:
            workbook_data: Workbook stream data
            boundsheets: List of (offset, name) tuples from BOUNDSHEET records
            
        Returns:
            Dictionary mapping sheet index to picture count
        """
        picture_counts: Dict[int, int] = {}
        
        offset = 0
        current_sheet_idx = -1
        
        while offset < len(workbook_data) - 4:
            # Check if we've crossed into a new sheet
            for i, (sheet_off, name) in enumerate(boundsheets):
                if offset == sheet_off:
                    current_sheet_idx = i
            
            record_type = struct.unpack('<H', workbook_data[offset:offset+2])[0]
            record_len = struct.unpack('<H', workbook_data[offset+2:offset+4])[0]
            record_data = workbook_data[offset+4:offset+4+record_len]
            
            # OBJ record (0x005D)
            if record_type == 0x005D and current_sheet_idx >= 0:
                # Parse sub-records to find ftCmo (Common Object Data)
                sub_offset = 0
                while sub_offset < len(record_data) - 4:
                    ft = struct.unpack('<H', record_data[sub_offset:sub_offset+2])[0]
                    cb = struct.unpack('<H', record_data[sub_offset+2:sub_offset+4])[0]
                    
                    if ft == 0x15:  # ftCmo - Common Object Data
                        if cb >= 18 and sub_offset + 6 <= len(record_data):
                            obj_type = struct.unpack('<H', record_data[sub_offset+4:sub_offset+6])[0]
                            if obj_type == 0x08:  # Picture
                                if current_sheet_idx not in picture_counts:
                                    picture_counts[current_sheet_idx] = 0
                                picture_counts[current_sheet_idx] += 1
                    
                    if ft == 0x00:  # End marker
                        break
                    
                    sub_offset += 4 + cb
            
            offset += 4 + record_len
        
        return picture_counts
    
    def _parse_biff_by_sheet(
        self,
        data: bytes,
        num_sheets: int,
    ) -> Dict[int, List[bytes]]:
        """
        Parse BIFF8 stream and extract images grouped by sheet.
        
        BIFF8 structure:
        - Workbook globals (BOF type=0x0005)
        - Sheet 1 (BOF type=0x0010)
        - Sheet 2 (BOF type=0x0010)
        - ...
        
        Args:
            data: Workbook stream data
            num_sheets: Number of sheets in workbook
            
        Returns:
            Dictionary mapping sheet index to list of image bytes
        """
        images_by_sheet: Dict[int, List[bytes]] = {i: [] for i in range(num_sheets)}
        
        offset = 0
        current_sheet_idx = -1  # -1 means workbook globals
        drawing_data_buffer = b''  # Buffer to accumulate MSODRAWING data
        
        while offset < len(data) - 4:
            try:
                # BIFF8 record header: 2 bytes type + 2 bytes length
                record_type = struct.unpack('<H', data[offset:offset+2])[0]
                record_len = struct.unpack('<H', data[offset+2:offset+4])[0]
                record_data = data[offset+4:offset+4+record_len]
                
                # BOF record - marks start of workbook globals or sheet
                if record_type == BIFF_BOF and len(record_data) >= 2:
                    bof_type = struct.unpack('<H', record_data[0:2])[0]
                    
                    # Process accumulated drawing data before moving to new section
                    if drawing_data_buffer:
                        extracted = self._find_images_by_signature(drawing_data_buffer)
                        if current_sheet_idx >= 0 and current_sheet_idx < num_sheets:
                            images_by_sheet[current_sheet_idx].extend(extracted)
                        drawing_data_buffer = b''
                    
                    if bof_type == 0x0010:  # Worksheet
                        current_sheet_idx += 1
                    elif bof_type == 0x0005:  # Workbook globals
                        current_sheet_idx = -1
                
                # MSODRAWING record - contains drawing/image data
                elif record_type == BIFF_MSODRAWING:
                    drawing_data_buffer += record_data
                
                # EOF record - marks end of current section
                elif record_type == BIFF_EOF:
                    # Process accumulated drawing data
                    if drawing_data_buffer:
                        extracted = self._find_images_by_signature(drawing_data_buffer)
                        if current_sheet_idx >= 0 and current_sheet_idx < num_sheets:
                            images_by_sheet[current_sheet_idx].extend(extracted)
                        drawing_data_buffer = b''
                
                # Move to next record
                offset += 4 + record_len
                
            except Exception as e:
                logger.debug(f"Error parsing BIFF record at offset {offset}: {e}")
                offset += 1
        
        # Process any remaining drawing data
        if drawing_data_buffer:
            extracted = self._find_images_by_signature(drawing_data_buffer)
            if current_sheet_idx >= 0 and current_sheet_idx < num_sheets:
                images_by_sheet[current_sheet_idx].extend(extracted)
        
        # Log extraction results
        total_images = sum(len(imgs) for imgs in images_by_sheet.values())
        logger.debug(f"Extracted {total_images} images across {num_sheets} sheets")
        for idx, imgs in images_by_sheet.items():
            if imgs:
                logger.debug(f"  Sheet {idx}: {len(imgs)} images")
        
        return images_by_sheet
    
    def _extract_images_from_ole(self, file_path: str) -> List[bytes]:
        """
        Extract images from OLE compound document streams.
        
        Args:
            file_path: Path to XLS file
            
        Returns:
            List of image bytes
        """
        images = []
        
        try:
            if not olefile.isOleFile(file_path):
                return images
            
            with olefile.OleFileIO(file_path) as ole:
                # Look for common image storage locations in XLS
                for entry in ole.listdir():
                    entry_path = '/'.join(entry)
                    entry_lower = entry_path.lower()
                    
                    # Check for image-related streams
                    # XLS stores images in various locations
                    if any(keyword in entry_lower for keyword in ['picture', 'image', 'media', 'mbd']):
                        try:
                            data = ole.openstream(entry).read()
                            extracted = self._extract_blip_images(data)
                            images.extend(extracted)
                        except Exception as e:
                            logger.debug(f"Error reading OLE stream {entry_path}: {e}")
                
                # Also check the Workbook stream for embedded images
                workbook_streams = ['Workbook', 'Book']
                for stream_name in workbook_streams:
                    if ole.exists(stream_name):
                        try:
                            data = ole.openstream(stream_name).read()
                            # Extract BLIP images from workbook stream
                            extracted = self._extract_blip_from_workbook(data)
                            images.extend(extracted)
                        except Exception as e:
                            logger.debug(f"Error reading {stream_name} stream: {e}")
                
        except Exception as e:
            logger.debug(f"OLE extraction failed: {e}")
        
        return images
    
    def _extract_images_from_biff(self, file_path: str) -> List[bytes]:
        """
        Extract images by parsing BIFF8 format records.
        
        Args:
            file_path: Path to XLS file
            
        Returns:
            List of image bytes
        """
        images = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Search for image signatures in the file
            images = self._find_images_by_signature(data)
            
        except Exception as e:
            logger.debug(f"BIFF extraction failed: {e}")
        
        return images
    
    def _extract_blip_images(self, data: bytes) -> List[bytes]:
        """
        Extract BLIP (Binary Large Image or Picture) format images.
        
        BLIP records contain image data in various formats (PNG, JPEG, etc.)
        
        Args:
            data: Raw binary data containing BLIP records
            
        Returns:
            List of extracted image bytes
        """
        images = []
        
        # Search for image signatures within the data
        images.extend(self._find_images_by_signature(data))
        
        return images
    
    def _extract_blip_from_workbook(self, data: bytes) -> List[bytes]:
        """
        Extract BLIP images from Workbook stream.
        
        XLS Workbook stream may contain MsoDrawing records with embedded images.
        
        Args:
            data: Workbook stream data
            
        Returns:
            List of extracted image bytes
        """
        images = []
        
        # MsoDrawing records (record type 0x00EC = 236) contain drawing data
        # which may include BLIP images
        offset = 0
        
        while offset < len(data) - 4:
            try:
                # BIFF8 record header: 2 bytes type + 2 bytes length
                record_type = struct.unpack('<H', data[offset:offset+2])[0]
                record_len = struct.unpack('<H', data[offset+2:offset+4])[0]
                
                # MsoDrawing record type
                if record_type == 0x00EC:  # MSODRAWING
                    record_data = data[offset+4:offset+4+record_len]
                    extracted = self._find_images_by_signature(record_data)
                    images.extend(extracted)
                
                # Move to next record
                offset += 4 + record_len
                
            except Exception:
                offset += 1
        
        return images
    
    def _find_images_by_signature(self, data: bytes) -> List[bytes]:
        """
        Find images in binary data by their file signatures.
        
        Args:
            data: Binary data to search
            
        Returns:
            List of extracted image bytes
        """
        images = []
        
        # Search for PNG images
        png_sig = b'\x89PNG\r\n\x1a\n'
        png_end = b'IEND\xaeB`\x82'
        pos = 0
        while True:
            start = data.find(png_sig, pos)
            if start == -1:
                break
            end = data.find(png_end, start)
            if end != -1:
                end += len(png_end)
                img_data = data[start:end]
                if len(img_data) > 100:  # Minimum reasonable image size
                    images.append(img_data)
            pos = start + 1
        
        # Search for JPEG images
        jpeg_start = b'\xff\xd8\xff'
        jpeg_end = b'\xff\xd9'
        pos = 0
        while True:
            start = data.find(jpeg_start, pos)
            if start == -1:
                break
            end = data.find(jpeg_end, start + 3)
            if end != -1:
                end += len(jpeg_end)
                img_data = data[start:end]
                if len(img_data) > 100:  # Minimum reasonable image size
                    images.append(img_data)
            pos = start + 1
        
        # Search for GIF images
        for gif_sig in [b'GIF87a', b'GIF89a']:
            pos = 0
            while True:
                start = data.find(gif_sig, pos)
                if start == -1:
                    break
                # GIF trailer is 0x3B (;)
                end = data.find(b'\x00;', start)
                if end != -1:
                    end += 2
                    img_data = data[start:end]
                    if len(img_data) > 100:
                        images.append(img_data)
                pos = start + 1
        
        # Search for BMP images
        bmp_sig = b'BM'
        pos = 0
        while True:
            start = data.find(bmp_sig, pos)
            if start == -1:
                break
            # BMP file size is stored at offset 2-6
            if start + 6 <= len(data):
                try:
                    file_size = struct.unpack('<I', data[start+2:start+6])[0]
                    if file_size > 100 and start + file_size <= len(data):
                        img_data = data[start:start+file_size]
                        # Validate BMP header
                        if len(img_data) > 54:  # Minimum BMP header size
                            images.append(img_data)
                except Exception:
                    pass
            pos = start + 1
        
        return images
    
    def process_all_images(
        self,
        file_path: str,
        sheet_name: Optional[str] = None,
    ) -> List[str]:
        """
        Extract and process all images from XLS file.
        
        Args:
            file_path: Path to XLS file
            sheet_name: Optional sheet name for naming
            
        Returns:
            List of image tag strings
        """
        results = []
        images = self.extract_images_from_xls(file_path)
        
        for idx, img_data in enumerate(images):
            tag = self.process_image(img_data, sheet_name=sheet_name, image_index=idx)
            if tag:
                results.append(tag)
        
        return results


__all__ = ["XLSImageProcessor"]
