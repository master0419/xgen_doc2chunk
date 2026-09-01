# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.5] - 2026-09-01

### Fixed
- **Excel**: A single cell (or a single-row / single-column block) was detected as a
  table, so one stray value occupied an entire chunk. Blocks smaller than 2x2 are now
  emitted as plain text, matching the minimum table size DOCX/HWPX/PDF already enforce.
  No values are dropped - a small block keeps its cell values as text.
- **Chunking**: `chunk_multi_sheet_content()` emitted one chunk per segment regardless
  of size, so `chunk_size` acted only as an upper bound and never as a fill target.
  Segments that fit are now packed into a shared chunk. Occupancy is measured on
  segment bodies, because the context prefix (metadata + sheet marker) is written once
  per chunk rather than once per segment. Oversized segments keep their previous
  handling, and the buffer is always flushed at a sheet boundary so segments from
  different sheets are never merged.
- **Chunking**: Long plain-text segments in multi-sheet content exceeded `chunk_size`
  by the context prefix length - the full `chunk_size` was passed to the text splitter
  before the prefix was prepended to every resulting chunk.
- **Chunking**: `extract_content_segments()` raised `NameError` when called with the
  default image pattern (`IMAGE_TAG_PATTERN` was referenced but never imported).

### Added
- **Excel**: `convert_xlsx_objects_to_blocks()` / `convert_xls_objects_to_blocks()`
  classify each detected object as `table` or `text`.
- **Excel**: `convert_xlsx_object_to_text()` / `convert_xls_object_to_text()` render a
  block smaller than the minimum table size as plain text.
- **Excel**: `LayoutRange.is_table_like()` plus `MIN_TABLE_ROWS` / `MIN_TABLE_COLS`
  constants in `excel_layout_detector`.

### Compatibility
- `convert_xlsx_objects_to_tables()` / `convert_xls_objects_to_tables()` keep their
  previous behavior (every detected object becomes a table) and remain exported.
- Non-Excel formats (PDF/DOCX/DOC/HWP/HWPX/PPTX) are unaffected; extraction and
  chunking output is byte-identical to 0.3.4.

## [0.3.4] - 2026-08-09

### Fixed
- **Chunking**: Rowspan blocks larger than the chunk limit produced a single
  oversized chunk that exceeded embedding model input limits. With
  `force_chunking` enabled, such blocks are now split at row boundaries
  (rows are never cut) and the spanning merged cells are re-issued into each
  continuation chunk with adjusted rowspan, so every chunk remains a
  self-contained, structurally valid table.
- **Chunking**: `force_chunking` was silently ignored for table-based files
  (CSV/TSV/XLSX/XLS); it is now propagated to the table chunking path.
  Default behavior (`force_chunking=False`) is unchanged.

### Added
- **Chunking**: `SpanningCell` dataclass and `extract_spanning_cells()` helper
  for rowspan-aware cell extraction with content.

## [0.2.26] - 2026-04-03

### Added
- **HWPX**: Extract text from shapes and improve section processing
- **HWPX**: Improve header/footer handling

### Changed
- Bump version to 0.2.26

## [0.2.25] - 2026-03-28

### Added
- **DOCX**: Clean field codes and improve run element extraction
- **DOC/DOCX**: Add header and footer extraction
- **Excel**: HTML content detection in Excel cell processing

### Changed
- **PDF**: Refine text extraction logic to exclude lines within table bounding boxes

## [0.2.24] - 2026-03-20

### Added
- **Chunking**: Implement small chunk merging to prevent table-title isolation
- **Chunking**: Allow backward merging when blocked by page boundaries

## [0.2.23] - 2026-03-15

### Improved
- **Excel**: Enhance merged cell handling in XLS and XLSX HTML conversion

## [0.2.22] - 2026-03-10

### Changed
- Update version retrieval mechanism in `__init__.py`

## [0.2.21] - 2026-03-05

### Changed
- Minor internal improvements and stabilization

## [0.2.20] - 2026-02-28

### Improved
- **Excel**: Enhance XLSX and XLS layout detection to consider cells with borders as valid

## [0.2.18] - 2026-02-22

### Changed
- **Excel**: Update HTML conversion to treat all cells as data cells without header distinction

## [0.2.17] - 2026-02-18

### Changed
- **Excel**: Remove textbox and image segment extraction from `sheet_processor` to prevent each image/textbox from occupying a separate chunk

### Added
- **Excel**: XLS and XLSX textbox extraction support
- **Excel**: Separate XLS and XLSX image handler refactoring

## [0.2.14] - 2026-02-12

### Fixed
- **Chunking**: Enhance `clean_chunks` to merge page-marker-only chunks with next chunk (solves skipped page numbers)

## [0.2.13] - 2026-02-08

### Added
- **Chunking**: Support for nested tables (tables within tables within tables) in protected region detection

## [0.2.12] - 2026-02-05

### Fixed
- **PDF**: Adjust Y gap threshold for table merging in `TableDetectionEngine` to prevent merging of separate tables

## [0.2.11] - 2026-02-02

### Changed
- **PDF**: Refactor import statements in `pdf_table_detection.py`

## [0.2.1] - 2026-01-30

### Fixed
- **PDF**: Enhance text extraction logic to handle table region extraction duplication problem

## [0.2.0] - 2026-01-28

### Changed
- Improve file extension handling in `DocumentProcessor`
- Major version bump: stabilization of core API

## [0.1.5x] - 2026-01-24 ~ 2026-01-27

### Added
- **PDF**: CJK compatibility handling and fragmented text reconstruction
- **Excel**: Table processing with context extraction and improved chunking logic (respects `chunk_size`)
- **Chunking**: Enhanced chunking logic for handling chunk size constraints

### Fixed
- **PDF**: Table quality validation criteria adjustment for paragraph text detection

## [0.1.4] - 2026-01-22

### Changed
- Refactor: Adjust validation criteria for paragraph text detection in `TableQualityValidator`
- Improve comments and documentation across processors (Korean → English)

## [0.1.2] - 2026-01-20

### Added
- **BedrockOCR**: AWS Bedrock Vision model support for OCR processing
  - Supports Claude 3.5 Sonnet and other Bedrock vision models
  - Full AWS credential configuration (access key, secret key, session token, region)
  - Configurable timeouts and retry settings
- **ImageFileHandler**: New handler for standalone image files (jpg, png, gif, bmp, webp)
  - Automatically uses OCR engine when available
  - Returns image tag format when OCR is not configured for later processing
- **PageTagProcessor**: Centralized page/slide/sheet tag processing system
  - Unified tag generation across all document handlers
  - Configurable tag prefixes and suffixes
- **Image pattern support for OCR**: Custom image tag patterns now passed to OCR engine
  - `ImageProcessor.get_pattern_string()` method for regex pattern generation
  - `BaseOCR.set_image_pattern()` and `set_image_pattern_from_string()` methods
  - OCR engines now recognize custom image tag formats

### Changed
- **DocumentProcessor**: OCR engine setter now invalidates handler registry for proper refresh
- **Handler registry**: ImageFileHandler automatically registered with OCR engine support
- **QUICKSTART.md**: Complete rewrite with comprehensive documentation
  - 3-stage processing pipeline documentation (File → Text → OCR → Chunks)
  - Detailed OCR configuration guide for all 5 engines
  - Tag customization examples (image, page, slide, sheet)
  - Complete API reference with all parameters

### Improved
- All Korean comments and docstrings in `img_processor.py` converted to English
- Enhanced OCR integration with custom pattern matching support
- Better separation of concerns with PageTagProcessor

## [0.1.0] - 2026-01-19

### Added
- Initial release of xgen_doc2chunk
- Multi-format document support (PDF, DOCX, DOC, XLSX, XLS, PPTX, PPT, HWP, HWPX)
- Intelligent text extraction with structure preservation
- Table detection and extraction with HTML formatting
- OCR integration (OpenAI, Anthropic, Google Gemini, vLLM)
- Smart chunking with semantic awareness
- Metadata extraction
- Support for 20+ code file formats
- Korean document support (HWP, HWPX)

### Features
- `DocumentProcessor` class for easy document processing
- Configurable chunk size and overlap
- Protected regions for code blocks
- Pluggable OCR engine architecture
- Automatic encoding detection for text files
- Chart and image extraction from Office documents

[0.2.26]: https://github.com/master0419/doc2chunk/compare/v0.2.25...v0.2.26
[0.2.25]: https://github.com/master0419/doc2chunk/compare/v0.2.24...v0.2.25
[0.2.24]: https://github.com/master0419/doc2chunk/compare/v0.2.23...v0.2.24
[0.2.23]: https://github.com/master0419/doc2chunk/compare/v0.2.22...v0.2.23
[0.2.22]: https://github.com/master0419/doc2chunk/compare/v0.2.21...v0.2.22
[0.2.21]: https://github.com/master0419/doc2chunk/compare/v0.2.20...v0.2.21
[0.2.20]: https://github.com/master0419/doc2chunk/compare/v0.2.18...v0.2.20
[0.2.18]: https://github.com/master0419/doc2chunk/compare/v0.2.17...v0.2.18
[0.2.17]: https://github.com/master0419/doc2chunk/compare/v0.2.14...v0.2.17
[0.2.14]: https://github.com/master0419/doc2chunk/compare/v0.2.13...v0.2.14
[0.2.13]: https://github.com/master0419/doc2chunk/compare/v0.2.12...v0.2.13
[0.2.12]: https://github.com/master0419/doc2chunk/compare/v0.2.11...v0.2.12
[0.2.11]: https://github.com/master0419/doc2chunk/compare/v0.2.1...v0.2.11
[0.2.1]: https://github.com/master0419/doc2chunk/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/master0419/doc2chunk/compare/v0.1.5...v0.2.0
[0.1.5x]: https://github.com/master0419/doc2chunk/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/master0419/doc2chunk/compare/v0.1.2...v0.1.4
[0.1.2]: https://github.com/master0419/doc2chunk/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/master0419/doc2chunk/releases/tag/v0.1.0
