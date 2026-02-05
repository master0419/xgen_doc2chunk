"""
Excel XLS 테이블 변환 모듈

XLS 시트를 Markdown 또는 HTML 테이블로 변환합니다.
병합셀이 있으면 HTML, 없으면 Markdown을 사용합니다.
layout_detect_range를 통해 실제 데이터 영역만 추출합니다.
object_detect를 통해 개별 객체(테이블)별로 청킹할 수 있습니다.
"""

import logging
from typing import Optional, List, Tuple, Set
import xlrd

from xgen_doc2chunk.core.processor.excel_helper.excel_layout_detector import (
    layout_detect_range_xls, object_detect_xls, LayoutRange, _has_border_xls
)

logger = logging.getLogger("document-processor")


def has_merged_cells_xls(sheet, layout: Optional[LayoutRange] = None) -> bool:
    """
    XLS 시트에 병합셀이 존재하는지 확인합니다.
    layout이 주어지면 해당 영역 내의 병합셀만 확인합니다.

    Args:
        sheet: xlrd Sheet 객체
        layout: 검사할 레이아웃 범위 (None이면 전체 시트)

    Returns:
        병합셀이 존재하면 True
    """
    try:
        if len(sheet.merged_cells) == 0:
            return False
        
        # layout이 없으면 전체 시트에 병합셀 존재 여부만 확인
        if layout is None:
            return True
        
        # layout 영역 내에 병합셀이 있는지 확인
        # xlrd merged_cells는 (rlo, rhi, clo, chi) 튜플, 0-based
        for (rlo, rhi, clo, chi) in sheet.merged_cells:
            # 1-based로 변환하여 비교
            mr_min_row = rlo + 1
            mr_max_row = rhi  # rhi는 exclusive이므로 +1 불필요
            mr_min_col = clo + 1
            mr_max_col = chi  # chi는 exclusive이므로 +1 불필요
            
            if (mr_min_row <= layout.max_row and
                mr_max_row >= layout.min_row and
                mr_min_col <= layout.max_col and
                mr_max_col >= layout.min_col):
                return True
        
        return False
    except Exception:
        return False


def convert_xls_sheet_to_table(sheet, wb, layout: Optional[LayoutRange] = None) -> str:
    """
    XLS 시트를 테이블로 변환합니다.
    병합셀이 없으면 Markdown, 있으면 HTML로 변환합니다.
    layout이 None이면 자동으로 감지합니다.

    Args:
        sheet: xlrd Sheet 객체
        wb: xlrd Workbook 객체
        layout: 변환할 레이아웃 범위 (None이면 자동 감지)

    Returns:
        변환된 테이블 문자열
    """
    # layout이 없으면 자동 감지
    if layout is None:
        layout = layout_detect_range_xls(sheet)
        if layout is None:
            logger.debug("No data found in XLS sheet")
            return ""
    
    if has_merged_cells_xls(sheet, layout):
        logger.debug("Merged cells detected in XLS, using HTML format")
        return convert_xls_sheet_to_html(sheet, wb, layout)
    else:
        logger.debug("No merged cells in XLS, using Markdown format")
        return convert_xls_sheet_to_markdown(sheet, wb, layout)


def convert_xls_sheet_to_markdown(sheet, wb, layout: Optional[LayoutRange] = None) -> str:
    """
    XLS 시트를 Markdown 테이블로 변환합니다.
    layout_detect_range를 통해 실제 데이터 영역만 추출합니다.

    Args:
        sheet: xlrd Sheet 객체
        wb: xlrd Workbook 객체
        layout: 변환할 레이아웃 범위 (None이면 자동 감지)

    Returns:
        Markdown 테이블 문자열
    """
    try:
        # layout이 없으면 자동 감지
        if layout is None:
            layout = layout_detect_range_xls(sheet)
            if layout is None:
                return ""

        md_parts = []
        row_count = 0

        # 1-based layout을 0-based로 변환하여 사용
        for row_idx in range(layout.min_row - 1, layout.max_row):  # 0-based
            cells = []
            row_has_content = False

            for col_idx in range(layout.min_col - 1, layout.max_col):  # 0-based
                cell_value = ""
                try:
                    value = sheet.cell_value(row_idx, col_idx)
                    if value:
                        cell_type = sheet.cell_type(row_idx, col_idx)
                        cell_value = _format_xls_cell_value(value, cell_type, wb)

                        if cell_value:
                            row_has_content = True
                except Exception:
                    pass

                # Markdown 테이블에서 파이프는 이스케이프 필요
                cell_value = cell_value.replace("|", "\\|")
                cell_value = cell_value.replace("\n", " ")
                cells.append(cell_value)

            if not row_has_content:
                continue

            row_str = "| " + " | ".join(cells) + " |"
            md_parts.append(row_str)
            row_count += 1

            # 첫 번째 데이터 행 다음에 구분선 추가
            if row_count == 1:
                separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                md_parts.append(separator)

        return "\n".join(md_parts) if md_parts else ""

    except Exception as e:
        logger.warning(f"Error converting XLS sheet to Markdown: {e}")
        return ""


def convert_xls_sheet_to_html(sheet, wb, layout: Optional[LayoutRange] = None) -> str:
    """
    XLS 시트를 HTML 테이블로 변환합니다.
    병합셀(rowspan/colspan)을 지원합니다.
    layout_detect_range를 통해 실제 데이터 영역만 추출합니다.
    
    병합셀이 있는 경우 빈 행도 테이블 구조의 일부이므로 포함합니다.

    Args:
        sheet: xlrd Sheet 객체
        wb: xlrd Workbook 객체
        layout: 변환할 레이아웃 범위 (None이면 자동 감지)

    Returns:
        HTML 테이블 문자열
    """
    try:
        # layout이 없으면 자동 감지
        if layout is None:
            layout = layout_detect_range_xls(sheet)
            if layout is None:
                return ""

        # 병합된 셀 정보 수집 (layout 영역 내만)
        # xlrd에서 merged_cells는 (rlo, rhi, clo, chi) 튜플 리스트
        # rlo, clo는 0-based, rhi, chi는 exclusive
        merged_cells_info = {}  # (row, col) -> (rowspan, colspan), 0-based
        skip_cells = set()  # 건너뛸 셀 (병합된 영역의 일부), 0-based

        for (rlo, rhi, clo, chi) in sheet.merged_cells:
            # layout 영역과 겹치는 병합 셀만 처리 (1-based로 변환하여 비교)
            mr_min_row = rlo + 1
            mr_max_row = rhi  # exclusive
            mr_min_col = clo + 1
            mr_max_col = chi  # exclusive
            
            if (mr_min_row <= layout.max_row and
                mr_max_row >= layout.min_row and
                mr_min_col <= layout.max_col and
                mr_max_col >= layout.min_col):
                
                rowspan = rhi - rlo
                colspan = chi - clo

                merged_cells_info[(rlo, clo)] = (rowspan, colspan)

                # 병합된 영역의 나머지 셀들은 건너뛰기
                for r in range(rlo, rhi):
                    for c in range(clo, chi):
                        if r != rlo or c != clo:
                            skip_cells.add((r, c))

        # HTML 생성
        html_parts = ["<table border='1'>"]
        has_data = False

        # 1-based layout을 0-based로 변환하여 사용
        for row_idx in range(layout.min_row - 1, layout.max_row):  # 0-based
            row_parts = ["<tr>"]

            for col_idx in range(layout.min_col - 1, layout.max_col):  # 0-based
                # 건너뛸 셀 확인 (병합된 영역의 일부)
                if (row_idx, col_idx) in skip_cells:
                    continue

                cell_value = ""
                try:
                    value = sheet.cell_value(row_idx, col_idx)
                    if value:
                        cell_type = sheet.cell_type(row_idx, col_idx)
                        cell_value = _format_xls_cell_value(value, cell_type, wb)

                        if cell_value:
                            has_data = True
                except Exception:
                    pass

                # HTML 이스케이프
                cell_value = _escape_html(cell_value)

                # 첫 번째 행은 헤더로 처리
                tag = "th" if row_idx == layout.min_row - 1 else "td"

                # 병합 속성
                attrs = []
                if (row_idx, col_idx) in merged_cells_info:
                    rowspan, colspan = merged_cells_info[(row_idx, col_idx)]
                    if rowspan > 1:
                        attrs.append(f"rowspan='{rowspan}'")
                    if colspan > 1:
                        attrs.append(f"colspan='{colspan}'")

                attr_str = " " + " ".join(attrs) if attrs else ""
                row_parts.append(f"<{tag}{attr_str}>{cell_value}</{tag}>")

            row_parts.append("</tr>")
            
            # 모든 행을 추가 (빈 행도 테이블 구조의 일부)
            html_parts.append("".join(row_parts))

        html_parts.append("</table>")

        if has_data:
            return "\n".join(html_parts)
        return ""

    except Exception as e:
        logger.warning(f"Error converting XLS sheet to HTML: {e}")
        return ""


def _format_xls_cell_value(value, cell_type, wb) -> str:
    """
    XLS 셀 값을 문자열로 포맷합니다.
    
    Args:
        value: 셀 값
        cell_type: xlrd 셀 타입
        wb: xlrd Workbook 객체
        
    Returns:
        포맷된 문자열
    """
    try:
        if cell_type == xlrd.XL_CELL_NUMBER:
            if value == int(value):
                return str(int(value))
            else:
                return str(value)
        elif cell_type == xlrd.XL_CELL_DATE:
            try:
                date_tuple = xlrd.xldate_as_tuple(value, wb.datemode)
                return f"{date_tuple[0]:04d}-{date_tuple[1]:02d}-{date_tuple[2]:02d}"
            except Exception:
                return str(value)
        else:
            return str(value).strip()
    except Exception:
        return str(value).strip() if value else ""


def _escape_html(text: str) -> str:
    """
    HTML 특수 문자를 이스케이프합니다.
    
    Args:
        text: 원본 텍스트
        
    Returns:
        이스케이프된 텍스트
    """
    if not text:
        return ""
    
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("\n", "<br>")
    
    return text


def convert_xls_objects_to_tables(sheet, wb, layout: Optional[LayoutRange] = None) -> List[str]:
    """
    XLS 시트에서 개별 객체(테이블)를 감지하고 각각을 테이블 문자열로 변환합니다.
    
    알고리즘:
    1. 테두리가 있는 영역을 먼저 개별 개체로 인식
    2. 테두리가 없는 값 영역을 감지
    3. 완전히 인접한 개체들을 병합
    4. 각 객체를 테이블로 변환
    
    Args:
        sheet: xlrd Sheet 객체
        wb: xlrd Workbook 객체
        layout: 탐색할 레이아웃 범위 (None이면 자동 감지)
    
    Returns:
        개별 객체 테이블 문자열 목록 (위→아래, 왼쪽→오른쪽 순서)
    """
    # Get overall layout
    if layout is None:
        layout = layout_detect_range_xls(sheet)
        if layout is None:
            return []
    
    # Detect bordered regions only (tables)
    table_regions = object_detect_xls(sheet, wb, layout)
    
    if not table_regions:
        # No bordered tables found - extract all values as plain text
        plain_text = _extract_all_values_as_text_xls(sheet, layout)
        return [plain_text] if plain_text else []
    
    results = []
    processed_cells: Set[Tuple[int, int]] = set()
    
    for table_region in table_regions:
        parts = []
        
        # 1. Extract context text above the table (title)
        title_text = _extract_context_above_xls(sheet, wb, table_region, layout, processed_cells)
        if title_text:
            parts.append(title_text)
        
        # 2. Convert bordered region to table
        table_str = convert_xls_sheet_to_table(sheet, wb, table_region)
        if table_str and table_str.strip():
            # Validate table has actual data
            if _has_table_data_xls(table_str):
                parts.append(table_str)
                # Mark table cells as processed
                _mark_region_processed_xls(table_region, processed_cells)
        
        # 3. Extract context text below the table (notes)
        notes_text = _extract_context_below_xls(sheet, wb, table_region, layout, processed_cells)
        if notes_text:
            parts.append(notes_text)
        
        if parts:
            results.append("\n\n".join(parts))
    
    # 4. Extract remaining value cells as plain text (not part of any table)
    remaining_text = _extract_remaining_values_as_text_xls(sheet, layout, processed_cells)
    if remaining_text:
        results.append(remaining_text)
    
    logger.debug(f"Converted {len(results)} objects to tables with context (XLS)")
    return results


def _extract_context_above_xls(
    sheet, wb, table_region: LayoutRange, layout: LayoutRange, 
    processed_cells: Set[Tuple[int, int]]
) -> Optional[str]:
    """
    Extract text from cells directly above the table (within 5 rows).
    Used for table titles. Only non-bordered cells are included.
    """
    context_lines = []
    
    # Search up to 5 rows above the table
    for row_offset in range(1, 6):
        row_idx = table_region.min_row - row_offset
        if row_idx < layout.min_row:
            break
        
        row_text = []
        has_value = False
        
        for col_idx in range(table_region.min_col, table_region.max_col + 1):
            if (row_idx, col_idx) in processed_cells:
                continue
            
            try:
                # XLS uses 0-based index
                value = sheet.cell_value(row_idx - 1, col_idx - 1)
                
                # Only include if cell has NO border
                if value and str(value).strip() and not _has_border_xls(sheet, wb, row_idx - 1, col_idx - 1):
                    row_text.append(str(value).strip())
                    processed_cells.add((row_idx, col_idx))
                    has_value = True
            except Exception:
                pass
        
        if has_value and row_text:
            context_lines.insert(0, " ".join(row_text))
        elif not has_value:
            # Stop if empty row found
            break
    
    return "\n".join(context_lines) if context_lines else None


def _extract_context_below_xls(
    sheet, wb, table_region: LayoutRange, layout: LayoutRange,
    processed_cells: Set[Tuple[int, int]]
) -> Optional[str]:
    """
    Extract text from cells directly below the table (within 5 rows).
    Used for table notes/footnotes. Only non-bordered cells are included.
    """
    context_lines = []
    
    # Search up to 5 rows below the table
    for row_offset in range(1, 6):
        row_idx = table_region.max_row + row_offset
        if row_idx > layout.max_row:
            break
        
        row_text = []
        has_value = False
        
        for col_idx in range(table_region.min_col, table_region.max_col + 1):
            if (row_idx, col_idx) in processed_cells:
                continue
            
            try:
                # XLS uses 0-based index
                value = sheet.cell_value(row_idx - 1, col_idx - 1)
                
                # Only include if cell has NO border
                if value and str(value).strip() and not _has_border_xls(sheet, wb, row_idx - 1, col_idx - 1):
                    row_text.append(str(value).strip())
                    processed_cells.add((row_idx, col_idx))
                    has_value = True
            except Exception:
                pass
        
        if has_value and row_text:
            context_lines.append(" ".join(row_text))
        elif not has_value:
            # Stop if empty row found
            break
    
    return "\n".join(context_lines) if context_lines else None


def _extract_all_values_as_text_xls(sheet, layout: LayoutRange) -> Optional[str]:
    """
    Extract all value cells as plain text when no bordered tables exist.
    """
    lines = []
    
    for row_idx in range(layout.min_row, layout.max_row + 1):
        row_values = []
        for col_idx in range(layout.min_col, layout.max_col + 1):
            try:
                # XLS uses 0-based index
                value = sheet.cell_value(row_idx - 1, col_idx - 1)
                if value and str(value).strip():
                    row_values.append(str(value).strip())
            except Exception:
                pass
        
        if row_values:
            lines.append(" | ".join(row_values))
    
    return "\n".join(lines) if lines else None


def _extract_remaining_values_as_text_xls(
    sheet, layout: LayoutRange, processed_cells: Set[Tuple[int, int]]
) -> Optional[str]:
    """
    Extract all remaining value cells as plain text.
    These are standalone values not associated with any bordered table.
    """
    lines = []
    
    for row_idx in range(layout.min_row, layout.max_row + 1):
        row_values = []
        for col_idx in range(layout.min_col, layout.max_col + 1):
            if (row_idx, col_idx) in processed_cells:
                continue
            
            try:
                # XLS uses 0-based index
                value = sheet.cell_value(row_idx - 1, col_idx - 1)
                if value and str(value).strip():
                    row_values.append(str(value).strip())
                    processed_cells.add((row_idx, col_idx))
            except Exception:
                pass
        
        if row_values:
            lines.append(" | ".join(row_values))
    
    return "\n".join(lines) if lines else None


def _mark_region_processed_xls(region: LayoutRange, processed_cells: Set[Tuple[int, int]]) -> None:
    """Mark all cells in a region as processed."""
    for row_idx in range(region.min_row, region.max_row + 1):
        for col_idx in range(region.min_col, region.max_col + 1):
            processed_cells.add((row_idx, col_idx))


def _has_table_data_xls(table_str: str) -> bool:
    """Check if table string has actual data (not just structure)."""
    lines = [line.strip() for line in table_str.strip().split('\n') if line.strip()]
    for line in lines:
        # Skip separator lines
        if '---' not in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if parts:
                return True
    return False
