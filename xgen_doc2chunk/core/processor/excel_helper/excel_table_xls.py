"""
Excel XLS 테이블 변환 모듈

XLS 시트를 Markdown 또는 HTML 테이블로 변환합니다.
병합셀이 있으면 HTML, 없으면 Markdown을 사용합니다.
layout_detect_range를 통해 실제 데이터 영역만 추출합니다.
object_detect를 통해 개별 객체(테이블)별로 청킹할 수 있습니다.
"""

import logging
from typing import Optional, List, Dict, Tuple
import xlrd

from xgen_doc2chunk.core.processor.excel_helper.excel_layout_detector import layout_detect_range_xls, object_detect_xls, LayoutRange
# 표 문자열 유효성 검사는 XLSX 모듈과 동일한 규칙을 사용한다
from xgen_doc2chunk.core.processor.excel_helper.excel_table_xlsx import _table_has_data

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
        layout = layout_detect_range_xls(sheet, wb)
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
            layout = layout_detect_range_xls(sheet, wb)
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
            layout = layout_detect_range_xls(sheet, wb)
            if layout is None:
                return ""

        # 병합된 셀 정보 수집 (layout 영역 내만)
        # xlrd에서 merged_cells는 (rlo, rhi, clo, chi) 튜플 리스트
        # rlo, clo는 0-based, rhi, chi는 exclusive
        merged_cells_info = {}  # (row, col) -> (rowspan, colspan), 0-based
        merged_partial_info = {}  # layout 외부에서 시작하는 병합셀의 layout 내부 앵커 → (actual_rowspan, actual_colspan)
        merged_value_override = {}  # layout 외부 앵커 값을 layout 내부 가상 앵커에 저장
        skip_cells = set()  # 건너뛸 셀 (병합된 영역의 일부), 0-based

        for (rlo, rhi, clo, chi) in sheet.merged_cells:
            # layout 영역과 겹치는 병합 셀만 처리 (1-based로 변환하여 비교)
            mr_min_row = rlo + 1
            mr_max_row = rhi  # exclusive in 0-based → 1-based max row
            mr_min_col = clo + 1
            mr_max_col = chi  # exclusive in 0-based → 1-based max col
            
            if (mr_min_row <= layout.max_row and
                mr_max_row >= layout.min_row and
                mr_min_col <= layout.max_col and
                mr_max_col >= layout.min_col):
                
                # 병합 시작점이 layout 내부인지 확인 (0-based 비교)
                start_in_layout = (rlo >= layout.min_row - 1 and clo >= layout.min_col - 1)

                if start_in_layout:
                    rowspan = rhi - rlo
                    colspan = chi - clo
                    merged_cells_info[(rlo, clo)] = (rowspan, colspan)

                    # 병합된 영역의 나머지 셀들은 건너뛰기
                    for r in range(rlo, rhi):
                        for c in range(clo, chi):
                            if r != rlo or c != clo:
                                skip_cells.add((r, c))
                else:
                    # 병합 시작점이 layout 외부인 경우: 가상 앵커로 처리
                    first_row_in_layout = max(rlo, layout.min_row - 1)
                    first_col_in_layout = max(clo, layout.min_col - 1)

                    actual_rowspan = min(rhi, layout.max_row) - first_row_in_layout
                    actual_colspan = min(chi, layout.max_col) - first_col_in_layout

                    merged_partial_info[(first_row_in_layout, first_col_in_layout)] = (actual_rowspan, actual_colspan)

                    # 원래 앵커(layout 외부)의 값을 읽어서 가상 앵커에 저장
                    try:
                        orig_value = sheet.cell_value(rlo, clo)
                        if orig_value:
                            orig_type = sheet.cell_type(rlo, clo)
                            merged_value_override[(first_row_in_layout, first_col_in_layout)] = _format_xls_cell_value(orig_value, orig_type, wb)
                    except Exception:
                        pass

                    # layout 내부의 병합 영역에서 가상 앵커를 제외하고 건너뛰기
                    for r in range(first_row_in_layout, min(rhi, layout.max_row)):
                        for c in range(first_col_in_layout, min(chi, layout.max_col)):
                            if r != first_row_in_layout or c != first_col_in_layout:
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

                # layout 외부에서 시작하는 병합셀의 가상 앵커: 원본 앵커의 값 사용
                if (row_idx, col_idx) in merged_value_override:
                    cell_value = merged_value_override[(row_idx, col_idx)]
                    if cell_value:
                        has_data = True

                # HTML 이스케이프
                cell_value = _escape_html(cell_value)

                # 모든 셀을 td로 처리 (헤더 구분 없음)
                tag = "td"

                # 병합 속성
                attrs = []
                if (row_idx, col_idx) in merged_cells_info:
                    rowspan, colspan = merged_cells_info[(row_idx, col_idx)]
                    if rowspan > 1:
                        attrs.append(f"rowspan='{rowspan}'")
                    if colspan > 1:
                        attrs.append(f"colspan='{colspan}'")
                elif (row_idx, col_idx) in merged_partial_info:
                    rowspan, colspan = merged_partial_info[(row_idx, col_idx)]
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


def _build_merged_value_override_xls(sheet, wb, layout: LayoutRange) -> Dict[Tuple[int, int], str]:
    """
    병합 셀의 시작점이 layout 밖에 있는 경우, layout 내 첫 번째 셀에 표시할 값을 계산합니다.

    layout 을 개별 객체 단위로 잘라서 변환할 때, 객체 경계를 가로지르는 병합 셀의
    값이 누락되는 것을 막기 위해 사용합니다.

    Args:
        sheet: xlrd Sheet 객체
        wb: xlrd Workbook 객체
        layout: 변환할 레이아웃 범위 (1-based)

    Returns:
        {(row, col): value} 매핑 (0-based 좌표)
    """
    merged_value_override: Dict[Tuple[int, int], str] = {}

    try:
        merged_ranges = sheet.merged_cells
    except Exception:
        return merged_value_override

    for (rlo, rhi, clo, chi) in merged_ranges:
        # xlrd: rlo/clo 는 0-based inclusive, rhi/chi 는 exclusive
        # layout 은 1-based 이므로 비교를 위해 1-based 로 변환
        mr_min_row, mr_max_row = rlo + 1, rhi
        mr_min_col, mr_max_col = clo + 1, chi

        if not (mr_min_row <= layout.max_row and
                mr_max_row >= layout.min_row and
                mr_min_col <= layout.max_col and
                mr_max_col >= layout.min_col):
            continue

        # 병합 시작점이 layout 안에 있으면 일반 경로에서 값을 읽으므로 override 불필요
        start_in_layout = (rlo >= layout.min_row - 1 and clo >= layout.min_col - 1)
        if start_in_layout:
            continue

        first_row_in_layout = max(rlo, layout.min_row - 1)
        first_col_in_layout = max(clo, layout.min_col - 1)

        try:
            orig_value = sheet.cell_value(rlo, clo)
            if orig_value:
                orig_type = sheet.cell_type(rlo, clo)
                merged_value_override[(first_row_in_layout, first_col_in_layout)] = \
                    _format_xls_cell_value(orig_value, orig_type, wb)
        except Exception:
            pass

    return merged_value_override


def convert_xls_object_to_text(sheet, wb, layout: LayoutRange) -> str:
    """
    표로 보기 어려운 작은 객체(1x1, 1xN, Nx1)를 일반 텍스트로 변환합니다.

    표 문법(파이프/구분선)을 붙이지 않고 셀 값만 남깁니다.
    한 행에 값이 여러 개면 " | " 로 이어 붙이고, 행 사이는 줄바꿈으로 구분합니다.

    Args:
        sheet: xlrd Sheet 객체
        wb: xlrd Workbook 객체
        layout: 변환할 객체 범위 (1-based)

    Returns:
        텍스트 문자열 (값이 없으면 빈 문자열)
    """
    try:
        merged_value_override = _build_merged_value_override_xls(sheet, wb, layout)

        lines: List[str] = []
        # 1-based layout 을 0-based 로 변환하여 사용
        for row_idx in range(layout.min_row - 1, layout.max_row):
            values: List[str] = []
            for col_idx in range(layout.min_col - 1, layout.max_col):
                cell_value = ""
                if (row_idx, col_idx) in merged_value_override:
                    cell_value = merged_value_override[(row_idx, col_idx)]
                else:
                    try:
                        value = sheet.cell_value(row_idx, col_idx)
                        if value:
                            cell_type = sheet.cell_type(row_idx, col_idx)
                            cell_value = _format_xls_cell_value(value, cell_type, wb)
                    except Exception:
                        cell_value = ""

                cell_value = cell_value.strip().replace("\n", " ") if cell_value else ""
                if cell_value:
                    values.append(cell_value)

            if values:
                lines.append(" | ".join(values))

        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Error converting XLS object to text: {e}")
        return ""


def convert_xls_objects_to_blocks(
    sheet,
    wb,
    layout: Optional[LayoutRange] = None
) -> List[Tuple[str, str]]:
    """
    XLS 시트의 개별 객체를 표/텍스트로 분류하여 변환합니다.

    object_detect_xls 로 감지한 각 객체에 대해 최소 크기(기본 2x2)를 검사하여,
    기준을 만족하면 표('table')로, 그렇지 않으면 일반 텍스트('text')로 변환합니다.
    DOCX/HWPX/PDF 가 사용하는 최소 표 크기 기준과 동일한 정책입니다.

    작은 블록을 버리지 않고 텍스트로 흘려보내므로 값 손실은 없습니다.

    Args:
        sheet: xlrd Sheet 객체
        wb: xlrd Workbook 객체
        layout: 탐색할 레이아웃 범위 (None이면 자동 감지)

    Returns:
        [(종류, 내용), ...] 목록. 종류는 'table' 또는 'text'
        (위→아래, 왼쪽→오른쪽 순서)
    """
    objects = object_detect_xls(sheet, wb, layout)

    if not objects:
        return []

    blocks: List[Tuple[str, str]] = []
    for obj_layout in objects:
        if obj_layout.is_table_like():
            table_str = convert_xls_sheet_to_table(sheet, wb, obj_layout)
            if _table_has_data(table_str):
                blocks.append(('table', table_str))
        else:
            text_str = convert_xls_object_to_text(sheet, wb, obj_layout)
            if text_str and text_str.strip():
                blocks.append(('text', text_str))

    table_count = sum(1 for kind, _ in blocks if kind == 'table')
    logger.debug(
        f"Converted {len(objects)} objects to {table_count} tables and "
        f"{len(blocks) - table_count} text blocks (XLS)"
    )
    return blocks


def convert_xls_objects_to_tables(sheet, wb, layout: Optional[LayoutRange] = None) -> List[str]:
    """
    XLS 시트에서 개별 객체(테이블)를 감지하고 각각을 테이블 문자열로 변환합니다.

    NOTE: 최소 크기 검사 없이 모든 객체를 표로 변환합니다.
          표/텍스트를 구분하려면 convert_xls_objects_to_blocks 를 사용하세요.

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
    objects = object_detect_xls(sheet, wb, layout)
    
    if not objects:
        return []
    
    tables = []
    for obj_layout in objects:
        table_str = convert_xls_sheet_to_table(sheet, wb, obj_layout)
        # 빈 테이블 필터링 (구분선만 있고 값이 없는 경우 제외)
        if _table_has_data(table_str):
            tables.append(table_str)

    logger.debug(f"Converted {len(tables)} objects to tables (XLS)")
    return tables
