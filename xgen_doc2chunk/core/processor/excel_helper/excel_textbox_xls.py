"""
XLS (BIFF8) 텍스트박스 및 도형 텍스트 추출 모듈

XLS 파일의 BIFF8 형식에서 TXO(Text Object) 레코드로부터 텍스트를 추출합니다.
텍스트박스, 도형(직사각형, 원, 삼각형 등) 내의 텍스트가 TXO 레코드에 저장됩니다.
"""

import struct
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# BIFF8 레코드 타입 상수
RECORD_BOF = 0x0809
RECORD_BOUNDSHEET = 0x0085
RECORD_TXO = 0x01B6           # Text Object
RECORD_CONTINUE = 0x003C      # Continue record (TXO의 텍스트 데이터)
RECORD_MSODRAWING = 0x00EC
RECORD_OBJ = 0x005D


def extract_textboxes_from_xls(file_path: str, sheet_names: List[str]) -> Dict[str, List[str]]:
    """
    XLS 파일에서 텍스트박스/도형 내의 텍스트를 추출합니다.

    XLS의 텍스트박스/도형 텍스트는 TXO(Text Object) 레코드에 저장됩니다.
    TXO 레코드 직후 CONTINUE 레코드에 실제 텍스트 데이터가 있습니다.

    Args:
        file_path: XLS 파일 경로
        sheet_names: 시트 이름 목록 (인덱스 순서)

    Returns:
        {시트명: [텍스트 리스트]} 형태의 딕셔너리
    """
    textboxes_by_sheet: Dict[str, List[str]] = {}

    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        # BIFF8 레코드 스트림 찾기 (OLE compound document)
        workbook_stream = _find_workbook_stream(data)
        if not workbook_stream:
            logger.debug("Could not find Workbook stream in XLS file")
            return textboxes_by_sheet

        # 시트 경계 파싱
        sheet_boundaries = _parse_sheet_boundaries_for_textbox(workbook_stream, sheet_names)

        # TXO 레코드에서 텍스트 추출
        textboxes_by_sheet = _extract_txo_texts(workbook_stream, sheet_boundaries, sheet_names)

        total_textboxes = sum(len(tb) for tb in textboxes_by_sheet.values())
        if total_textboxes > 0:
            logger.info(f"Total extracted {total_textboxes} textboxes/shapes from XLS")

    except Exception as e:
        logger.warning(f"Error extracting textboxes from XLS: {e}")
        import traceback
        logger.debug(traceback.format_exc())

    return textboxes_by_sheet


def _find_workbook_stream(data: bytes) -> Optional[bytes]:
    """
    OLE compound document에서 Workbook 스트림을 찾습니다.

    Args:
        data: 파일 바이너리 데이터

    Returns:
        Workbook 스트림 바이트 (없으면 None)
    """
    try:
        import olefile
        if not olefile.isOleFile(data):
            return None

        from io import BytesIO
        ole = olefile.OleFileIO(BytesIO(data))

        # Workbook 또는 Book 스트림 찾기
        stream_name = None
        if ole.exists('Workbook'):
            stream_name = 'Workbook'
        elif ole.exists('Book'):
            stream_name = 'Book'
        else:
            ole.close()
            return None

        workbook_stream = ole.openstream(stream_name).read()
        ole.close()
        return workbook_stream

    except Exception as e:
        logger.debug(f"Error finding workbook stream: {e}")
        return None


def _parse_sheet_boundaries_for_textbox(
    stream: bytes,
    sheet_names: List[str]
) -> List[Tuple[str, int, int]]:
    """
    Workbook 스트림에서 시트 경계를 파싱합니다.

    Args:
        stream: Workbook 스트림 바이트
        sheet_names: 시트 이름 목록

    Returns:
        [(시트명, 시작오프셋, 끝오프셋)] 리스트
    """
    boundaries: List[Tuple[str, int, int]] = []
    sheet_offsets: List[Tuple[str, int]] = []

    offset = 0
    stream_len = len(stream)

    while offset + 4 <= stream_len:
        rec_type = struct.unpack_from('<H', stream, offset)[0]
        rec_len = struct.unpack_from('<H', stream, offset + 2)[0]

        if rec_type == RECORD_BOUNDSHEET:
            # BOUNDSHEET 레코드에서 시트 시작 오프셋 추출
            if rec_len >= 4:
                sheet_bof_offset = struct.unpack_from('<I', stream, offset + 4)[0]
                # 시트 이름 추출
                if rec_len >= 8:
                    name_len = stream[offset + 8] if offset + 8 < stream_len else 0
                    encoding_flag = stream[offset + 9] if offset + 9 < stream_len else 0
                    name_start = offset + 10
                    if encoding_flag == 0:  # Compressed (8-bit)
                        name = stream[name_start:name_start + name_len].decode('latin-1', errors='ignore')
                    else:  # UTF-16LE
                        name = stream[name_start:name_start + name_len * 2].decode('utf-16-le', errors='ignore')
                    sheet_offsets.append((name, sheet_bof_offset))

        offset += 4 + rec_len

    # 경계 계산 (각 시트의 끝 = 다음 시트의 시작)
    for i, (name, start) in enumerate(sheet_offsets):
        if i + 1 < len(sheet_offsets):
            end = sheet_offsets[i + 1][1]
        else:
            end = stream_len
        boundaries.append((name, start, end))

    return boundaries


def _extract_txo_texts(
    stream: bytes,
    sheet_boundaries: List[Tuple[str, int, int]],
    sheet_names: List[str]
) -> Dict[str, List[str]]:
    """
    TXO 레코드에서 텍스트를 추출합니다.

    TXO 레코드 구조:
    - Offset 0-1: grbit (옵션)
    - Offset 2-3: rot (회전)
    - Offset 4-9: reserved
    - Offset 10-11: cchText (텍스트 길이)
    - Offset 12-13: cbRuns (서식 실행 크기)

    TXO 직후 CONTINUE 레코드에 실제 텍스트가 있습니다.

    Args:
        stream: Workbook 스트림 바이트
        sheet_boundaries: 시트 경계 정보
        sheet_names: 시트 이름 목록

    Returns:
        {시트명: [텍스트 리스트]} 딕셔너리
    """
    textboxes_by_sheet: Dict[str, List[str]] = {}
    for name in sheet_names:
        textboxes_by_sheet[name] = []

    offset = 0
    stream_len = len(stream)
    pending_txo: Optional[Tuple[int, str]] = None  # (expected_text_len, sheet_name)

    while offset + 4 <= stream_len:
        rec_type = struct.unpack_from('<H', stream, offset)[0]
        rec_len = struct.unpack_from('<H', stream, offset + 2)[0]
        rec_data_start = offset + 4
        rec_data_end = offset + 4 + rec_len

        # 현재 오프셋이 어느 시트에 속하는지 확인
        current_sheet = _get_sheet_for_offset(offset, sheet_boundaries, sheet_names)

        if rec_type == RECORD_TXO:
            # TXO 레코드 파싱
            if rec_len >= 14:
                cchText = struct.unpack_from('<H', stream, rec_data_start + 10)[0]
                if cchText > 0:
                    # 다음 CONTINUE 레코드에서 텍스트를 기대
                    pending_txo = (cchText, current_sheet)
                    logger.debug(f"Found TXO at offset {offset}, expecting {cchText} chars for sheet '{current_sheet}'")

        elif rec_type == RECORD_CONTINUE and pending_txo is not None:
            # CONTINUE 레코드에서 텍스트 추출
            expected_len, sheet_name = pending_txo
            pending_txo = None

            if rec_len > 0:
                text = _decode_txo_continue_text(stream[rec_data_start:rec_data_end], expected_len)
                if text and text.strip():
                    # 시트 이름이 유효한 경우에만 추가
                    if sheet_name and sheet_name in textboxes_by_sheet:
                        textboxes_by_sheet[sheet_name].append(text.strip())
                        logger.debug(f"Extracted text from TXO: '{text[:50]}...' -> sheet '{sheet_name}'")
                    elif sheet_names:
                        # 첫 번째 시트에 귀속
                        textboxes_by_sheet[sheet_names[0]].append(text.strip())
                        logger.debug(f"Extracted text from TXO: '{text[:50]}...' -> sheet '{sheet_names[0]}' (default)")

        offset += 4 + rec_len

    return textboxes_by_sheet


def _get_sheet_for_offset(
    offset: int,
    sheet_boundaries: List[Tuple[str, int, int]],
    sheet_names: List[str]
) -> str:
    """
    주어진 오프셋이 어느 시트에 속하는지 반환합니다.

    Args:
        offset: 스트림 내 오프셋
        sheet_boundaries: 시트 경계 정보
        sheet_names: 시트 이름 목록

    Returns:
        시트 이름 (찾지 못하면 첫 번째 시트)
    """
    for name, start, end in sheet_boundaries:
        if start <= offset < end:
            return name

    # 경계를 찾지 못하면 첫 번째 시트에 귀속
    if sheet_names:
        return sheet_names[0]
    return "Sheet1"


def _decode_txo_continue_text(data: bytes, expected_len: int) -> Optional[str]:
    """
    TXO CONTINUE 레코드의 텍스트를 디코딩합니다.

    CONTINUE 레코드 구조:
    - Offset 0: fHighByte (0 = 압축 8비트, 1 = UTF-16LE)
    - Offset 1+: 텍스트 데이터

    Args:
        data: CONTINUE 레코드 데이터
        expected_len: 예상 텍스트 길이 (문자 수)

    Returns:
        디코딩된 텍스트 (없으면 None)
    """
    if len(data) < 1:
        return None

    try:
        fHighByte = data[0]
        text_data = data[1:]

        if fHighByte == 0:
            # 압축된 8비트 문자 (Latin-1 또는 CP1252)
            text = text_data[:expected_len].decode('cp1252', errors='ignore')
        else:
            # UTF-16LE
            byte_len = min(expected_len * 2, len(text_data))
            text = text_data[:byte_len].decode('utf-16-le', errors='ignore')

        return text

    except Exception as e:
        logger.debug(f"Error decoding TXO text: {e}")
        return None
