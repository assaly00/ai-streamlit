import requests
import json
import os
import time
from .base import BaseNode
from .state import ParseState

DEFAULT_CONFIG = {
    "ocr": True,
    "coordinates": True,
    "output_formats": "['html', 'text', 'markdown']",
    "model": "document-parse",
    "base64_encoding": "['figure', 'chart', 'table']",
}


class DocumentParseNode(BaseNode):
    def __init__(self, api_key, use_ocr=True, verbose=False, **kwargs):
        """
        DocumentParse 클래스의 생성자

        :param api_key: Upstage API 인증을 위한 API 키
        :param config: API 요청에 사용할 설정값. None인 경우 기본 설정 사용
        """
        super().__init__(verbose=verbose, **kwargs)
        self.api_key = api_key
        self.config = DEFAULT_CONFIG
        if use_ocr:
            self.config["ocr"] = True

    # test를 위해서 아래 코드를 주석처리합니다. by april
    def _upstage_layout_analysis(self, input_file):
        """
        Upstage의 Document Parse API를 호출하여 문서 분석을 수행합니다.

        :param input_file: 분석할 PDF 파일의 경로
        :return: 분석 결과가 저장된 JSON 파일의 경로
        """
        
        output_file = os.path.splitext(input_file)[0] + ".json"
        if os.path.exists(output_file):
            print("기존파일을 사용합니다.")
            return output_file
        

        # API 요청 헤더 설정
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # 분석할 PDF 파일 열기
        files = {"document": open(input_file, "rb")}

        # API 요청 보내기
        response = requests.post(
            "https://api.upstage.ai/v1/document-ai/document-parse",
            headers=headers,
            data=self.config,
            files=files,
        )

        # API 응답 처리 및 결과 저장
        if response.status_code == 200:
            # 분석 결과를 저장할 JSON 파일 경로 생성
            output_file = os.path.splitext(input_file)[0] + ".json"

            print(f"API 요청 성공. 결과를 {output_file}에 저장합니다.")
            #print(response.json())

            # 분석 결과를 JSON 파일로 저장
            # 중국어 인코딩 문제로 수정 by april
            #with open(output_file, "w") as f:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, ensure_ascii=False)
                

            return output_file
        else:
            # API 요청이 실패한 경우 예외 발생
            raise ValueError(f"API 요청 실패. 상태 코드: {response.status_code}")
        
    # 아래는 테스트용
    # def _upstage_layout_analysis(self, input_file):
        
    #     output_file = os.path.splitext(input_file)[0] + ".json"
    #     print("기존파일을 사용합니다.")

    #     return output_file
       

    def parse_start_end_page(self, filepath):
        # 파일명에서 페이지 번호 추출 (예: WorldEnergyOutlook2024_0040_0049.pdf)
        filename = os.path.basename(filepath)
        # .pdf 확장자 제거
        name_without_ext = filename.rsplit(".", 1)[0]

        # 파일명 형식 검증
        try:
            # 파일명이 최소 9자 이상이어야 함
            if len(name_without_ext) < 9:
                return (-1, -1)

            # 마지막 9자리 추출 (예: 0040_0049)
            page_numbers = name_without_ext[-9:]

            # 형식이 ####_#### 인지 검증 (숫자4개_숫자4개)
            if not (
                page_numbers[4] == "_"
                and page_numbers[:4].isdigit()
                and page_numbers[5:].isdigit()
            ):
                return (-1, -1)

            # 시작 페이지와 끝 페이지 추출
            start_page = int(page_numbers[:4])
            end_page = int(page_numbers[5:])

            # 시작 페이지가 끝 페이지보다 크면 검증 실패
            if start_page > end_page:
                return (-1, -1)

            return (start_page, end_page)

        except (IndexError, ValueError):
            return (-1, -1)

    def execute(self, state: ParseState):
        """
        주어진 입력 파일에 대해 문서 분석을 실행합니다.

        :param input_file: 분석할 PDF 파일의 경로
        :return: 분석 결과가 저장된 JSON 파일의 경로
        """
        start_time = time.time()
        self.log(f"Start Parsing: {state['working_filepath']}")

        filepath = state["working_filepath"]
        parsed_json = self._upstage_layout_analysis(filepath)

        # 파일명에서 시작 페이지 추출
        start_page, _ = self.parse_start_end_page(filepath)
        page_offset = start_page - 1 if start_page != -1 else 0

        # 중국어 인코딩 문제로 수정 by april, encoding='utf-8' 추가
        with open(parsed_json, "r", encoding='utf-8') as f:
            data = json.load(f)

            # 페이지 번호와 ID 재계산
            for element in data["elements"]:
                element["page"] += page_offset

        metadata = {
            "api": data.pop("api"),
            "model": data.pop("model"),
            "usage": data.pop("usage"),
        }

        duration = time.time() - start_time
        self.log(f"Finished Parsing in {duration:.2f} seconds")

        return {"metadata": [metadata], "raw_elements": [data["elements"]]}


class PostDocumentParseNode(BaseNode):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def execute(self, state: ParseState):
        elements_list = state["raw_elements"]
        id_counter = 0  # ID를 순차적으로 부여하기 위한 카운터
        post_processed_elements = []

        for elements in elements_list:
            for element in elements:
                elem = element.copy()
                # ID 순차적으로 부여
                elem["id"] = id_counter
                id_counter += 1

                post_processed_elements.append(elem)

        self.log(f"Total Post-processed Elements: {id_counter}")

        pages_count = 0
        metadata = state["metadata"]

        for meta in metadata:
            for k, v in meta.items():
                if k == "usage":
                    pages_count += int(v["pages"])

        total_cost = pages_count * 0.01

        self.log(f"Total Cost: ${total_cost:.2f}")

        # 재정렬된 elements를 state에 업데이트
        return {
            "elements_from_parser": post_processed_elements,
            "total_cost": total_cost,
        }


class WorkingQueueNode(BaseNode):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def execute(self, state: ParseState):
        working_filepath = state.get("working_filepath", None)
        # working_filepath가 없거나 비어있는 경우 첫번째 파일 선택
        if (
            "working_filepath" not in state
            or state["working_filepath"] is None
            or state["working_filepath"] == ""
        ):
            if len(state["split_filepaths"]) > 0:
                working_filepath = state["split_filepaths"][0]
            else:
                working_filepath = "<<FINISHED>>"
        else:
            if working_filepath == "<<FINISHED>>":
                return {"working_filepath": "<<FINISHED>>"}

            # 현재 작업중인 파일의 인덱스 찾기
            current_index = state["split_filepaths"].index(working_filepath)
            # 다음 파일이 있으면 다음 파일을 선택, 없으면 FINISHED 표시
            if current_index + 1 < len(state["split_filepaths"]):
                working_filepath = state["split_filepaths"][current_index + 1]
            else:
                working_filepath = "<<FINISHED>>"
        return {"working_filepath": working_filepath}


def continue_parse(state: ParseState):
    if state["working_filepath"] == "<<FINISHED>>":
        return False
    else:
        return True
