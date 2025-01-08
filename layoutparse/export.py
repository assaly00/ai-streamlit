import re
import os
import io
import base64
from PIL import Image
from io import StringIO
import pandas as pd
from bs4 import BeautifulSoup
from .base import BaseNode
from .state import ParseState

# 문서 파싱 결과를 다양한 형식(이미지, HTML, 마크다운, CSV)으로 내보내는 모듈입니다.
# 각 클래스는 특정 형식으로 변환하는 기능을 담당합니다.


class ExportImage(BaseNode):
    """문서에서 추출한 이미지를 PNG 파일로 저장하는 클래스입니다.

    base64로 인코딩된 이미지 데이터를 디코딩하여 PNG 파일로 저장합니다.
    저장된 이미지는 카테고리별로 분류되어 저장됩니다.
    """

    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def save_to_png(self, base64_encoding, dirname, basename, category, page, index):
        # base64 디코딩
        image_data = base64.b64decode(base64_encoding)

        # 바이트 데이터를 이미지로 변환
        image = Image.open(io.BytesIO(image_data))

        # dirname 내에 images 폴더와 하위 카테고리 폴더 생성
        image_dir = os.path.join(dirname, "images", category)
        os.makedirs(image_dir, exist_ok=True)

        # basename_prefix를 사용하여 이미지 파일명 생성
        base_prefix = os.path.splitext(basename)[0]
        image_filename = (
            f"{base_prefix.upper()}_{category.upper()}_Page_{page+1}_Index_{index}.png"
        )
        image_path = os.path.join(image_dir, image_filename)
        abs_image_path = os.path.abspath(image_path)

        # 이미지 저장
        image.save(abs_image_path)
        return abs_image_path

    def execute(self, state: ParseState):
        # 경로
        filepath = state["filepath"]

        # filepath가 집합인지 확인 by april
        if isinstance(filepath, set):
            filepath = next(iter(filepath))


        dirname = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        for elem in state["elements_from_parser"]:
            if elem["category"] in ["figure", "chart", "table"]:
                # base64 인코딩이 있는지 확인
                base64_encoding = elem.get("base64_encoding")
                image_path = self.save_to_png(
                    base64_encoding,
                    dirname,
                    basename,
                    elem["category"],
                    elem["page"],
                    elem["id"],
                )
                elem["png_filepath"] = image_path

        return {"elements_from_parser": state["elements_from_parser"]}


class ExportHTML(BaseNode):
    """문서 내용을 HTML 형식으로 변환하여 저장하는 클래스입니다.

    이미지가 포함된 경우 base64 인코딩을 통해 HTML 내에 직접 삽입합니다.
    텍스트의 줄바꿈 처리를 선택적으로 할 수 있습니다.
    """

    def __init__(self, ignore_new_line_in_text=False, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.ignore_new_line_in_text = ignore_new_line_in_text

    def _add_base64_src_to_html(self, html, base64_encoding):
        """HTML 태그에 src 속성을 추가하는 함수"""
        if not base64_encoding:
            return html

        # base64 인코딩된 이미지 데이터를 src에 직접 추가
        pattern = r"<img([^>]*)>"
        replacement = f'<img\\1 src="data:image/png;base64,{base64_encoding}">'
        return re.sub(pattern, replacement, html)

    def execute(self, state: ParseState):
        # 원본 파일의 전체 경로를 유지하면서 확장자만 .html로 변경
        filepath = state["filepath"]

        # filepath가 집합인지 확인 by april
        if isinstance(filepath, set):
            filepath = next(iter(filepath))


        dirname = os.path.dirname(filepath)
        basename = os.path.basename(filepath)
        html_basename = os.path.splitext(basename)[0] + ".html"
        html_filepath = os.path.join(dirname, html_basename)

        # category: table, figure, chart, heading1, header, footer, caption, paragraph, equation, list, index, footnote
        # https://console.upstage.ai/docs/capabilities/document-parse

        # full_markdown 내용을 파일로 저장
        with open(html_filepath, "w", encoding="utf-8") as f:
            for elem in state["elements_from_parser"]:
                # 주석 처리된 요소는 제외
                if elem["category"] in ["header", "footer", "footnote"]:
                    continue

                if elem["category"] in ["figure", "chart", "table"]:
                    # base64 인코딩이 있는지 확인
                    base64_encoding = elem.get("base64_encoding")

                    # HTML에 src 속성 추가
                    modified_html = self._add_base64_src_to_html(
                        elem["content"]["html"], base64_encoding
                    )
                    f.write(modified_html)
                else:
                    if self.ignore_new_line_in_text:
                        f.write(elem["content"]["html"].replace("<br>", " "))
                    else:
                        f.write(elem["content"]["html"])

        self.log(f"HTML 파일이 성공적으로 생성되었습니다: {html_filepath}")

        return {"export": [html_filepath]}


class ExportMarkdown(BaseNode):
    """문서 내용을 마크다운 형식으로 변환하여 저장하는 클래스입니다.

    이미지는 로컬 파일 경로를 참조하는 방식으로 저장됩니다.
    테이블은 마크다운 테이블 문법으로 변환됩니다.
    텍스트의 줄바꿈 처리를 선택적으로 할 수 있습니다.
    """

    def __init__(
        self,
        ignore_new_line_in_text=False,
        show_image=True,
        verbose=False,
        **kwargs,
    ):
        super().__init__(verbose=verbose, **kwargs)
        self.ignore_new_line_in_text = ignore_new_line_in_text
        self.show_image = show_image
        self.separator = "\n\n"

    def _add_src_to_markdown(self, png_filepath):
        if not png_filepath:
            return ""
        return f"![]({png_filepath})"

    def execute(self, state: ParseState):
        # 원본 파일의 전체 경로를 유지하면서 확장자만 .md로 변경
        filepath = state["filepath"]

        # filepath가 집합인지 확인 by april
        if isinstance(filepath, set):
            filepath = next(iter(filepath))

        dirname = os.path.abspath(os.path.dirname(filepath))
        basename = os.path.basename(filepath)
        md_basename = os.path.splitext(basename)[0] + ".md"
        md_filepath = os.path.join(dirname, md_basename)

        # full_markdown 내용을 파일로 저장
        with open(md_filepath, "w", encoding="utf-8") as f:
            for elem in state["elements_from_parser"]:
                # 주석 처리된 요소는 제외
                if elem["category"] in ["header", "footer", "footnote"]:
                    continue

                if elem["category"] in ["figure", "chart"]:
                    # png_filepath가 있는지 확인
                    if self.show_image:
                        png_filepath = elem.get("png_filepath")
                        modified_md = self._add_src_to_markdown(png_filepath)
                        f.write(modified_md + self.separator)

                elif elem["category"] in ["table"]:
                    # png_filepath가 있는지 확인
                    if self.show_image:
                        png_filepath = elem.get("png_filepath")
                        modified_md = self._add_src_to_markdown(png_filepath)
                        f.write(modified_md + self.separator)
                    # markdown 형식의 테이블 추가
                    f.write(elem["content"]["markdown"] + self.separator)

                elif elem["category"] in ["paragraph"]:
                    if self.ignore_new_line_in_text:
                        f.write(
                            elem["content"]["markdown"].replace("\n", " ")
                            + self.separator
                        )
                    else:
                        f.write(elem["content"]["markdown"] + self.separator)
                else:
                    f.write(elem["content"]["markdown"] + self.separator)

        self.log(f"마크다운 파일이 성공적으로 생성되었습니다: {md_filepath}")

        return {"export": [md_filepath]}


class ExportTableCSV(BaseNode):
    """문서에서 추출한 테이블을 CSV 형식으로 저장하는 클래스입니다.

    HTML 형식의 테이블을 파싱하여 CSV 파일로 변환합니다.
    각 테이블은 개별 CSV 파일로 저장되며, 파일명에는 페이지 번호와 인덱스가 포함됩니다.
    """

    def __init__(self, verbose=False, **kwargs):
        super().__init__(verbose=verbose, **kwargs)

    def execute(self, state: ParseState):
        # 원본 파일의 전체 경로를 유지하면서 확장자만 .csv로 변경
        filepath = state["filepath"]

        # filepath가 집합인지 확인 by april
        if isinstance(filepath, set):
            filepath = next(iter(filepath))

        dirname = os.path.dirname(filepath)
        # dirname 내에 tables 폴더와 하위 카테고리 폴더 생성
        table_dir = os.path.join(dirname, "tables")
        os.makedirs(table_dir, exist_ok=True)

        basename = os.path.basename(filepath)
        base_without_ext = os.path.splitext(basename)[0]

        csv_filepaths = []

        # 테이블 데이터 추출 및 변환
        for elem in state["elements_from_parser"]:
            if elem["category"] == "table":
                # BeautifulSoup으로 HTML 파싱
                soup = BeautifulSoup(elem["content"]["html"], "html.parser")

                # 불규칙한 문자 정리
                for td in soup.find_all("td"):
                    td.string = (
                        td.get_text(strip=True).replace("\\t", " ").replace("\t", " ")
                    )

                # 정리된 HTML을 문자열로 변환
                cleaned_html = str(soup)
                cleaned_html_io = StringIO(cleaned_html)

                # pandas로 테이블 파싱
                try:
                    parsed_tables = pd.read_html(cleaned_html_io)
                    for table in parsed_tables:
                        # 각 테이블마다 개별 CSV 파일 생성 (페이지 번호와 인덱스 번호 포함)
                        csv_filename = f"{base_without_ext.upper()}_TABLE_Page_{elem['page']}_Index_{elem['id']}.csv"
                        csv_filepath = os.path.join(table_dir, csv_filename)
                        absolute_path = os.path.abspath(csv_filepath)

                        # CSV 파일로 저장
                        table.to_csv(absolute_path, index=False, encoding="utf-8-sig")
                        csv_filepaths.append(absolute_path)
                        elem["csv_filepath"] = absolute_path
                        self.log(
                            f"CSV 파일이 성공적으로 생성되었습니다: {absolute_path}"
                        )
                except Exception as e:
                    self.log(f"테이블 파싱 중 오류 발생: {str(e)}")
                    continue

        if csv_filepaths:
            return {
                "elements_from_parser": state["elements_from_parser"],
                "export": csv_filepaths,
            }
        else:
            self.log("변환할 테이블이 없습니다.")
            return {"elements_from_parser": state["elements_from_parser"], "export": []}
