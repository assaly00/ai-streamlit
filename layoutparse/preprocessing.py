from layoutparse.state import ParseState
from .element import Element
import base64
import os
from .base import BaseNode
from langchain_core.documents import Document


IMAGE_TYPES = ["figure", "chart"]
TEXT_TYPES = ["text", "equation", "caption", "paragraph", "list", "index", "heading1"]
TABLE_TYPES = ["table"]


class CreateElementsNode(BaseNode):
    def __init__(self, verbose=False, add_newline=True, **kwargs):
        super().__init__(verbose=verbose, **kwargs)
        self.add_newline = add_newline
        self.newline = "\n" if add_newline else ""

    def _save_base64_image(self, base64_str, basename, page_num, element_id, directory):
        """base64 인코딩된 이미지를 파일로 저장하는 함수"""
        img_filename = f"{basename}_Page_{page_num}_Index_{element_id}.png"
        # 원본 파일과 동일한 디렉토리에 저장
        img_filepath = os.path.join(directory, img_filename)
        img_data = base64.b64decode(base64_str)
        with open(img_filepath, "wb") as f:
            f.write(img_data)
        return img_filepath

    def execute(self, state: ParseState) -> ParseState:
        post_processed_elements = []
        directory = os.path.dirname(state["filepath"])
        base_filename = os.path.splitext(os.path.basename(state["filepath"]))[0]

        for element in state["elements_from_parser"]:
            elem = None
            if element["category"] in ["footnote", "header", "footer"]:
                continue

            if element["category"] in ["equation"]:
                # (markdown only)
                # equation
                elem = Element(
                    category=element["category"],
                    content=element["content"]["markdown"] + self.newline,
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    page=element["page"],
                    id=element["id"],
                )

            elif element["category"] in ["table"]:
                # (markdown + image crop/image save)
                # table
                image_filename = self._save_base64_image(
                    element["base64_encoding"],
                    base_filename,
                    element["page"],
                    element["id"],
                    directory,
                )
                elem = Element(
                    category=element["category"],
                    content=element["content"]["markdown"] + self.newline,
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    base64_encoding=element["base64_encoding"],
                    image_filename=image_filename,
                    page=element["page"],
                    id=element["id"],
                    coordinates=element["coordinates"],
                )

            elif element["category"] in ["figure", "chart"]:
                # (markdown + image crop/image save)
                # figure, chart
                image_filename = self._save_base64_image(
                    element["base64_encoding"],
                    base_filename,
                    element["page"],
                    element["id"],
                    directory,
                )

                elem = Element(
                    category=element["category"],
                    content=element["content"]["markdown"] + self.newline,
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    base64_encoding=element["base64_encoding"],
                    image_filename=image_filename,
                    page=element["page"],
                    id=element["id"],
                    coordinates=element["coordinates"],
                )
            elif element["category"] in ["heading1"]:
                # (text w/ heading)
                # heading1
                elem = Element(
                    category=element["category"],
                    content=f'# {element["content"]["text"]}{self.newline}',
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    page=element["page"],
                    id=element["id"],
                )
            elif element["category"] in ["caption", "paragraph", "list", "index"]:
                # (text)
                # caption, paragraph, list
                elem = Element(
                    category=element["category"],
                    content=element["content"]["text"] + self.newline,
                    html=element["content"]["html"],
                    markdown=element["content"]["markdown"],
                    page=element["page"],
                    id=element["id"],
                )

            if elem is not None:
                post_processed_elements.append(elem)

        return {"elements": post_processed_elements}


class MergeEntityNode(BaseNode):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def execute(self, state: ParseState) -> ParseState:
        elements = state["elements"]

        for elem in state["extracted_image_entities"]:
            for e in elements:
                if elem.id == e.id:
                    e.entity = elem.entity
                    break

        for elem in state["extracted_table_entities"]:
            for e in elements:
                if elem.id == e.id:
                    e.entity = elem.entity
                    break

        return {"elements": elements}


class ReconstructElementsNode(BaseNode):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def _add_src_to_markdown(self, image_filename):
        """마크다운 이미지 문법에 src 경로를 추가하는 함수"""
        abs_image_path = os.path.abspath(image_filename)
        image_md = f"![](file:///{abs_image_path})"
        return image_md

    def execute(self, state: ParseState) -> ParseState:
        elements = state["elements"]
        filepath = state["filepath"]

        # filepath가 집합인지 확인 by april
        if isinstance(filepath, set):
            filepath = next(iter(filepath))

        pages = sorted(list(state["texts_by_page"].keys()))
        max_page = pages[-1]

        reconstructed_elements = dict()
        for page_num in range(max_page + 1):
            reconstructed_elements[int(page_num)] = {
                "text": "",
                "image": [],
                "table": [],
            }

        for elem in elements:
            if elem.category in TABLE_TYPES:
                table_elem = {
                    "content": elem.content + "\n\n" + elem.entity,
                    "metadata": {
                        "table": elem.content,
                        "entity": elem.entity,
                        "page": elem.page,
                        "source": filepath,
                    },
                }
                reconstructed_elements[elem.page]["table"].append(table_elem)
            elif elem.category in IMAGE_TYPES:
                image_elem = {
                    "content": self._add_src_to_markdown(elem.image_filename)
                    + "\n\n"
                    + elem.entity,
                    "metadata": {
                        "image": self._add_src_to_markdown(elem.image_filename),
                        "entity": elem.entity,
                        "page": elem.page,
                        "source": filepath,
                    },
                }
                reconstructed_elements[elem.page]["image"].append(image_elem)
            elif elem.category in TEXT_TYPES:
                reconstructed_elements[elem.page]["text"] += elem.content

        return {"reconstructed_elements": reconstructed_elements}


class LangChainDocumentNode(BaseNode):
    def __init__(self, splitter, verbose=False):
        super().__init__(verbose)
        self.splitter = splitter

    def execute(self, state: ParseState) -> ParseState:
        reconstructed_elements = state["reconstructed_elements"]
        filepath = state["filepath"]

        # filepath가 집합인지 확인 by april
        if isinstance(filepath, set):
            filepath = next(iter(filepath))

            
        documents = []
        for page_num, page_data in reconstructed_elements.items():
            text = page_data["text"]
            split_texts = self.splitter.split_text(text)
            for split_text in split_texts:
                documents.append(
                    Document(
                        page_content=split_text,
                        metadata={"page": page_num, "source": filepath},
                    )
                )
            images = page_data["image"]
            for image in images:
                documents.append(
                    Document(page_content=image["content"], metadata=image["metadata"])
                )
            tables = page_data["table"]
            for table in tables:
                documents.append(
                    Document(page_content=table["content"], metadata=table["metadata"])
                )
                documents.append(
                    Document(
                        page_content=table["metadata"]["entity"],
                        metadata=table["metadata"],
                    )
                )

        return {"documents": documents}
