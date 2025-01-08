from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from langchain_teddynote.models import MultiModal
from .state import ParseState
from .base import BaseNode
from .preprocessing import IMAGE_TYPES, TEXT_TYPES, TABLE_TYPES
from langchain_core.prompts import load_prompt


class PageElementsExtractorNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "PageElementsExtractorNode"

    def execute(self, state: ParseState) -> ParseState:
        elements = state["elements"]
        elements_by_page = dict()
        max_page = 0

        # 최대 페이지 번호 찾기
        for elem in elements:
            page_num = int(elem.page)
            max_page = max(max_page, page_num)
            if page_num not in elements_by_page:
                elements_by_page[page_num] = []
            if elem.category in (IMAGE_TYPES + TABLE_TYPES):
                elements_by_page[page_num] = []
            elements_by_page[page_num].append(elem)

        texts_by_page = dict()
        images_by_page = dict()
        tables_by_page = dict()

        # 0부터 max_page까지 모든 페이지에 대해 빈 문자열 초기화
        for page_num in range(max_page + 1):
            texts_by_page[page_num] = ""
            images_by_page[page_num] = []
            tables_by_page[page_num] = []

        # 실제 컨텐츠 채우기
        for page_num, elems in elements_by_page.items():
            print(f"Page {page_num}:")
            for elem in elems:
                if elem.category in IMAGE_TYPES:
                    images_by_page[page_num].append(elem)
                elif elem.category in TABLE_TYPES:
                    tables_by_page[page_num].append(elem)
                else:
                    texts_by_page[page_num] += elem.content

        return {
            "texts_by_page": texts_by_page,
            "images_by_page": images_by_page,
            "tables_by_page": tables_by_page,
        }


# IMAGE_ENTITY_EXTRACTOR_SYSTEM_PROMPT = """Extract key information and insights from an image based on the provided context.

# Given the context related to the image, analyze and interpret the image to generate a structured output that includes a title, key details, entities, and hypothetical questions. Ensure that your response is coherent and follows the specified format.

# # Steps

# 1. **Analyze the Context**: Understand the context provided in relation to the image. This will guide the interpretation and extraction process.
# 2. **Title Generation**: Create a concise and descriptive title that encapsulates the main theme or subject of the image.
# 3. **Details Extraction**: Identify and articulate key insights and details visible in the image.
# 4. **Entity Identification**: Recognize and list significant entities or objects present in the image.
# 5. **Hypothetical Questions**: Formulate relevant hypothetical questions that arise from the content of the image, encouraging deeper inquiry or reflection.
# """

# IMAGE_ENTITY_EXTRACTOR_USER_PROMPT = """# Output Format

# - The output should be structured using the following tags:
#   - `<image>`: Wrap the entire output.
#   - `<title>`: Enclose the generated title.
#   - `<details>`: Include detailed insights extracted from the image.
#   - `<entities>`: List the identified entities.
#   - `<hypothetical_questions>`: Present the formulated hypothetical questions.
# - Ensure all sections are filled appropriately, maintaining clarity and relevance to the context.
# - The output must be written in {language}.

# # Example

# **Input**:
# Here is the context related to the image:
# {context}

# **Output**:
# <image>
# <title>
# The Rise of Artificial Intelligence in Modern Technology
# </title>
# <details>
# The image depicts the integration of AI in various technological devices, highlighting advancements in automation and data processing.
# </details>
# <entities>
# AI algorithms, robotics, smart devices
# </entities>
# <hypothetical_questions>
# - How will AI continue to evolve in the next decade?
# - What are the ethical implications of AI in everyday life?
# </hypothetical_questions>
# </image>

# # Notes

# - Use the provided context to inform and enhance the extraction process.
# - Ensure that the hypothetical questions are thought-provoking and relevant to the image's theme.
# - Maintain clarity and coherence throughout the response.
# - Be sure to include numerical values, proper nouns, terms, and teminologies.
# """

# TABLE_ENTITY_EXTRACTOR_SYSTEM_PROMPT = """Extract key information and insights from an image based on the provided context. The image given is a cropped image of a table from a document.

# Given the context related to the image, analyze and interpret the image to generate a structured output that includes a title, key details, entities, and hypothetical questions. Ensure that your response is coherent and follows the specified format.

# # Steps

# 1. **Analyze the Context**: Understand the context provided in relation to the image. This will guide the interpretation and extraction process.
# 2. **Title Generation**: Create a concise and descriptive title that encapsulates the main theme or subject of the image.
# 3. **Details Extraction**: Identify and articulate key insights and details visible in the image. Be sure to include numerical values.
# 4. **Entity Identification**: Recognize and list significant entities or objects present in the image.
# 5. **Hypothetical Questions**: Formulate relevant hypothetical questions that arise from the content of the image, encouraging deeper inquiry or reflection.
# """

# TABLE_ENTITY_EXTRACTOR_USER_PROMPT = """# Output Format

# - The output should be structured using the following tags:
#   - `<table>`: Wrap the entire output.
#   - `<title>`: Enclose the generated title.
#   - `<details>`: Include detailed insights extracted from the image.
#   - `<entities>`: List the identified entities.
#   - `<hypothetical_questions>`: Present the formulated hypothetical questions.
# - Ensure all sections are filled appropriately, maintaining clarity and relevance to the context.
# - The output must be written in {language}.

# # Example

# **Input**:
# Here is the context related to the image:
# {context}

# **Output**:
# <table>
# <title>
# The Financial Performance of the Company
# </title>
# <details>
# The table shows the financial performance of the company over the past year.
# </details>
# <entities>
# Revenue, Employees, Profit, Expenses, Net Income
# </entities>
# <hypothetical_questions>
# - What is the total revenue of the company?
# - What is the total number of employees?
# </hypothetical_questions>
# </table>

# # Notes

# - Use the provided context to inform and enhance the extraction process.
# - Ensure that the hypothetical questions are thought-provoking and relevant to the image(table's) theme.
# - Maintain clarity and coherence throughout the response.
# - Be sure to include numerical values, proper nouns, terms, and teminologies.
# """


@chain
def image_entity_extractor(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o-mini",  # 모델명
    )

    system_prompt = load_prompt("prompts/IMAGE-SYSTEM-PROMPT.yaml", encoding="utf-8")

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        context = data_batch["context"]
        image_path = data_batch["image"]
        language = data_batch["language"]
        user_prompt_template = load_prompt(
            "prompts/IMAGE-USER-PROMPT.yaml", encoding="utf-8"
        ).format(context=context, language=language)
        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    # 이미지 파일로 부터 질의
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer


@chain
def table_entity_extractor(data_batches):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model_name="gpt-4o-mini",  # 모델명
    )

    system_prompt = load_prompt("prompts/TABLE-SYSTEM-PROMPT.yaml", encoding="utf-8")

    image_paths = []
    system_prompts = []
    user_prompts = []

    for data_batch in data_batches:
        context = data_batch["context"]
        image_path = data_batch["image"]
        language = data_batch["language"]
        user_prompt_template = load_prompt(
            "prompts/TABLE-USER-PROMPT.yaml", encoding="utf-8"
        ).format(context=context, language=language)

        image_paths.append(image_path)
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template)

    # 멀티모달 객체 생성
    multimodal_llm = MultiModal(llm)

    # 이미지 파일로 부터 질의
    answer = multimodal_llm.batch(
        image_paths, system_prompts, user_prompts, display_image=False
    )
    return answer


class ImageEntityExtractorNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ImageEntityExtractorNode"

    def execute(self, state: ParseState) -> ParseState:
        # 모든 페이지의 이미지 요소들을 하나의 리스트로 합치기
        images_files = []
        for page_images in state["images_by_page"].values():
            images_files.extend(page_images)

        BATCH_SIZE = 10
        language = state["language"]
        extracted_image_entities = []

        # 이미지를 배치 크기로 나누어 처리
        for i in range(0, len(images_files), BATCH_SIZE):
            batch = images_files[i : i + BATCH_SIZE]
            batch_data = []
            for image_element in batch:
                batch_data.append(
                    {
                        "image": image_element.image_filename,
                        "context": state["texts_by_page"][image_element.page],
                        "language": language,
                    }
                )
            # 배치 단위로 처리
            batch_result = image_entity_extractor.invoke(batch_data)
            # Element 인스턴스로 변환하고 결과 추가
            for j, result in enumerate(batch_result):
                # Element 객체의 얕은 복사 대신 새로운 객체 생성
                element = batch[j].copy()  # 원본 Element 객체 사용
                element.entity = result
                extracted_image_entities.append(element)
        return {"extracted_image_entities": extracted_image_entities}


class TableEntityExtractorNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "TableEntityExtractorNode"

    def execute(self, state: ParseState) -> ParseState:
        # 모든 페이지의 테이블 요소들을 하나의 리스트로 합치기
        tables_files = []
        for page_tables in state["tables_by_page"].values():
            tables_files.extend(page_tables)

        BATCH_SIZE = 10
        language = state["language"]
        extracted_table_entities = []

        # 이미지를 배치 크기로 나누어 처리
        for i in range(0, len(tables_files), BATCH_SIZE):
            batch = tables_files[i : i + BATCH_SIZE]
            batch_data = []
            for table_element in batch:
                batch_data.append(
                    {
                        "image": table_element.image_filename,
                        "context": state["texts_by_page"][table_element.page],
                        "language": language,
                    }
                )
            # 배치 단위로 처리
            batch_result = table_entity_extractor.invoke(batch_data)
            # Element 인스턴스로 변환하고 결과 추가
            for j, result in enumerate(batch_result):
                # Element 객체의 얕은 복사 대신 새로운 객체 생성
                element = batch[j].copy()  # 원본 Element 객체 사용
                element.entity = result
                extracted_table_entities.append(element)
        return {"extracted_table_entities": extracted_table_entities}
