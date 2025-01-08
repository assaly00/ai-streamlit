import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging
from dotenv import load_dotenv
import os
# PDF 파싱과 관련된 필수 모듈들을 임포트합니다
from layoutparse.teddynote_parser import (
    create_upstage_parser_graph,  # Upstage 파서 그래프를 생성하는 함수
    create_export_graph,  # 결과를 내보내는 그래프를 생성하는 함수
)

# 그래프 관련 기본 모듈들을 임포트합니다
from langgraph.graph import StateGraph, END  # 상태 그래프와 종료 상태를 위한 모듈
from layoutparse.state import ParseState  # 파싱 상태를 관리하는 클래스
from langchain_teddynote.graphs import visualize_graph  # 그래프 시각화를 위한 모듈
from langgraph.checkpoint.memory import MemorySaver  # 체크포인트 저장을 위한 모듈
# UUID 모듈을 가져와서 고유 식별자를 생성할 수 있게 합니다
import uuid
# Langchain의 실행 설정을 위한 RunnableConfig를 가져옵니다
from langchain_core.runnables import RunnableConfig
# 그래프 실행 결과를 스트리밍하기 위한 유틸리티 함수를 가져옵니다
#from langchain_teddynote.messages import stream_graph
from langgraph.graph.state import CompiledStateGraph
from typing import Any, Dict, List, Callable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] Upstage PDF Parser")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    print("Create .cache directory")
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    print("Create .cache/files directory")
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.set_page_config(page_title="LLM Playground 💬", page_icon="💬")
st.title("AI OCR - Upstage LLM 기반 💬")


########################################################################
# LangGraph 전체 워크플로우 구성
########################################################################
# Upstage 파서 그래프를 생성합니다 - 배치 크기 30으로 설정하고 상세 로그를 출력합니다
upstage_parser_graph = create_upstage_parser_graph(
    batch_size=30, test_page=None, verbose=True
)
# 결과를 마크다운으로 내보내는 그래프를 생성합니다
export_graph = create_export_graph(show_image_in_markdown=True)

# 전체 워크플로우를 담을 부모 그래프를 생성합니다
parent_workflow = StateGraph(ParseState)

# Upstage 파서 노드를 워크플로우에 추가합니다
parent_workflow.add_node("upstage_parser", upstage_parser_graph)
# 결과 내보내기 노드를 워크플로우에 추가합니다
parent_workflow.add_node("export_output", export_graph)

# 파서에서 내보내기로 이어지는 엣지를 추가합니다
parent_workflow.add_edge("upstage_parser", "export_output")

# 워크플로우의 시작점을 Upstage 파서로 설정합니다
parent_workflow.set_entry_point("upstage_parser")

# 메모리 체크포인터를 사용하여 그래프를 컴파일합니다
parent_graph = parent_workflow.compile(checkpointer=MemorySaver())

# 프롬프트 템플릿을 정의합니다
prompt_content = ""
html_content = ""
markdown_content = ""
ocrjson_content = ""

########################################################################


# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None


# 탭을 생성
html_tab, markdown_tab, convjson_tab, message_tab = st.tabs(["HTML", "Markdown", "변환JSON", "대화내용"])

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")
    tab1, tab2 = st.tabs(["LLM OCR", "JSON 변환"])

    ## LLM OCR 탭
    # 파일 업로드
    tab1.uploaded_file = tab1.file_uploader("파일 업로드", type=["pdf"])

    ## JSON 변환 탭
    # 모델 선택 메뉴
    tab2.selected_model = tab2.selectbox(
        "LLM 선택", ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    ) 
 
    user_selected_prompt = tab2.selectbox("JSON변환 프롬프트 선택", ["invoice-latest", "chinese-test"])
    translate_btn = tab2.button("JSON 변환", key="apply")
        

# 이전 대화를 출력
def print_messages():
    for message_tab.chat_message in st.session_state["messages"]:
        message_tab.chat_message(message_tab.chat_message.role).write(message_tab.chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

def stream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable = None,
):
    """
    LangGraph의 실행 결과를 스트리밍하여 출력하는 함수입니다.

    Args:
        graph (CompiledStateGraph): 실행할 컴파일된 LangGraph 객체
        inputs (dict): 그래프에 전달할 입력값 딕셔너리
        config (RunnableConfig): 실행 설정
        node_names (List[str], optional): 출력할 노드 이름 목록. 기본값은 빈 리스트
        callback (Callable, optional): 각 청크 처리를 위한 콜백 함수. 기본값은 None
            콜백 함수는 {"node": str, "content": str} 형태의 딕셔너리를 인자로 받습니다.

    Returns:
        None: 함수는 스트리밍 결과를 출력만 하고 반환값은 없습니다.
    """
    print("\n🚀 LangGraph 실행을 시작합니다 🚀")
    
    prev_node = ""
    for chunk_msg, metadata in graph.stream(inputs, config, stream_mode="messages"):
        curr_node = metadata["langgraph_node"]

        # node_names가 비어있거나 현재 노드가 node_names에 있는 경우에만 처리
        if not node_names or curr_node in node_names:
            # 콜백 함수가 있는 경우 실행
            if callback:
                callback({"node": curr_node, "content": chunk_msg.content})
            # 콜백이 없는 경우 기본 출력
            else:
                # 노드가 변경된 경우에만 구분선 출력
                if curr_node != prev_node:
                    # add_message("assistant", "\n" + "=" * 50)
                    # add_message("assistant", f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
                    # add_message("assistant", "- " * 25)
                    print("\n" + "=" * 50)
                    print(f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
                    print("- " * 25)
                print(chunk_msg.content, end="", flush=True)
                #add_message("assistant", chunk_msg.content, end="", flush=True)

            prev_node = curr_node

# 전역 변수 선언
config = None
@st.cache_resource(show_spinner="업로드한 파일을 분석중입니다...")
def call_upstage_parser(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.    
    file_content = file.read()
    # 현재 스크립트의 절대 경로 얻기
    current_directory = os.path.abspath("")
    # 파일 경로 생성
    file_path = os.path.join(current_directory, ".cache", "files", file.name)
    #file_path = f"./.cache/files/{file.name}"
    #file_path = f"D:/vs-workspace/langchain-kr/19-Streamlit/01-MyProject/.cache/files/{file.name}"    
    with open(file_path, "wb") as f:
        f.write(file_content)
    

    print(f"filepath type: {type(file_path)}, value: {file_path}")

    # 그래프 실행을 위한 상세 설정을 구성합니다
    config = RunnableConfig(
        # 재귀 호출의 최대 깊이를 300으로 제한하여 무한 루프를 방지합니다
        recursion_limit=300,
        # 실행마다 고유한 스레드 ID를 생성하여 독립적인 실행을 보장합니다
        configurable={"thread_id": str(uuid.uuid4())},
    )

    # 분석할 PDF 파일의 경로를 입력값으로 설정합니다
    inputs = {
        # PDF 파일의 실제 경로를 지정합니다        
        "filepath": {file_path},
    }

    print(inputs)

    # 설정된 그래프를 스트리밍 방식으로 실행하고 결과를 실시간으로 확인합니다
    stream_graph(parent_graph, inputs, config=config)    
    return


# 체인 생성
def create_chain(model_name="gpt-4o"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        #{"context": markdown_content, "question": RunnablePassthrough()}
        {"context": StrOutputParser().pipe(lambda x: markdown_content), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# JSON변환 체인 생성
@st.cache_resource(show_spinner="지정한 JSON포맷으로 변환중입니다...")
def exec_trans_chain(_prompt, model_name="gpt-4o"):
    # JSON 출력 파서 초기화
    parser = JsonOutputParser()
    # ChatGPT 모델을 초기화하고 구조화된 출력을 설정합니다
    llm = ChatOpenAI(model=model_name, temperature=0)
    # 프롬프트와 LLM을 체인으로 연결합니다
    chain = _prompt | llm | parser
    # 지시사항을 프롬프트에 주입합니다.
    _prompt = _prompt.partial(format_instructions=parser.get_format_instructions())        
    # 체인을 호출하여 쿼리 실행
    response = chain.invoke({"ocr_data": markdown_content, "format_instructions": "Return a JSON object"})

    with convjson_tab:
        convjson_tab.json(response)

    # 새로운 파일이 업로드 되었으므로 채팅내용 
    # 삭제
    st.session_state["messages"] = []
    return response


# 파일이 업로드 되었을 때
if tab1.uploaded_file:
    # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정...)
    #retriever = embed_file(uploaded_file)
    call_upstage_parser(tab1.uploaded_file)
    #chain = create_chain(retriever, model_name=selected_model)
    #st.session_state["chain"] = chain

    filename = os.path.abspath("") + "/.cache/files/" + os.path.splitext(tab1.uploaded_file.name)[0]    
    # ocrjson_file_path = filename + "_0000_0000.json"
    # #ocrjson_file_path = f"D:/vs-workspace/langchain-kr/19-Streamlit/01-MyProject/.cache/files/Terminal Invoice Samples10_0000_0000.json"            
    # with open(ocrjson_file_path, "r", encoding="utf-8") as file:
    #     ocrjson_content = file.read()
    #     with ocrjson_tab:
    #         st.json(ocrjson_content)

    #html_file_path = f"D:/vs-workspace/langchain-kr/19-Streamlit/01-MyProject/.cache/files/Terminal Invoice Samples10.html"
    html_file_path = filename + ".html"
    with open(html_file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
        with html_tab:
            html_tab.write(html_content, unsafe_allow_html=True)

    #markdown_file_path = f"D:/vs-workspace/langchain-kr/19-Streamlit/01-MyProject/.cache/files/Terminal Invoice Samples10.md"            
    markdown_file_path = filename + ".md"
    with open(markdown_file_path, "r", encoding="utf-8") as file:
        markdown_content = file.read()
        with markdown_tab:
            markdown_tab.markdown(markdown_content)

    # 파일 업로드 후 chain 생성
    chain = create_chain(model_name=tab2.selected_model)
    st.session_state["chain"] = chain                
    tab1.markdown(f"✅ OCR 추출이 완료되었습니다.")    




# 프롬프트 적용 버튼이 눌리면...
if translate_btn:    
    prompt = load_prompt(f"prompts/ocr-{user_selected_prompt}.yaml", encoding="utf8")
    exec_trans_chain(prompt, model_name=tab2.selected_model)
    # 이전 상태의 그래프 값을 가져오는 함수입니다
    #previous_state = upstage_parser_graph.get_state(config).values
    # 파서가 추출한 요소들 중 마지막 30개 요소에서 5개만 슬라이싱하여 확인합니다 (디버깅 목적)
    #previous_state["elements_from_parser"][-30:-25]
    #previous_state["elements_from_parser"]
    tab2.markdown(f"✅ JSON 변환이 완료되었습니다.")    
   
    

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = message_tab.chat_input("업로드한 파일에 대해 궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []
    
# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        message_tab.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with message_tab.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")
