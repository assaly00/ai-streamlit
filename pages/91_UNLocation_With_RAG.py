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
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.storage import LocalFileStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.embeddings import CacheBackedEmbeddings

from dotenv import load_dotenv
import os
import faiss

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] UN/LOCODE with RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# cache 저장 경로 지정
store = LocalFileStore("./cache/embeddings")

st.title("UN/LOCODE with RAG💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")
    embed_file_btn = st.button("데이터분석 시작", key="apply")

    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "검색용 LLM 선택", ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )

# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="UN/LOCODE 데이터를 임베딩중입니다...")
def embed_data():        
    embeddings = OpenAIEmbeddings()        
    vectorstore = load_local_db(embeddings)
    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever

def load_local_db(embeddings):
    print("db 불러오는중...")    
    try:
        # 저장된 데이터를 로드
        vectorstore = FAISS.load_local(
            folder_path="faiss_db",
            index_name="faiss_index",
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        print("저장된 DB를 반환합니다")            
    except Exception:
        # 저장된 데이터가 없을 경우
        vectorstore = create_db(embeddings)

    return vectorstore


cached_embedder = None
def create_db(embeddings):
    
    # 단계 1: 문서 로드(Load Documents)
    # CSV 로더 생성
    loader = CSVLoader(file_path="./data/2024-1 UNLOCODE CodeList.csv", encoding='latin-1')
    #loader = CSVLoader(file_path="./data/UNLOCODECodeList-Test.csv", encoding='latin-1')    
    docs = loader.load()
    print(len(docs))
    
    # 캐시를 지원하는 임베딩 생성
    # cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    #     underlying_embeddings=embeddings,
    #     document_embedding_cache=store,
    #     namespace=embeddings.model,  # 기본 임베딩과 저장소를 사용하여 캐시 지원 임베딩을 생성
    # )
    # print(f"embedding 생성완료")

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)
    print(f"분할된 청크의수: {len(split_documents)}")    

    # 단계 4: DB 생성(Create DB) 및 저장
    # 캐시를 지원하는 벡터스토어 생성
    #vectorstore = FAISS.from_documents(documents=split_documents, embedding=cached_embedder)
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    print(f"vector store 생성완료")

    # 로컬 Disk 에 저장
    vectorstore.save_local(folder_path="faiss_db", index_name="faiss_index")
    print(f"vector store 로컬 저장 완료")
    
    return vectorstore


# 체인 생성
def create_chain(retriever, model_name="gpt-4o"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("prompts/uncode-only-rag.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 파일이 업로드 되었을 때
if embed_file_btn:
    # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정...)
    retriever = embed_data()
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain
    st.markdown(f"✅ 데이터분석이 완료되었습니다.")    

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
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
        warning_msg.error("데이터분석 버튼을 눌러주세요")
