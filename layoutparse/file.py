import pickle
import os


def save_documents_to_pkl(documents, filepath):
    """
    Langchain Document 리스트를 pickle 파일로 저장하는 함수

    Args:
        documents: Langchain Document 객체 리스트
        filepath: 원본 파일 경로 (예: path/to/filename.pdf)
    """
    # 확장자 제거하고 절대 경로로 변환
    abs_path = os.path.abspath(filepath)
    base_path = os.path.splitext(abs_path)[0]
    pkl_path = f"{base_path}.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump(documents, f)


def load_documents_from_pkl(filepath):
    """
    Pickle 파일에서 Langchain Document 리스트를 불러오는 함수

    Args:
        filepath: 원본 파일 경로 (예: path/to/filename.pdf)
    Returns:
        Langchain Document 객체 리스트
    """
    # 확장자 제거하고 절대 경로로 변환
    abs_path = os.path.abspath(filepath)
    base_path = os.path.splitext(abs_path)[0]
    pkl_path = f"{base_path}.pkl"

    with open(pkl_path, "rb") as f:
        documents = pickle.load(f)
    return documents
