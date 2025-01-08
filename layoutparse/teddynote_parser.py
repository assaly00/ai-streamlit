import os
from layoutparse.utils import SplitPDFFilesNode
from layoutparse.state import ParseState
from layoutparse.upstage import (
    DocumentParseNode,
    PostDocumentParseNode,
    WorkingQueueNode,
    continue_parse,
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from layoutparse.preprocessing import (
    CreateElementsNode,
    MergeEntityNode,
    ReconstructElementsNode,
    LangChainDocumentNode,
)
from layoutparse.export import ExportHTML, ExportMarkdown, ExportTableCSV, ExportImage
from layoutparse.extractor import (
    PageElementsExtractorNode,
    ImageEntityExtractorNode,
    TableEntityExtractorNode,
)
from langchain_teddynote.graphs import visualize_graph
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_upstage_parser_graph(
    batch_size: int = 30,
    test_page: int = None,
    verbose: bool = True,
    visualize: bool = False    
):
    split_pdf_node = SplitPDFFilesNode(
        batch_size=batch_size, test_page=test_page, verbose=verbose
    )

    document_parse_node = DocumentParseNode(
        api_key=os.environ["UPSTAGE_API_KEY"], verbose=verbose
    )

    post_document_parse_node = PostDocumentParseNode(verbose=verbose)
    working_queue_node = WorkingQueueNode(verbose=verbose)

    # LangGraph 생성
    workflow = StateGraph(ParseState)

    # 노드들을 정의합니다.
    workflow.add_node("split_pdf_node", split_pdf_node)
    workflow.add_node("document_parse_node", document_parse_node)
    workflow.add_node("post_document_parse_node", post_document_parse_node)
    workflow.add_node("working_queue_node", working_queue_node)

    # 각 노드들을 연결합니다.
    workflow.add_edge("split_pdf_node", "working_queue_node")
    workflow.add_conditional_edges(
        "working_queue_node",
        continue_parse,
        {True: "document_parse_node", False: "post_document_parse_node"},
    )
    workflow.add_edge("document_parse_node", "working_queue_node")

    workflow.set_entry_point("split_pdf_node")

    document_parse_graph = workflow.compile(checkpointer=MemorySaver())
    if visualize:
        visualize_graph(document_parse_graph)
    return document_parse_graph


def create_export_graph(
    ignore_new_line_in_text=True,
    show_image_in_markdown=False,
    verbose=True,
    visualize=False,
):
    # 워크플로우 생성
    export_workflow = StateGraph(ParseState)

    export_image = ExportImage(verbose=verbose)
    export_html = ExportHTML(
        ignore_new_line_in_text=ignore_new_line_in_text, verbose=verbose
    )
    export_markdown = ExportMarkdown(
        ignore_new_line_in_text=ignore_new_line_in_text,
        show_image=show_image_in_markdown,
        verbose=verbose,
    )
    export_table_csv = ExportTableCSV(verbose=verbose)

    export_workflow.add_node("export_image", export_image)
    export_workflow.add_node("export_html", export_html)
    export_workflow.add_node("export_markdown", export_markdown)
    export_workflow.add_node("export_table_to_csv", export_table_csv)

    export_workflow.add_edge("export_image", "export_html")
    export_workflow.add_edge("export_image", "export_markdown")
    export_workflow.add_edge("export_image", "export_table_to_csv")

    export_workflow.add_edge("export_html", END)
    export_workflow.add_edge("export_markdown", END)
    export_workflow.add_edge("export_table_to_csv", END)

    export_workflow.set_entry_point("export_image")

    export_graph = export_workflow.compile()
    if visualize:
        visualize_graph(export_graph)
    return export_graph


def create_document_parse_graph():
    # PDF 분할 노드 생성
    split_pdf_node = SplitPDFFilesNode(batch_size=30, test_page=None, verbose=True)
    document_parse_node = DocumentParseNode(
        api_key=os.environ["UPSTAGE_API_KEY"], verbose=True
    )
    post_document_parse_node = PostDocumentParseNode(verbose=True)
    working_queue_node = WorkingQueueNode(verbose=True)

    # 첫 번째 워크플로우 생성
    workflow = StateGraph(ParseState)
    workflow.add_node("split_pdf_node", split_pdf_node)
    workflow.add_node("document_parse_node", document_parse_node)
    workflow.add_node("post_document_parse_node", post_document_parse_node)
    workflow.add_node("working_queue_node", working_queue_node)

    workflow.add_edge("split_pdf_node", "working_queue_node")
    workflow.add_conditional_edges(
        "working_queue_node",
        continue_parse,
        {True: "document_parse_node", False: "post_document_parse_node"},
    )
    workflow.add_edge("document_parse_node", "working_queue_node")
    workflow.set_entry_point("split_pdf_node")
    parser_graph = workflow.compile()

    # 후처리 노드들 생성
    create_elements_node = CreateElementsNode(verbose=True)
    export_html = ExportHTML(verbose=True)
    export_markdown = ExportMarkdown(verbose=True)
    export_table_csv = ExportTableCSV(verbose=True)
    page_elements_extractor_node = PageElementsExtractorNode(verbose=True)
    image_entity_extractor_node = ImageEntityExtractorNode(verbose=True)
    table_entity_extractor_node = TableEntityExtractorNode(verbose=True)
    merge_entity_node = MergeEntityNode(verbose=True)
    reconstruct_elements_node = ReconstructElementsNode(verbose=True)
    langchain_document_node = LangChainDocumentNode(
        verbose=True,
        splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0),
    )

    # 후처리 워크플로우 생성
    post_process_workflow = StateGraph(ParseState)

    # 노드 추가
    post_process_workflow.add_node("document_parse", parser_graph)
    post_process_workflow.add_node("create_elements_node", create_elements_node)
    post_process_workflow.add_node("export_html", export_html)
    post_process_workflow.add_node("export_markdown", export_markdown)
    post_process_workflow.add_node("export_table_csv", export_table_csv)
    post_process_workflow.add_node(
        "page_elements_extractor_node", page_elements_extractor_node
    )
    post_process_workflow.add_node(
        "image_entity_extractor_node", image_entity_extractor_node
    )
    post_process_workflow.add_node(
        "table_entity_extractor_node", table_entity_extractor_node
    )
    post_process_workflow.add_node("merge_entity_node", merge_entity_node)
    post_process_workflow.add_node(
        "reconstruct_elements_node", reconstruct_elements_node
    )
    post_process_workflow.add_node("langchain_document_node", langchain_document_node)

    # 엣지 연결
    post_process_workflow.add_edge("document_parse", "create_elements_node")
    post_process_workflow.add_edge("create_elements_node", "export_html")
    post_process_workflow.add_edge("create_elements_node", "export_markdown")
    post_process_workflow.add_edge("create_elements_node", "export_table_csv")
    post_process_workflow.add_edge(
        "create_elements_node", "page_elements_extractor_node"
    )
    post_process_workflow.add_edge(
        "page_elements_extractor_node", "image_entity_extractor_node"
    )
    post_process_workflow.add_edge(
        "page_elements_extractor_node", "table_entity_extractor_node"
    )
    post_process_workflow.add_edge("image_entity_extractor_node", "merge_entity_node")
    post_process_workflow.add_edge("export_html", END)
    post_process_workflow.add_edge("export_markdown", END)
    post_process_workflow.add_edge("export_table_csv", END)
    post_process_workflow.add_edge("table_entity_extractor_node", "merge_entity_node")
    post_process_workflow.add_edge("merge_entity_node", "reconstruct_elements_node")
    post_process_workflow.add_edge(
        "reconstruct_elements_node", "langchain_document_node"
    )
    post_process_workflow.add_edge("langchain_document_node", END)

    post_process_workflow.set_entry_point("document_parse")

    memory = MemorySaver()
    return post_process_workflow.compile(checkpointer=memory)
