import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.task10_generation import generate_with_citation
from src.local_store import source_label
from src.task9_retrieval_pipeline import retrieve


def init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "query" not in st.session_state:
        st.session_state.query = ""


def add_message(query: str, answer: str, sources: list[dict], retrieval_source: str) -> None:
    st.session_state.history.append(
        {
            "query": query,
            "answer": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
        }
    )


def build_contextual_query(query: str) -> str:
    recent = st.session_state.history[-3:]
    if not recent:
        return query

    turns = []
    for item in recent:
        turns.append(f"User: {item['query']}\nAssistant: {item['answer'][:700]}")
    memory = "\n\n".join(turns)
    return f"Lịch sử hội thoại gần đây:\n{memory}\n\nCâu hỏi hiện tại: {query}"


def extractive_fallback_answer(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    parts = []
    for chunk in chunks[:3]:
        content = " ".join(chunk.get("content", "").split())
        if not content:
            continue
        citation = source_label(chunk.get("metadata", {}))
        parts.append(f"{content[:520]}... [{citation}]")

    if not parts:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    return "\n\n".join(parts)


def run_pipeline(query: str) -> dict:
    contextual_query = build_contextual_query(query)
    try:
        return generate_with_citation(contextual_query)
    except Exception as exc:
        chunks = retrieve(contextual_query, top_k=5, score_threshold=0.0, use_reranking=True)
        return {
            "answer": (
                "Generation API đang lỗi hoặc hết quota, app chuyển sang câu trả lời "
                "extractive từ context đã retrieve.\n\n"
                f"{extractive_fallback_answer(query, chunks)}"
            ),
            "sources": chunks,
            "retrieval_source": f"hybrid_fallback_after_generation_error: {exc.__class__.__name__}",
        }


def render_conversation() -> None:
    for idx, item in enumerate(reversed(st.session_state.history), start=1):
        st.markdown(f"### Cuộc hội thoại #{len(st.session_state.history) - idx + 1}")
        st.markdown(f"**Bạn:** {item['query']}")
        st.markdown(f"**Bot:** {item['answer']}")
        st.markdown(
            f"**Nguồn retrieval:** `{item['retrieval_source']}` | số chunk: {len(item['sources'])}"
        )

        if item["sources"]:
            with st.expander("Xem chi tiết chunk và metadata", expanded=False):
                for i, chunk in enumerate(item["sources"], start=1):
                    metadata = chunk.get("metadata", {})
                    label = source_label(metadata)
                    st.markdown(
                        f"**Chunk {i}** — score: {chunk.get('score', 0.0):.4f} — source: {label}"
                    )
                    st.write(chunk.get("content", ""))
                    st.markdown(
                        f"_Metadata:_ `{metadata.get('path', metadata.get('filename', 'unknown'))}` | type: `{metadata.get('type', 'unknown')}`"
                    )
                    st.divider()
        st.markdown("---")


def handle_query() -> None:
    query = st.session_state.query.strip()
    if not query:
        return

    try:
        with st.spinner("Đang truy vấn retrieval và sinh câu trả lời..."):
            result = run_pipeline(query)
    except Exception as exc:
        st.error(
            "Lỗi khi chạy pipeline. Kiểm tra biến môi trường và kết nối Weaviate/PageIndex."
        )
        st.error(str(exc))
        return

    add_message(
        query=query,
        answer=result.get("answer", "Không có câu trả lời."),
        sources=result.get("sources", []),
        retrieval_source=result.get("retrieval_source", "unknown"),
    )
    st.session_state.query = ""


def main() -> None:
    st.set_page_config(
        page_title="DrugLaw RAG Chatbot",
        layout="wide",
    )
    init_state()

    st.title("RAG Chatbot Pháp luật ma tuý")
    st.write(
        "Ứng dụng demo chatbot retrieval + generation có citation. "
        "Câu trả lời lấy từ sources trong bộ dữ liệu pháp luật và báo chí."
    )

    with st.sidebar:
        st.header("Cấu hình")
        st.markdown("- `GENERATION_PROVIDER`: OpenAI hoặc PageIndex")
        st.markdown(f"- Generation provider: `{os.getenv('GENERATION_PROVIDER', 'pageindex')}`")
        st.markdown(f"- OpenAI model: `{os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')}`")
        st.markdown(f"- Weaviate collection: `{os.getenv('WEAVIATE_COLLECTION', 'DrugLawDocs')}`")
        st.markdown(f"- PageIndex key: `{bool(os.getenv('PAGEINDEX_API_KEY'))}`")
        if st.button("Xoá hội thoại"):
            st.session_state.history = []

    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input(
            "Nhập câu hỏi của bạn",
            key="query",
            placeholder="Ví dụ: Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?",
            on_change=handle_query,
        )
        st.button("Gửi", on_click=handle_query)

        if st.session_state.history:
            render_conversation()
        else:
            st.info("Nhập câu hỏi và ấn Gửi để khởi chạy chatbot.")

    with col2:
        st.header("Hướng dẫn nhanh")
        st.markdown(
            "1. Điền câu hỏi vào ô trên.\n"
            "2. Nếu chưa có kết quả, kiểm tra biến môi trường `OPENAI_API_KEY`, `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `PAGEINDEX_API_KEY`.\n"
            "3. Bot sẽ trả lời kèm citation và hiển thị source chunk."
        )
        st.markdown("---")
        st.header("Lưu ý")
        st.markdown(
            "- App sử dụng pipeline `src/task10_generation.py` để tạo câu trả lời có citation.\n"
            "- Nếu generation API lỗi quota, app fallback sang câu trả lời extractive từ Task9."
        )


if __name__ == "__main__":
    main()
