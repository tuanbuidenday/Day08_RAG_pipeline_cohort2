"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
MANIFEST_PATH = Path(__file__).parent.parent / "data" / "pageindex_manifest.json"
PAGEINDEX_UPLOAD_DIR = Path(__file__).parent.parent / "data" / "pageindex_uploads"


def _has_real_api_key() -> bool:
    return bool(PAGEINDEX_API_KEY and PAGEINDEX_API_KEY not in {"pi_xxx", "xxx"})


def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _escape_pdf_text(text: str) -> str:
    safe = (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .encode("latin-1", errors="replace")
        .decode("latin-1")
    )
    return safe


def _write_simple_pdf(text: str, output_path: Path) -> None:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        while len(line) > 90:
            lines.append(line[:90])
            line = line[90:]
        lines.append(line)

    pages = [lines[i : i + 48] for i in range(0, len(lines), 48)] or [[""]]
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_object_ids = []
    for page_lines in pages:
        content_ops = ["BT", "/F1 11 Tf", "50 790 Td", "14 TL"]
        for line in page_lines:
            content_ops.append(f"({_escape_pdf_text(line)}) Tj")
            content_ops.append("T*")
        content_ops.append("ET")
        stream = "\n".join(content_ops).encode("latin-1", errors="replace")
        content_id = len(objects) + 1
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        page_id = len(objects) + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        page_object_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_object_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_ids)} >>".encode("ascii")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    output_path.write_bytes(bytes(pdf))


def _ensure_pageindex_pdfs() -> list[Path]:
    pdf_paths = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        text = md_file.read_text(encoding="utf-8")
        relative = md_file.relative_to(STANDARDIZED_DIR)
        pdf_path = PAGEINDEX_UPLOAD_DIR / relative.with_suffix(".pdf")
        if not pdf_path.exists() or pdf_path.stat().st_mtime < md_file.stat().st_mtime:
            _write_simple_pdf(text, pdf_path)
        pdf_paths.append(pdf_path)
    return pdf_paths


def _extract_retrieval_items(payload: dict, fallback_metadata: dict) -> list[dict]:
    candidates = (
        payload.get("results")
        or payload.get("retrieval_results")
        or payload.get("chunks")
        or payload.get("nodes")
        or []
    )
    if isinstance(candidates, dict):
        candidates = candidates.get("results") or candidates.get("items") or []

    items = []
    for candidate in candidates:
        if isinstance(candidate, str):
            content = candidate
            score = 1.0
            metadata = fallback_metadata
        else:
            content = (
                candidate.get("content")
                or candidate.get("text")
                or candidate.get("markdown")
                or candidate.get("node_text")
                or ""
            )
            score = candidate.get("score") or candidate.get("relevance_score") or 1.0
            metadata = {**fallback_metadata, **candidate.get("metadata", {})}
        if content:
            items.append(
                {
                    "content": content,
                    "score": float(score),
                    "metadata": metadata,
                    "source": "pageindex",
                }
            )
    return items


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not _has_real_api_key():
        raise RuntimeError("Missing PAGEINDEX_API_KEY for production PageIndex upload")

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    existing = {item["path"]: item for item in _load_manifest()}
    uploaded = list(existing.values())

    for pdf_file in _ensure_pageindex_pdfs():
        rel_path = str(pdf_file.relative_to(STANDARDIZED_DIR.parent.parent))
        if rel_path in existing:
            print(f"  ✓ Already uploaded: {pdf_file.name} -> {existing[rel_path]['doc_id']}")
            continue

        response = client.submit_document(str(pdf_file))
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex upload did not return doc_id: {response}")
        item = {
            "doc_id": doc_id,
            "path": rel_path,
            "filename": pdf_file.name,
            "type": pdf_file.parent.name,
        }
        uploaded.append(item)
        print(f"  ✓ Uploaded: {pdf_file.name} -> {doc_id}")

    MANIFEST_PATH.write_text(
        json.dumps(uploaded, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return uploaded


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not _has_real_api_key():
        raise RuntimeError("Missing PAGEINDEX_API_KEY for production PageIndex search")

    manifest = _load_manifest()
    if not manifest:
        manifest = upload_documents()

    from pageindex import PageIndexAPIError, PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    all_results: list[dict] = []
    max_docs = int(os.getenv("PAGEINDEX_MAX_QUERY_DOCS", "3"))
    for doc in manifest[:max_docs]:
        doc_id = doc["doc_id"]
        if not client.is_retrieval_ready(doc_id):
            continue

        try:
            query_response = client.submit_query(doc_id=doc_id, query=query, thinking=False)
        except PageIndexAPIError:
            continue
        retrieval_id = query_response.get("retrieval_id") or query_response.get("id")
        if not retrieval_id:
            continue

        payload = {}
        for _ in range(12):
            payload = client.get_retrieval(retrieval_id)
            status = str(payload.get("status", "")).lower()
            if status in {"completed", "complete", "succeeded", "success", "ready", "done"}:
                break
            if payload.get("results") or payload.get("retrieval_results"):
                break
            time.sleep(1)

        all_results.extend(_extract_retrieval_items(payload, doc))

    all_results.sort(key=lambda item: item["score"], reverse=True)
    if all_results:
        return all_results[:top_k]
    return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
