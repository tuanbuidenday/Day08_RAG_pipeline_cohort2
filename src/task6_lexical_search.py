"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from .production_clients import WEAVIATE_COLLECTION, connect_weaviate
from .local_store import tokenize

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    from weaviate.classes.query import MetadataQuery

    client = connect_weaviate()
    try:
        collection = client.collections.get(WEAVIATE_COLLECTION)
        response = collection.query.bm25(
            query=query,
            limit=top_k,
            return_metadata=MetadataQuery(score=True),
        )
        results = []
        for obj in response.objects:
            props = obj.properties
            score = obj.metadata.score
            results.append(
                {
                    "content": props.get("content", ""),
                    "score": float(score or 0.0),
                    "metadata": {
                        "source": props.get("source", ""),
                        "filename": props.get("filename", ""),
                        "path": props.get("path", ""),
                        "type": props.get("doc_type", ""),
                        "chunk_index": props.get("chunk_index", 0),
                    },
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results
    finally:
        client.close()


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
