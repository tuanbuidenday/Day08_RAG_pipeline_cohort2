"""Task 5 — Semantic Search Module."""

from .production_clients import WEAVIATE_COLLECTION, connect_weaviate, embed_texts

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    from weaviate.classes.query import MetadataQuery

    query_embedding = embed_texts([query], task="retrieval.query")[0]
    client = connect_weaviate()
    try:
        collection = client.collections.get(WEAVIATE_COLLECTION)
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        results = []
        for obj in response.objects:
            props = obj.properties
            distance = obj.metadata.distance
            score = 1.0 - float(distance or 0.0)
            results.append(
                {
                    "content": props.get("content", ""),
                    "score": score,
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
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
