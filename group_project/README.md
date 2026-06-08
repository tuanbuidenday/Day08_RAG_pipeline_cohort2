# Bài Tập Nhóm - Search Engine / RAG Chatbot

## Mục Tiêu

Nhóm triển khai cả 2 deliverable trong README gốc:

1. RAG Chatbot trả lời câu hỏi về pháp luật ma túy và tin tức nghệ sĩ Việt Nam liên quan đến ma túy.
2. RAG Evaluation Pipeline với golden dataset, 4 metric và so sánh A/B.

## Tính Năng Đã Hoàn Thành

| Hạng mục | Trạng thái | File |
|---|---|---|
| Giao diện chat Streamlit | Done | `group_project/app.py` |
| Trả lời có citation từ Task10 | Done | `src/task10_generation.py` |
| Follow-up questions bằng conversation memory | Done | `group_project/app.py` |
| Hiển thị source documents/chunks | Done | `group_project/app.py` |
| Golden dataset >= 15 Q&A | Done, 16 cases | `group_project/evaluation/golden_dataset.json` |
| Evaluation 4 metrics | Done | `group_project/evaluation/eval_pipeline.py` |
| A/B comparison | Done, `hybrid_rerank` vs `dense_only` | `group_project/evaluation/results.md` |
| Báo cáo worst performers và đề xuất | Done | `group_project/evaluation/results.md` |

## Kiến Trúc Hệ Thống

```text
User
  |
  v
Streamlit Chat UI (group_project/app.py)
  |
  +--> Conversation memory: 3 lượt hỏi đáp gần nhất
  |
  v
Task10 Generation with Citation
  |
  v
Task9 Retrieval Pipeline
  |
  +--> Task5 Semantic Search: Weaviate Cloud + Jina embeddings
  +--> Task6 Lexical Search: Weaviate BM25
  +--> Task7 Fusion/Reranking: RRF + Jina reranker
  +--> Task8 PageIndex fallback khi retrieval yếu
  |
  v
Context chunks + metadata + citation labels
  |
  v
Answer + source documents displayed in UI
```

Evaluation pipeline:

```text
golden_dataset.json
  |
  +--> Config A: hybrid_rerank
  +--> Config B: dense_only
  |
  v
Deterministic DeepEval/RAGAS-style scorer
  |
  +--> faithfulness
  +--> answer_relevance
  +--> context_recall
  +--> context_precision
  |
  v
results.md
```

## Dữ Liệu

- Văn bản pháp luật: Luật Phòng, chống ma túy 2021; Nghị định 105/2021/NĐ-CP; Bộ luật Hình sự 2015 sửa đổi 2017, chương các tội phạm về ma túy.
- Báo chí: 6 bài báo thật trong `data/landing/news/article_01.json` đến `article_06.json`, đã chuẩn hóa sang Markdown trong `data/standardized/news/`.
- Vector store: Weaviate Cloud collection `DrugLawDocs`.
- Embedding: Jina `jina-embeddings-v3`.

## Evaluation

Framework lựa chọn: evaluator nội bộ theo phong cách DeepEval/RAGAS, giữ đúng 4 trục metric trong README. Lý do dùng scorer deterministic là để pipeline chạy ổn định trong demo khi OpenAI/PageIndex generation hoặc LLM judge hết quota.

Metric:

- `faithfulness`: câu trả lời extractive có nằm trong retrieved context không.
- `answer_relevance`: câu trả lời có khớp câu hỏi và expected answer không.
- `context_recall`: retrieved context có bao phủ expected answer/context không.
- `context_precision`: tỉ lệ chunk retrieve được có evidence hữu ích.

A/B configs:

- `hybrid_rerank`: semantic + BM25, RRF merge, Jina rerank.
- `dense_only`: chỉ dùng semantic search trên vector Weaviate.

## Hướng Dẫn Chạy

Cài dependencies:

```bash
pip install -r requirements.txt
```

Tạo/chỉnh `.env` theo `.env.example`, tối thiểu cần:

```bash
JINA_API_KEY=...
WEAVIATE_URL=...
WEAVIATE_API_KEY=...
PAGEINDEX_API_KEY=...
EMBEDDING_PROVIDER=jina
GENERATION_PROVIDER=pageindex
```

Index lại dữ liệu nếu cần:

```bash
python -m src.task4_chunking_indexing
```

Chạy chatbot:

```bash
streamlit run group_project/app.py
```

Chạy evaluation:

```bash
python group_project/evaluation/eval_pipeline.py
```

Kết quả evaluation được ghi vào:

```text
group_project/evaluation/results.md
```

## Phân Công Công Việc

Nhóm 1 thành viên.

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|---|---|---|---|
| Bùi Văn Tuân | 2A202601006 | Tích hợp Task9/Task10 vào Streamlit chatbot; xây dựng golden dataset 16 Q&A; viết evaluation pipeline, A/B comparison, results.md; hoàn thiện README và demo local | Done |

## Ghi Chú Demo

App ưu tiên gọi Task10 để sinh câu trả lời có citation. Nếu generation API trả lỗi quota, app tự chuyển sang câu trả lời extractive từ Task9 để vẫn hiển thị được evidence và source documents.
