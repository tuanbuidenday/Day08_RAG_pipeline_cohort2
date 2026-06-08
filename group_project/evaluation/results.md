# RAG Evaluation Results

## Setup

- Golden dataset: 16 câu hỏi thật từ văn bản pháp luật và 6 bài báo đã crawl.
- Evaluator: DeepEval/RAGAS-style deterministic scorer, dùng cùng 4 trục trong README.
- Lý do không dùng LLM judge trực tiếp: script cần chạy ổn định khi OpenAI/PageIndex hết quota trong buổi demo.
- A/B configs: `hybrid_rerank` so với `dense_only`.

## Overall Scores

| Config | Faithfulness | Answer relevance | Context recall | Context precision | Overall |
|---|---:|---:|---:|---:|---:|
| hybrid_rerank | 1.000 | 0.441 | 0.661 | 0.825 | 0.732 |
| dense_only | 1.000 | 0.425 | 0.617 | 0.825 | 0.717 |

Config tốt nhất theo overall score: `hybrid_rerank`.

## Worst Performers

| Config | Case | Overall | Weakest metric | Top source | Question |
|---|---|---:|---|---|---|
| dense_only | legal_001 | 0.358 | answer_relevance=0.096 | nghi-dinh-105-2021, 2021 | Điều 249 Bộ luật Hình sự quy định tội tàng trữ trái phép chất ma túy như thế nào? |
| hybrid_rerank | legal_001 | 0.384 | answer_relevance=0.037 | bo-luat-hinh-su-2015-chuong-ma-tuy, 2015 | Điều 249 Bộ luật Hình sự quy định tội tàng trữ trái phép chất ma túy như thế nào? |
| hybrid_rerank | legal_010 | 0.453 | answer_relevance=0.029 | nghi-dinh-105-2021, 2021 | Nghị định 105/2021/NĐ-CP quy định việc xác định tình trạng nghiện ma túy nhằm mục đích gì? |
| dense_only | legal_005 | 0.469 | answer_relevance=0.188 | article_04, 2026 | Luật Phòng, chống ma túy 2021 quy định những hình thức cai nghiện ma túy nào? |
| hybrid_rerank | legal_005 | 0.592 | answer_relevance=0.240 | nghi-dinh-105-2021, 2021 | Luật Phòng, chống ma túy 2021 quy định những hình thức cai nghiện ma túy nào? |

## Recommendations

1. Tăng coverage của golden dataset cho từng điều luật cụ thể, nhất là các câu hỏi có số điều và định lượng khối lượng ma túy.
2. Bổ sung metadata `article_title`, `published_date`, `law_article` vào chunk để citation rõ hơn và context precision cao hơn.
3. Tách chunk pháp luật theo điều/khoản thay vì chỉ theo kích thước ký tự để giảm nhiễu khi hỏi về Điều 249-255.
4. Khi có quota LLM judge, có thể thay scorer hiện tại bằng DeepEval hoặc RAGAS thật mà vẫn giữ cùng schema kết quả.

## Per-case Details

### hybrid_rerank

| Case | Category | Overall | Sources |
|---|---|---:|---|
| legal_001 | criminal_law | 0.384 | bo-luat-hinh-su-2015-chuong-ma-tuy, 2015, nghi-dinh-105-2021, 2021, nghi-dinh-105-2021, 2021 |
| legal_002 | criminal_law | 0.670 | nghi-dinh-105-2021, 2021, article_03, 2026, article_05, 2026 |
| legal_003 | criminal_law | 0.701 | nghi-dinh-105-2021, 2021, article_05, 2026, article_02, 2026 |
| legal_004 | criminal_law | 0.816 | article_01, 2026, article_02, 2026, article_03, 2026 |
| legal_005 | prevention_law | 0.592 | nghi-dinh-105-2021, 2021, article_02, 2026, article_02, 2026 |
| legal_006 | prevention_law | 0.684 | article_01, 2026, nghi-dinh-105-2021, 2021, article_02, 2026 |
| legal_007 | prevention_law | 0.654 | nghi-dinh-105-2021, 2021, article_03, 2026, article_02, 2026 |
| legal_008 | decree | 0.657 | nghi-dinh-105-2021, 2021, article_06, 2026, article_02, 2026 |
| legal_009 | decree | 0.710 | article_04, 2026, article_02, 2026, article_01, 2026 |
| legal_010 | decree | 0.453 | nghi-dinh-105-2021, 2021, nghi-dinh-105-2021, 2021, nghi-dinh-105-2021, 2021 |
| news_001 | news | 0.958 | article_01, 2026, article_01, 2026, article_01, 2026 |
| news_002 | news | 0.922 | article_02, 2026, article_02, 2026, article_01, 2026 |
| news_003 | news | 0.876 | article_03, 2026, article_03, 2026, article_03, 2026 |
| news_004 | news | 0.916 | article_04, 2026, article_04, 2026, article_01, 2026 |
| news_005 | news | 0.824 | article_05, 2026, article_05, 2026, article_01, 2026 |
| news_006 | news | 0.893 | article_06, 2026, article_01, 2026, article_02, 2026 |

### dense_only

| Case | Category | Overall | Sources |
|---|---|---:|---|
| legal_001 | criminal_law | 0.358 | nghi-dinh-105-2021, 2021, article_04, 2026, bo-luat-hinh-su-2015-chuong-ma-tuy, 2015 |
| legal_002 | criminal_law | 0.670 | nghi-dinh-105-2021, 2021, article_05, 2026, article_03, 2026 |
| legal_003 | criminal_law | 0.649 | nghi-dinh-105-2021, 2021, article_05, 2026, article_04, 2026 |
| legal_004 | criminal_law | 0.821 | article_03, 2026, article_01, 2026, article_05, 2026 |
| legal_005 | prevention_law | 0.469 | article_04, 2026, article_06, 2026, article_02, 2026 |
| legal_006 | prevention_law | 0.627 | nghi-dinh-105-2021, 2021, article_04, 2026, article_06, 2026 |
| legal_007 | prevention_law | 0.640 | nghi-dinh-105-2021, 2021, article_04, 2026, article_06, 2026 |
| legal_008 | decree | 0.597 | article_06, 2026, article_04, 2026, article_01, 2026 |
| legal_009 | decree | 0.670 | article_04, 2026, article_03, 2026, article_05, 2026 |
| legal_010 | decree | 0.601 | article_04, 2026, article_06, 2026, article_01, 2026 |
| news_001 | news | 0.958 | article_01, 2026, article_01, 2026, article_01, 2026 |
| news_002 | news | 0.888 | article_01, 2026, article_02, 2026, article_02, 2026 |
| news_003 | news | 0.867 | article_03, 2026, article_03, 2026, article_03, 2026 |
| news_004 | news | 0.924 | article_04, 2026, article_04, 2026, article_04, 2026 |
| news_005 | news | 0.818 | article_05, 2026, article_05, 2026, article_05, 2026 |
| news_006 | news | 0.910 | article_06, 2026, article_06, 2026, article_02, 2026 |
