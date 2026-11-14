# Document sources and attribution

All three sample PDFs are **original educational documents written for this project** by Claude (an AI assistant made by Anthropic) on 2026-11-14.
They are **not** copies of external publications. The environment in which this project was built could not download third-party PDFs (external document hosts were blocked), so original, clearly labelled material was authored instead. The same attribution appears in the footer of every page and in each PDF's metadata.

| File | Title | Author / origin | Licence |
|---|---|---|---|
| `machine_learning_fundamentals.pdf` | Machine Learning Fundamentals: Data, Training and Prediction | Claude (AI assistant by Anthropic), written for this assignment | MIT (same as project) |
| `ai_in_healthcare.pdf` | Artificial Intelligence in Healthcare: Benefits, Risks and Responsible Use | Claude (AI assistant by Anthropic), written for this assignment | MIT (same as project) |
| `data_security_and_privacy.pdf` | Data Security and Privacy in the Age of AI | Claude (AI assistant by Anthropic), written for this assignment | MIT (same as project) |

## Using your own documents

Any PDF placed in this folder is picked up automatically by `python main.py` and by the Streamlit *Document Ingestion* page. If your assignment requires third-party documents (for example public papers or reports), download them and copy them here, then re-run `python main.py --reset`. Remember to add their attribution to this file.

Scripts used to build the samples: `scripts/sample_content.py`, `scripts/generate_sample_documents.py`.
