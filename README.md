# 🧠 Provider-Agnostic Semantic Keyword Clusterer & Intent Classifier

**Live App:** [https://seo-semantic-clusterer.streamlit.app/](https://seo-semantic-clusterer.streamlit.app/)

A scalable keyword grouping and topical authority architecture tool. Combines local hierarchical clustering (`AgglomerativeClustering`) with character n-gram TF-IDF or neural embeddings to organize raw keyword exports into Pillar Page hubs and classify search intent (`Informational`, `Commercial`, `Transactional`).

## Key Capabilities
- **Zero-Lock-In Architecture:** Runs 100% locally with zero API keys, or integrates with **Google Gemini, Anthropic Claude, OpenAI, and Groq/OpenRouter** via auto-updated model dropdowns.
- **Hybrid Cost-Saving Pipeline:** Decouples vectorization from LLM reasoning so providers without native embedding endpoints (such as Anthropic Claude) run seamlessly with minimal token usage.
- **Adjustable Cluster Tightness:** Interactive cosine distance threshold slider to tune broad pillar hubs vs. granular long-tail clusters.
- **Flexible Input & Export:** Supports direct keyword pasting or `.csv` / `.xlsx` uploads with one-click CSV export.

## Tech Stack
`Python` • `Streamlit` • `Scikit-Learn` • `Pandas` • `LiteLLM`

## Run Locally
```bash
git clone [https://github.com/mashkoorvohra/seo-semantic-clusterer.git](https://github.com/mashkoorvohra/seo-semantic-clusterer.git)
cd seo-semantic-clusterer
pip install -r requirements.txt
streamlit run app.py
