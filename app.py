import streamlit as st
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import litellm
import requests
import json
import re

st.set_page_config(page_title="Universal Semantic Keyword Clusterer", layout="wide")
st.title("🧠 Provider-Agnostic Semantic Keyword Clusterer")
st.markdown("Groups raw keywords into topical Pillar Page clusters and classifies search intent from **pasted lists, CSV, or Excel (.xlsx)**—runs **100% free locally** or with **any AI provider**.")

@st.cache_data(ttl=3600)
def fetch_live_models(provider_name: str, key: str) -> list:
    defaults = {
        "Google Gemini": ["gemini-4-hydrogen", "gemini-4-carbon", "gemini-4-argon", "gemini-3.5-flash-lite"],
        "Anthropic Claude": ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"],
        "OpenAI": ["gpt-6.2-astra", "gpt-6.2-sol", "gpt-6-luna", "gpt-5.4-mini"],
        "Groq / OpenRouter": ["groq/llama-3.3-70b-versatile", "openrouter/anthropic/claude-sonnet-5-5"]
    }
    if not key:
        return defaults.get(provider_name, [])
    try:
        if provider_name == "Google Gemini":
            r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}", timeout=6)
            models = [m["name"].replace("models/", "") for m in r.json().get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
            newest = [m for m in models if any(v in m for v in ["gemini-4", "gemini-3.8", "gemini-3.5"])]
            return newest[::-1] if newest else defaults["Google Gemini"]
        elif provider_name == "Anthropic Claude":
            r = requests.get("https://api.anthropic.com/v1/models", headers={"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=6)
            models = [m["id"] for m in r.json().get("data", [])]
            return models if models else defaults["Anthropic Claude"]
        elif provider_name == "OpenAI":
            r = requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=6)
            models = [m["id"] for m in r.json().get("data", []) if any(v in m["id"] for v in ["gpt-6", "gpt-5", "o3"])]
            return sorted(models, reverse=True) if models else defaults["OpenAI"]
    except Exception:
        pass
    return defaults.get(provider_name, [])

# Sidebar Controls
st.sidebar.header("⚙️ AI Engine Settings")
provider = st.sidebar.selectbox(
    "Select AI Engine",
    ["Free Local Engine (No API Key Needed)", "Google Gemini", "Anthropic Claude", "OpenAI", "Groq / OpenRouter"]
)

api_key = ""
selected_model = ""
embedding_mode = "Local TF-IDF (Free & Universal)"

if provider != "Free Local Engine (No API Key Needed)":
    api_key = st.sidebar.text_input(f"Enter {provider} API Key", type="password")
    available_models = fetch_live_models(provider, api_key)
    if provider == "Groq / OpenRouter":
        selected_model = st.sidebar.text_input("Model ID", value=available_models[0])
    else:
        selected_model = st.sidebar.selectbox("Select Model (Auto-Updated)", available_models)
        prefix_map = {"Google Gemini": "gemini/", "Anthropic Claude": "anthropic/", "OpenAI": "openai/"}
        if not selected_model.startswith(prefix_map[provider]):
            selected_model = prefix_map[provider] + selected_model

    if provider == "Google Gemini":
        embedding_mode = st.sidebar.radio("Vector Engine", ["Gemini Neural Embeddings (gemini-embedding-2)", "Local TF-IDF (Free & Universal)"])
    elif provider == "OpenAI":
        embedding_mode = st.sidebar.radio("Vector Engine", ["OpenAI Neural Embeddings", "Local TF-IDF (Free & Universal)"])

distance_threshold = st.sidebar.slider("Cluster Tightness (Lower = More Specific)", 0.30, 0.95, 0.72, 0.05)

def classify_intent_local(kw: str) -> str:
    kw_lower = kw.lower()
    if any(re.search(rf"\b{re.escape(t)}\b", kw_lower) for t in ["hire", "buy", "book", "quote", "consultant", "agency", "services", "software", "platform", "demo", "trial"]):
        return "Transactional"
    if any(re.search(rf"\b{re.escape(t)}\b", kw_lower) for t in ["best", "top", "cost", "price", "pricing", "vs", "review", "reviews", "comparison", "alternatives"]):
        return "Commercial"
    return "Informational"

def name_cluster_local(kw_list: list) -> str:
    stop_words = {"in", "for", "of", "the", "a", "an", "to", "how", "what", "is", "best", "top", "cost", "price", "guide"}
    words = [w for kw in kw_list for w in re.findall(r"[a-zA-Z0-9]+", kw.lower()) if w not in stop_words and len(w) > 2]
    return (" ".join(pd.Series(words).value_counts().head(3).index.tolist()).title() + " Hub") if words else kw_list[0].title()

# Flexible Input Method (Neutral Demo, Paste, or CSV/Excel Upload)
input_mode = st.radio(
    "Choose Keyword Input Source:",
    ["Neutral Demo Keywords / Paste List", "Upload CSV or Excel File (.csv / .xlsx)"],
    horizontal=True
)

keywords = []
if input_mode == "Neutral Demo Keywords / Paste List":
    default_keywords = """cloud security audit checklist
best cloud security posture management tools
enterprise cloud compliance pricing
how to prevent cloud data breaches
technical seo audit checklist
core web vitals optimization guide
how to fix crawl budget issues
b2b crm software comparison
enterprise crm implementation cost
buy crm platform for sales teams"""
    raw_text = st.text_area("Paste Keywords (One per line — minimum 2):", value=default_keywords, height=200)
    keywords = list(dict.fromkeys([k.strip() for k in raw_text.split("\n") if k.strip()]))
else:
    uploaded = st.file_uploader("Upload any Ahrefs, Semrush, GSC, or custom keyword file", type=["csv", "xlsx"])
    if uploaded:
        up_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        cols = list(up_df.columns)
        guess_idx = next((i for i, c in enumerate(cols) if any(k in str(c).lower() for k in ["keyword", "query", "term"])), 0)
        kw_col = st.selectbox("Select Keyword Column:", cols, index=guess_idx)
        keywords = list(dict.fromkeys(up_df[kw_col].dropna().astype(str).str.strip().tolist()))
        st.caption(f"Loaded {len(keywords)} unique keywords from `{uploaded.name}`.")

if st.button("Cluster Keywords & Classify Intent", type="primary"):
    if len(keywords) < 2:
        st.warning("Please provide at least 2 unique keywords.")
        st.stop()
    if provider != "Free Local Engine (No API Key Needed)" and not api_key:
        st.warning(f"Please paste your {provider} API key in the sidebar, or use 'Free Local Engine'.")
        st.stop()

    try:
        with st.spinner("Clustering keywords..."):
            if "Gemini Neural" in embedding_mode:
                embed_res = litellm.embedding(model="gemini/gemini-embedding-2", input=keywords, api_key=api_key)
                vectors = [item["embedding"] for item in embed_res.data]
            elif "OpenAI Neural" in embedding_mode:
                embed_res = litellm.embedding(model="openai/text-embedding-3-small", input=keywords, api_key=api_key)
                vectors = [item["embedding"] for item in embed_res.data]
            else:
                vectors = TfidfVectorizer(ngram_range=(1, 2), analyzer="char_wb", min_df=1).fit_transform(keywords).toarray()

            labels = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average").fit_predict(vectors)
            df = pd.DataFrame({"Keyword": keywords, "Cluster_ID": labels})

            if provider == "Free Local Engine (No API Key Needed)":
                df["Cluster Topic"] = df["Cluster_ID"].map(df.groupby("Cluster_ID")["Keyword"].apply(list).apply(name_cluster_local).to_dict())
                df["Search Intent"] = df["Keyword"].apply(classify_intent_local)
                active_label = "Local Engine (0 API Calls)"
            else:
                grouped = df.groupby("Cluster_ID")["Keyword"].apply(list).to_dict()
                prompt = f"Analyze these SEO keyword clusters. For each Cluster_ID, return a JSON object mapping the ID (as string) to {{'topic': 'Pillar Page Title', 'intent': 'Informational|Commercial|Transactional'}}. Clusters: {json.dumps(grouped)}. Return ONLY raw JSON."
                resp = litellm.completion(model=selected_model, messages=[{"role": "user", "content": prompt}], api_key=api_key)
                clean_json = resp.choices[0].message.content.replace("```json", "").replace("```", "").strip()
                meta = json.loads(clean_json)
                df["Cluster Topic"] = df["Cluster_ID"].astype(str).map(lambda x: meta.get(x, {}).get("topic", "General Hub"))
                df["Search Intent"] = df["Cluster_ID"].astype(str).map(lambda x: meta.get(x, {}).get("intent", "Informational"))
                active_label = selected_model

            final_df = df[["Keyword", "Cluster Topic", "Search Intent"]].sort_values(["Cluster Topic", "Search Intent"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Keywords Clustered", len(final_df))
            c2.metric("Pillar Clusters", final_df["Cluster Topic"].nunique())
            c3.metric("Active Engine", active_label)
            st.dataframe(final_df, use_container_width=True, hide_index=True)
            st.download_button("Download Clustered CSV", data=final_df.to_csv(index=False).encode("utf-8"), file_name="seo_keyword_clusters.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
