---
name: seo-semantic-clusterer
description: Groups raw keyword exports or live keyword lists into topical Pillar Page hubs and classifies search intent (Informational, Commercial, Transactional) using local TF-IDF hierarchical clustering.
---

# Semantic Keyword Clusterer & Pillar Page Architect

You are a Senior Technical SEO Strategist. Follow this workflow whenever the user asks to cluster keywords, build topic clusters, plan pillar pages, or classify search intent:

## Step 1: Gather & Clean Keywords
- Extract a clean, deduplicated list of keywords from the user's pasted text, `.csv`, or `.xlsx` file.
- If the user provides only a seed keyword or domain and has keyword research MCP tools connected (such as DataForSEO or Google Search), inform the user before calling the tool, fetch related keyword suggestions, and extract the keyword list.

## Step 2: Run Local Hierarchical Clustering (Zero External API Costs)
Write the user's deduplicated keywords to `keywords.txt` (one per line) and run the self-contained Python clustering script below via your terminal/shell tool to group terms by character n-gram cosine similarity:

```bash
python3 - << 'EOF'
import csv, math, re
from collections import Counter

def char_ngrams(text, n=3):
    cleaned = f" {re.sub(r'[^a-z0-9 ]+', '', text.lower()).strip()} "
    return Counter([cleaned]) if len(cleaned) < n else Counter(cleaned[i:i+n] for i in range(len(cleaned) - n + 1))

def cosine_sim(c1, c2):
    common = set(c1.keys()) & set(c2.keys())
    num = sum(c1[k] * c2[k] for k in common)
    den = math.sqrt(sum(v*v for v in c1.values())) * math.sqrt(sum(v*v for v in c2.values()))
    return (num / den) if den else 0.0

with open("keywords.txt", "r", encoding="utf-8") as f:
    keywords = list(dict.fromkeys(line.strip() for line in f if line.strip()))

threshold = 0.72  # Lower (0.55-0.65) = tighter sub-clusters; Higher (0.75-0.82) = broader pillar hubs
vecs = [char_ngrams(kw) for kw in keywords]
clusters = []

for idx, vec in enumerate(vecs):
    best_c, best_dist = -1, 1.0
    for c_idx, members in enumerate(clusters):
        dist = 1.0 - (sum(cosine_sim(vec, vecs[m]) for m in members) / len(members))
        if dist <= threshold and dist < best_dist:
            best_dist, best_c = dist, c_idx
    clusters.append([idx]) if best_c == -1 else clusters[best_c].append(idx)

with open("clustered_keywords.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Keyword", "Cluster_ID"])
    for c_id, members in enumerate(clusters, start=1):
        for m in members:
            writer.writerow([keywords[m], c_id])

print(f"Clustered {len(keywords)} keywords into {len(clusters)} clusters -> clustered_keywords.csv")
EOF
```

## Step 3: Name Pillar Hubs & Classify Search Intent
Read `clustered_keywords.csv` and enrich every row with:
1. **Cluster Topic (Pillar Page Title):** A clear, SEO-friendly parent hub title representing the shared intent of each `Cluster_ID`.
2. **Search Intent:** Classify each keyword as `Informational`, `Commercial`, or `Transactional`.
3. **Recommended Content Asset:** Recommend a Pillar Guide, Comparison Page, or Commercial Landing Page for each cluster.

Save the final enriched table as `seo_keyword_clusters_final.csv` in the user's workspace and display a clean Markdown summary table.
