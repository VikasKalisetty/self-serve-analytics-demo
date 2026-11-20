import os
from pathlib import Path

import chromadb
import yaml
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from google import genai

load_dotenv()

BASE_DIR = Path(__file__).parent
CONTEXT_DIR = BASE_DIR / "context"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom ChromaDB embedding function using the official google-genai SDK."""

    def __init__(self, model_name: str = "gemini-embedding-001"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=list(input),
        )
        return [emb.values for emb in response.embeddings]


def load_yaml(filename: str) -> dict:
    with open(CONTEXT_DIR / filename, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("Initializing ChromaDB with Gemini embeddings...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = GeminiEmbeddingFunction()

    # Reset collections for clean, idempotent re-indexing
    for col_name in ["enterprise_kpis", "semantic_layer"]:
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

    kpi_col = client.create_collection(
        name="enterprise_kpis",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    sem_col = client.create_collection(
        name="semantic_layer",
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # 1. Index Enterprise KPI Definitions
    kpi_data = load_yaml("kpi_definitions.yaml")["kpis"]
    kpi_ids, kpi_docs, kpi_metas = [], [], []

    for k in kpi_data:
        doc_text = (
            f"KPI: {k['name']} ({k['id']})\n"
            f"Synonyms: {', '.join(k.get('synonyms', []))}\n"
            f"Definition: {k['business_definition'].strip()}\n"
            f"Formula: {k['formula_logic'].strip()}\n"
            f"Mandatory Filters: {', '.join(k.get('mandatory_filters', [])) or 'None'}\n"
            f"Required Tables: {', '.join(k.get('required_tables', []))}\n"
            f"Notes: {k.get('notes', '').strip()}"
        )
        kpi_ids.append(k["id"])
        kpi_docs.append(doc_text)
        kpi_metas.append({
            "doc_type": "kpi",
            "kpi_name": k["name"],
            "version": k["version"],
            "owner": k["owner"],
            "required_tables": ", ".join(k.get("required_tables", [])),
        })

    kpi_col.add(ids=kpi_ids, documents=kpi_docs, metadatas=kpi_metas)
    print(f"  [Indexed] enterprise_kpis -> {len(kpi_ids)} KPI documents")

    # 2. Index Semantic Layer (Tables + Joins + Global Rules)
    sem_data = load_yaml("semantic_layer.yaml")
    sem_ids, sem_docs, sem_metas = [], [], []

    for t in sem_data["tables"]:
        doc_text = (
            f"Table: {t['name']} ({t['type']})\n"
            f"Grain: {t['grain']}\n"
            f"Description: {t['description']}\n"
            f"Key Columns:\n- " + "\n- ".join(t["key_columns"])
        )
        sem_ids.append(f"TABLE-{t['name']}")
        sem_docs.append(doc_text)
        sem_metas.append({"doc_type": "table", "table_name": t["name"], "table_type": t["type"]})

    for j in sem_data["joins"]:
        doc_text = (
            f"Canonical Join ({j['id']}): {j['left_table']} JOIN {j['right_table']}\n"
            f"ON {j['join_condition']}\n"
            f"Usage: {j['usage']}"
        )
        sem_ids.append(j["id"])
        sem_docs.append(doc_text)
        sem_metas.append({
            "doc_type": "join",
            "left_table": j["left_table"],
            "right_table": j["right_table"],
        })

    for r in sem_data["global_rules"]:
        doc_text = f"Global SQL Rule ({r['id']} - {r['title']}): {r['description'].strip()}"
        sem_ids.append(r["id"])
        sem_docs.append(doc_text)
        sem_metas.append({"doc_type": "rule", "title": r["title"]})

    sem_col.add(ids=sem_ids, documents=sem_docs, metadatas=sem_metas)
    print(f"  [Indexed] semantic_layer  -> {len(sem_ids)} documents (9 tables, 10 joins, 3 global rules)")

    # 3. Quick Retrieval Sanity Check
    test_query = "What is our Purchase Price Variance (PPV) by vendor tier?"
    print(f"\n--- Retrieval Sanity Check: '{test_query}' ---")

    kpi_res = kpi_col.query(query_texts=[test_query], n_results=2)
    print("Top KPIs retrieved:")
    for doc_id, meta, dist in zip(kpi_res["ids"][0], kpi_res["metadatas"][0], kpi_res["distances"][0]):
        print(f"  - {doc_id}: {meta['kpi_name']} (similarity: {1 - dist:.3f})")

    sem_res = sem_col.query(query_texts=[test_query], n_results=4)
    print("Top Semantic Docs retrieved:")
    for doc_id, meta, dist in zip(sem_res["ids"][0], sem_res["metadatas"][0], sem_res["distances"][0]):
        print(f"  - {doc_id} [{meta['doc_type']}] (similarity: {1 - dist:.3f})")


if __name__ == "__main__":
    main()
