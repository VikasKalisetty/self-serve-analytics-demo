import json
import os
from pathlib import Path
from typing import Any

import chromadb
import duckdb
import pandas as pd
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "lakehouse.duckdb"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "gemini-embedding-001"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=list(input),
        )
        return [emb.values for emb in response.embeddings]


# --- Structured Output Schemas ---

class GroundingPlan(BaseModel):
    is_answerable: bool = Field(description="False if the question is outside procurement/inventory data.")
    decline_reason: str = Field(default="", description="Polite explanation if is_answerable is False.")
    metric_name: str = Field(description="Primary KPI or metric being analyzed.")
    business_definition: str = Field(description="Plain-English definition of how the metric will be calculated.")
    formula_summary: str = Field(description="Formula logic to be applied.")
    mandatory_filters: list[str] = Field(default_factory=list, description="Key filters enforced (e.g. latest snapshot, non-void, active parts).")
    tables_to_use: list[str] = Field(default_factory=list, description="List of DuckDB tables needed.")
    join_strategy: str = Field(description="How tables will be joined (e.g. composite keys with company).")


class SQLGenerationResult(BaseModel):
    chain_of_thought: list[str] = Field(description="Step-by-step reasoning: tables, joins, filters, aggregations.")
    sql: str = Field(description="Valid, executable DuckDB SQL query (SELECT only, no markdown fences).")


class VizAndSummary(BaseModel):
    executive_summary: str = Field(description="2-sentence plain-English business takeaway from the query results.")
    chart_type: str = Field(description="One of: 'kpi_card', 'bar', 'line', 'table'.")
    x_col: str = Field(default="", description="Column name for x-axis (empty for kpi_card or table).")
    y_col: str = Field(default="", description="Numeric column name for y-axis (empty for table).")
    chart_title: str = Field(default="", description="Clean title for the chart.")


class AnalyticsEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")
        self.genai_client = genai.Client(api_key=api_key)
        self.embed_fn = GeminiEmbeddingFunction()
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.kpi_col = self.chroma_client.get_collection("enterprise_kpis", embedding_function=self.embed_fn)
        self.sem_col = self.chroma_client.get_collection("semantic_layer", embedding_function=self.embed_fn)

    def get_raw_schema(self) -> str:
        """Used when Enterprise Context is OFF: raw table & column names only, zero business rules."""
        con = duckdb.connect(str(DB_PATH), read_only=True)
        rows = con.execute("""
            SELECT table_name, string_agg(column_name || ' (' || data_type || ')', ', ') AS cols
            FROM information_schema.columns
            WHERE table_schema = 'main'
            GROUP BY table_name
            ORDER BY table_name
        """).fetchall()
        con.close()
        return "\n".join(f"- {tbl}: {cols}" for tbl, cols in rows)

    def retrieve_context(self, question: str) -> dict[str, Any]:
        """Retrieves Top-K KPIs, Semantic Layer tables/joins, and all Global Rules from ChromaDB."""
        kpi_res = self.kpi_col.query(query_texts=[question], n_results=3)
        sem_res = self.sem_col.query(query_texts=[question], n_results=7)
        rules_res = self.sem_col.get(ids=["RULE-001", "RULE-002", "RULE-003"])

        kpis = []
        for doc_id, doc, meta, dist in zip(
            kpi_res["ids"][0], kpi_res["documents"][0], kpi_res["metadatas"][0], kpi_res["distances"][0]
        ):
            kpis.append({
                "id": doc_id,
                "name": meta.get("kpi_name", doc_id),
                "version": meta.get("version", ""),
                "owner": meta.get("owner", ""),
                "similarity": round(1 - dist, 3),
                "content": doc,
            })

        sem_docs = []
        seen_ids = set()
        for doc_id, doc, meta, dist in zip(
            sem_res["ids"][0], sem_res["documents"][0], sem_res["metadatas"][0], sem_res["distances"][0]
        ):
            seen_ids.add(doc_id)
            sem_docs.append({
                "id": doc_id,
                "doc_type": meta.get("doc_type", "semantic"),
                "similarity": round(1 - dist, 3),
                "content": doc,
            })

        for doc_id, doc, meta in zip(rules_res["ids"], rules_res["documents"], rules_res["metadatas"]):
            if doc_id not in seen_ids:
                sem_docs.append({
                    "id": doc_id,
                    "doc_type": "rule",
                    "similarity": 1.0,
                    "content": doc,
                })

        return {"kpis": kpis, "semantic": sem_docs}

    def create_grounding_plan(self, question: str, context_on: bool) -> dict[str, Any]:
        """Step 1 & 2: Retrieve context (if ON) and generate a structured Grounding Verification Plan."""
        if context_on:
            retrieved = self.retrieve_context(question)
            kpi_block = "\n\n".join(k["content"] for k in retrieved["kpis"])
            sem_block = "\n\n".join(s["content"] for s in retrieved["semantic"])
            prompt = f"""
You are an Enterprise AI Analytics Architect.
A user asked: "{question}"

Use ONLY the governed Enterprise Context (KPI Definitions) and Semantic Layer below to plan the query.
If the question asks about data that does not exist in this Procurement & Inventory lakehouse (e.g., HR headcount, CRM leads, social media), set is_answerable=False and explain politely.

--- ENTERPRISE CONTEXT (KPI DEFINITIONS) ---
{kpi_block}

--- SEMANTIC LAYER (TABLES, JOINS, GLOBAL RULES) ---
{sem_block}

Return a structured GroundingPlan specifying the exact governed definition, formula, mandatory filters, tables, and composite join strategy.
"""
        else:
            retrieved = {"kpis": [], "semantic": []}
            raw_schema = self.get_raw_schema()
            prompt = f"""
You are a basic SQL assistant without access to an Enterprise Context or Semantic Layer.
A user asked: "{question}"

Here is the raw database schema (table and column names only):
{raw_schema}

If the question cannot be answered from these tables at all, set is_answerable=False.
Otherwise, make your best ungrounded guess at the metric definition, formula, and tables to use.
"""

        resp = self.genai_client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GroundingPlan,
                temperature=0.1,
            ),
        )
        plan = GroundingPlan.model_validate_json(resp.text)
        return {
            "plan": plan.model_dump(),
            "retrieved": retrieved,
            "context_on": context_on,
        }

    def generate_and_execute_sql(self, question: str, grounding_bundle: dict[str, Any]) -> dict[str, Any]:
        """Step 3, 4 & 5: Generate CoT + DuckDB SQL, execute with self-healing retries, and create viz + summary."""
        plan = grounding_bundle["plan"]
        retrieved = grounding_bundle["retrieved"]
        context_on = grounding_bundle["context_on"]
        raw_schema = self.get_raw_schema()

        if context_on:
            kpi_block = "\n\n".join(k["content"] for k in retrieved["kpis"])
            sem_block = "\n\n".join(s["content"] for s in retrieved["semantic"])
            context_section = f"""
--- GOVERNED ENTERPRISE CONTEXT ---
{kpi_block}

--- SEMANTIC LAYER ---
{sem_block}

--- CONFIRMED GROUNDING PLAN ---
{json.dumps(plan, indent=2)}
"""
            instruction = (
                "Strictly enforce ALL mandatory filters, exact KPI formulas, and multi-company composite joins "
                "(company = company) from the Enterprise Context and Confirmed Grounding Plan."
            )
        else:
            context_section = f"""
--- RAW DATABASE SCHEMA (NO ENTERPRISE CONTEXT) ---
{raw_schema}
"""
            instruction = (
                "Write a straightforward DuckDB SQL query based only on the raw schema above. "
                "Do not invent hidden business rules."
            )

        base_prompt = f"""
You are a DuckDB SQL expert.
User Question: "{question}"

{context_section}

Instructions:
- {instruction}
- Output ONLY a single valid DuckDB SELECT query (no markdown backticks, no comments at the end).
- Round dollar amounts and percentages to 2 decimal places using ROUND(..., 2).
- Limit output to at most 50 rows unless aggregating a small number of groups.
"""

        cot_steps: list[str] = []
        sql_query = ""
        df = pd.DataFrame()
        error_msg = ""

        # Execute with up to 2 self-healing retries
        for attempt in range(1, 4):
            prompt = base_prompt if attempt == 1 else (
                f"{base_prompt}\n\nPREVIOUS SQL FAILED WITH DUCKDB ERROR:\n"
                f"SQL:\n{sql_query}\nError:\n{error_msg}\nFix the SQL and explain the fix in chain_of_thought."
            )
            resp = self.genai_client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SQLGenerationResult,
                    temperature=0.0,
                ),
            )
            result = SQLGenerationResult.model_validate_json(resp.text)
            sql_query = result.sql.strip().rstrip(";")
            if attempt == 1:
                cot_steps = result.chain_of_thought
            else:
                cot_steps.append(f"[Self-Healing Retry #{attempt - 1}] Fixed error: {error_msg}")
                cot_steps.extend(result.chain_of_thought)

            try:
                con = duckdb.connect(str(DB_PATH), read_only=True)
                df = con.execute(sql_query).df()
                con.close()
                error_msg = ""
                break
            except Exception as e:
                error_msg = str(e)

        if error_msg:
            return {
                "success": False,
                "error": error_msg,
                "sql": sql_query,
                "chain_of_thought": cot_steps,
                "df": pd.DataFrame(),
                "viz": None,
            }

        # Step 5: Auto-Visualization & Executive Summary
        preview = df.head(15).to_markdown(index=False) if not df.empty else "No rows returned."
        viz_prompt = f"""
User Question: "{question}"
Returned Columns & Types: {dict(df.dtypes.astype(str))}
Row Count: {len(df)}
Sample Data:
{preview}

1. Write a concise 2-sentence executive summary answering the user's question with exact numbers from the data.
2. Choose the best chart_type:
   - 'kpi_card' if there is 1 row and 1-2 numeric metrics
   - 'line' if there is a date/month/snapshot_date column with multiple rows
   - 'bar' if there is a categorical column + numeric column (2 to 20 rows)
   - 'table' otherwise
"""
        viz_resp = self.genai_client.models.generate_content(
            model=LLM_MODEL,
            contents=viz_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VizAndSummary,
                temperature=0.2,
            ),
        )
        viz = VizAndSummary.model_validate_json(viz_resp.text).model_dump()

        return {
            "success": True,
            "error": "",
            "sql": sql_query,
            "chain_of_thought": cot_steps,
            "df": df,
            "viz": viz,
        }
