import json
import os
import re
from pathlib import Path
from typing import Any, TypeVar

import chromadb
import duckdb
import pandas as pd
import yaml
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from sarvamai import SarvamAI

load_dotenv()

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "lakehouse.duckdb"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-105b")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _kpi_label_variants(kpi: dict[str, Any]) -> list[tuple[str, list[str]]]:
    name = kpi["name"]
    labels = [
        name,
        re.sub(r"\([^)]*\)", "", name),
        *kpi.get("synonyms", []),
    ]
    variants = []
    seen = set()
    for label in labels:
        words = _tokens(label)
        if words and words[-1] in {"rate", "metric", "kpi"}:
            words = words[:-1]
        normalized = tuple(words)
        if normalized and normalized not in seen:
            seen.add(normalized)
            variants.append((label, words))
    return variants


def find_exact_kpi_match(
    question: str, kpi_definitions: list[dict[str, Any]]
) -> tuple[dict[str, Any], str] | None:
    question_tokens = _tokens(question)
    matches = []
    for kpi in kpi_definitions:
        for label, label_tokens in _kpi_label_variants(kpi):
            width = len(label_tokens)
            if any(
                question_tokens[index:index + width] == label_tokens
                for index in range(len(question_tokens) - width + 1)
            ):
                matches.append((width, kpi, label))

    if not matches:
        return None

    longest = max(width for width, _, _ in matches)
    best_matches = [match for match in matches if match[0] == longest]
    best_by_id = {match[1]["id"]: match for match in best_matches}
    if len(best_by_id) != 1:
        return None

    _, kpi, label = next(iter(best_by_id.values()))
    return kpi, label


def _format_kpi_document(kpi: dict[str, Any]) -> str:
    return (
        f"KPI: {kpi['name']} ({kpi['id']})\n"
        f"Synonyms: {', '.join(kpi.get('synonyms', []))}\n"
        f"Definition: {kpi['business_definition'].strip()}\n"
        f"Formula: {kpi['formula_logic'].strip()}\n"
        f"Mandatory Filters: {', '.join(kpi.get('mandatory_filters', [])) or 'None'}\n"
        f"Required Tables: {', '.join(kpi.get('required_tables', []))}\n"
        f"Notes: {kpi.get('notes', '').strip()}"
    )


def _kpi_selection_instruction(question: str, retrieved: dict[str, Any]) -> str:
    match = retrieved["kpi_match"]
    if match["is_exact"]:
        return (
            f"An exact KPI name/synonym match was found: {match['matched_label']!r} "
            f"for {match['matched_kpi_name']!r}. Use that KPI definition only; "
            "do not replace it with a more semantically similar KPI."
        )

    instruction = (
        "No KPI name or synonym exactly matches the requested metric. Treat every "
        "embedding-retrieved KPI below as reference material only, not as the selected "
        "KPI. Define the metric named in the user's question and carry over only the "
        "parts of a reference definition that measure that requested metric. Do not "
        "copy unrelated formula conditions just because they appear in a retrieved KPI."
    )
    question_tokens = _tokens(question)
    asks_on_time_delivery = (
        "otd" in question_tokens
        or any(
            question_tokens[index:index + 3] == ["on", "time", "delivery"]
            for index in range(len(question_tokens) - 2)
        )
    )
    has_otif_reference = any(
        "otif" in _tokens(kpi["name"])
        or any(
            _tokens(kpi["name"])[index:index + 2] == ["in", "full"]
            for index in range(len(_tokens(kpi["name"])) - 1)
        )
        for kpi in retrieved["kpis"]
    )
    if asks_on_time_delivery and has_otif_reference:
        instruction += (
            " For the requested On-Time Delivery metric derived from an OTIF reference, "
            "use the reference's posted-receipt filter and days_early_late BETWEEN -3 "
            "AND 3 timeliness condition, but omit is_complete_receipt and "
            "rejected_quantity = 0 because those are In-Full/quality criteria."
        )
    return instruction


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

class RewrittenQuery(BaseModel):
    is_followup: bool = Field(description="True if the user's message depends on prior conversation turns.")
    standalone_question: str = Field(description="Self-contained analytics question with all prior context resolved.")


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


T = TypeVar("T", bound=BaseModel)


class AnalyticsEngine:
    def __init__(self):
        sarvam_api_key = os.getenv("SARVAM_API_KEY")
        if not sarvam_api_key:
            raise ValueError("SARVAM_API_KEY not found. Add it to your .env file.")
        self.sarvam_client = SarvamAI(api_subscription_key=sarvam_api_key)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")
        self.genai_client = genai.Client(api_key=api_key)
        self.embed_fn = GeminiEmbeddingFunction()
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.kpi_col = self.chroma_client.get_collection("enterprise_kpis", embedding_function=self.embed_fn)
        self.sem_col = self.chroma_client.get_collection("semantic_layer", embedding_function=self.embed_fn)
        with (BASE_DIR / "context" / "kpi_definitions.yaml").open(encoding="utf-8") as file:
            self.kpi_definitions = yaml.safe_load(file)["kpis"]
        self.kpi_definitions_by_id = {kpi["id"]: kpi for kpi in self.kpi_definitions}

    def _generate_structured_response(
        self, prompt: str, response_model: type[T], temperature: float
    ) -> T:
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        response = self.sarvam_client.chat.completions(
            model=SARVAM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return exactly one valid JSON object that conforms to the JSON Schema "
                        "provided in the user message. Do not include markdown or extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nRequired JSON Schema:\n{schema}",
                },
            ],
            temperature=temperature,
            reasoning_effort=None,
            max_tokens=4096,
        )
        choice = next(iter(response.choices), None)
        if choice is None:
            raise ValueError("Sarvam returned no completion choices.")

        message = choice.message
        content = message.content
        if not isinstance(content, str) or not content.strip():
            finish_reason = getattr(choice, "finish_reason", "unknown")
            reasoning_present = bool(getattr(message, "reasoning_content", None))
            raise ValueError(
                "Sarvam returned no visible content "
                f"(finish_reason={finish_reason}, reasoning_content_present={reasoning_present})."
            )
        # Strip accidental markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        return response_model.model_validate_json(cleaned)

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

    def rewrite_question(
        self, question: str, chat_history: list[dict[str, Any]]
    ) -> RewrittenQuery:
        """Resolves follow-up prompts (e.g. 'filter by company') into a standalone question using chat history."""
        if not chat_history:
            return RewrittenQuery(is_followup=False, standalone_question=question)

        history_lines = []
        for m in chat_history[-6:]:
            if m["role"] == "user":
                history_lines.append(f"User: {m['content']}")
            elif not m.get("declined"):
                metric = m.get("plan", {}).get("metric_name", "")
                sql = m.get("sql", "")
                history_lines.append(
                    f"Assistant (Metric: {metric}): {m.get('summary', '')}\nPrevious SQL: {sql}"
                )

        if not history_lines:
            return RewrittenQuery(is_followup=False, standalone_question=question)

        history_text = "\n".join(history_lines)
        prompt = f"""
You are a conversational query rewriter for an enterprise analytics assistant.
Given the conversation history and the user's latest message, rewrite the latest message into a
complete, standalone analytics question.
- If the latest message is a follow-up (e.g., "filter down by company", "only for COMP01", "break that down by vendor"),
  merge it with the prior metric, entities, and filters from the most recent turn.
- If the latest message is a brand new analytics question, return it unchanged and set is_followup=False.

--- CONVERSATION HISTORY ---
{history_text}

--- LATEST USER MESSAGE ---
{question}
"""
        return self._generate_structured_response(prompt, RewrittenQuery, temperature=0.0)

    def retrieve_context(self, question: str) -> dict[str, Any]:
        """Uses exact KPI names first and semantic retrieval only for reference KPIs."""
        exact_match = find_exact_kpi_match(question, self.kpi_definitions)
        if exact_match:
            matched_kpi, matched_label = exact_match
            kpis = [{
                "id": matched_kpi["id"],
                "name": matched_kpi["name"],
                "version": matched_kpi.get("version", ""),
                "owner": matched_kpi.get("owner", ""),
                "similarity": 1.0,
                "match_type": "exact",
                "content": _format_kpi_document(matched_kpi),
            }]
            kpi_match = {
                "is_exact": True,
                "matched_kpi_id": matched_kpi["id"],
                "matched_kpi_name": matched_kpi["name"],
                "matched_label": matched_label,
            }
        else:
            kpi_res = self.kpi_col.query(query_texts=[question], n_results=3)
            kpis = []
            for doc_id, meta, dist in zip(
                kpi_res["ids"][0], kpi_res["metadatas"][0], kpi_res["distances"][0]
            ):
                definition = self.kpi_definitions_by_id.get(doc_id)
                if definition is None:
                    continue
                kpis.append({
                    "id": doc_id,
                    "name": definition["name"],
                    "version": definition.get("version", ""),
                    "owner": definition.get("owner", ""),
                    "similarity": round(1 - dist, 3),
                    "match_type": "reference",
                    "content": _format_kpi_document(definition),
                })
            kpi_match = {
                "is_exact": False,
                "matched_kpi_id": "",
                "matched_kpi_name": "",
                "matched_label": "",
            }

        sem_res = self.sem_col.query(query_texts=[question], n_results=7)
        rules_res = self.sem_col.get(ids=["RULE-001", "RULE-002", "RULE-003"])

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

        return {"kpis": kpis, "semantic": sem_docs, "kpi_match": kpi_match}

    def create_grounding_plan(
        self,
        question: str,
        context_on: bool,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Step 0, 1 & 2: Rewrite follow-up, retrieve context (if ON), and generate Grounding Plan."""
        rewritten = self.rewrite_question(question, chat_history or [])
        resolved_q = rewritten.standalone_question

        if context_on:
            retrieved = self.retrieve_context(resolved_q)
            kpi_block = "\n\n".join(k["content"] for k in retrieved["kpis"])
            sem_block = "\n\n".join(s["content"] for s in retrieved["semantic"])
            kpi_selection = _kpi_selection_instruction(resolved_q, retrieved)
            prompt = f"""
You are an Enterprise AI Analytics Architect.
A user asked: "{resolved_q}"

Use the KPI Selection Guardrail below to decide whether a retrieved KPI is an exact match or reference-only.
Use the Semantic Layer for available tables, joins, and global rules.
If the question asks about data that does not exist in this Procurement & Inventory lakehouse (e.g., HR headcount, CRM leads, social media), set is_answerable=False and explain politely.

--- KPI SELECTION GUARDRAIL ---
{kpi_selection}

--- ENTERPRISE CONTEXT (KPI DEFINITIONS) ---
{kpi_block}

--- SEMANTIC LAYER (TABLES, JOINS, GLOBAL RULES) ---
{sem_block}

If an exact KPI match exists, return its governed definition and formula. Otherwise, name and define the metric requested by the user, deriving only relevant logic from reference KPIs. Return its formula, mandatory filters, tables, and composite join strategy.
"""
        else:
            retrieved = {
                "kpis": [],
                "semantic": [],
                "kpi_match": {
                    "is_exact": False,
                    "matched_kpi_id": "",
                    "matched_kpi_name": "",
                    "matched_label": "",
                },
            }
            raw_schema = self.get_raw_schema()
            prompt = f"""
You are a basic SQL assistant without access to an Enterprise Context or Semantic Layer.
A user asked: "{resolved_q}"

Here is the raw database schema (table and column names only):
{raw_schema}

If the question cannot be answered from these tables at all, set is_answerable=False.
Otherwise, make your best ungrounded guess at the metric definition, formula, and tables to use.
"""

        plan = self._generate_structured_response(prompt, GroundingPlan, temperature=0.1)
        return {
            "plan": plan.model_dump(),
            "retrieved": retrieved,
            "context_on": context_on,
            "standalone_question": resolved_q,
            "is_followup": rewritten.is_followup,
        }

    def generate_and_execute_sql(self, question: str, grounding_bundle: dict[str, Any]) -> dict[str, Any]:
        """Step 3, 4 & 5: Generate CoT + DuckDB SQL, execute with self-healing retries, and create viz + summary."""
        resolved_q = grounding_bundle.get("standalone_question", question)
        plan = grounding_bundle["plan"]
        retrieved = grounding_bundle["retrieved"]
        context_on = grounding_bundle["context_on"]
        raw_schema = self.get_raw_schema()

        if context_on:
            kpi_block = "\n\n".join(k["content"] for k in retrieved["kpis"])
            sem_block = "\n\n".join(s["content"] for s in retrieved["semantic"])
            kpi_selection = _kpi_selection_instruction(resolved_q, retrieved)
            context_section = f"""
--- KPI SELECTION GUARDRAIL ---
{kpi_selection}

--- KPI CONTEXT (EXACT MATCH OR REFERENCE ONLY AS INDICATED ABOVE) ---
{kpi_block}

--- SEMANTIC LAYER ---
{sem_block}

--- CONFIRMED GROUNDING PLAN ---
{json.dumps(plan, indent=2)}
"""
            if retrieved["kpi_match"]["is_exact"]:
                instruction = (
                    "Enforce the exact-matched KPI's formula, mandatory filters, and multi-company "
                    "composite joins (company = company) from the Confirmed Grounding Plan."
                )
            else:
                instruction = (
                    "Implement the requested derived metric from the Confirmed Grounding Plan. "
                    "Treat retrieved KPI formulas as reference-only; do not apply unrelated conditions. "
                    "Preserve multi-company composite joins (company = company) where applicable."
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
User Question: "{resolved_q}"

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
            result = self._generate_structured_response(
                prompt, SQLGenerationResult, temperature=0.0
            )
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
User Question: "{resolved_q}"
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
        viz = self._generate_structured_response(
            viz_prompt, VizAndSummary, temperature=0.2
        ).model_dump()

        return {
            "success": True,
            "error": "",
            "sql": sql_query,
            "chain_of_thought": cot_steps,
            "df": df,
            "viz": viz,
        }
