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

from index_context import get_categorical_value_context

load_dotenv()

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "lakehouse.duckdb"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-105b")
MAX_DIMENSION_VALUES = 30


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _quote_sql_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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
    columns_to_use: list[str] = Field(default_factory=list, description="Fully-qualified table.column names needed for the metric, filters, grouping, and joins.")
    join_strategy: str = Field(description="How tables will be joined (e.g. composite keys with company).")
    grounding_considerations: list[str] = Field(default_factory=list, description="Visible explanation of KPI selection, table selection, column selection, and applicable semantic rules.")


class ContextRouting(BaseModel):
    is_answerable: bool = Field(description="False if no selected lakehouse table can answer the question.")
    decline_reason: str = Field(default="", description="Polite explanation if the question is not answerable.")
    reference_kpi_ids: list[str] = Field(default_factory=list, description="Only materially relevant existing KPI IDs to use as reference; empty when none applies.")
    tables_to_use: list[str] = Field(default_factory=list, description="Lakehouse tables required to answer the question.")
    selection_reason: str = Field(description="Brief reason for the KPI reference decision and table selection.")


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
        with (BASE_DIR / "context" / "semantic_layer.yaml").open(encoding="utf-8") as file:
            self.semantic_layer = yaml.safe_load(file)

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

    def _table_catalog_for_routing(self) -> str:
        tables = "\n\n".join(
            "\n".join([
                f"Table: {table['name']} ({table['type']})",
                f"Aliases: {', '.join(table.get('aliases', [])) or 'None'}",
                f"Grain: {table['grain']}",
                f"Description: {table['description']}",
                "Semantic columns: " + "; ".join(table.get("key_columns", [])),
            ])
            for table in self.semantic_layer["tables"]
        )
        joins = "\n".join(
            f"- {join['id']}: {join['left_table']} JOIN {join['right_table']} "
            f"ON {join['join_condition']}"
            for join in self.semantic_layer["joins"]
        )
        return f"{tables}\n\nCanonical join paths:\n{joins}"

    def format_column_grain_catalog(
        self,
        table: dict[str, Any],
        physical_columns: list[tuple[Any, ...]] | None = None,
    ) -> list[str]:
        table_name = table["name"]
        default_grain = table.get("default_column_grain", table["grain"])
        overrides = table.get("column_grains", {})
        if physical_columns is None:
            con = duckdb.connect(str(DB_PATH), read_only=True)
            try:
                physical_columns = con.execute(
                    f"DESCRIBE SELECT * FROM {_quote_sql_identifier(table_name)}"
                ).fetchall()
            finally:
                con.close()

        return [
            f"{table_name}.{column[0]} — {overrides.get(column[0], default_grain)}"
            for column in physical_columns
        ]

    def _selected_table_context(
        self, table_names: list[str]
    ) -> tuple[str, list[dict[str, Any]], set[str]]:
        table_definitions = {
            table["name"]: table for table in self.semantic_layer["tables"]
        }
        unknown_tables = set(table_names) - set(table_definitions)
        if unknown_tables:
            raise ValueError(
                "Selected tables are not in the semantic layer: "
                + ", ".join(sorted(unknown_tables))
            )

        con = duckdb.connect(str(DB_PATH), read_only=True)
        documents = []
        allowed_columns: set[str] = set()
        try:
            for table_name in table_names:
                table = table_definitions[table_name]
                columns = con.execute(
                    f"DESCRIBE SELECT * FROM {_quote_sql_identifier(table_name)}"
                ).fetchall()
                column_grain_catalog = self.format_column_grain_catalog(table, columns)
                allowed_columns.update(
                    f"{table_name}.{column[0]}" for column in columns
                )
                document = "\n".join([
                    f"Table: {table_name} ({table['type']})",
                    f"Aliases: {', '.join(table.get('aliases', [])) or 'None'}",
                    f"Grain: {table['grain']}",
                    f"Description: {table['description']}",
                    "Column granularities:\n- " + "\n- ".join(column_grain_catalog),
                    "Semantic column descriptions:\n- "
                    + "\n- ".join(table.get("key_columns", [])),
                    "Physical columns and types:\n- "
                    + "\n- ".join(
                        f"{column[0]} ({column[1]})" for column in columns
                    ),
                    get_categorical_value_context(con, table).strip(),
                ]).strip()
                documents.append({
                    "id": f"TABLE-{table_name}",
                    "doc_type": "table",
                    "similarity": 1.0,
                    "content": document,
                })
        finally:
            con.close()

        selected = set(table_names)
        selected_joins = [
            join for join in self.semantic_layer["joins"]
            if join["left_table"] in selected and join["right_table"] in selected
        ]
        for join in selected_joins:
            documents.append({
                "id": join["id"],
                "doc_type": "join",
                "similarity": 1.0,
                "content": (
                    f"Canonical Join ({join['id']}): {join['left_table']} JOIN "
                    f"{join['right_table']}\nON {join['join_condition']}\n"
                    f"Usage: {join['usage']}"
                ),
            })
            allowed_columns.update(
                f"{table}.{column}"
                for table, column in re.findall(
                    r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
                    join["join_condition"],
                )
            )

        for rule in self.semantic_layer["global_rules"]:
            documents.append({
                "id": rule["id"],
                "doc_type": "rule",
                "similarity": 1.0,
                "content": (
                    f"Global SQL Rule ({rule['id']} - {rule['title']}): "
                    f"{rule['description'].strip()}"
                ),
            })

        return (
            "\n\n".join(doc["content"] for doc in documents),
            documents,
            allowed_columns,
        )

    def find_dimension_exploration(self, question: str) -> dict[str, Any] | None:
        question_tokens = _tokens(question)
        asks_for_dimensions = any(
            token in {
                "dimension", "dimensions", "attribute", "attributes", "fields",
            }
            for token in question_tokens
        ) and any(
            token in {
                "what", "which", "list", "show", "available", "relevant",
                "possible", "describe", "provide",
            }
            for token in question_tokens
        )
        listing_phrases = (
            ["what", "categories"],
            ["which", "categories"],
            ["available", "categories"],
            ["list", "categories"],
            ["what", "values"],
            ["which", "values"],
            ["available", "values"],
            ["list", "values"],
        )
        asks_for_categories = any(
            question_tokens[index:index + len(phrase)] == phrase
            for phrase in listing_phrases
            for index in range(len(question_tokens) - len(phrase) + 1)
        )
        if not asks_for_dimensions and not asks_for_categories:
            return None

        for exploration in self.semantic_layer.get("dimension_explorations", []):
            for term in exploration.get("match_terms", []):
                term_tokens = _tokens(term)
                if any(
                    question_tokens[index:index + len(term_tokens)] == term_tokens
                    for index in range(len(question_tokens) - len(term_tokens) + 1)
                ):
                    return exploration

        table_matches = []
        for table in self.semantic_layer["tables"]:
            aliases = [table["name"], *table.get("aliases", [])]
            for alias in aliases:
                alias_tokens = _tokens(alias)
                if any(
                    question_tokens[index:index + len(alias_tokens)] == alias_tokens
                    for index in range(len(question_tokens) - len(alias_tokens) + 1)
                ):
                    table_matches.append((len(alias_tokens), table))

        selected_table = None
        if table_matches:
            longest = max(match[0] for match in table_matches)
            best_tables = {
                table["name"]: table
                for width, table in table_matches
                if width == longest
            }
            if len(best_tables) == 1:
                selected_table = next(iter(best_tables.values()))

        tables = [selected_table] if selected_table else self.semantic_layer["tables"]
        return {
            "id": f"TABLE-DIMENSIONS-{selected_table['name']}" if selected_table else "ALL-TABLE-DIMENSIONS",
            "name": f"{selected_table['name']} dimensions" if selected_table else "Lakehouse dimensions",
            "description": (
                "Lists the semantic layer's categorical dimensions and their distinct values "
                "from the current lakehouse data."
            ),
            "base_table": None,
            "tables": tables,
            "dimensions": [
                {
                    "name": f"{table['name']}.{column.replace('_', ' ').title()}",
                    "table": table["name"],
                    "column": column,
                    "grain": table["grain"],
                }
                for table in tables
                for column in table.get("categorical_columns", [])
            ],
        }

    def run_dimension_exploration(
        self, exploration: dict[str, Any]
    ) -> tuple[pd.DataFrame, str, str]:
        if exploration.get("base_table") is None:
            return self._run_table_dimension_exploration(exploration)

        base_table = exploration["base_table"]
        joins_by_id = {join["id"]: join for join in self.semantic_layer["joins"]}
        source_tables = {base_table} | {
            dimension.get("table", base_table)
            for dimension in exploration["dimensions"]
        }
        con = duckdb.connect(str(DB_PATH), read_only=True)
        rows = []
        sql_statements = []

        try:
            table_columns = {
                table: {
                    row[0]
                    for row in con.execute(
                        f"DESCRIBE SELECT * FROM {_quote_sql_identifier(table)}"
                    ).fetchall()
                }
                for table in source_tables
            }

            for dimension in exploration["dimensions"]:
                source_table = dimension.get("table", base_table)
                joins = ""
                if source_table != base_table:
                    join = joins_by_id.get(dimension.get("join_id", ""))
                    if (
                        join is None
                        or join["left_table"] != base_table
                        or join["right_table"] != source_table
                    ):
                        raise ValueError(
                            f"Missing valid semantic join for {base_table} to {source_table}."
                        )
                    joins = (
                        f" LEFT JOIN {join['right_table']} "
                        f"ON {join['join_condition']}"
                    )

                if dimension.get("derived") == "void_status":
                    status_columns = ("release_status", "po_status")
                    missing = set(status_columns) - table_columns[base_table]
                    if missing:
                        raise ValueError(
                            f"{base_table} is missing columns required for void status: "
                            f"{', '.join(sorted(missing))}."
                        )
                    conditions = [
                        f"UPPER(TRIM(COALESCE(CAST({base_table}.{_quote_sql_identifier(column)} "
                        f"AS VARCHAR), ''))) = 'VOID'"
                        for column in status_columns
                    ]
                    value_expr = f"CASE WHEN {' OR '.join(conditions)} THEN 'Void' ELSE 'Non-void' END"
                else:
                    column = dimension["column"]
                    if column not in table_columns[source_table]:
                        raise ValueError(f"{source_table}.{column} is not present in the lakehouse.")
                    value_expr = f"{source_table}.{_quote_sql_identifier(column)}"

                sql = (
                    f"SELECT {_sql_string(dimension['name'])} AS dimension, "
                    f"{_sql_string(dimension['grain'])} AS grain, "
                    f"COALESCE(CAST({value_expr} AS VARCHAR), 'Unknown') AS value "
                    f"FROM {_quote_sql_identifier(base_table)}{joins} "
                    "GROUP BY 1, 2, 3 ORDER BY 3"
                    f" LIMIT {MAX_DIMENSION_VALUES + 1}"
                )
                sql_statements.append(sql)
                dimension_rows = con.execute(sql).fetchall()
                value_labels = {
                    str(key).casefold(): value
                    for key, value in dimension.get("value_labels", {}).items()
                }
                values = [
                    value_labels.get(str(row[2]).casefold(), row[2])
                    for row in dimension_rows
                ]
                notes = dimension.get("notes", "")
                if len(values) > MAX_DIMENSION_VALUES:
                    values_text = "High cardinality; values not listed"
                else:
                    values_text = "; ".join(str(value) for value in values) or "No values"
                if dimension.get("derived") == "void_status" and "Void" not in values:
                    notes = (notes + " " if notes else "") + "No Void values are present in the current data."
                rows.append({
                    "Dimension": dimension["name"],
                    "Grain": dimension["grain"],
                    "Values": values_text,
                    "Notes": notes,
                })
        finally:
            con.close()

        result = pd.DataFrame(
            rows,
            columns=["Dimension", "Grain", "Values", "Notes"],
        )
        has_void_releases = (
            not result.empty
            and any(
                value.strip().casefold() == "void"
                for values in result.loc[result["Dimension"] == "Void status", "Values"]
                for value in values.split(";")
            )
        )
        summary = f"Distinct dimension values from the current lakehouse for {exploration['name']}."
        if not has_void_releases:
            if any(dimension["name"] == "Void status" for dimension in exploration["dimensions"]):
                summary += " No Void releases are present in the current data."
        return result, summary, "\n\n".join(sql_statements)

    def _run_table_dimension_exploration(
        self, exploration: dict[str, Any]
    ) -> tuple[pd.DataFrame, str, str]:
        rows = []
        sql_statements = []
        con = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            for dimension in exploration["dimensions"]:
                table = dimension["table"]
                column = dimension["column"]
                available_columns = {
                    row[0]
                    for row in con.execute(
                        f"DESCRIBE SELECT * FROM {_quote_sql_identifier(table)}"
                    ).fetchall()
                }
                if column not in available_columns:
                    raise ValueError(f"{table}.{column} is not present in the lakehouse.")

                sql = (
                    f"SELECT DISTINCT COALESCE(CAST({_quote_sql_identifier(column)} AS VARCHAR), "
                    "'Unknown') AS value "
                    f"FROM {_quote_sql_identifier(table)} ORDER BY 1 "
                    f"LIMIT {MAX_DIMENSION_VALUES + 1}"
                )
                sql_statements.append(sql)
                values = [row[0] for row in con.execute(sql).fetchall()]
                if len(values) > MAX_DIMENSION_VALUES:
                    values_text = "High cardinality; values not listed"
                else:
                    values_text = "; ".join(str(value) for value in values) or "No values"
                rows.append({
                    "Dimension": dimension["name"],
                    "Grain": dimension["grain"],
                    "Values": values_text,
                    "Notes": "",
                })
        finally:
            con.close()

        result = pd.DataFrame(
            rows,
            columns=["Dimension", "Grain", "Values", "Notes"],
        )
        return (
            result,
            f"Distinct dimension values from the current lakehouse for {exploration['name']}.",
            "\n\n".join(sql_statements),
        )

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
        """Finds an exact KPI first; otherwise returns candidates for relevance routing."""
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
            return {
                "kpis": kpis,
                "semantic": [],
                "kpi_match": kpi_match,
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

        return {
            "kpis": kpis,
            "semantic": [],
            "kpi_match": kpi_match,
        }

    def create_grounding_plan(
        self,
        question: str,
        context_on: bool,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Step 0, 1 & 2: Rewrite follow-up, retrieve context (if ON), and generate Grounding Plan."""
        if context_on:
            exploration = self.find_dimension_exploration(question)
            if exploration:
                df, summary, sql = self.run_dimension_exploration(exploration)
                if exploration.get("base_table") is None:
                    tables = {table["name"] for table in exploration["tables"]}
                else:
                    tables = {
                        exploration["base_table"],
                        *(dimension.get("table", exploration["base_table"])
                          for dimension in exploration["dimensions"]),
                    }
                return {
                    "plan": {
                        "is_answerable": True,
                        "decline_reason": "",
                        "metric_name": exploration["name"],
                        "business_definition": exploration["description"].strip(),
                        "formula_summary": "Distinct observed values per dimension; no row counts.",
                        "mandatory_filters": [],
                        "tables_to_use": sorted(tables),
                        "join_strategy": "Use the semantic layer's documented composite joins.",
                    },
                    "retrieved": {
                        "kpis": [],
                        "semantic": [{
                            "id": f"DIMENSION-{exploration['id']}",
                            "doc_type": "dimension_exploration",
                            "similarity": 1.0,
                            "content": exploration["description"].strip(),
                        }],
                        "kpi_match": {
                            "is_exact": False,
                            "matched_kpi_id": "",
                            "matched_kpi_name": "",
                            "matched_label": "",
                        },
                    },
                    "context_on": True,
                    "standalone_question": question,
                    "is_followup": False,
                    "semantic_exploration": {
                        "summary": summary,
                        "df": df,
                        "sql": sql,
                    },
                }

        rewritten = self.rewrite_question(question, chat_history or [])
        resolved_q = rewritten.standalone_question

        if context_on:
            retrieved = self.retrieve_context(resolved_q)
            exact_match = retrieved["kpi_match"]["is_exact"]
            candidate_kpis = retrieved["kpis"]
            candidate_ids = {kpi["id"] for kpi in candidate_kpis}
            catalog = self._table_catalog_for_routing()
            candidate_block = "\n\n".join(
                f"Candidate KPI ID: {kpi['id']}\n{kpi['content']}"
                for kpi in candidate_kpis
            ) or "No KPI candidates were retrieved."
            exact_kpi = candidate_kpis[0] if exact_match else None
            exact_required_tables = (
                self.kpi_definitions_by_id[exact_kpi["id"]].get("required_tables", [])
                if exact_kpi else []
            )
            routing_prompt = f"""
You are the context router for a procurement and inventory analytics lakehouse.
Question: "{resolved_q}"

Decide whether any retrieved KPI is materially relevant to the requested metric.
- If an exact KPI is identified below, it is locked as the governed KPI. Do not replace
  it or return a different KPI reference.
- Otherwise, use a candidate KPI only as a reference when its definition or formula
  genuinely helps derive the requested metric. Choose no references when all candidates
  are merely the nearest semantic matches but do not help answer the question.
- Choose only tables from the supplied catalog that can answer the question. Include
  tables needed for requested grouping/filter dimensions as well as KPI facts.
- If a relevant KPI has required tables, include those tables. Do not invent table names
  or KPI IDs. Mark out of scope only when no catalog table can answer the question.

Exact KPI, if present: {exact_kpi['id'] if exact_kpi else 'none'}
Required tables for exact KPI: {exact_required_tables or 'none'}

--- KPI CANDIDATES ---
{candidate_block}

--- SEMANTIC TABLE CATALOG (DESCRIPTIONS ONLY; NO PHYSICAL COLUMNS YET) ---
{catalog}
"""
            routing = self._generate_structured_response(
                routing_prompt, ContextRouting, temperature=0.0
            ).model_dump()
            valid_table_names = {
                table["name"] for table in self.semantic_layer["tables"]
            }
            selected_reference_ids = (
                [exact_kpi["id"]]
                if exact_kpi
                else [
                    kpi_id for kpi_id in routing["reference_kpi_ids"]
                    if kpi_id in candidate_ids
                ]
            )
            invalid_reference_ids = (
                set(routing["reference_kpi_ids"]) - candidate_ids
                if not exact_kpi else set()
            )
            selected_kpis = [
                kpi for kpi in candidate_kpis
                if kpi["id"] in selected_reference_ids
            ]
            required_tables = [
                table_name
                for kpi_id in selected_reference_ids
                for table_name in self.kpi_definitions_by_id[kpi_id].get("required_tables", [])
            ]
            invalid_required_tables = set(required_tables) - valid_table_names
            if invalid_required_tables:
                raise ValueError(
                    "KPI definitions reference tables missing from the semantic layer: "
                    + ", ".join(sorted(invalid_required_tables))
                )
            requested_tables = [
                table_name for table_name in routing["tables_to_use"]
                if table_name in valid_table_names
            ]
            invalid_selected_tables = (
                set(routing["tables_to_use"]) - valid_table_names
            )
            selected_table_names = list(dict.fromkeys([
                *requested_tables,
                *required_tables,
            ]))
            if not routing["is_answerable"] or not selected_table_names:
                reason = routing["decline_reason"] or (
                    "I could not identify a lakehouse table that can answer this question."
                )
                if invalid_selected_tables:
                    reason += (
                        " The table selector returned names outside the semantic layer: "
                        + ", ".join(sorted(invalid_selected_tables))
                        + "."
                    )
                plan = {
                    "is_answerable": False,
                    "decline_reason": reason,
                    "metric_name": resolved_q,
                    "business_definition": "",
                    "formula_summary": "",
                    "mandatory_filters": [],
                    "tables_to_use": [],
                    "columns_to_use": [],
                    "join_strategy": "",
                    "grounding_considerations": [routing["selection_reason"]],
                }
                retrieved["kpis"] = selected_kpis
                retrieved["routing"] = routing
                return {
                    "plan": plan,
                    "retrieved": retrieved,
                    "context_on": context_on,
                    "standalone_question": resolved_q,
                    "is_followup": rewritten.is_followup,
                }

            semantic_context, semantic_docs, allowed_columns = (
                self._selected_table_context(selected_table_names)
            )
            retrieved["kpis"] = selected_kpis
            retrieved["semantic"] = semantic_docs
            retrieved["routing"] = routing
            kpi_block = "\n\n".join(kpi["content"] for kpi in selected_kpis)
            kpi_selection = _kpi_selection_instruction(resolved_q, retrieved)
            if exact_kpi:
                selection_note = (
                    f"Exact KPI match: {exact_kpi['name']} ({exact_kpi['id']}); "
                    "its governed definition and formula are locked."
                )
            elif selected_kpis:
                selection_note = (
                    "No exact KPI matched. The following KPI(s) were selected as "
                    "reference-only and may be adapted only where relevant: "
                    + ", ".join(kpi["name"] for kpi in selected_kpis)
                    + "."
                )
            else:
                selection_note = (
                    "No exact or materially relevant KPI was found. Define the requested "
                    "metric from the question and selected table context; do not borrow "
                    "formula conditions from unrelated KPI candidates."
                )
            plan_prompt = f"""
You are an enterprise analytics grounding planner.
Question: "{resolved_q}"

--- KPI SELECTION ---
{selection_note}
{kpi_selection}

--- SELECTED KPI DEFINITION(S) ---
{kpi_block or "None. Do not infer a KPI from unrelated candidates."}

--- TABLE ROUTING DECISION ---
{routing["selection_reason"]}
Use only these selected tables: {selected_table_names}

--- SELECTED TABLE DESCRIPTIONS, PHYSICAL COLUMNS, DATA VALUES, JOINS, AND RULES ---
{semantic_context}

Ground the requested metric against the selected tables. Select only physical columns
present in the supplied schemas. In columns_to_use, return fully-qualified names in the
form table.column for every metric, filter, grouping, and join-key column needed.
Return tables_to_use from the selected table list only.
- For an exact KPI, preserve its governed name, definition, formula, and mandatory filters.
- For a reference KPI, define the metric the user actually asked for and reuse only
  relevant logic; do not copy unrelated conditions from the reference.
- If no KPI is relevant, define the metric from the question and available semantic
  context without fabricating a governed KPI.
- Explain the KPI decision, table selection, selected columns, and semantic rules as
  concise grounding_considerations that can be shown to the user.
"""
            plan = self._generate_structured_response(
                plan_prompt, GroundingPlan, temperature=0.1
            ).model_dump()
            plan["tables_to_use"] = selected_table_names
            invalid_columns = [
                column for column in plan["columns_to_use"]
                if column not in allowed_columns
            ]
            if plan["is_answerable"] and (
                invalid_columns or not plan["columns_to_use"]
            ):
                details = (
                    "Unknown columns: " + ", ".join(invalid_columns)
                    if invalid_columns
                    else "No columns were selected."
                )
                plan["is_answerable"] = False
                plan["decline_reason"] = (
                    "The grounding planner could not produce a valid schema-grounded "
                    f"column list. {details}"
                )
            if exact_kpi:
                definition = self.kpi_definitions_by_id[exact_kpi["id"]]
                plan["metric_name"] = definition["name"]
                plan["business_definition"] = definition["business_definition"].strip()
                plan["formula_summary"] = definition["formula_logic"].strip()
                plan["mandatory_filters"] = definition.get("mandatory_filters", [])
            deterministic_considerations = [
                selection_note,
                f"Table selection: {routing['selection_reason']}",
                (
                    "Schema-validated columns: "
                    + ", ".join(plan["columns_to_use"])
                    if plan["columns_to_use"]
                    else "Column selection failed validation; SQL generation will not proceed."
                ),
            ]
            if invalid_reference_ids:
                deterministic_considerations.append(
                    "Ignored unknown KPI reference IDs returned by the router: "
                    + ", ".join(sorted(invalid_reference_ids))
                    + "."
                )
            if invalid_selected_tables:
                deterministic_considerations.append(
                    "Ignored unknown table names returned by the router: "
                    + ", ".join(sorted(invalid_selected_tables))
                    + "."
                )
            plan["grounding_considerations"] = list(dict.fromkeys([
                *deterministic_considerations,
                *plan["grounding_considerations"],
            ]))
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
Otherwise, make your best ungrounded guess at the metric definition and formula.
Return tables_to_use and fully-qualified table.column values in columns_to_use.
Explain your table and column choices in grounding_considerations.
"""
            plan = self._generate_structured_response(
                prompt, GroundingPlan, temperature=0.1
            ).model_dump()
        return {
            "plan": plan,
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

--- FINALIZED COLUMN ALLOWLIST ---
{", ".join(plan.get("columns_to_use", [])) or "No columns were finalized."}
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

--- FINALIZED COLUMN ALLOWLIST ---
{", ".join(plan.get("columns_to_use", [])) or "No columns were finalized."}
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
- Use only the finalized columns above; documented composite join keys may also be used
  for the supplied canonical joins. Do not invent columns or switch to unselected tables.
- Respect each selected column's declared grain. Before counting or deduplicating rows,
  use the complete primary/composite grain; do not count only po_release_number,
  pack_slip, vendor_num, or another child identifier unless it is the actual requested grain.
- For PO releases, count or SELECT DISTINCT complete release identities using
  company + po_number + po_line + po_release_number. For receipts use
  company + pack_slip + pack_line. Preserve snapshot_date for inventory snapshot grains.
- For user-supplied string filters, use the canonical values listed in the data context. Match
  case-insensitively with LOWER(TRIM(column)) = LOWER(TRIM('user value')); use exact matching
  rather than wildcards unless the user asks for a partial match.
- Do not wrap or normalize columns in JOIN conditions. Preserve the documented composite-key joins.
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
