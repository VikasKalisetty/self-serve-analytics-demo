import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import yaml

from engine import AnalyticsEngine, ContextRouting, GroundingPlan


BASE_DIR = Path(__file__).resolve().parents[1]


class FakeKpiCollection:
    def __init__(self, kpi_ids: list[str]):
        self.kpi_ids = kpi_ids

    def query(self, **kwargs):
        return {
            "ids": [self.kpi_ids],
            "metadatas": [[{} for _ in self.kpi_ids]],
            "distances": [[0.08 for _ in self.kpi_ids]],
        }


class GroundingFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.semantic_layer = yaml.safe_load(
            (BASE_DIR / "context" / "semantic_layer.yaml").read_text(encoding="utf-8")
        )
        cls.kpi_definitions = yaml.safe_load(
            (BASE_DIR / "context" / "kpi_definitions.yaml").read_text(encoding="utf-8")
        )["kpis"]

    def make_engine(self, kpi_ids: list[str] | None = None) -> AnalyticsEngine:
        engine = object.__new__(AnalyticsEngine)
        engine.semantic_layer = self.semantic_layer
        engine.kpi_definitions = self.kpi_definitions
        engine.kpi_definitions_by_id = {
            kpi["id"]: kpi for kpi in self.kpi_definitions
        }
        if kpi_ids is not None:
            engine.kpi_col = FakeKpiCollection(kpi_ids)
        return engine

    @staticmethod
    def plan(columns: list[str]) -> GroundingPlan:
        return GroundingPlan(
            is_answerable=True,
            metric_name="Model metric",
            business_definition="Model definition",
            formula_summary="Model formula",
            mandatory_filters=[],
            tables_to_use=[],
            columns_to_use=columns,
            join_strategy="Use documented joins.",
            grounding_considerations=["The model selected these columns."],
        )

    def test_exact_kpi_uses_governed_definition_and_selected_schema(self):
        engine = self.make_engine()
        routing = ContextRouting(
            is_answerable=True,
            tables_to_use=["fct_purchase_orders"],
            selection_reason="Open PO value is measured from PO releases.",
        )
        generated_plan = self.plan([
            "fct_purchase_orders.open_quantity",
            "fct_purchase_orders.unit_price",
        ])
        with patch.object(
            engine,
            "_generate_structured_response",
            side_effect=[routing, generated_plan],
        ) as generate:
            bundle = engine.create_grounding_plan(
                "What is Open PO Value by company?", context_on=True
            )

        definition = engine.kpi_definitions_by_id["KPI-004"]
        self.assertEqual(
            [call.args[1] for call in generate.call_args_list],
            [ContextRouting, GroundingPlan],
        )
        self.assertEqual(bundle["plan"]["metric_name"], definition["name"])
        self.assertEqual(
            bundle["plan"]["formula_summary"], definition["formula_logic"].strip()
        )
        self.assertEqual(bundle["plan"]["mandatory_filters"], definition["mandatory_filters"])
        self.assertIn(
            "fct_purchase_orders.open_quantity", bundle["plan"]["columns_to_use"]
        )
        self.assertTrue(any(
            item.startswith("Schema-validated columns:")
            for item in bundle["plan"]["grounding_considerations"]
        ))

    def test_related_kpi_is_reference_and_table_join_context_is_selected(self):
        engine = self.make_engine(["KPI-002"])
        routing = ContextRouting(
            is_answerable=True,
            reference_kpi_ids=["KPI-002"],
            tables_to_use=["fct_po_receipts", "dim_vendor"],
            selection_reason="Receipt timing and supplier tier answer the request.",
        )
        generated_plan = self.plan([
            "fct_po_receipts.days_early_late",
            "fct_po_receipts.receipt_status",
            "dim_vendor.supplier_tier",
        ])
        with patch.object(
            engine,
            "_generate_structured_response",
            side_effect=[routing, generated_plan],
        ):
            bundle = engine.create_grounding_plan(
                "What is On-Time Delivery by supplier tier?", context_on=True
            )

        self.assertEqual([kpi["id"] for kpi in bundle["retrieved"]["kpis"]], ["KPI-002"])
        self.assertEqual(bundle["retrieved"]["kpis"][0]["match_type"], "reference")
        self.assertIn("JOIN-05", {
            document["id"] for document in bundle["retrieved"]["semantic"]
        })

    def test_unrelated_kpi_candidate_is_not_forwarded_as_reference(self):
        engine = self.make_engine(["KPI-002"])
        routing = ContextRouting(
            is_answerable=True,
            reference_kpi_ids=[],
            tables_to_use=["dim_vendor"],
            selection_reason="Supplier status is available in the vendor dimension.",
        )
        generated_plan = self.plan(["dim_vendor.vendor_status"])
        with patch.object(
            engine,
            "_generate_structured_response",
            side_effect=[routing, generated_plan],
        ):
            bundle = engine.create_grounding_plan(
                "List active supplier statuses", context_on=True
            )

        self.assertEqual(bundle["retrieved"]["kpis"], [])
        self.assertIn(
            "No exact or materially relevant KPI was found",
            bundle["plan"]["grounding_considerations"][0],
        )

    def test_out_of_scope_route_stops_before_column_planning(self):
        engine = self.make_engine(["KPI-002"])
        routing = ContextRouting(
            is_answerable=False,
            decline_reason="This lakehouse has no CRM lead data.",
            selection_reason="No selected table can answer the CRM question.",
        )
        with patch.object(
            engine, "_generate_structured_response", return_value=routing
        ) as generate:
            bundle = engine.create_grounding_plan(
                "How many CRM leads converted?", context_on=True
            )

        self.assertFalse(bundle["plan"]["is_answerable"])
        self.assertEqual(generate.call_count, 1)
        self.assertIn("CRM lead data", bundle["plan"]["decline_reason"])

    def test_every_physical_column_has_effective_grain_metadata(self):
        engine = self.make_engine()
        con = duckdb.connect(str(BASE_DIR / "data" / "lakehouse.duckdb"), read_only=True)
        try:
            for table in engine.semantic_layer["tables"]:
                columns = con.execute(
                    f'DESCRIBE SELECT * FROM "{table["name"]}"'
                ).fetchall()
                catalog = engine.format_column_grain_catalog(table, columns)
                self.assertEqual(len(catalog), len(columns))
                for column in columns:
                    expected = (
                        table["name"] + "." + column[0] + " — "
                        + table.get("column_grains", {}).get(
                            column[0],
                            table.get("default_column_grain", table["grain"]),
                        )
                    )
                    self.assertIn(expected, catalog)
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
