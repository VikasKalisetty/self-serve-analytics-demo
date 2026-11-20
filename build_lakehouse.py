import random
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
from faker import Faker

# Deterministic seeds for repeatable demo & evaluation benchmarks
SEED = 42
random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

BASE_DIR = Path(__file__).parent
GOLD_DIR = BASE_DIR / "data" / "gold"
DB_PATH = BASE_DIR / "data" / "lakehouse.duckdb"
GOLD_DIR.mkdir(parents=True, exist_ok=True)

COMPANIES = ["COMP01", "COMP02"]
SOURCE_SYSTEM = "EPICOR_ERP"
REPORT_DATE = date(2026, 10, 31)
START_DATE = date(2026, 1, 1)


def build_dim_employee() -> pd.DataFrame:
    rows = []
    departments = ["Procurement", "Supply Chain", "Warehouse Ops", "Quality"]
    divisions = ["Manufacturing", "Distribution"]
    locations = ["Chicago, IL", "Austin, TX", "Detroit, MI", "Columbus, OH"]

    for comp in COMPANIES:
        for i in range(1, 21):
            emp_num = f"EMP-{i:03d}"
            first = fake.first_name()
            last = fake.last_name()
            is_buyer = i <= 8
            is_approver = i in (1, 2, 9, 10)
            is_wh_mgr = i in (11, 12)
            is_inv_worker = i >= 13

            title = (
                "Senior Buyer" if i <= 3 else
                "Buyer" if is_buyer else
                "Procurement Manager" if is_approver else
                "Warehouse Manager" if is_wh_mgr else
                "Inventory Specialist"
            )
            mgr_id = "EMP-001" if i > 1 else None
            mgr_name = "Alex Rivera" if i > 1 else None
            hire_dt = date(2020, 1, 1) + timedelta(days=random.randint(0, 1800))

            rows.append({
                "company": comp,
                "employee_num": emp_num,
                "employee_name": f"{first} {last}",
                "first_name": first,
                "last_name": last,
                "employee_email": f"{first.lower()}.{last.lower()}@{comp.lower()}.com",
                "employee_status": "Active",
                "job_title": title,
                "department": "Procurement" if (is_buyer or is_approver) else random.choice(departments),
                "division": random.choice(divisions),
                "location": random.choice(locations),
                "manager_id": mgr_id,
                "manager_name": mgr_name,
                "is_buyer": is_buyer,
                "is_inventory_worker": is_inv_worker,
                "is_warehouse_manager": is_wh_mgr,
                "is_approver": is_approver,
                "hire_date": hire_dt,
                "termination_date": None,
                "source_system": SOURCE_SYSTEM,
            })
    return pd.DataFrame(rows)


def build_dim_warehousebin() -> pd.DataFrame:
    rows = []
    wh_defs = [
        ("WH-MAIN", "Central Distribution Hub", "Central", "Chicago", "IL"),
        ("WH-PROD", "Plant Production Warehouse", "Production", "Detroit", "MI"),
        ("WH-REG1", "South Regional Depot", "Regional", "Austin", "TX"),
    ]
    bin_defs = [
        ("BIN-A01", "Primary Storage A1", "Standard", "Zone-A", "A", "01", "1", True, False, False, True),
        ("BIN-A02", "Bulk Rack A2", "Bulk", "Zone-A", "A", "02", "2", True, False, False, False),
        ("BIN-RCV", "Inbound Receiving Dock", "Receiving", "Zone-R", "R", "01", "1", True, True, False, False),
        ("BIN-INSP", "Quality Inspection Hold", "Inspection", "Zone-Q", "Q", "01", "1", False, False, False, False),
        ("BIN-SHP", "Outbound Shipping Bay", "Shipping", "Zone-S", "S", "01", "1", True, False, True, False),
    ]
    for comp in COMPANIES:
        for wh_code, wh_name, wh_type, city, state in wh_defs:
            for (
                bin_num, bin_name, bin_type, zone, aisle, rack, level,
                is_net, is_rcv, is_shp, is_def
            ) in bin_defs:
                max_cap = random.choice([500, 1000, 2000, 5000])
                curr_cap = random.randint(int(max_cap * 0.2), int(max_cap * 0.85))
                rows.append({
                    "company": comp,
                    "warehouse_code": wh_code,
                    "warehouse_name": wh_name,
                    "warehouse_description": f"{wh_name} ({comp})",
                    "warehouse_type": wh_type,
                    "warehouse_location": f"{city}, {state}",
                    "warehouse_city": city,
                    "warehouse_state": state,
                    "warehouse_country": "USA",
                    "bin_num": bin_num,
                    "bin_name": bin_name,
                    "bin_description": f"{bin_name} in {wh_code}",
                    "bin_type": bin_type,
                    "bin_zone": zone,
                    "bin_aisle": aisle,
                    "bin_rack": rack,
                    "bin_level": level,
                    "max_capacity": float(max_cap),
                    "current_capacity": float(curr_cap),
                    "available_capacity": float(max_cap - curr_cap),
                    "capacity_uom": "EA",
                    "is_active": True,
                    "is_nettable": is_net,
                    "is_receiving_bin": is_rcv,
                    "is_shipping_bin": is_shp,
                    "is_default_bin": is_def,
                    "source_system": SOURCE_SYSTEM,
                })
    return pd.DataFrame(rows)


def build_dim_vendor() -> pd.DataFrame:
    rows = []
    categories = ["Raw material", "Components", "Services", "Other"]
    tiers = ["Strategic", "Preferred", "Transactional"]
    terms = ["NET30", "NET45", "NET60", "2/10 NET30"]

    for comp in COMPANIES:
        for i in range(1, 26):
            v_num = f"VEND-{i:03d}"
            tier = random.choices(tiers, weights=[0.20, 0.35, 0.45], k=1)[0]
            risk = "Low" if tier == "Strategic" else random.choices(["Low", "Medium", "High"], weights=[0.5, 0.35, 0.15], k=1)[0]
            rating = round(random.uniform(4.2, 5.0) if tier == "Strategic" else random.uniform(2.8, 4.6), 2)
            rows.append({
                "company": comp,
                "vendor_num": v_num,
                "vendor_name": f"{fake.company()} Supply",
                "vendor_type": random.choice(["Manufacturer", "Distributor", "OEM"]),
                "vendor_category": random.choice(categories),
                "vendor_status": "Active",
                "supplier_tier": tier,
                "is_preferred_supplier": tier in ("Strategic", "Preferred"),
                "is_strategic_supplier": tier == "Strategic",
                "is_approved_supplier": True,
                "contact_name": fake.name(),
                "contact_email": fake.company_email(),
                "contact_phone": fake.phone_number(),
                "address": fake.street_address(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "country": "USA",
                "postal_code": fake.postcode(),
                "payment_terms": random.choice(terms),
                "currency_code": "USD",
                "tax_id": f"{random.randint(10,99)}-{random.randint(1000000,9999999)}",
                "lead_time_days": random.choice([7, 10, 14, 21, 30]),
                "supplier_rating": rating,
                "supplier_risk_category": risk,
                "source_system": SOURCE_SYSTEM,
            })
    return pd.DataFrame(rows)


def build_dim_part() -> pd.DataFrame:
    rows = []
    part_classes = {
        "FAST": ("Fasteners", "Components", 1.5, 12.0),
        "ELEC": ("Electronics", "Components", 25.0, 220.0),
        "RAW": ("Raw Metals", "Raw material", 15.0, 140.0),
        "HYDR": ("Hydraulics", "Components", 45.0, 380.0),
        "PACK": ("Packaging", "Other", 2.0, 18.0),
    }
    class_keys = list(part_classes.keys())

    for comp in COMPANIES:
        for i in range(1, 51):
            p_num = f"PART-{i:04d}"
            p_cls = class_keys[(i - 1) % len(class_keys)]
            p_grp, p_cat, low, high = part_classes[p_cls]
            std_cost = round(random.uniform(low, high), 2)
            avg_cost = round(std_cost * random.uniform(0.96, 1.05), 2)
            min_stock = random.choice([20, 50, 100, 200])
            safety = int(min_stock * 0.5)
            reorder = int(min_stock * 1.4)
            max_stock = int(min_stock * 4.0)
            abc = random.choices(["A", "B", "C"], weights=[0.2, 0.3, 0.5], k=1)[0]

            rows.append({
                "company": comp,
                "part_num": p_num,
                "part_description": f"{p_grp} {fake.word().capitalize()} Assembly-{i}",
                "part_type": "Purchased",
                "part_class": p_cls,
                "part_category": p_cat,
                "product_group": p_grp,
                "product_family": f"{p_grp} Series",
                "product_code": f"PC-{p_cls}",
                "uom": "EA",
                "inventory_uom": "EA",
                "purchase_uom": "EA",
                "is_purchased": True,
                "is_manufactured": False,
                "is_stock_item": True,
                "is_active": True,
                "minimum_stock_level": float(min_stock),
                "maximum_stock_level": float(max_stock),
                "reorder_point": float(reorder),
                "safety_stock_quantity": float(safety),
                "minimum_order_quantity": float(random.choice([10, 25, 50])),
                "maximum_order_quantity": float(random.choice([500, 1000, 2500])),
                "standard_cost": std_cost,
                "average_cost": avg_cost,
                "minimum_price": round(std_cost * 0.90, 2),
                "maximum_price": round(std_cost * 1.25, 2),
                "default_lead_time_days": random.choice([7, 14, 21, 28]),
                "abc_classification": abc,
                "criticality": "High" if abc == "A" else ("Medium" if abc == "B" else "Low"),
                "shelf_life_days": random.choice([None, 180, 365, 730]),
                "source_system": SOURCE_SYSTEM,
            })
    return pd.DataFrame(rows)


def build_dim_supplier_price_list(dim_part: pd.DataFrame, dim_vendor: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp in COMPANIES:
        parts = dim_part[dim_part["company"] == comp]
        vendors = dim_vendor[dim_vendor["company"] == comp]["vendor_num"].tolist()

        for p in parts.itertuples(index=False):
            # Assign 2 to 3 suppliers per part
            chosen_vendors = random.sample(vendors, k=random.choice([2, 3]))
            for idx, v_num in enumerate(chosen_vendors):
                quoted = round(p.standard_cost * random.uniform(0.92, 1.12), 2)
                lt = max(5, p.default_lead_time_days + random.randint(-4, 7))
                rows.append({
                    "company": comp,
                    "part_num": p.part_num,
                    "vendor_num": v_num,
                    "vendor_part_num": f"VP-{v_num[-3:]}-{p.part_num[-4:]}",
                    "vendor_part_description": f"Supplier Spec: {p.part_description}",
                    "quoted_unit_price": quoted,
                    "minimum_price": round(quoted * 0.92, 2),
                    "maximum_price": round(quoted * 1.10, 2),
                    "currency_code": "USD",
                    "price_uom": "EA",
                    "minimum_order_quantity": p.minimum_order_quantity,
                    "maximum_order_quantity": p.maximum_order_quantity,
                    "order_multiple_quantity": float(random.choice([5, 10, 25])),
                    "standard_lead_time_days": lt,
                    "minimum_lead_time_days": max(3, lt - 3),
                    "maximum_lead_time_days": lt + 7,
                    "effective_from_date": date(2025, 1, 1),
                    "effective_to_date": date(2026, 12, 31),
                    "price_break_quantity": float(random.choice([100, 250, 500])),
                    "discount_percentage": random.choice([0.0, 2.5, 5.0, 7.5]),
                    "is_preferred_supplier": idx == 0,
                    "is_primary_supplier": idx == 0,
                    "is_active": True,
                    "last_quoted_date": date(2026, 1, 15),
                    "last_purchase_date": REPORT_DATE - timedelta(days=random.randint(5, 90)),
                    "last_purchase_price": quoted,
                    "source_system": SOURCE_SYSTEM,
                })
    return pd.DataFrame(rows)


def build_po_and_receipts(
    dim_employee: pd.DataFrame,
    dim_supplier_price_list: pd.DataFrame,
    dim_warehousebin: pd.DataFrame,
):
    dim_po_rows = []
    fct_po_rows = []
    fct_rcv_rows = []

    ship_vias = ["UPS Ground", "FedEx Freight", "DHL Express", "Dedicated Truck"]
    freight_terms = ["FOB Origin", "FOB Destination", "Prepaid & Add"]
    po_types = ["Standard", "Blanket", "Expedited"]

    total_days = (REPORT_DATE - START_DATE).days

    for comp in COMPANIES:
        buyers = dim_employee[(dim_employee["company"] == comp) & (dim_employee["is_buyer"])]["employee_num"].tolist()
        approvers = dim_employee[(dim_employee["company"] == comp) & (dim_employee["is_approver"])]["employee_num"].tolist()
        spl = dim_supplier_price_list[dim_supplier_price_list["company"] == comp]
        rcv_bins = dim_warehousebin[
            (dim_warehousebin["company"] == comp) & (dim_warehousebin["is_receiving_bin"])
        ][["warehouse_code", "bin_num"]].to_dict("records")

        # Group parts by vendor for realistic multi-line POs
        vendor_parts = spl.groupby("vendor_num")
        vendor_list = list(vendor_parts.groups.keys())

        pack_counter = 1

        # 600 POs per company (1,200 POs total)
        for po_idx in range(1, 601):
            po_num = f"PO-{10000 + po_idx}"
            v_num = random.choice(vendor_list)
            buyer_id = random.choice(buyers)
            approver_id = random.choice(approvers)
            order_dt = START_DATE + timedelta(days=random.randint(0, total_days))
            po_type = random.choices(po_types, weights=[0.78, 0.12, 0.10], k=1)[0]
            is_approved = random.random() < 0.96
            approval_dt = order_dt + timedelta(days=random.randint(0, 2)) if is_approved else None

            v_part_df = vendor_parts.get_group(v_num)
            num_lines = random.choices([1, 2, 3], weights=[0.50, 0.35, 0.15], k=1)[0]
            chosen_rows = v_part_df.sample(n=min(num_lines, len(v_part_df)), replace=False)

            po_all_closed = True
            po_release_records = []

            for line_idx, p_row in enumerate(chosen_rows.itertuples(index=False), start=1):
                num_releases = random.choices([1, 2], weights=[0.80, 0.20], k=1)[0]
                for rel_idx in range(1, num_releases + 1):
                    lt = int(p_row.standard_lead_time_days)
                    due_dt = order_dt + timedelta(days=lt + (rel_idx - 1) * 14)
                    need_by_dt = due_dt - timedelta(days=random.randint(0, 3))
                    is_confirmed = random.random() < 0.90
                    confirmed_dt = order_dt + timedelta(days=random.randint(1, 3)) if is_confirmed else None
                    promised_dt = due_dt + timedelta(days=random.choice([0, 0, 0, 2, 5, -2])) if is_confirmed else due_dt

                    order_qty = float(random.choice([25, 50, 100, 150, 200, 300]))
                    # Intentional Maverick Spend / PPV trap: ~15% of POs priced 5-18% above quoted price
                    if random.random() < 0.15:
                        unit_price = round(p_row.quoted_unit_price * random.uniform(1.05, 1.18), 2)
                    else:
                        unit_price = round(p_row.quoted_unit_price * random.uniform(0.98, 1.01), 2)

                    # Simulate receipts if due date is near or before REPORT_DATE
                    received_qty = 0.0
                    if due_dt <= REPORT_DATE + timedelta(days=5):
                        fulfill_roll = random.random()
                        if fulfill_roll < 0.72:
                            # Complete receipt (1 shipment)
                            rcpt_qty = order_qty
                            delay = random.choices([-2, 0, 0, 0, 2, 5, 10], weights=[0.1, 0.4, 0.15, 0.1, 0.1, 0.1, 0.05], k=1)[0]
                            rcpt_dt = min(due_dt + timedelta(days=delay), REPORT_DATE)
                            received_qty = rcpt_qty
                            bin_choice = random.choice(rcv_bins)

                            insp_req = random.random() < 0.35
                            rej_qty = float(random.choice([0, 0, 0, 5, 10])) if insp_req else 0.0
                            rej_qty = min(rej_qty, rcpt_qty)
                            acc_qty = rcpt_qty - rej_qty
                            is_ret = rej_qty > 0 and random.random() < 0.5

                            fct_rcv_rows.append({
                                "company": comp,
                                "pack_slip": f"PS-{pack_counter:06d}",
                                "pack_line": 1,
                                "po_number": po_num,
                                "po_line": line_idx,
                                "po_release_number": rel_idx,
                                "vendor_num": v_num,
                                "part_num": p_row.part_num,
                                "warehouse_code": bin_choice["warehouse_code"],
                                "bin_num": bin_choice["bin_num"],
                                "receipt_date": rcpt_dt,
                                "arrival_date": rcpt_dt - timedelta(days=random.choice([0, 1])),
                                "received_quantity": rcpt_qty,
                                "accepted_quantity": acc_qty,
                                "rejected_quantity": rej_qty,
                                "received_uom": "EA",
                                "unit_cost": unit_price,
                                "extended_cost": round(rcpt_qty * unit_price, 2),
                                "receipt_status": "Posted",
                                "is_complete_receipt": True,
                                "is_partial_receipt": False,
                                "is_returned": is_ret,
                                "days_early_late": (rcpt_dt - due_dt).days,
                                "inspection_required": insp_req,
                                "inspection_status": "Failed" if rej_qty > 0 else ("Passed" if insp_req else "Not Required"),
                                "source_system": SOURCE_SYSTEM,
                                "record_created_date": datetime.combine(rcpt_dt, datetime.min.time()),
                            })
                            pack_counter += 1

                        elif fulfill_roll < 0.87:
                            # Partial receipt
                            rcpt_qty = float(int(order_qty * random.uniform(0.4, 0.75)))
                            rcpt_dt = min(due_dt + timedelta(days=random.randint(-1, 6)), REPORT_DATE)
                            received_qty = rcpt_qty
                            bin_choice = random.choice(rcv_bins)

                            fct_rcv_rows.append({
                                "company": comp,
                                "pack_slip": f"PS-{pack_counter:06d}",
                                "pack_line": 1,
                                "po_number": po_num,
                                "po_line": line_idx,
                                "po_release_number": rel_idx,
                                "vendor_num": v_num,
                                "part_num": p_row.part_num,
                                "warehouse_code": bin_choice["warehouse_code"],
                                "bin_num": bin_choice["bin_num"],
                                "receipt_date": rcpt_dt,
                                "arrival_date": rcpt_dt,
                                "received_quantity": rcpt_qty,
                                "accepted_quantity": rcpt_qty,
                                "rejected_quantity": 0.0,
                                "received_uom": "EA",
                                "unit_cost": unit_price,
                                "extended_cost": round(rcpt_qty * unit_price, 2),
                                "receipt_status": "Posted",
                                "is_complete_receipt": False,
                                "is_partial_receipt": True,
                                "is_returned": False,
                                "days_early_late": (rcpt_dt - due_dt).days,
                                "inspection_required": False,
                                "inspection_status": "Not Required",
                                "source_system": SOURCE_SYSTEM,
                                "record_created_date": datetime.combine(rcpt_dt, datetime.min.time()),
                            })
                            pack_counter += 1

                    open_qty = max(0.0, order_qty - received_qty)
                    is_closed = open_qty == 0.0
                    is_open = not is_closed
                    if is_open:
                        po_all_closed = False

                    days_to_due = (due_dt - REPORT_DATE).days if is_open else 0
                    days_past_due = max(0, (REPORT_DATE - due_dt).days) if is_open else 0
                    is_late = is_open and (REPORT_DATE > due_dt)

                    po_release_records.append({
                        "company": comp,
                        "po_number": po_num,
                        "po_line": line_idx,
                        "po_release_number": rel_idx,
                        "vendor_num": v_num,
                        "part_num": p_row.part_num,
                        "buyer_id": buyer_id,
                        "order_date": order_dt,
                        "due_date": due_dt,
                        "supplier_confirmed_date": confirmed_dt,
                        "promised_date": promised_dt,
                        "need_by_date": need_by_dt,
                        "order_quantity": order_qty,
                        "received_quantity": received_qty,
                        "open_quantity": open_qty,
                        "unit_price": unit_price,
                        "extended_price": round(order_qty * unit_price, 2),
                        "currency_code": "USD",
                        "uom": "EA",
                        "release_status": "Closed" if is_closed else "Open",
                        "po_status": "Closed" if po_all_closed else "Open",
                        "is_open": is_open,
                        "is_closed": is_closed,
                        "is_confirmed": is_confirmed,
                        "is_late": is_late,
                        "days_to_due_date": days_to_due,
                        "days_past_due": days_past_due,
                        "lead_time_days": (due_dt - order_dt).days,
                        "source_system": SOURCE_SYSTEM,
                        "record_created_date": datetime.combine(order_dt, datetime.min.time()),
                        "record_updated_date": datetime.combine(REPORT_DATE, datetime.min.time()),
                    })

            # Sync overall po_status across header and releases
            overall_status = "Closed" if po_all_closed else "Open"
            for rec in po_release_records:
                rec["po_status"] = overall_status
                fct_po_rows.append(rec)

            dim_po_rows.append({
                "company": comp,
                "po_number": po_num,
                "vendor_num": v_num,
                "buyer_id": buyer_id,
                "order_date": order_dt,
                "po_status": overall_status,
                "po_type": po_type,
                "currency_code": "USD",
                "terms_code": random.choice(["NET30", "NET45", "NET60"]),
                "ship_via_code": random.choice(ship_vias),
                "freight_terms": random.choice(freight_terms),
                "ship_to_name": f"{comp} Plant Receiving",
                "ship_to_address": fake.street_address(),
                "ship_to_city": "Chicago" if comp == "COMP01" else "Austin",
                "ship_to_state": "IL" if comp == "COMP01" else "TX",
                "ship_to_country": "USA",
                "approval_status": "Approved" if is_approved else "Pending",
                "approved_by": approver_id if is_approved else None,
                "approval_date": approval_dt,
                "is_approved": is_approved,
                "is_open": not po_all_closed,
                "is_closed": po_all_closed,
                "is_blanket_order": po_type == "Blanket",
                "buyer_group": "Direct Materials" if comp == "COMP01" else "Indirect & MRO",
                "source_system": SOURCE_SYSTEM,
            })

    return pd.DataFrame(dim_po_rows), pd.DataFrame(fct_po_rows), pd.DataFrame(fct_rcv_rows)


def build_fct_inventory_position(dim_part: pd.DataFrame, dim_warehousebin: pd.DataFrame) -> pd.DataFrame:
    """Periodic monthly snapshot fact (6 months) so historical trends AND snapshot-filtering traps work."""
    snapshot_dates = [
        date(2026, 5, 31),
        date(2026, 6, 30),
        date(2026, 7, 31),
        date(2026, 8, 31),
        date(2026, 9, 30),
        date(2026, 10, 31),
    ]
    rows = []

    for comp in COMPANIES:
        parts = dim_part[dim_part["company"] == comp]
        bins = dim_warehousebin[
            (dim_warehousebin["company"] == comp) & (dim_warehousebin["bin_num"].isin(["BIN-A01", "BIN-A02", "BIN-INSP"]))
        ]

        # Assign each part to 2 warehouse/bin locations
        part_locs = []
        bin_records = bins.to_dict("records")
        for p in parts.itertuples(index=False):
            chosen_bins = random.sample(bin_records, k=2)
            for b in chosen_bins:
                part_locs.append((p, b))

        for snap_dt in snapshot_dates:
            for p, b in part_locs:
                is_insp_bin = b["bin_num"] == "BIN-INSP"
                # Generate realistic stockout / understock / normal / overstock mix
                roll = random.random()
                if roll < 0.08:
                    on_hand = 0.0
                elif roll < 0.25:
                    on_hand = float(random.randint(1, max(2, int(p.minimum_stock_level) - 1)))
                elif roll < 0.80:
                    on_hand = float(random.randint(int(p.minimum_stock_level), int(p.maximum_stock_level)))
                else:
                    on_hand = float(random.randint(int(p.maximum_stock_level) + 10, int(p.maximum_stock_level * 1.5)))

                allocated = round(on_hand * random.uniform(0.0, 0.25), 0)
                reserved = round(on_hand * random.uniform(0.0, 0.10), 0)
                insp_qty = on_hand if is_insp_bin else 0.0
                hold_qty = round(on_hand * 0.05, 0) if (not is_insp_bin and random.random() < 0.1) else 0.0
                non_net = insp_qty + hold_qty
                avail = max(0.0, on_hand - allocated - reserved - non_net)
                in_transit = float(random.choice([0, 0, 0, 50, 100]))
                inv_val = round(on_hand * p.average_cost, 2)
                daily_usage = max(1.0, p.minimum_stock_level / 15.0)
                doi = round(on_hand / daily_usage, 1)

                if on_hand == 0:
                    status = "Stockout"
                elif on_hand < p.minimum_stock_level:
                    status = "Understock"
                elif on_hand > p.maximum_stock_level:
                    status = "Overstock"
                else:
                    status = "Normal"

                rows.append({
                    "company": comp,
                    "part_num": p.part_num,
                    "warehouse_code": b["warehouse_code"],
                    "bin_num": b["bin_num"],
                    "on_hand_quantity": on_hand,
                    "available_quantity": avail,
                    "allocated_quantity": allocated,
                    "reserved_quantity": reserved,
                    "non_nettable_quantity": non_net,
                    "inspection_quantity": insp_qty,
                    "hold_quantity": hold_qty,
                    "in_transit_quantity": in_transit,
                    "inventory_value": inv_val,
                    "average_unit_cost": p.average_cost,
                    "last_receipt_date": snap_dt - timedelta(days=random.randint(1, 30)),
                    "last_issue_date": snap_dt - timedelta(days=random.randint(1, 15)),
                    "last_count_date": snap_dt - timedelta(days=random.randint(5, 60)),
                    "minimum_stock_level": p.minimum_stock_level,
                    "maximum_stock_level": p.maximum_stock_level,
                    "reorder_point": p.reorder_point,
                    "safety_stock_quantity": p.safety_stock_quantity,
                    "days_of_inventory": doi,
                    "stock_status": status,
                    "is_below_minimum": on_hand < p.minimum_stock_level,
                    "is_above_maximum": on_hand > p.maximum_stock_level,
                    "snapshot_date": snap_dt,
                    "source_system": SOURCE_SYSTEM,
                })

    return pd.DataFrame(rows)


def main():
    print("Building Epicor-based Procurement & Inventory Lakehouse (9 tables)...")

    dim_employee = build_dim_employee()
    dim_warehousebin = build_dim_warehousebin()
    dim_vendor = build_dim_vendor()
    dim_part = build_dim_part()
    dim_supplier_price_list = build_dim_supplier_price_list(dim_part, dim_vendor)
    dim_po, fct_purchase_orders, fct_po_receipts = build_po_and_receipts(
        dim_employee, dim_supplier_price_list, dim_warehousebin
    )
    fct_inventory_position = build_fct_inventory_position(dim_part, dim_warehousebin)

    tables = {
        "dim_employee": dim_employee,
        "dim_warehousebin": dim_warehousebin,
        "dim_vendor": dim_vendor,
        "dim_part": dim_part,
        "dim_supplier_price_list": dim_supplier_price_list,
        "dim_po": dim_po,
        "fct_purchase_orders": fct_purchase_orders,
        "fct_po_receipts": fct_po_receipts,
        "fct_inventory_position": fct_inventory_position,
    }

    # 1. Write Parquet files (Gold Layer)
    for name, df in tables.items():
        pq_path = GOLD_DIR / f"{name}.parquet"
        df.to_parquet(pq_path, index=False)
        print(f"  [Parquet] {name:<25} -> {len(df):>6,} rows | {len(df.columns):>2} cols")

    # 2. Load into DuckDB
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))
    for name in tables:
        pq_path = (GOLD_DIR / f"{name}.parquet").as_posix()
        con.execute(
            f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{pq_path}')"
        )

    # 3. Built-in ERP Trap Sanity Check: Periodic Snapshot Inflation
    naive_inv, governed_inv = con.execute("""
        SELECT
            ROUND(SUM(inventory_value), 2) AS naive_total_inv,
            ROUND(SUM(CASE WHEN snapshot_date = (SELECT MAX(snapshot_date) FROM fct_inventory_position)
                           THEN inventory_value ELSE 0 END), 2) AS latest_snapshot_inv
        FROM fct_inventory_position;
    """).fetchone()

    con.close()

    print(f"\nDuckDB lakehouse created at: {DB_PATH}")
    print("\n--- Built-in ERP Trap Sanity Check (Total Inventory Valuation) ---")
    print(f"  Ungrounded SQL (No Snapshot Filter): ${naive_inv:,.2f}")
    print(f"  Governed KPI   (Latest Snapshot)   : ${governed_inv:,.2f}")
    print(f"  Overstatement Prevented            : ${(naive_inv - governed_inv):,.2f}")


if __name__ == "__main__":
    main()
