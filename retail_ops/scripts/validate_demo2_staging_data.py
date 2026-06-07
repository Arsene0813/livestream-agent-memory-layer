from pathlib import Path
import csv

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

METRICS_FILE = "demo2_store_period_metrics.csv"
SEARCH_FILE = "demo2_top_search_terms.csv"
SKU_SALES_FILE = "demo2_top_skus_by_sales_volume.csv"
SKU_AMOUNT_FILE = "demo2_top_skus_by_transaction_amount.csv"

EXPECTED_STORES = {"B", "C", "D", "E", "F"}

EXPECTED_METRICS_FIELDS = [
    "store_id",
    "period_month",
    "period_start",
    "period_end",
    "region_type",
    "store_type",
    "transaction_amount",
    "transaction_orders",
    "valid_orders",
    "invalid_orders",
    "estimated_income_proxy",
    "average_order_value",
    "exposure_users",
    "exposure_times",
    "store_average_rank",
    "entry_conversion_rate_pct",
    "entry_users",
    "entry_times",
    "order_users",
    "order_times",
    "order_conversion_rate_pct",
    "order_amount",
    "payment_users",
    "payment_amount",
    "payment_conversion_rate_pct",
    "search_exposure_users",
    "search_average_rank",
    "search_entry_users",
    "merchant_list_exposure_users",
    "merchant_list_average_rank",
    "merchant_list_entry_users",
    "activity_zone_exposure_users",
    "activity_zone_entry_users",
    "order_page_exposure_users",
    "order_page_entry_users",
    "other_exposure_users",
    "other_entry_users",
    "activity_original_transaction_amount",
    "activity_orders",
    "activity_cost",
    "merchant_subsidy_amount",
    "platform_subsidy_amount",
    "activity_cost_ratio_pct",
    "refund_amount",
    "full_refund_orders",
    "refund_orders_all_or_partial",
    "business_district_rank",
]

EXPECTED_SEARCH_FIELDS = [
    "store_id",
    "period_month",
    "period_start",
    "period_end",
    "search_term_rank",
    "search_term",
    "search_term_en",
    "search_term_exposure_times",
    "search_term_click_times",
    "search_term_order_times",
]

EXPECTED_SKU_FIELDS = [
    "store_id",
    "period_month",
    "period_start",
    "period_end",
    "sku_rank",
    "sku_name",
    "sku_name_en",
    "sku_transaction_amount",
    "sales_volume",
    "sku_category_note",
]

FORBIDDEN_ALIAS_FIELDS = {
    "store_exposure_users",
    "store_exposure_times",
    "entry_visits",
    "order_submissions",
    "full_or_partial_refund_orders",
    "business_area_rank",
}


def read_csv_checked(filename, expected_fields):
    path = DATA_DIR / filename

    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        actual_fields = reader.fieldnames

        if actual_fields != expected_fields:
            raise SystemExit(
                f"{filename} header mismatch.\n"
                f"Expected: {expected_fields}\n"
                f"Actual:   {actual_fields}"
            )

        rows = list(reader)

    for i, row in enumerate(rows, start=2):
        if None in row:
            raise SystemExit(
                f"{filename} row {i} has extra columns. "
                f"This usually means a comma was not quoted correctly."
            )

    return rows


def require_same_period(row, filename):
    if row["period_month"] != "2026-03":
        raise SystemExit(f"{filename}: bad period_month for store {row['store_id']}")

    if row["period_start"] != "2026-03-01":
        raise SystemExit(f"{filename}: bad period_start for store {row['store_id']}")

    if row["period_end"] != "2026-03-31":
        raise SystemExit(f"{filename}: bad period_end for store {row['store_id']}")


def check_three_rows_per_store(rows, filename):
    counts = {}

    for row in rows:
        store_id = row["store_id"]
        counts[store_id] = counts.get(store_id, 0) + 1

    expected_counts = {store_id: 3 for store_id in EXPECTED_STORES}

    if counts != expected_counts:
        raise SystemExit(f"{filename}: expected 3 rows per store, got {counts}")


def check_metrics_rows(rows):
    if len(rows) != 5:
        raise SystemExit(f"{METRICS_FILE}: expected 5 rows, got {len(rows)}")

    stores = {row["store_id"] for row in rows}

    if stores != EXPECTED_STORES:
        raise SystemExit(f"{METRICS_FILE}: unexpected stores: {sorted(stores)}")

    overlap = FORBIDDEN_ALIAS_FIELDS.intersection(EXPECTED_METRICS_FIELDS)

    if overlap:
        raise SystemExit(f"{METRICS_FILE}: forbidden alias fields found: {sorted(overlap)}")

    for row in rows:
        require_same_period(row, METRICS_FILE)

        transaction_amount = float(row["transaction_amount"])
        transaction_orders = float(row["transaction_orders"])
        exposure_users = float(row["exposure_users"])
        entry_users = float(row["entry_users"])
        order_users = float(row["order_users"])
        payment_users = float(row["payment_users"])
        activity_cost = float(row["activity_cost"])
        activity_original_transaction_amount = float(row["activity_original_transaction_amount"])

        checks = [
            ("average_order_value", transaction_amount / transaction_orders),
            ("entry_conversion_rate_pct", entry_users / exposure_users * 100),
            ("order_conversion_rate_pct", order_users / entry_users * 100),
            ("payment_conversion_rate_pct", payment_users / order_users * 100),
            ("activity_cost_ratio_pct", activity_cost / activity_original_transaction_amount * 100),
        ]

        for field, expected_value in checks:
            actual_value = float(row[field])

            if abs(actual_value - expected_value) > 0.02:
                raise SystemExit(
                    f"{METRICS_FILE}: {row['store_id']} {field} mismatch. "
                    f"actual={actual_value:.4f}, expected={expected_value:.4f}"
                )


def check_search_rows(rows):
    if len(rows) != 15:
        raise SystemExit(f"{SEARCH_FILE}: expected 15 rows, got {len(rows)}")

    check_three_rows_per_store(rows, SEARCH_FILE)

    for row in rows:
        require_same_period(row, SEARCH_FILE)

        if not row["search_term"]:
            raise SystemExit(f"{SEARCH_FILE}: missing search_term")

        if not row["search_term_en"]:
            raise SystemExit(f"{SEARCH_FILE}: missing search_term_en for {row['search_term']}")

        exposure = int(row["search_term_exposure_times"])
        clicks = int(row["search_term_click_times"])
        orders = int(row["search_term_order_times"])

        if exposure < clicks:
            raise SystemExit(f"{SEARCH_FILE}: clicks exceed exposure for {row['search_term']}")

        if clicks < orders:
            raise SystemExit(f"{SEARCH_FILE}: orders exceed clicks for {row['search_term']}")


def check_sku_rows(rows, filename, amount_required, sales_required):
    if len(rows) != 15:
        raise SystemExit(f"{filename}: expected 15 rows, got {len(rows)}")

    check_three_rows_per_store(rows, filename)

    for row in rows:
        require_same_period(row, filename)

        if not row["sku_name"]:
            raise SystemExit(f"{filename}: missing sku_name")

        if not row["sku_name_en"]:
            raise SystemExit(f"{filename}: missing sku_name_en for {row['sku_name']}")

        if row["sku_category_note"] != "not_classified":
            raise SystemExit(f"{filename}: sku_category_note must be not_classified")

        if amount_required and not row["sku_transaction_amount"]:
            raise SystemExit(f"{filename}: missing sku_transaction_amount for {row['sku_name']}")

        if sales_required and not row["sales_volume"]:
            raise SystemExit(f"{filename}: missing sales_volume for {row['sku_name']}")


def main():
    metrics = read_csv_checked(METRICS_FILE, EXPECTED_METRICS_FIELDS)
    search = read_csv_checked(SEARCH_FILE, EXPECTED_SEARCH_FIELDS)
    sku_sales = read_csv_checked(SKU_SALES_FILE, EXPECTED_SKU_FIELDS)
    sku_amount = read_csv_checked(SKU_AMOUNT_FILE, EXPECTED_SKU_FIELDS)

    check_metrics_rows(metrics)
    check_search_rows(search)
    check_sku_rows(sku_sales, SKU_SALES_FILE, amount_required=False, sales_required=True)
    check_sku_rows(sku_amount, SKU_AMOUNT_FILE, amount_required=True, sales_required=False)

    print("Demo 2 staging data validation PASSED.")
    print("Checked store-period rows: 5")
    print("Checked top-search-term rows: 15")
    print("Checked top-SKU-by-sales-volume rows: 15")
    print("Checked top-SKU-by-transaction-amount rows: 15")
    print("Checked English helper columns: search_term_en and sku_name_en")
    print("Checked canonical field names and key percentage formulas.")


if __name__ == "__main__":
    main()
