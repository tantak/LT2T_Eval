"""
Compare model task1_logic (from aggregation jsonl) to gold logic and table.

- Loads model output (jsonl) and gold (json), joins by id.
- Option 1: Executes gold logic on gold table and compares final result to model.
- Option 2: Derives expected filter rows from gold and compares to model output_rows.
- Option 3: Verifies model evidence_cells against gold table_cont.

Usage:
  python compare_aggregation_logic.py

Paths are relative to this script's directory; override with env or args if needed.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

# Paths relative to TT-data folder
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_JSONL = SCRIPT_DIR / "round1_split_by_logic_type_updated" / "TT_deepseek-v3.2_output_one_shot_with_operations_nested" / "TT_deepseek-v3.2_output_one_shot_with_operations_nested_aggregation.jsonl"
GOLD_JSON = SCRIPT_DIR / "data_seprated_actions" / "aggregation_sample_100_gold.json"


def load_gold_by_id(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {item["id"]: item for item in items if "id" in item}


def load_model_lines(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def col_name_to_index(header: list[str], name: str) -> Optional[int]:
    try:
        return header.index(name)
    except ValueError:
        return None


def annotation_to_filter_spec(gold: dict) -> Optional[tuple[int, str, Any, int, str]]:
    """From gold annotation + table_header, get (filter_col_idx, criterion, value, agg_col_idx, agg_type)."""
    ann = gold.get("annotation") or {}
    subset = ann.get("subset") or {}
    logic = gold.get("logic") or {}
    header = gold.get("table_header") or []
    if not header and not logic:
        return None
    # annotation col/subset.col are often 1-based string indices
    def to_idx(s: Any) -> int:
        if s is None:
            return -1
        i = int(s) if isinstance(s, str) else int(s)
        return i - 1 if i > 0 else i  # treat as 1-based if positive

    filter_col = to_idx(subset.get("col"))
    if filter_col < 0:
        args = _inner_filter_args(logic)
        if args and len(args) > 1 and isinstance(args[1], str):
            filter_col = col_name_to_index(header, args[1])
            if filter_col is None:
                filter_col = 0
    agg_col = to_idx(ann.get("col"))
    if agg_col < 0:
        args = logic.get("args") or []
        if len(args) >= 2 and isinstance(args[0], dict):
            a2 = (args[0].get("args") or [])
            if len(a2) >= 2 and isinstance(a2[1], str):
                agg_col = col_name_to_index(header, a2[1])
                if agg_col is None:
                    agg_col = 0
    criterion = (subset.get("criterion") or "equal").replace(" ", "_").lower()
    inner_args = _inner_filter_args(logic)
    value = subset.get("value")
    if value is None and inner_args and len(inner_args) > 2:
        value = inner_args[2]
    agg_type = (ann.get("type") or "sum").lower()
    return (filter_col, criterion, value, agg_col, agg_type)


def _inner_filter_args(logic: dict):
    if not logic:
        return []
    args = logic.get("args") or []
    for a in args:
        if isinstance(a, dict) and (a.get("func") or "").startswith("filter"):
            return a.get("args") or []
        if isinstance(a, dict):
            inner = _inner_filter_args(a)
            if inner:
                return inner
    return []


def expected_filter_rows(table_cont: list[list], filter_col: int, criterion: str, value: Any) -> list[int]:
    """Return row indices that satisfy the filter. Criterion: equal, greater_than_eq, less, not_equal, etc."""
    rows = []
    for i, row in enumerate(table_cont):
        if filter_col >= len(row):
            continue
        cell = row[filter_col]
        # normalize for comparison
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
            try:
                cell_n = float(cell) if "." in str(cell) else int(cell)
                val_n = float(value) if "." in str(value) else int(value)
            except (ValueError, TypeError):
                cell_n, val_n = str(cell).strip(), str(value).strip()
        else:
            cell_n, val_n = str(cell).strip().lower(), str(value).strip().lower()

        if criterion in ("equal", "eq"):
            if cell_n == val_n:
                rows.append(i)
        elif criterion in ("greater_than_eq", "greater_eq", "gte"):
            try:
                if cell_n >= val_n:
                    rows.append(i)
            except TypeError:
                if str(cell_n) >= str(val_n):
                    rows.append(i)
        elif criterion in ("less_than", "less", "lt"):
            try:
                if cell_n < val_n:
                    rows.append(i)
            except TypeError:
                if str(cell_n) < str(val_n):
                    rows.append(i)
        elif criterion in ("not_equal", "not_eq"):
            if cell_n != val_n:
                rows.append(i)
        else:
            if str(cell_n) == str(val_n):
                rows.append(i)
    return rows


def expected_aggregate(table_cont: list[list], row_indices: list[int], agg_col: int, agg_type: str) -> Optional[float]:
    values = []
    for i in row_indices:
        if 0 <= i < len(table_cont) and agg_col < len(table_cont[i]):
            raw = table_cont[i][agg_col]
            try:
                v = float(re.sub(r"[^\d.-]", "", str(raw)))
                values.append(v)
            except (ValueError, TypeError):
                pass
    if not values:
        return None
    if agg_type in ("sum", "total"):
        return sum(values)
    if agg_type in ("average", "avg", "mean"):
        return sum(values) / len(values)
    return sum(values)  # default sum


def gold_root_result(gold: dict) -> Any:
    r = (gold.get("logic") or {}).get("result")
    if r is not None:
        return r
    return (gold.get("annotation") or {}).get("result")


def model_root_result(model: dict) -> Any:
    logic = model.get("task1_logic") or {}
    return logic.get("result")


def normalize_result(v: Any) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    s = str(v).strip()
    try:
        f = float(s.replace(",", ""))
        if f == int(f):
            return str(int(f))
        return str(round(f, 4))
    except ValueError:
        return s.lower()


def collect_model_filter_output_rows(node: dict, acc: list[tuple[int, list[int]]]) -> None:
    """Collect (ind, output_rows) for every filter-like node."""
    if not isinstance(node, dict):
        return
    func = (node.get("func") or "")
    if "filter" in func.lower() and "output_rows" in node:
        acc.append((node.get("ind", -1), node.get("output_rows") or []))
    for a in node.get("args") or []:
        if isinstance(a, dict):
            collect_model_filter_output_rows(a, acc)


def collect_evidence_cells(node: dict, acc: list[dict]) -> None:
    for k, v in (node or {}).items():
        if k == "evidence_cells" and isinstance(v, list):
            acc.extend(v)
        elif isinstance(v, dict):
            collect_evidence_cells(v, acc)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, dict):
                    collect_evidence_cells(x, acc)


def main():
    gold_by_id = load_gold_by_id(GOLD_JSON)
    print(f"Loaded {len(gold_by_id)} gold samples from {GOLD_JSON.name}")

    results = {
        "match_id": 0,
        "final_result_match": 0,
        "filter_rows_match": 0,
        "evidence_cells_ok": 0,
        "total_compared": 0,
        "missing_gold": 0,
        "errors": [],
    }

    for model in load_model_lines(MODEL_JSONL):
        mid = model.get("id")
        if not mid:
            continue
        gold = gold_by_id.get(mid)
        if not gold:
            results["missing_gold"] += 1
            continue

        results["total_compared"] += 1
        table_cont = gold.get("table_cont") or []
        table_header = gold.get("table_header") or []

        # --- Option 1: final result ---
        gold_result = gold.get("annotation") or {}
        gold_result_val = gold_result.get("result")
        logic_result = (gold.get("logic") or {}).get("result")
        model_final = model_root_result(model)
        if gold_result_val is not None:
            expected_str = normalize_result(gold_result_val)
        else:
            expected_str = normalize_result(logic_result)
        model_str = normalize_result(model_final)
        if expected_str == model_str:
            results["final_result_match"] += 1
        else:
            results["errors"].append({
                "id": mid,
                "type": "final_result",
                "expected": expected_str,
                "model": model_str,
            })

        # --- Option 2: filter rows ---
        spec = annotation_to_filter_spec(gold)
        if spec:
            filter_col, criterion, value, agg_col, agg_type = spec
            expected_rows = expected_filter_rows(table_cont, filter_col, criterion, value)
            filter_pairs = []
            collect_model_filter_output_rows(model.get("task1_logic") or {}, filter_pairs)
            # take first filter node's output_rows (aggregation typically has one main filter)
            model_rows = sorted(filter_pairs[0][1]) if filter_pairs else []
            expected_sorted = sorted(expected_rows)
            if model_rows == expected_sorted:
                results["filter_rows_match"] += 1
            else:
                results["errors"].append({
                    "id": mid,
                    "type": "filter_rows",
                    "expected": expected_sorted,
                    "model": model_rows,
                })

        # --- Option 3: evidence_cells vs table ---
        cells = []
        collect_evidence_cells(model.get("task1_logic") or {}, cells)
        all_ok = True
        for c in cells:
            r, col = c.get("row"), c.get("col")
            val = c.get("value")
            if r is None or col is None or not (0 <= r < len(table_cont) and 0 <= col < len(table_cont[r])):
                all_ok = False
                break
            if str(table_cont[r][col]).strip() != str(val).strip():
                all_ok = False
                break
        if all_ok and cells:
            results["evidence_cells_ok"] += 1
        elif not cells:
            pass  # no cells to check
        else:
            results["errors"].append({"id": mid, "type": "evidence_cells"})

    print(f"\nCompared {results['total_compared']} samples (model ids present in gold).")
    print(f"Missing gold for {results['missing_gold']} model ids.")
    print(f"Final result match: {results['final_result_match']} / {results['total_compared']}")
    print(f"Filter rows match:  {results['filter_rows_match']} / {results['total_compared']}")
    print(f"Evidence cells OK: {results['evidence_cells_ok']} (where present)")
    if results["errors"]:
        print(f"\nFirst 5 errors: {results['errors'][:5]}")


if __name__ == "__main__":
    main()
