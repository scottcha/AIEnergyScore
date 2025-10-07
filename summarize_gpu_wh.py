#!/usr/bin/env python3
import sys, json, math
from pathlib import Path
from typing import Dict, Any

def to_wh(value, unit: str) -> float:
    unit = (unit or "").strip().lower()
    if unit == "wh":  return float(value)
    if unit == "kwh": return float(value) * 1000.0
    if unit in ("j", "joule", "joules"): return float(value) / 3600.0  # 1 Wh = 3600 J
    # Fallback: treat as kWh
    return float(value) * 1000.0

def extract_gpu_wh_from_phase(phase_node: Dict[str, Any]) -> float:
    """Return GPU energy in Wh for a given phase node, or 0.0 if missing."""
    if not isinstance(phase_node, dict):
        return 0.0
    energy = phase_node.get("energy") or {}
    if not isinstance(energy, dict):
        return 0.0
    gpu = energy.get("gpu", None)
    if gpu is None:
        return 0.0
    unit = energy.get("unit", "kWh")
    try:
        return to_wh(gpu, unit)
    except Exception:
        return 0.0

def find_report(start: Path) -> Path:
    if start.is_file():
        return start
    candidate = start / "benchmark_report.json"
    if candidate.exists():
        return candidate
    matches = list(start.rglob("benchmark_report.json"))
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError("benchmark_report.json not found")

def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/results")
    report_path = find_report(target)
    rep = json.loads(report_path.read_text())

    # Auto-discover phases at the top level that contain energy.gpu
    per_phase_wh = {}
    for phase_name, node in rep.items():
        wh = extract_gpu_wh_from_phase(node)
        # ignore non-positive values; keeps JSON clean per your requirement
        if wh > 0:
            per_phase_wh[phase_name] = wh

    total_wh = round(sum(per_phase_wh.values()) + 1e-12, 2)

    out_dir = report_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print a readable, dynamic breakdown
    print("\n===============================")
    print(f"GPU Energy (Wh): {total_wh:,.2f}")
    if per_phase_wh:
        parts = [f"{k}={v:.2f} Wh" for k, v in sorted(per_phase_wh.items())]
        print("(" + ", ".join(parts) + ")")
    print("===============================\n")

    (out_dir / "GPU_ENERGY_WH.txt").write_text(f"{total_wh:.2f}\n")

    # Only include non-zero phases in JSON
    summary = {"units": "Wh", "total": total_wh}
    for k, v in sorted(per_phase_wh.items()):
        summary[f"{k}_wh"] = round(v, 2)
    (out_dir / "GPU_ENERGY_SUMMARY.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
