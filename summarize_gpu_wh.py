#!/usr/bin/env python3
import sys, json
from pathlib import Path

PHASES = ("preprocess", "prefill", "decode")

def to_wh(value, unit: str) -> float:
    unit = (unit or "").strip().lower()
    if unit == "wh":  return float(value)
    if unit == "kwh": return float(value) * 1000.0
    if unit in ("j", "joule", "joules"): return float(value) / 3600.0  # 1 Wh = 3600 J
    # Fallback: treat as kWh
    return float(value) * 1000.0

def extract_phase_gpu_wh(rep: dict, phase: str) -> float:
    node = rep.get(phase) or {}
    energy = node.get("energy") or {}
    gpu = energy.get("gpu")
    unit = energy.get("unit", "kWh")
    if gpu is None: return 0.0
    return to_wh(gpu, unit)

def find_report(start: Path) -> Path:
    if start.is_file():
        return start
    # Prefer an obvious file name
    candidate = start / "benchmark_report.json"
    if candidate.exists():
        return candidate
    # Otherwise pick the most recent matching file under start
    matches = list(start.rglob("benchmark_report.json"))
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError("benchmark_report.json not found")

def main():
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/results")
    report_path = find_report(target)
    rep = json.loads(report_path.read_text())

    per_phase = {ph: extract_phase_gpu_wh(rep, ph) for ph in PHASES}
    total_wh = round(sum(per_phase.values()) + 1e-12, 2)  # 2 decimals

    out_dir = report_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Human + CI friendly
    print("\n===============================")
    print(f"GPU Energy (Wh): {total_wh:,.2f}")
    print("(preprocess={:.2f} Wh, prefill={:.2f} Wh, decode={:.2f} Wh)".format(
        per_phase["preprocess"], per_phase["prefill"], per_phase["decode"]
    ))
    print("===============================\n")

    (out_dir / "GPU_ENERGY_WH.txt").write_text(f"{total_wh:.2f}\n")
    (out_dir / "GPU_ENERGY_SUMMARY.json").write_text(json.dumps({
        "units": "Wh",
        "total": total_wh,
        **{f"{k}_wh": round(v, 2) for k, v in per_phase.items()}
    }, indent=2))

if __name__ == "__main__":
    main()
