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

def extract_phase_gpu_wh(rep: dict, phase: str) -> float:
    """Extract GPU energy for a specific phase (optimum-benchmark format)"""
    node = rep.get(phase) or {}
    energy = node.get("energy") or {}
    gpu = energy.get("gpu")
    unit = energy.get("unit", "kWh")
    try:
        return to_wh(gpu, unit)
    except Exception:
        return 0.0

def detect_format(rep: dict) -> str:
    """Detect whether this is optimum-benchmark or ai_energy_benchmarks format"""
    # ai_energy_benchmarks format has 'energy' key at top level with 'gpu_energy_wh'
    if "energy" in rep and "gpu_energy_wh" in rep["energy"]:
        return "ai_energy_benchmarks"
    # optimum-benchmark format has phase keys (preprocess, prefill, decode)
    if any(phase in rep for phase in PHASES):
        return "optimum"
    # Default to optimum for backward compatibility
    return "optimum"

def extract_ai_energy_benchmarks_format(rep: dict) -> tuple[float, dict]:
    """Extract energy from ai_energy_benchmarks format"""
    energy = rep.get("energy", {})
    total_wh = float(energy.get("gpu_energy_wh", 0.0))

    # Create a compatible breakdown (no phases in ai_energy_benchmarks)
    per_phase = {
        "preprocess": 0.0,
        "prefill": 0.0,
        "decode": total_wh  # Assign all energy to decode phase for compatibility
    }

    return total_wh, per_phase

def extract_optimum_format(rep: dict) -> tuple[float, dict]:
    """Extract energy from optimum-benchmark format"""
    per_phase = {ph: extract_phase_gpu_wh(rep, ph) for ph in PHASES}
    total_wh = round(sum(per_phase.values()) + 1e-12, 2)  # 2 decimals
    return total_wh, per_phase

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

    # Detect format and extract energy accordingly
    format_type = detect_format(rep)

    if format_type == "ai_energy_benchmarks":
        print(f"Detected ai_energy_benchmarks result format")
        total_wh, per_phase = extract_ai_energy_benchmarks_format(rep)
    else:
        print(f"Detected optimum-benchmark result format")
        total_wh, per_phase = extract_optimum_format(rep)
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
    if format_type == "optimum":
        print("(preprocess={:.2f} Wh, prefill={:.2f} Wh, decode={:.2f} Wh)".format(
            per_phase["preprocess"], per_phase["prefill"], per_phase["decode"]
        ))
    print("===============================\n")

    (out_dir / "GPU_ENERGY_WH.txt").write_text(f"{total_wh:.2f}\n")
    (out_dir / "GPU_ENERGY_SUMMARY.json").write_text(json.dumps({
        "units": "Wh",
        "total": total_wh,
        "format": format_type,
        **{f"{k}_wh": round(v, 2) for k, v in per_phase.items()}
    }, indent=2))

if __name__ == "__main__":
    main()
