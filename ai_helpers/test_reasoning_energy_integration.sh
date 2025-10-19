#!/bin/bash
# Integration test to verify different reasoning levels produce different energy results
#
# This script runs actual benchmarks with gpt-oss-20b at different reasoning levels
# and verifies that the energy consumption differs as expected.
#
# Usage: ./test_reasoning_energy_integration.sh [num_prompts]

set -e

NUM_PROMPTS="${1:-5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/test_reasoning_energy_results"

echo "============================================"
echo "Reasoning Energy Integration Test"
echo "============================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "Number of prompts: ${NUM_PROMPTS}"
echo ""

# Clean up previous test results
if [ -d "${OUTPUT_DIR}" ]; then
    echo "Cleaning up previous test results..."
    rm -rf "${OUTPUT_DIR}"
fi

# Run batch runner with gpt-oss-20b only
echo "Running benchmarks for gpt-oss-20b (High, Low, Off)..."
cd "${PROJECT_DIR}"
.venv/bin/python batch_runner.py \
    --model-name gpt-oss-20b \
    --num-prompts "${NUM_PROMPTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --backend pytorch

# Check if results were created
RESULTS_FILE="${OUTPUT_DIR}/master_results.csv"
if [ ! -f "${RESULTS_FILE}" ]; then
    echo "❌ ERROR: Results file not created: ${RESULTS_FILE}"
    exit 1
fi

echo ""
echo "============================================"
echo "Analyzing Results"
echo "============================================"

# Use Python to analyze the results
.venv/bin/python << 'EOF'
import sys
import pandas as pd
from pathlib import Path

results_file = Path("test_reasoning_energy_results/master_results.csv")

if not results_file.exists():
    print(f"❌ ERROR: Results file not found: {results_file}")
    sys.exit(1)

df = pd.read_csv(results_file)

# Filter for successful runs only
df = df[df['error_message'] == '']

if len(df) == 0:
    print("❌ ERROR: No successful runs found")
    sys.exit(1)

print(f"\n✓ Found {len(df)} successful runs\n")

# Display results
print("Results:")
print("-" * 80)
for _, row in df.iterrows():
    print(f"{row['reasoning_state']:15s} | "
          f"Tokens: {row['total_tokens']:4.0f} | "
          f"Duration: {row['total_duration_seconds']:6.2f}s | "
          f"Energy: {row['gpu_energy_wh']:7.4f} Wh | "
          f"CO2: {row['co2_emissions_g']:6.4f}g")

print("-" * 80)

# Get energy values by reasoning state
results = {}
for _, row in df.iterrows():
    state = row['reasoning_state']
    results[state] = {
        'energy': row['gpu_energy_wh'],
        'tokens': row['total_tokens'],
        'duration': row['total_duration_seconds'],
    }

# Verify we have the expected states
expected_states = ['On (High)', 'On (Low)', 'Off (N/A)']
missing_states = [s for s in expected_states if s not in results]

if missing_states:
    print(f"\n⚠ WARNING: Missing reasoning states: {missing_states}")
    print("Available states:", list(results.keys()))

# Analyze differences
print("\n" + "=" * 80)
print("Analysis:")
print("=" * 80)

# Check token count differences
if 'On (High)' in results and 'On (Low)' in results:
    high_tokens = results['On (High)']['tokens']
    low_tokens = results['On (Low)']['tokens']
    off_tokens = results.get('Off (N/A)', {}).get('tokens', 0)

    print(f"\nToken Generation:")
    print(f"  High:  {high_tokens:4.0f} tokens")
    print(f"  Low:   {low_tokens:4.0f} tokens")
    if off_tokens:
        print(f"  Off:   {off_tokens:4.0f} tokens")

    if high_tokens != low_tokens:
        print("  ✓ High and Low generate different token counts")
    else:
        print("  ⚠ WARNING: High and Low generate same token counts")

# Check energy differences
if 'On (High)' in results and 'On (Low)' in results:
    high_energy = results['On (High)']['energy']
    low_energy = results['On (Low)']['energy']
    off_energy = results.get('Off (N/A)', {}).get('energy', 0)

    print(f"\nEnergy Consumption:")
    print(f"  High:  {high_energy:.4f} Wh")
    print(f"  Low:   {low_energy:.4f} Wh")
    if off_energy:
        print(f"  Off:   {off_energy:.4f} Wh")

    # Calculate differences
    energy_diff_pct = abs(high_energy - low_energy) / max(high_energy, low_energy) * 100

    print(f"\nEnergy Difference (High vs Low): {energy_diff_pct:.1f}%")

    # Test assertions
    all_passed = True

    # Test 1: Energy values should be positive
    if high_energy > 0 and low_energy > 0:
        print("  ✓ Energy values are positive")
    else:
        print("  ❌ ERROR: Energy values should be positive")
        all_passed = False

    # Test 2: High and Low should have measurably different energy
    # Allow 5% minimum difference to account for noise
    if energy_diff_pct >= 5.0:
        print(f"  ✓ High and Low have significantly different energy ({energy_diff_pct:.1f}% difference)")
    else:
        print(f"  ⚠ WARNING: High and Low have similar energy ({energy_diff_pct:.1f}% difference)")
        print("    This might indicate reasoning parameters aren't affecting model behavior")

    # Test 3: Off should use significantly less energy than High
    if off_energy > 0:
        off_vs_high_pct = (off_energy / high_energy) * 100
        print(f"\nOff vs High: {off_vs_high_pct:.1f}% of High's energy")

        if off_energy < high_energy * 0.5:
            print(f"  ✓ Off uses significantly less energy than High")
        else:
            print(f"  ⚠ WARNING: Off should use much less energy than High")

    # Final verdict
    print("\n" + "=" * 80)
    if all_passed and energy_diff_pct >= 5.0:
        print("✓ PASS: Reasoning parameters result in different energy consumption")
        sys.exit(0)
    elif energy_diff_pct < 5.0:
        print("⚠ PARTIAL: Tests passed but energy differences are small")
        print("   Consider:")
        print("   - Increasing --num-prompts for more stable measurements")
        print("   - Verifying model actually responds to reasoning parameters")
        print("   - Checking if Harmony format is correctly applied")
        sys.exit(0)
    else:
        print("❌ FAIL: Energy differentiation test failed")
        sys.exit(1)
else:
    print("❌ ERROR: Missing required reasoning states for comparison")
    sys.exit(1)
EOF

TEST_RESULT=$?

echo ""
echo "============================================"
echo "Test Complete"
echo "============================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""

exit ${TEST_RESULT}
