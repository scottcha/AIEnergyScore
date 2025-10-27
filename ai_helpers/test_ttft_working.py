#!/usr/bin/env python3
"""
Quick test to verify TTFT tracking is working with the updated ai_energy_benchmarks package.
"""

import sys
from pathlib import Path

# Add ai_energy_benchmarks to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ai_energy_benchmarks"))

from ai_energy_benchmarks.backends.pytorch import PyTorchBackend


def test_ttft_tracking():
    """Test that TTFT tracking is present in the updated package."""

    print("=" * 80)
    print("TTFT Tracking Verification Test")
    print("=" * 80)
    print()

    # Check if enable_streaming parameter exists in run_inference signature
    import inspect
    sig = inspect.signature(PyTorchBackend.run_inference)
    params = list(sig.parameters.keys())

    print("✓ PyTorchBackend.run_inference parameters:")
    for param in params:
        print(f"  - {param}")

    if 'enable_streaming' in params:
        print("\n✅ SUCCESS: 'enable_streaming' parameter found!")
        print("   TTFT tracking is available in the updated package.")
    else:
        print("\n❌ FAILED: 'enable_streaming' parameter NOT found!")
        print("   The package may not be updated correctly.")
        return False

    print()
    print("=" * 80)
    print("Package Update Status")
    print("=" * 80)
    print()

    # Check package version
    try:
        import ai_energy_benchmarks
        print(f"✓ ai_energy_benchmarks version: {ai_energy_benchmarks.__version__}")
        print(f"✓ Package location: {ai_energy_benchmarks.__file__}")
    except Exception as e:
        print(f"⚠ Could not get package info: {e}")

    print()
    print("=" * 80)
    print("Why TTFT is 0 in your batch_runner results")
    print("=" * 80)
    print()
    print("The batch_runner.py uses DOCKER for PyTorch backend execution.")
    print("The Docker image contains the OLD version of ai_energy_benchmarks")
    print("without TTFT tracking.")
    print()
    print("To get TTFT tracking in Docker:")
    print("  1. Rebuild the Docker image with the updated wheel:")
    print("     cd /home/scott/src/AIEnergyScore")
    print("     # Update Dockerfile to use new wheel")
    print("     docker build -t your-image-name .")
    print()
    print("  2. OR modify batch_runner.py to skip Docker and run directly")
    print("     (only for testing - not recommended for production)")
    print()
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_ttft_tracking()
    sys.exit(0 if success else 1)
