# AIEnergyScore Dependency Management

## Overview

AIEnergyScore uses standard Python dependency management with `requirements.txt` as the single source of truth for all Python package versions, including `ai_energy_benchmarks`.

## Dependency Version Location

All Python package versions are specified in:
```
AIEnergyScore/requirements.txt
```

This includes the `ai_energy_benchmarks` dependency:
```
ai_energy_benchmarks==0.0.4
```

## How It Works

### requirements.txt
- Single source of truth for all Python dependencies
- Specifies exact versions with `==` for reproducibility
- Located at `AIEnergyScore/requirements.txt`

### Dockerfile
- Installs all dependencies from `requirements.txt`
- Uses TestPyPI index for `ai_energy_benchmarks` (pre-release)
- Falls back to PyPI for other dependencies

### build.sh
- Simple Docker build script
- No version management logic needed
- Just builds the image using the Dockerfile

## Updating ai_energy_benchmarks Version

To update the `ai_energy_benchmarks` version:

1. Edit `AIEnergyScore/requirements.txt`
2. Update the version line:
   ```
   ai_energy_benchmarks==0.0.5
   ```
3. Rebuild the Docker image:
   ```bash
   cd ~/src
   ./AIEnergyScore/build.sh
   ```

## Why This Approach?

✅ **Standard Python practice**: Uses requirements.txt as expected
✅ **Single source of truth**: Version defined once, used everywhere
✅ **Dependency clarity**: Clear that ai_energy_benchmarks is a dependency, not part of AIEnergyScore
✅ **Simple maintenance**: Standard pip workflow
✅ **Version pinning**: Ensures reproducible builds

## TestPyPI Installation

The Dockerfile installs from TestPyPI because `ai_energy_benchmarks` is currently in pre-release:

```dockerfile
RUN pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    -r requirements.txt
```

When `ai_energy_benchmarks` is published to production PyPI, this can be simplified to:
```dockerfile
RUN pip install -r requirements.txt
```

## Local Development

For local development with a custom wheel:
1. Uncomment the local wheel installation section in the Dockerfile
2. Build the wheel in `ai_energy_benchmarks/`:
   ```bash
   cd ~/src/ai_energy_benchmarks
   python -m build
   ```
3. Build the Docker image (it will copy the wheel from the build context)
