# HydroPhonoKit Installation Guide

## Quick Start

### For Users (Recommended)

```bash
# Clone the repository
git clone https://github.com/hydrophonokit/hydrophonokit.git
cd hydrophonokit

# Install in development mode (editable)
pip install -e .

# Verify installation
hydrophonokit help
```

### For Developers

```bash
# Clone and install with dev dependencies
git clone https://github.com/hydrophonokit/hydrophonokit.git
cd hydrophonokit
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check hydrophonokit/

# Format code
black hydrophonokit/
```

---

## Installation Methods

### Method 1: pip (Recommended)

```bash
pip install hydrophonokit
```

### Method 2: From Source

```bash
git clone https://github.com/hydrophonokit/hydrophonokit.git
cd hydrophonokit
pip install .
```

### Method 3: Editable Install (for development)

```bash
git clone https://github.com/hydrophonokit/hydrophonokit.git
cd hydrophonokit
pip install -e ".[all]"
```

---

## Optional Dependencies

HydroPhonoKit has several optional features that can be installed separately:

### Symmetry-Adapted Force Constants (Recommended)

Improves force constant accuracy by respecting crystal symmetry:

```bash
pip install hydrophonokit[symfc]
```

### Automatic K-Path Generation

Automatically determines high-symmetry points for band structure:

```bash
pip install hydrophonokit[seekpath]
```

### All Optional Features

```bash
pip install hydrophonokit[all]
```

---

## System Requirements

- **Python**: 3.9, 3.10, 3.11, or 3.12
- **Operating System**: Linux, macOS, or Windows
- **RAM**: 8 GB minimum, 16 GB recommended
- **Disk**: 500 MB for package + dependencies

---

## HPC Cluster Setup

On HPC clusters, VASP is typically loaded via environment modules. HydroPhonoKit handles the rest:

```bash
# Load VASP module (cluster-specific)
module load vasp/6.5.1

# Install HydroPhonoKit in your home directory
pip install --user hydrophonokit

# Or use a virtual environment
python -m venv ~/hydrophonokit-env
source ~/hydrophonokit-env/bin/activate
pip install hydrophonokit[symfc,seekpath]
```

---

## Verify Installation

```bash
# Check version
hydrophonokit --version

# Run help
hydrophonokit help

# Test with a simple analysis (requires VASP output)
hydrophonokit analyze --source /path/to/vasp_relaxed
```

---

## Troubleshooting

### Issue: `hydrophonokit: command not found`

**Solution**: Ensure your Python bin directory is in PATH:

```bash
# For --user installs
export PATH=$HOME/.local/bin:$PATH

# Or reinstall with --force-reinstall
pip install --force-reinstall hydrophonokit
```

### Issue: Missing optional dependency

**Solution**: Install the feature you need:

```bash
# For symfc support
pip install hydrophonokit[symfc]

# For seekpath support
pip install hydrophonokit[seekpath]
```

### Issue: NumPy version conflict

**Solution**: Upgrade NumPy:

```bash
pip install --upgrade "numpy>=1.21.0,<3.0.0"
```

---

## Uninstall

```bash
pip uninstall hydrophonokit
```

---

## Next Steps

After installation, see:
- [CHANGES.md](CHANGES.md) - Full changelog
- [DOCUMENTATION.md](DOCUMENTATION.md) - User guide
- `hydrophonokit help` - CLI help
