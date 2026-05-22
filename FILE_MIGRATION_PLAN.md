# HydroPhonoKit File Management System — Restructuring Plan

> **Version:** 2.6.0
> **Date:** April 12, 2026
> **Objective:** Reorganize project structure for clarity, maintainability, and error-free operation

---

## 🎯 Current Problems

| Problem | Location | Impact |
|---------|----------|--------|
| Mixed file types in root | `/` root | Hard to navigate, confuses new contributors |
| Documentation scattered | Root + multiple `.md` files | Inconsistent updates, hard to find |
| No examples directory | Missing | Users don't know how to use the system |
| No data directory | Missing | Test data mixed with source code |
| No scripts directory | Missing | Utility scripts have no home |
| Build artifacts not isolated | `hydrophonokit.egg-info/` | Clutters project root |
| No clear separation of concerns | Overall | Hard to maintain long-term |

---

## 📋 Proposed Structure

```
hydrophonokit/                              # Project root
│
├── src/                                # Source code (PEP 517 compliant)
│   └── hydrophonokit/                      # Main Python package
│       ├── __init__.py                 # Package init + public API
│       ├── __main__.py                 # Entry point for `python -m hydrophonokit`
│       ├── cli.py                      # CLI interface
│       │
│       ├── core/                       # Core workflow modules
│       │   ├── analyzer.py             # Material analysis
│       │   ├── generator.py            # Workspace generation
│       │   ├── verifier.py             # Workspace validation
│       │   ├── inspector.py            # VASP output inspection
│       │   ├── postprocessor.py        # Results analysis
│       │   └── templates.py            # VASP INCAR/KPOINTS/SLURM templates
│       │
│       ├── science/                    # Scientific analysis modules
│       │   ├── physics.py              # Physical constants & formulas
│       │   ├── elastic.py              # Elastic constants
│       │   ├── eos.py                  # Equation of State
│       │   ├── qha.py                  # Quasi-Harmonic Approximation
│       │   ├── anharmonic.py           # Phonon linewidths/lifetimes
│       │   ├── h2_molecule.py          # H2 gas properties
│       │   └── h_storage.py            # Hydrogen storage thermodynamics
│       │
│       └── utils/                      # Utility modules
│           ├── __init__.py
│           └── (future utility modules)
│
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── conftest.py                     # Pytest fixtures & config
│   ├── test_core/                      # Core module tests
│   │   ├── test_analyzer.py
│   │   ├── test_generator.py
│   │   ├── test_verifier.py
│   │   └── test_postprocessor.py
│   ├── test_science/                   # Science module tests
│   │   ├── test_physics.py
│   │   ├── test_elastic.py
│   │   ├── test_eos_qha.py
│   │   ├── test_anharmonic.py
│   │   └── test_h_storage.py
│   └── data/                           # Test data files
│       ├── sample_contcar/
│       └── sample_vasprun.xml
│
├── docs/                               # Documentation
│   ├── index.md                        # Documentation home
│   ├── installation.md                 # Installation guide
│   ├── cli.md                          # CLI usage guide
│   ├── api/                            # API reference
│   │   ├── core.md
│   │   ├── science.md
│   │   └── utils.md
│   ├── workflows/                      # Workflow guides
│   │   ├── basic_workflow.md
│   │   ├── elastic_constants.md
│   │   ├── qha.md
│   │   ├── anharmonicity.md
│   │   └── h_storage.md
│   ├── development/                    # Developer docs
│   │   ├── architecture.md
│   │   ├── contributing.md
│   │   ├── file_management.md
│   │   └── testing.md
│   └── changelog.md                    # CHANGES.md content
│
├── examples/                           # Usage examples
│   ├── basic/                          # Basic workflow examples
│   │   ├── 01_analyze_material.py
│   │   ├── 02_build_workspace.py
│   │   └── 03_postprocess.py
│   ├── advanced/                       # Advanced analysis examples
│   │   ├── elastic_constants.py
│   │   ├── qha_analysis.py
│   │   ├── anharmonic_properties.py
│   │   └── h_storage_analysis.py
│   └── data/                           # Sample input data
│       ├── MgH2/
│       │   ├── CONTCAR
│       │   ├── INCAR
│       │   └── ...
│       └── Si/
│           └── ...
│
├── scripts/                            # Utility scripts
│   ├── setup_local_env.sh              # Local environment setup
│   ├── run_tests.sh                    # Test runner
│   ├── build_docs.sh                   # Documentation build
│   └── clean.sh                        # Cleanup build artifacts
│
├── config/                             # Configuration files
│   ├── pyproject.toml                  # Project metadata
│   ├── ruff.toml                       # Linter config
│   ├── mypy.ini                        # Type checker config
│   └── pytest.ini                      # Test config
│
├── .github/                            # GitHub-specific files
│   ├── workflows/                      # CI/CD workflows
│   │   ├── tests.yml
│   │   ├── lint.yml
│   │   └── docs.yml
│   ├── ISSUE_TEMPLATE/                 # Issue templates
│   └── PULL_REQUEST_TEMPLATE.md
│
├── .gitignore                          # Git ignore rules
├── LICENSE                             # MIT License
├── README.md                           # Project overview
├── ROADMAP.md                          # Future development
└── PLAN_PHASE5_H_STORAGE.md            # H-storage plan (reference)
```

---

## 🗺️ Migration Plan

### Phase 1: Create New Structure (Day 1)

**Step 1.1:** Create directory structure
```bash
mkdir -p src/hydrophonokit/{core,science,utils}
mkdir -p tests/{test_core,test_science,data}
mkdir -p docs/{api,workflows,development}
mkdir -p examples/{basic,advanced,data}
mkdir -p scripts
mkdir -p config
mkdir -p .github/{workflows,ISSUE_TEMPLATE}
```

**Step 1.2:** Move source files
```bash
# Move core modules
mv hydrophonokit/analyzer.py src/hydrophonokit/core/
mv hydrophonokit/generator.py src/hydrophonokit/core/
mv hydrophonokit/verifier.py src/hydrophonokit/core/
mv hydrophonokit/inspector.py src/hydrophonokit/core/
mv hydrophonokit/postprocessor.py src/hydrophonokit/core/
mv hydrophonokit/templates.py src/hydrophonokit/core/

# Move science modules
mv hydrophonokit/physics.py src/hydrophonokit/science/
mv hydrophonokit/elastic.py src/hydrophonokit/science/
mv hydrophonokit/eos.py src/hydrophonokit/science/
mv hydrophonokit/qha.py src/hydrophonokit/science/
mv hydrophonokit/anharmonic.py src/hydrophonokit/science/
mv hydrophonokit/h2_molecule.py src/hydrophonokit/science/
mv hydrophonokit/h_storage.py src/hydrophonokit/science/

# Move CLI and package files
mv hydrophonokit/cli.py src/hydrophonokit/
mv hydrophonokit/__init__.py src/hydrophonokit/
mv hydrophonokit/__main__.py src/hydrophonokit/
```

**Step 1.3:** Move config files
```bash
mv pyproject.toml config/
mv setup.py config/
mv requirements.txt config/
mv requirements-dev.txt config/
mv MANIFEST.in config/
```

**Step 1.4:** Move documentation
```bash
mv INSTALL.md docs/installation.md
mv CLI_GUIDE.md docs/cli.md
mv DOCUMENTATION.md docs/development/architecture.md
mv FILE_MANAGEMENT.md docs/development/file_management.md
mv CHANGES.md docs/changelog.md
```

**Step 1.5:** Move tests
```bash
mv tests/test_elastic.py tests/test_science/
mv tests/test_eos_qha.py tests/test_science/
mv tests/test_anharmonic.py tests/test_science/
mv tests/test_h_storage.py tests/test_science/
```

**Step 1.6:** Move examples (create new)
```bash
# Create example scripts (see examples/ section below)
```

---

### Phase 2: Update Imports (Day 2)

**Step 2.1:** Update `src/hydrophonokit/__init__.py`

```python
"""HydroPhonoKit — Material-Aware Phonon Workflow Engine"""
__version__ = "2.6.0"

# Core modules
from .core.analyzer import MaterialAnalyzer
from .core.generator import PhononGenerator
from .core.verifier import SystemVerifier
from .core.postprocessor import PhononPostProcessor
from .core.templates import (
    get_incar_born, get_incar_force,
    get_kpoints_supercell, make_slurm_script
)

# Science modules
from .science.physics import (
    H_PLANCK, K_BOLTZMANN, R_GAS, THZ_TO_CM, BORN_FACTOR,
    bose_einstein, helmholtz_free_energy, phonon_entropy,
    heat_capacity_cv, dulong_petit_limit, zero_point_energy,
    check_dynamical_stability, slack_thermal_conductivity,
)
from .science.elastic import ElasticConstantsExtractor
from .science.qha import QHAEngine
from .science.anharmonic import AnharmonicCalculator
from .science.h_storage import HStorageAnalyzer
```

**Step 2.2:** Update all internal imports

```python
# Before: from .physics import ...
# After:  from .science.physics import ...

# Before: from .analyzer import ...
# After:  from .core.analyzer import ...
```

**Files to update:**
- `src/hydrophonokit/cli.py`
- `src/hydrophonokit/core/generator.py`
- `src/hydrophonokit/core/postprocessor.py`
- `src/hydrophonokit/science/qha.py`
- `src/hydrophonokit/science/anharmonic.py`
- `src/hydrophonokit/science/h_storage.py`
- `src/hydrophonokit/science/elastic.py`

**Step 2.3:** Update `config/pyproject.toml`

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["hydrophonokit*"]
```

**Step 2.4:** Update `src/hydrophonokit/__main__.py`

```python
from hydrophonokit.cli import main
main()
```

---

### Phase 3: Create Infrastructure (Day 3)

**Step 3.1:** Create `tests/conftest.py`

```python
"""Pytest fixtures and configuration for HydroPhonoKit tests."""
import pytest
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_vasp_dir(tmp_path):
    """Create a temporary VASP directory with minimal files."""
    vasp_dir = tmp_path / "vasp_relaxed"
    vasp_dir.mkdir()
    
    # Create minimal CONTCAR, INCAR, etc.
    # (implementation depends on test needs)
    
    return vasp_dir

@pytest.fixture
def sample_phonon_workspace(tmp_path):
    """Create a temporary phonon workspace."""
    ws = tmp_path / "phonon_ws"
    ws.mkdir()
    # Create workspace structure
    return ws
```

**Step 3.2:** Create `scripts/run_tests.sh`

```bash
#!/bin/bash
set -e

echo "Running HydroPhonoKit tests..."
cd "$(dirname "$0")/.."

python -m pytest tests/ -v --tb=short "$@"

echo "Tests complete!"
```

**Step 3.3:** Create `scripts/clean.sh`

```bash
#!/bin/bash
set -e

echo "Cleaning build artifacts..."
cd "$(dirname "$0")/.."

rm -rf build/
rm -rf dist/
rm -rf *.egg-info
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

echo "Cleanup complete!"
```

**Step 3.4:** Create `docs/index.md`

```markdown
# HydroPhonoKit Documentation

Welcome to HydroPhonoKit v2.6.0!

## Getting Started
- [Installation](installation.md)
- [Quick Start](workflows/basic_workflow.md)
- [CLI Guide](cli.md)

## Workflows
- [Basic Phonon Workflow](workflows/basic_workflow.md)
- [Elastic Constants](workflows/elastic_constants.md)
- [Quasi-Harmonic Approximation](workflows/qha.md)
- [Anharmonicity](workflows/anharmonicity.md)
- [Hydrogen Storage](workflows/h_storage.md)

## Development
- [Architecture](development/architecture.md)
- [File Management](development/file_management.md)
- [Testing](development/testing.md)
- [Contributing](development/contributing.md)
```

---

### Phase 4: Update Build System (Day 4)

**Step 4.1:** Update `config/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "hydro-phonokit"
version = "2.6.0"
description = "Material-aware scientific framework for automated VASP/Phonopy phonon calculations"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.8.0",
    "phonopy>=2.20.0",
    "spglib>=2.0.0",
    "pymatgen>=2023.0.0",
    "matplotlib>=3.5.0",
]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["hydrophonokit*"]
```

**Step 4.2:** Create root `setup.py` (for backwards compatibility)

```python
"""Legacy setup.py — delegates to pyproject.toml"""
from setuptools import setup
setup()
```

**Step 4.3:** Update `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
*.egg

# Virtual environments
.venv/
venv/

# IDE
.vscode/
.idea/

# Test artifacts
.coverage
htmlcov/
.pytest_cache/

# Build artifacts
*.whl
*.tar.gz

# OS files
.DS_Store
Thumbs.db
```

---

### Phase 5: Testing & Validation (Day 5)

**Step 5.1:** Run all existing tests
```bash
cd hydrophonokit
python -m pytest tests/ -v
```

**Step 5.2:** Verify CLI works
```bash
python -m hydrophonokit --help
python -m hydrophonokit help
```

**Step 5.3:** Verify package installs correctly
```bash
pip install -e .
hydrophonokit --help
```

**Step 5.4:** Test import structure
```python
from hydrophonokit import MaterialAnalyzer, PhononGenerator
from hydrophonokit.science.physics import THZ_TO_CM
from hydrophonokit.core.analyzer import ELEMENT_DB
```

**Step 5.5:** Verify documentation links
- Check all internal links in `docs/`
- Verify code examples work

---

## 📊 Migration Timeline

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 1 | Create new structure | 2 hours | None |
| 2 | Update imports | 4 hours | Phase 1 |
| 3 | Create infrastructure | 3 hours | Phase 1 |
| 4 | Update build system | 2 hours | Phase 2 |
| 5 | Testing & validation | 3 hours | Phase 4 |
| **Total** | | **~14 hours** | |

---

## ⚠️ Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Import errors | Build failure | Comprehensive testing after each phase |
| Broken CLI | User impact | Test CLI after Phase 2 |
| Lost documentation | Knowledge loss | Git commit before migration |
| Test failures | Confidence loss | Run full test suite after each phase |

---

## ✅ Post-Migration Checklist

- [ ] All imports updated
- [ ] `pyproject.toml` points to `src/` layout
- [ ] CLI works: `hydrophonokit --help`
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Package installs: `pip install -e .`
- [ ] Documentation links work
- [ ] Examples run without errors
- [ ] Git history preserved
- [ ] CI/CD pipeline updated (if applicable)

---

## 📁 Benefits of New Structure

| Benefit | Description |
|---------|-------------|
| **PEP 517 compliant** | Modern Python packaging standard |
| **Clear separation** | Core vs Science vs Utils |
| **Scalable** | Easy to add new modules |
| **Testable** | Organized test structure |
| **Documented** | Centralized documentation |
| **Example-driven** | Users can learn from examples |
| **Maintainable** | Clear file locations |

---

*End of File Management Restructuring Plan — HydroPhonoKit v2.6.0*
