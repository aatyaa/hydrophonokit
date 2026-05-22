# HydroPhonoKit Postprocessor Module — Development Plan

> **Module:** `hydrophonokit/postprocessor.py`
> **Current Version:** v2.1 (752 lines, monolithic)
> **Target Version:** v2.7 (refactored, robust, enhanced)
> **Date:** April 12, 2026

---

## 📊 Current State Analysis

### Module Overview
- **File:** `hydrophonokit/postprocessor.py`
- **Lines:** 752
- **Class:** `PhononPostProcessor` (single monolithic class)
- **Phases:** 6 (Data → IFC → Bands/DOS → Thermo → H-Analysis → HTML Report)
- **Plotting:** 4 methods (band structure, pDOS, thermodynamics, H-modes)
- **Tests:** None

### Current Architecture
```
PhononPostProcessor
├── __init__()              # Path setup, output dir creation
├── execute_pipeline()      # Main orchestrator (calls all phases)
├── _phase1_data()          # Load phonopy, parse forces, extract Born charges
├── _phase2_force_constants()  # symfc or phonopy IFC computation
├── _phase3_bands_dos()     # Band structure (seekpath) + partial DOS
├── _phase4_thermodynamics()  # F(T), S(T), Cv(T) + validations
├── _phase5_h_analysis()    # H-mode decomposition (lib/bend/stretch)
├── _plot_all()             # Calls all plotting methods
├── _plot_band_structure()  # Band structure plot
├── _plot_partial_dos()     # Element-projected DOS plot
├── _plot_thermodynamics()  # F/S/Cv vs T (3-panel)
├── _plot_h_modes()         # H vibrational mode decomposition
└── _phase6_reporting()     # HTML report generation
```

---

## 🔍 Issues Found

### 1. Monolithic Architecture (Severity: 🔴 HIGH)

**Problem:** Everything in one 752-line class with no separation of concerns.

```python
# Data loading, physics, and plotting all mixed together
class PhononPostProcessor:
    def _phase1_data(self): ...      # Data extraction
    def _phase4_thermodynamics(self): ...  # Physics computation
    def _plot_band_structure(self): ...    # Visualization
    def _phase6_reporting(self): ...       # Report generation
```

**Impact:**
- Hard to test individual phases
- Hard to extend without modifying core file
- Hard to maintain — 752 lines is too large for single responsibility
- Plotting logic is tightly coupled to computation logic

### 2. No Error Recovery (Severity: 🔴 HIGH)

**Problem:** If one phase fails, entire pipeline crashes with no partial results.

```python
def execute_pipeline(self):
    self._phase1_data()           # If this fails, nothing saved
    self._phase2_force_constants()  # No try/except
    self._phase3_bands_dos()      # If seekpath fails, silent fallback OK
    self._phase4_thermodynamics() # No error handling
    ...
```

**Impact:**
- Lost compute time if late-phase failure
- No partial results saved
- User has to debug and re-run everything

### 3. Hardcoded Plot Settings (Severity: 🟡 MEDIUM)

**Problem:** Only 8 element colors defined, global matplotlib config modification.

```python
# Only 8 elements supported in pDOS!
COLORS = {
    'Na':'#E74C3C', 'Ca':'#3498DB', 'B':'#2ECC71', 'H':'#F39C12',
    'O':'#9B59B6', 'Ti':'#1ABC9C', 'Li':'#E67E22', 'N':'#3498DB'
}

# Global modification — affects all subsequent plots in user's session
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12, ...
})
```

**Impact:**
- Missing colors for most elements (falls back to gray `#888888`)
- Side effects on user's matplotlib configuration
- No theme support for different publication styles

### 4. No Data Export Standardization (Severity: 🟡 MEDIUM)

**Problem:** Results scattered across multiple formats with no unified export.

```
band.yaml                      # Phonopy native format
thermodynamic_properties.dat   # Custom text format
FORCE_CONSTANTS                # Phonopy native format
BORN                           # Phonopy native format
Phonon_Analysis_Report.html    # Simple HTML, no raw data
```

**Impact:**
- No single file contains all results
- Hard to compare runs programmatically
- No metadata (version, timestamp, input files) in exports

### 5. Missing Scientific Features (Severity: 🟡 MEDIUM)

**What's missing:**
- ❌ Group velocity computation (`v_g = ∂ω/∂q`)
- ❌ Mode-specific thermodynamics (per-element F, S, Cv)
- ❌ Grüneisen parameter extraction from phonon data
- ❌ Debye-Waller factors (mean square displacement)
- ❌ Phonon mean free path estimation
- ❌ Cumulative thermal conductivity
- ❌ Fat band plots (element-projected band structure)
- ❌ Phonon convergence checks (mesh density, supercell size)

### 6. No Caching (Severity: 🟢 LOW)

**Problem:** Re-parses vasprun.xml and re-computes force constants every run.

```python
# Always re-parses all vasprun.xml files
for i in range(n_disp):
    vr = Vasprun(vpath)  # Slow: 10-60 seconds per file
    forces = vr.read_forces()
```

**Impact:**
- Slow re-runs (minutes for large systems)
- No incremental processing
- Wasted I/O on large vasprun.xml files (10MB-2GB each)

---

## 📋 Development Plan

### Phase 1: Architecture Refactoring (Week 1)

**Goal:** Split monolithic module into focused submodules without changing behavior.

**Proposed Structure:**
```
src/hydrophonokit/postprocessor/
├── __init__.py               # Public API exports
├── core.py                   # Main orchestrator (PhononPostProcessor)
├── data_loader.py            # Phase 1: Force/Born charge extraction
├── ifc_computer.py           # Phase 2: Force constants (symfc/phonopy)
├── bands_dos.py              # Phase 3: Band structure + DOS computation
├── thermodynamics.py         # Phase 4: F(T), S(T), Cv(T) + validations
├── hydrogen.py               # Phase 5: H-mode analysis
├── plotting.py               # All plotting methods (PhononPlotter class)
├── reporting.py              # Phase 6: HTML/PDF report generation
└── export.py                 # Standardized data export (JSON/CSV)
```

**Tasks:**
- [ ] Create directory structure
- [ ] Extract `_phase1_data()` → `data_loader.py`
- [ ] Extract `_phase2_force_constants()` → `ifc_computer.py`
- [ ] Extract `_phase3_bands_dos()` → `bands_dos.py`
- [ ] Extract `_phase4_thermodynamics()` → `thermodynamics.py`
- [ ] Extract `_phase5_h_analysis()` → `hydrogen.py`
- [ ] Extract all `_plot_*()` → `plotting.py` (create `PhononPlotter` class)
- [ ] Extract `_phase6_reporting()` → `reporting.py`
- [ ] Create `export.py` with `PhononResultsExporter` class
- [ ] Update `core.py` to use new modules
- [ ] Update `__init__.py` with public API
- [ ] Verify all existing functionality works identically

**Proposed Core API:**
```python
class PhononPostProcessor:
    def __init__(self, workspace_dir, profile, output_dir=None):
        self.data_loader = DataLoader(workspace_dir, profile)
        self.ifc_computer = IFCComputer(profile)
        self.bands_dos = BandsDOSComputer(profile)
        self.thermo = ThermodynamicsComputer(profile)
        self.hydrogen = HydrogenAnalyzer(profile)
        self.plotter = PhononPlotter()
        self.exporter = PhononResultsExporter()
        self.reporting = ReportGenerator()

    def execute_pipeline(self, config=None):
        """Execute all phases with error handling."""
        results = {}
        results['data'] = self.data_loader.load()
        results['ifc'] = self.ifc_computer.compute(self.data_loader.phonon)
        results['bands'] = self.bands_dos.compute(self.data_loader.phonon)
        results['thermo'] = self.thermo.compute(self.data_loader.phonon)
        if 'H' in self.profile.elements:
            results['hydrogen'] = self.hydrogen.analyze(results['bands'])
        self.plotter.plot_all(results, self.output_dir)
        self.exporter.export(results, self.output_dir)
        self.reporting.generate(results, self.output_dir)
        return results
```

**Deliverable:** Same behavior, cleaner architecture, testable modules.

---

### Phase 2: Robustness & Error Handling (Week 2)

**Goal:** Make pipeline resilient to partial failures.

**Tasks:**
- [ ] Add try/except around each phase
- [ ] Implement phase status tracking: `{'completed': bool, 'error': str, 'duration': float}`
- [ ] Add `--skip-phases` CLI option
- [ ] Implement dry-run mode (validates inputs without computing)
- [ ] Add progress callbacks (for GUI/progress bar integration)
- [ ] Create `PostprocessorConfig` dataclass
- [ ] Save partial results on failure

**Proposed Config API:**
```python
from dataclasses import dataclass

@dataclass
class PostprocessorConfig:
    skip_born_if_missing: bool = True
    fallback_to_phonopy_if_symfc_fails: bool = True
    max_dos_mesh: tuple = (20, 20, 20)
    cache_force_constants: bool = True
    skip_phases: list = None          # ['bands', 'hydrogen']
    dry_run: bool = False
    verbose: bool = True
    save_partial_results: bool = True

# Usage
config = PostprocessorConfig(
    skip_born_if_missing=True,
    cache_force_constants=True,
    skip_phases=['hydrogen']  # Skip H-analysis for non-hydrides
)

results = postprocessor.execute(config)
print(results.phase_status)
# {
#     'data': {'completed': True, 'error': None, 'duration': 12.3},
#     'ifc': {'completed': True, 'error': None, 'duration': 45.2},
#     'bands': {'completed': False, 'error': 'seekpath failed', 'duration': 2.1},
#     'thermo': {'completed': True, 'error': None, 'duration': 30.5},
#     ...
# }
```

**Proposed Error Recovery:**
```python
def _safe_execute(self, phase_name, func, *args, **kwargs):
    """Execute a phase with error handling and status tracking."""
    start = time.time()
    try:
        result = func(*args, **kwargs)
        self.phase_status[phase_name] = {
            'completed': True,
            'error': None,
            'duration': time.time() - start
        }
        return result
    except Exception as e:
        self.phase_status[phase_name] = {
            'completed': False,
            'error': str(e),
            'duration': time.time() - start
        }
        if self.config.save_partial_results:
            self._save_partial_results()
        if self.config.verbose:
            print(f"  [!] Phase '{phase_name}' failed: {e}")
        return None
```

**Deliverable:** Resilient pipeline with partial result saving and detailed status reports.

---

### Phase 3: Scientific Enhancements (Week 3)

**Goal:** Add missing physics capabilities.

**New Features:**

#### 3.1 Group Velocities
```python
def compute_group_velocities(self, q_mesh=None):
    """Compute phonon group velocities: v_g = ∂ω/∂q.

    Uses central finite difference on phonon dispersion.
    Essential for thermal conductivity and mean free path.

    Returns:
        dict: {
            'q_points': array (n_q, 3),
            'frequencies': array (n_q, n_bands),
            'group_velocities': array (n_q, n_bands, 3),  # m/s
        }
    """
```

#### 3.2 Mode-Resolved Thermodynamics
```python
def compute_partial_thermodynamics(self, temperature_range=None):
    """Compute F(T), S(T), Cv(T) contribution per element.

    S(T) = Σ_i S_i(T) where S_i is from atom i's partial DOS.
    Enables understanding which atoms dominate entropy.

    Returns:
        dict: {
            'total': {'F': ..., 'S': ..., 'Cv': ...},
            'by_element': {
                'Mg': {'F': ..., 'S': ..., 'Cv': ...},
                'H': {'F': ..., 'S': ..., 'Cv': ...},
            }
        }
    """
```

#### 3.3 Grüneisen Parameters
```python
def compute_mode_gruneisen(self, freq_at_V1, freq_at_V2, V1, V2):
    """Compute mode Grüneisen parameter: γ_i = -d(ln ω_i)/d(ln V).

    Requires phonon calculations at 2+ volumes.
    Essential for thermal expansion (QHA) and anharmonicity.

    Returns:
        array: γ_i for each mode (n_q, n_bands)
    """
```

#### 3.4 Debye-Waller Factors
```python
def compute_debye_waller(self, temperature=300):
    """Compute mean square displacement <u²> for each atom.

    <u²>_i = (ℏ/2M_i) × Σ_qj |e_i(qj)|² / ω(qj) × coth(ℏω/2kT)

    Essential for X-ray/neutron diffraction and B-factor analysis.

    Returns:
        dict: {
            'mean_square_displacement': array (n_atoms, 3),  # A²
            'isotropic_B_factor': array (n_atoms),           # A²
        }
    """
```

#### 3.5 Phonon Mean Free Path
```python
def compute_mean_free_path(self, linewidths=None):
    """Compute phonon mean free path: ℓ = v_g / (2Γ).

    Requires group velocities and linewidths (from anharmonic module).
    Essential for understanding thermal conductivity limits.

    Returns:
        dict: {
            'mean_free_path': array (n_q, n_bands),  # nm
            'cumulative_kappa': array (sorted by MFP),
        }
    """
```

**Deliverable:** 5 new scientific capabilities with full documentation.

---

### Phase 4: Visualization Upgrade (Week 4)

**Goal:** Publication-ready plots with theming.

**Tasks:**
- [ ] Create `PhononPlotter` class with configurable themes
- [ ] Add interactive plots (plotly optional dependency)
- [ ] Add phonon fat bands (element-projected band structure)
- [ ] Add cumulative thermal conductivity plot
- [ ] Add phonon convergence plots (mesh density, supercell size)
- [ ] Support multiple materials overlay
- [ ] Export to PDF/SVG/PNG with consistent styling

**Proposed API:**
```python
class PhononPlotter:
    THEMES = {
        'nature': {
            'band_color': '#1E3A8A',
            'imaginary_color': '#EF4444',
            'stable_badge': '#D1FAE5',
            'unstable_badge': '#FEE2E2',
            'font': 'serif',
            'dpi': 300,
        },
        'science': {
            'band_color': '#000000',
            'imaginary_color': '#FF0000',
            ...
        },
        'dark': {
            'band_color': '#60A5FA',
            'background': '#1a1a2e',
            ...
        }
    }

    def __init__(self, theme='nature', interactive=False):
        self.theme = self.THEMES[theme]
        self.interactive = interactive

    def plot_band_structure(self, band_dict, highlight_imaginary=True,
                           save_path=None):
        """Plot phonon band structure with stability badge."""

    def plot_fat_bands(self, band_dict, pdos, element_projection='all',
                      save_path=None):
        """Plot element-projected ('fat') band structure.

        Band width ∝ element's contribution to that mode.
        """

    def plot_partial_dos(self, pdos, group_by=None, save_path=None):
        """Plot partial DOS with optional element grouping.

        group_by: None, 'metal/nonmetal', or custom dict
        """

    def plot_thermodynamics(self, thermo_data, show_dulong_petit=True,
                           save_path=None):
        """Plot F(T), S(T), Cv(T) with Dulong-Petit limit."""

    def plot_cumulative_kappa(self, mfp_data, save_path=None):
        """Plot cumulative thermal conductivity vs mean free path.

        Shows what fraction of κ comes from phonons with ℓ < x.
        """

    def save(self, path, format='pdf'):
        """Save all plots in specified format."""
```

**Example Fat Band Output:**
```
Band Structure (Fat Bands)
==========================
Band width ∝ element contribution:
  Mg: blue (metallic character)
  H:  red (hydrogen modes)

  Γ → X → M → Γ → Z
  ════════════════════
  ┃    ╭───╮         ┃  ← Mg-dominated (wide blue)
  ┃   ╭╯   ╰╮        ┃
  ┃  ╭╯ H-H ╰╮       ┃  ← H-H stretch (wide red)
  ┃ ╭╯       ╰╮      ┃
  ┃╭╯         ╰╮     ┃
  ┗━━━━━━━━━━━━━┗━━━┃  ← Acoustic modes
  0   5   10   15   20 THz
```

**Deliverable:** 6 publication-ready plot types with 3 themes.

---

### Phase 5: Standardized Export (Week 5)

**Goal:** Single JSON export with all results and metadata.

**Tasks:**
- [ ] Create `PhononResults` dataclass
- [ ] Implement `to_json()`, `to_csv()`, `to_xarray()` methods
- [ ] Add metadata (version, timestamp, input files, git hash)
- [ ] Create export templates (minimal, full, publication)
- [ ] Add validation (schema check on export)

**Proposed Data Structure:**
```json
{
  "metadata": {
    "hydrophonokit_version": "2.7.0",
    "timestamp": "2026-04-12T14:30:00Z",
    "formula": "MgH2",
    "space_group": "P4_2/mnm (136)",
    "crystal_system": "Tetragonal",
    "n_atoms_primitive": 6,
    "n_atoms_supercell": 96,
    "input_files": {
      "phonopy_disp_yaml": "workspace/phonopy_disp.yaml",
      "born_outcar": "workspace/01_born/OUTCAR",
      "force_directories": "workspace/02_displacements/disp-*"
    },
    "git_hash": "abc123def"
  },
  "force_constants": {
    "method": "symfc",
    "symfc_version": "1.2.0",
    "asr_applied": true,
    "asr_type": "reciprocal",
    "nac_applied": true,
    "born_factor": 14.399652,
    "max_fc_eV_A2": 45.67,
    "sum_rule_error_eV_A3": 1.2e-6
  },
  "band_structure": {
    "k_path": "Γ → X → M → Γ → Z",
    "n_q_points": 101,
    "n_bands": 18,
    "min_frequency_THz": -0.45,
    "max_frequency_THz": 42.3,
    "is_stable": false,
    "n_imaginary_modes": 2,
    "imaginary_modes": [
      {"q": [0.0, 0.0, 0.0], "frequency_THz": -0.45, "band": 3},
      {"q": [0.1, 0.0, 0.0], "frequency_THz": -0.12, "band": 4}
    ],
    "band_data_file": "band.yaml"
  },
  "density_of_states": {
    "mesh": [15, 15, 15],
    "smearing": "Gaussian",
    "sigma_THz": 0.2,
    "total_dos_file": "phonon_dos_partial.png",
    "by_element": {
      "Mg": {"peak_THz": 8.5, "integrated_states": 45.2},
      "H": {"peak_THz": 37.3, "integrated_states": 18.7}
    }
  },
  "thermodynamics": {
    "ZPE_kJ_mol": 156.789,
    "temperature_range_K": [0, 1000],
    "temperature_step_K": 10,
    "validation": {
      "third_law_ok": true,
      "entropy_at_0K": 2.3e-8,
      "dulong_petit_check": {
        "limit_J_molK": 598.64,
        "at_1000K_J_molK": 580.2,
        "deviation_pct": 3.1,
        "acceptable": true
      }
    },
    "at_300K": {
      "F_kJ_mol": 142.3,
      "S_J_molK": 89.4,
      "Cv_J_molK": 72.1
    },
    "at_500K": {
      "F_kJ_mol": 118.7,
      "S_J_molK": 125.6,
      "Cv_J_molK": 88.3
    },
    "data_file": "thermodynamic_properties.dat"
  },
  "hydrogen_analysis": {
    "present": true,
    "n_H_atoms": 2,
    "mode_decomposition": {
      "librational_pct": 12.3,
      "bending_pct": 18.7,
      "stretching_pct": 69.0
    },
    "peak_stretching": {
      "frequency_THz": 37.3,
      "frequency_cm": 1245,
      "assignment": "Mg-H stretch (ionic hydride)"
    },
    "hydride_type": "ionic",
    "data_file": "H_mode_analysis.png"
  },
  "phase_status": {
    "data": {"completed": true, "error": null, "duration_s": 12.3},
    "ifc": {"completed": true, "error": null, "duration_s": 45.2},
    "bands": {"completed": true, "error": null, "duration_s": 30.1},
    "thermo": {"completed": true, "error": null, "duration_s": 25.7},
    "hydrogen": {"completed": true, "error": null, "duration_s": 8.4}
  },
  "total_duration_s": 121.7
}
```

**Export API:**
```python
class PhononResultsExporter:
    def export_json(self, results, path, template='full'):
        """Export results as JSON.

        template: 'minimal' (metadata + key results only),
                  'full' (all data),
                  'publication' (full + plot paths + citations)
        """

    def export_csv(self, results, path):
        """Export thermodynamics and DOS as CSV."""

    def export_xarray(self, results):
        """Export as xarray Dataset for advanced analysis."""

    def validate(self, results):
        """Validate results against schema.

        Returns: {'valid': bool, 'errors': list}
        """
```

**Deliverable:** Standardized, validated export with full metadata.

---

### Phase 6: Caching & Performance (Week 6)

**Goal:** Skip already-computed phases for faster re-runs.

**Tasks:**
- [ ] Add `--use-cache` option
- [ ] Cache force constants if `FORCE_CONSTANTS` exists and is valid
- [ ] Cache band structure if `band.yaml` exists
- [ ] Cache DOS if `dos_data.npz` exists
- [ ] Parallelize DOS computation (per-element projection)
- [ ] Add memory usage monitoring for large systems
- [ ] Add progress bar (tqdm optional)

**Proposed Cache API:**
```python
class PhononCache:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or '.hydrophonokit_cache'
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_force_constants(self, workspace_hash):
        """Load cached force constants if available and valid."""
        path = os.path.join(self.cache_dir, f'fc_{workspace_hash}.npz')
        if os.path.exists(path):
            meta = self._load_metadata(path)
            if self._is_valid(meta):
                return self._load(path)
        return None

    def save_force_constants(self, fc, workspace_hash, metadata):
        """Save force constants with metadata for validation."""
        path = os.path.join(self.cache_dir, f'fc_{workspace_hash}.npz')
        np.savez_compressed(path, fc=fc, metadata=json.dumps(metadata))

    def _is_valid(self, metadata):
        """Check if cache entry is still valid.

        Checks:
        - phonopy_disp.yaml hasn't changed (mtime, hash)
        - Force constant shape matches current system
        - NAC params match (if applicable)
        """
```

**Performance Expectations:**
| Operation | First Run | Cached Run | Speedup |
|-----------|-----------|------------|---------|
| Force extraction (100 disp) | 5-15 min | < 10 sec | 30-90x |
| Force constants (symfc) | 1-5 min | < 5 sec | 12-60x |
| Band structure | 30 sec | < 5 sec | 6x |
| DOS (15³ mesh) | 2-3 min | < 10 sec | 12-18x |

**Deliverable:** Fast re-runs with validated caching.

---

## 📊 Priority Matrix

| Priority | Phase | Impact | Effort | Risk |
|----------|-------|--------|--------|------|
| 🔴 P0 | 1. Architecture Refactoring | Enables all future work | 1 week | Low (move-only) |
| 🔴 P0 | 2. Error Handling | Prevents data loss | 3 days | Low |
| 🟠 P1 | 5. Standardized Export | Reproducibility | 2 days | Low |
| 🟠 P1 | 3.1 Group Velocities | Essential for transport | 1 day | Medium |
| 🟡 P2 | 3. Scientific Enhancements | Advanced analysis | 3 days | Medium |
| 🟡 P2 | 4. Visualization | Publication quality | 3 days | Low |
| 🟢 P3 | 6. Caching | Convenience | 2 days | Low |

---

## 🎯 Implementation Order

```
Week 1: Architecture Refactoring (Phase 1)
    ↓
Week 2: Error Handling (Phase 2)
    ↓
Week 3: Scientific Enhancements (Phase 3)
    ↓
Week 4: Visualization Upgrade (Phase 4)
    ↓
Week 5: Standardized Export (Phase 5)
    ↓
Week 6: Caching & Performance (Phase 6)
```

**Total Timeline:** 6 weeks
**Total New Code:** ~3,500 lines
**Refactored Code:** ~752 lines (reorganized, not rewritten)

---

## ⚠️ Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing behavior | User workflow disruption | Comprehensive regression tests before each phase |
| Import path changes | CLI breakage | Update all imports systematically, test CLI after Phase 1 |
| symfc API changes | IFC computation fails | Pin symfc version in pyproject.toml |
| Large vasprun.xml memory | OOM on big systems | Streaming XML parsing, memory monitoring |
| Plotting theme conflicts | User matplotlib config broken | Use figure-level rcParams, not global |

---

## ✅ Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Code organization | Lines per module | < 200 |
| Test coverage | Postprocessor tests | > 80% |
| Error recovery | Partial failure handling | Save results, report status |
| Export quality | Metadata completeness | All fields populated |
| Plot quality | Publication-ready | 3 themes, PDF export |
| Performance | Cached re-run time | < 30 sec (vs 15+ min first run) |

---

*End of Postprocessor Development Plan — HydroPhonoKit v2.7.0 Target*
