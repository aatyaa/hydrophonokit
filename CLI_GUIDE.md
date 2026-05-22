# HydroPhonoKit CLI — Terminal Interface Guide

> **Version:** 2.6.0
> **Scope:** Complete guide to using the HydroPhonoKit command-line interface

---

## 🎯 Quick Start

```bash
# Show help and available commands
hydrophonokit --help

# Start interactive wizard
hydrophonokit

# Run a command directly
hydrophonokit analyze --source /path/to/vasp_relaxed
```

---

## 📋 Available Commands

### Core Workflow

| Command | Description |
|---------|-------------|
| `hydrophonokit analyze` | Analyze a relaxed VASP directory (read-only) |
| `hydrophonokit build` | Generate phonon calculation workspace |
| `hydrophonokit postprocess` | Analyze completed VASP phonon results |

### Advanced Analysis

| Command | Description |
|---------|-------------|
| `hydrophonokit elastic` | Extract elastic constants from phonon dispersion |
| `hydrophonokit qha` | Quasi-Harmonic Approximation (thermal expansion) |
| `hydrophonokit anharmonic` | Phonon linewidths, lifetimes, thermal conductivity |
| `hydrophonokit hstorage` | Hydrogen storage thermodynamics & DOE targets |

---

## 🔧 Command Details

### 1. Analyze

Analyze a relaxed VASP directory without generating any files.

```bash
hydrophonokit analyze --source /path/to/vasp_relaxed
```

**Required:**
- `--source`: Path to directory containing CONTCAR, INCAR, OUTCAR, POTCAR

**Output:**
- Material profile (formula, symmetry, composition)
- Convergence status (forces, stress, pressure)
- Electronic structure (bandgap, insulator/metal)
- Phonon readiness score
- Recommended supercell, displacement, VDW settings

---

### 2. Build

Generate a complete phonon calculation workspace.

```bash
hydrophonokit build --source /path/to/vasp_relaxed --output /path/to/workspace
```

**Required:**
- `--source`: Path to relaxed VASP directory
- `--output`: Path for output workspace

**Output:**
- `00_unitcell/POSCAR` — Reference unit cell
- `01_born/` — DFPT Born charge calculation (if insulator)
- `02_displacements/` — Displaced supercells with INCARs
- `phonopy_disp.yaml` — Phonopy displacement data
- `submit_born.sh` — Script to submit Born calculation
- `submit_all_disps.sh` — Script to submit all displacements
- `PHONON_PLAN.txt` — Summary report

---

### 3. Postprocess

Analyze completed VASP phonon results.

```bash
hydrophonokit postprocess --source /path/to/vasp_relaxed --workspace /path/to/completed_ws
```

**Required:**
- `--source`: Path to relaxed VASP directory
- `--workspace`: Path to completed phonon workspace

**Output:**
- `phonon_band_structure.png` — Band structure plot
- `phonon_dos_partial.png` — Partial DOS plot
- `phonon_thermodynamics.png` — F(T), S(T), Cv(T) plots
- `thermodynamic_properties.dat` — Raw thermodynamic data
- `Phonon_Analysis_Report.html` — HTML summary report
- (If H present) `H_mode_analysis.png` — H mode decomposition

---

### 4. Elastic Constants

Extract elastic constants from phonon dispersion.

```bash
hydrophonokit elastic --source /path/to/vasp_relaxed --workspace /path/to/phonon_ws
```

**Output:**
- C11, C12, C44 (cubic crystals)
- Sound velocities (LA, TA1, TA2)
- Bulk modulus, shear modulus, Young's modulus, Poisson's ratio
- Debye temperature from elastic data
- Mechanical stability check (Born-Huang criteria)
- `elastic_constants.json` — Full results

---

### 5. Quasi-Harmonic Approximation

Compute thermal expansion from multi-volume phonon calculations.

```bash
hydrophonokit qha --source qha_input.json --output qha_results/
```

**Input JSON Format:**
```json
{
  "formula": "Si",
  "volumes": [38.0, 39.0, 40.0, 41.0, 42.0],
  "energies_DFT": [-10.50, -10.55, -10.58, -10.54, -10.48],
  "free_energies_T": [
    [F(0K), F(10K), ..., F(1000K)],
    ...
  ]
}
```

**Output:**
- V(T) — Equilibrium volume vs temperature
- α(T) — Thermal expansion coefficient
- B(T) — Bulk modulus vs temperature
- C_p(T) — Heat capacity at constant pressure
- γ(T) — Gruneisen parameter
- `qha_results.png` — 4-panel plot
- `qha_data.json` — Full data

---

### 6. Anharmonic Properties

Compute phonon linewidths, lifetimes, and frequency shifts.

```bash
hydrophonokit anharmonic --source /path/to/vasp_relaxed --workspace /path/to/phonon_ws
```

**Output:**
- Phonon linewidths Γ(q,j,T)
- Phonon lifetimes τ(q,j,T) = 1/(2Γ)
- Frequency shifts Δω(T)
- Average lifetime vs temperature
- Slack model thermal conductivity κ(T)
- `anharmonic_results.png` — 4-panel plot
- `anharmonic_data.json` — Full data

---

### 7. Hydrogen Storage

Complete H-storage thermodynamics and DOE target evaluation.

```bash
hydrophonokit hstorage --hydride /path/to/hydride --dehydride /path/to/dehydride
```

**Optional Arguments:**
- `--h2-energy`: DFT energy of H₂ molecule in eV (default: -6.77)
- `--n-h2`: Moles of H₂ released (auto-computed from n_h)

**Input Data:**
Each directory should contain either:
- `h_storage_data.json` — Pre-computed data
- `thermodynamic_properties.dat` — From HydroPhonoKit postprocessing

**Output:**
- ΔH(T), ΔS(T), ΔG(T) vs temperature
- Desorption temperature T_des at various pressures
- Gravimetric capacity (wt%)
- Volumetric capacity (g/L)
- Van't Hoff plot (ln P vs 1/T)
- DOE 2025 target evaluation
- H-vibrational mode decomposition
- `h_thermodynamics.png` — ΔH, ΔG, ΔS plots
- `vant_hoff.png` — Van't Hoff plot
- `h_storage_data.json` — Full data

---

## 🎨 Interactive Wizard

Run `hydrophonokit` without arguments to start the interactive wizard:

```
 ╔══════════════════════════════════════════════════════╗
 ║                                                      ║
 ║   ____  __                          __ __ _ __      ║
 ║  / __ \/ /_  ____  ____  ____     / //_/(_) /_     ║
 ║ / /_/ / __ \/ __ \/ __ \/ __ \   / ,<  / / __/    ║
 ║/ ____/ / / / /_/ / / / / /_/ /  / /| |/ / /_      ║
 ║/_/   /_/ /_/\____/_/ /_/\____/  /_/ |_/_/\__/     ║
 ║                                        v2.6.0              ║
 ║         Material-Aware Phonon Workflow Engine       ║
 ║                                                      ║
 ╚══════════════════════════════════════════════════════╝

  Welcome to HydroPhonoKit Interactive Wizard!
  Run 'hydrophonokit --help' for direct command usage.

  What would you like to do?
    1. 📊 Analyze a relaxed VASP directory (read-only)
    2. 🏗️  Build a phonon workspace (analyze + generate)
    3. 📈 Post-process completed VASP phonon results
    4. 🔬 Extract elastic constants
    5. 🌡️  Quasi-Harmonic Approximation (thermal expansion)
    6. 📉 Compute anharmonic properties (linewidths, lifetimes)
    7. 💧 Hydrogen storage thermodynamics & DOE targets
    8. Exit

  Select (1-8):
```

---

## 🚨 Error Messages

The CLI provides clear error messages:

```
  [❌] ERROR: --source is required for 'analyze'.
    Usage: python -m hydrophonokit analyze --source <PATH>
```

```
  [❌] ERROR: Source directory is missing: CONTCAR, INCAR
    Directory: /path/to/input
```

```
  [❌] UNEXPECTED ERROR: ...
  This may be a bug in HydroPhonoKit. Please report it.
  [traceback follows]
```

---

## 💡 Tips

1. **Use short options:** `-s` for `--source`, `-o` for `--output`, `-w` for `--workspace`
2. **Tab completion:** Works with bash/zsh for directory paths
3. **Pipe output:** `hydrophonokit analyze -s DIR 2>&1 | tee analysis.log`
4. **Check requirements:** `hydrophonokit --help` shows full command reference
5. **Version info:** Banner shows current version on every run

---

## 📖 Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `SLURM_EMAIL` | Email for SLURM notifications | None (disabled) |
| `MPLBACKEND` | Matplotlib backend | Agg (non-interactive) |

---

## 📊 Example Workflow

```bash
# Step 1: Analyze relaxed structure
hydrophonokit analyze -s ./BaTiO3_relax

# Step 2: Build workspace
hydrophonokit build -s ./BaTiO3_relax -o ./BaTiO3_phonon

# → Upload to HPC, run VASP jobs

# Step 3: Postprocess results
hydrophonokit postprocess -s ./BaTiO3_relax -w ./BaTiO3_phonon

# Step 4: Advanced analysis
hydrophonokit elastic -s ./BaTiO3_relax -w ./BaTiO3_phonon
hydrophonokit anharmonic -s ./BaTiO3_relax -w ./BaTiO3_phonon
```

---

*End of CLI Guide — HydroPhonoKit v2.6.0*
