# HydroPhonoKit

**A Material-Aware Scientific Framework for Automated VASP/Phonopy Phonon Calculations**

---

## 🌟 Overview

**HydroPhonoKit** is an advanced, high-performance Python framework designed to automate, compute, and post-process phonon and thermodynamic properties of materials, with a particular focus on hydrogen storage systems and metal hydrides. It bridges the gap between raw Density Functional Theory (DFT) calculations (like VASP) and macroscopic thermodynamic properties.

Developed and maintained exclusively by **Attia A Gadallah** (ava7312@psu.edu).

---

## 📂 System Architecture & Components

The framework is modular, separating core physics, automated pipeline processing, and publication-ready visualizations.

```
hydrophonokit/
├── hydrophonokit/
│   ├── postprocessor/        # Pipeline orchestration & data extraction
│   ├── visualization/        # Publication-quality plotting
│   ├── analyzer.py           # Material properties & DFT analytics
│   ├── physics.py            # CODATA 2018 physical constants & formulas
│   ├── eos.py                # Equations of State (Vinet, Birch-Murnaghan)
│   ├── qha.py                # Quasi-Harmonic Approximation engine
│   ├── h2_molecule.py        # H2 gas partition functions
│   ├── h_storage.py          # Hydride thermodynamics & DOE target matching
│   ├── elastic.py            # Elastic constants & moduli calculator
│   └── anharmonic.py         # Slack thermal conductivity & Grüneisen parameters
```

### 1. Core Scientific Engine (`hydrophonokit/`)
*   **`physics.py`**: The mathematical backbone of the framework. Built using CODATA 2018 physical constants, it computes the Bose-Einstein distribution, vibrational partition functions, zero-point energy (ZPE), vibrational internal energy, Helmholtz free energy, entropy, and constant-volume heat capacity ($C_v$).
*   **`eos.py`**: Fits energy-volume ($E-V$) data curves to extract equilibrium volume ($V_0$), bulk modulus ($B_0$), and its pressure derivative ($B_0'$). Supported models include:
    *   **Vinet Equation of State** (most accurate for high compression).
    *   **Birch-Murnaghan (3rd order)**.
    *   **Murnaghan**.
*   **`qha.py` (Quasi-Harmonic Approximation)**: Integrates $F(V, T)$ Helmholtz free energy curves over temperature and volume grids. It determines the thermal expansion coefficient ($\alpha(T)$), bulk modulus temperature dependence ($B_0(T)$), and computes the constant-pressure heat capacity ($C_p(T) = C_v(T) + V T \alpha^2 B_0$).
*   **`h2_molecule.py`**: Computes the thermodynamic properties of $H_2$ gas. It evaluates translation (Sackur-Tetrode), rotation (distinguishing between *ortho* and *para* states with a 3:1 high-temperature limit ratio), and vibration to get absolute gas entropy $S_{H_2}(T, P)$ and enthalpy $H_{H_2}(T)$ without relying on constant empirical approximations.
*   **`h_storage.py`**: Focuses on hydrogen storage systems. It computes:
    *   Reaction enthalpy $\Delta H(T)$ and entropy $\Delta S(T)$ for hydride decomposition ($MH_x \rightarrow M + \frac{x}{2}H_2$).
    *   Equilibrium pressure plateau ($P_{eq}(T)$) via the Van 't Hoff relation: $\ln(P_{eq}/P_0) = \frac{2}{x} \left( \frac{\Delta H}{RT} - \frac{\Delta S}{R} \right)$.
    *   Decomposition temperature ($T_{dec}$) at $1\text{ bar}$.
    *   Gravimetric and volumetric hydrogen storage capacities compared directly with **US Department of Energy (DOE) targets**.
*   **`elastic.py`**: Analyzes the elastic tensor to compute Voigt-Reuss-Hill bounds for Bulk ($B$), Shear ($G$), Young's ($E$) moduli, Poisson's ratio ($\nu$), and longitudinal/transverse acoustic velocities.
*   **`anharmonic.py`**: Estimates lattice thermal conductivity ($\kappa_L(T)$) using the Slack model, based on Debye temperatures ($\theta_D$), acoustic velocities, and the mode Grüneisen parameter ($\gamma$).

### 2. Pipeline Post-Processor (`hydrophonokit/postprocessor/`)
Orchestrates data extraction and executes an isolated, sequential 6-phase pipeline:
*   **Phase 1: Data Loading (`data_loader.py`)**: Parses VASP outputs (`OUTCAR`, `vasprun.xml`) and Phonopy files.
*   **Phase 2: IFC & Born Corrections (`ifc_computer.py`)**: Computes Second-Order Interatomic Force Constants (IFCs) and handles Non-Analytical Term Corrections (Born effective charges and dielectric tensors) to resolve LO-TO splitting in polar materials.
*   **Phase 3: Phonon Dispersion & DOS (`bands_dos.py`)**: Generates high-symmetry band paths and density of states grids.
*   **Phase 4: Thermo Integration & Debye-Waller (`thermodynamics.py`, `debye_waller.py`)**: Evaluates vibrational integrals and atomic mean-square displacements.
*   **Phase 5: Mode-Resolved Transport (`mode_resolved_thermo.py`, `group_velocities.py`)**: Calculates phonon group velocities and mode-specific thermal properties.
*   **Phase 6: Hydrogen Storage Assessment (`hydrogen.py`)**: Combines solid state thermodynamics with the gas phase to export hydride metrics.
*   **Caching Engine (`caching.py`)**: Caches intermediate steps to prevent repeating heavy computational tasks when running analysis repeatedly.
*   **Enhanced Exporters (`enhanced_export.py`)**: Outputs standardized, human-readable structured JSON reports and publication-ready CSVs containing full temperature grids.

### 3. Visualization Suite (`hydrophonokit/visualization/`)
Generates premium, publication-quality vector plots:
*   `band_plots.py` & `dos_plots.py`: Combined phonon dispersion and DOS plots.
*   `thermo_plots.py`: Free energy, entropy, heat capacity ($C_v$, $C_p$), and Debye temperature plots.
*   `hydrogen_plots.py`: Van 't Hoff plots, P-C isothermal curves, and DOE targets comparisons.
*   `transport_plots.py`: Mode-resolved group velocities and Slack lattice thermal conductivity profiles.
*   `themes.py` & `publication_formatter.py`: Applies consistent aesthetics with elegant HSL color palettes, customizable grid layout rules, and standard fonts (LaTeX-ready).

---

## 🔬 Scientific Foundations & Equations

### 1. Vibrational Helmholtz Free Energy
$$F_{vib}(T) = r \int_0^\infty \left[ \frac{\hbar \omega}{2} + k_B T \ln\left(1 - e^{-\frac{\hbar \omega}{k_B T}}\right) \right] g(\omega) d\omega$$
where $g(\omega)$ is the phonon density of states and $r$ is the number of degrees of freedom.

### 2. Vinet Equation of State
$$E(V) = E_0 + \frac{4 B_0 V_0}{(B_0' - 1)^2} \left[ 1 - (1 - x)\exp(x) \right]$$
where $x = \frac{3}{2}(B_0' - 1)\left[ 1 - \left(\frac{V}{V_0}\right)^{1/3} \right]$.

### 3. Sackur-Tetrode Equation (Hydrogen Translation)
$$S_{trans} = R \left[ \ln\left( \frac{V_m}{\Lambda^3} \right) + 2.5 \right], \quad \Lambda = \frac{h}{\sqrt{2\pi m k_B T}}$$

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Attia-A-Gadallah/hydrophonokit.git
cd hydrophonokit

# Install requirements
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

---

## 🚀 Usage Guide

### 1. Command Line Interface (CLI)

The CLI is the quickest way to execute pipelines and generate reports.

#### Run a Complete Post-Processing Pipeline
```bash
hydrophonokit run --workspace ./vasp_calculation --output ./results --cache
```
*   `--workspace`: Directory containing VASP calculation files (`POSCAR`, `OUTCAR`, `FORCE_SETS`, etc.).
*   `--output`: Output directory where CSVs, JSONs, and PDF plots will be saved.
*   `--cache`: Enables intermediate caching to speed up subsequent runs.

#### Generate Specific Plots
```bash
hydrophonokit plot --json ./results/phonon_results.json --type bands_dos --output ./plots
hydrophonokit plot --json ./results/phonon_results.json --type thermo --output ./plots
```

#### Run Hydrogen Storage Analysis
```bash
hydrophonokit analyze --type hydrogen --hydride MgH2 --metal Mg --output ./hydrogen_results
```

---

### 2. Programmatic Python API (Code Examples)

For custom scientific scripts, you can interface directly with the Python API.

#### Example A: Executing the Post-Processor Pipeline
```python
import os
from hydrophonokit.postprocessor.core import PhononPostProcessor, PostprocessorConfig
from hydrophonokit.analyzer import MaterialProfile

# 1. Define the material metadata
profile = MaterialProfile(
    formula="MgH2",
    space_group="P42/mnm",
    elements={"Mg": 1, "H": 2}
)

# 2. Configure the pipeline options
config = PostprocessorConfig(
    use_cache=True,
    cache_dir="./.hydrophonokit_cache",
    temperature_min=0,
    temperature_max=1000,
    temperature_step=10
)

# 3. Initialize and run the post-processor
workspace = "./MgH2_dft_data"
processor = PhononPostProcessor(workspace_dir=workspace, profile=profile, config=config)
results = processor.execute_pipeline()

# 4. Access computed thermodynamic values
zpe = results['thermodynamics']['zpe'] # Zero Point Energy
entropy_300 = results['thermodynamics']['entropy'][30] # Entropy at 300K (index 30 for step=10)
print(f"Zero-Point Energy: {zpe:.4f} kJ/mol")
print(f"Entropy at 300 K: {entropy_300:.4f} J/(mol·K)")
```

#### Example B: Fitting an Equation of State (QHA)
```python
import numpy as np
from hydrophonokit.eos import fit_eos

# Example volumes (in A^3) and energies (in eV) from DFT calculations
volumes = np.array([55.2, 57.1, 59.0, 61.0, 63.1])
energies = np.array([-12.35, -12.48, -12.52, -12.45, -12.28])

# Fit to Vinet EOS
eos_parameters, fit_curve = fit_eos(volumes, energies, eos_type="vinet")

E0, V0, B0_GPa, B0_prime = eos_parameters
print(f"Equilibrium Volume: {V0:.3f} A^3")
print(f"Bulk Modulus: {B0_GPa:.2f} GPa")
print(f"Pressure Derivative B0': {B0_prime:.2f}")
```

#### Example C: Plotting Phonon Dispersion and DOS
```python
from hydrophonokit.visualization.figure_factory import PhononFigureFactory
from hydrophonokit.postprocessor.enhanced_export import PhononResultsExporter

# Load exported results dictionary (or obtain it from pipeline execution)
exporter = PhononResultsExporter()
results = exporter.load_results("./results/phonon_results.json")

# Instantiate visual factory with publication theme
factory = PhononFigureFactory(style="publication")

# Create a combined Band Structure & DOS plot
fig = factory.create_bands_dos_plot(
    bands_data=results['bands'],
    dos_data=results['dos']
)

# Save figure as vector graphic
fig.savefig("./plots/phonon_dispersion_dos.pdf", dpi=300, bbox_inches="tight")
```

---

## 📊 Sample Outputs & Formats

HydroPhonoKit exports highly standardized, human-readable data files for direct use in publication plotting or supplementary materials.

### 1. Thermodynamic Properties Text File (`thermodynamic_properties.dat`)
This file contains the calculated free energy ($F$), entropy ($S$), and constant-volume heat capacity ($C_v$) calculated over the defined temperature grid, prefixed with thermodynamic boundary limits and zero-point energy:

```text
# PhononKit v2.7 -- Thermodynamic Properties (Harmonic Approximation)
# Reference: Born & Huang, Dynamical Theory of Crystal Lattices (1954)
#
# Material: Na2Ca2B6H24
# Atoms per cell: 34
# Dulong-Petit limit: 848.08 J/(mol·K)
# ZPE: 619.091 kJ/mol
#
# T(K)   F(kJ/mol)   S(J/mol·K)   Cv(J/mol·K)
   0.0     619.0906       0.0000       0.0000
  10.0     619.0812       2.8511       5.6020
  20.0     619.0178      10.4861      18.9892
  30.0     618.8604      21.4330      36.5808
  40.0     618.5820      34.5414      55.5970
  ...
```

### 2. Structured JSON Output (`phonon_results.json`)
The complete analysis results are dumped as a validated schema-compliant JSON file. This includes metadata, convergence warnings, and detailed phase-by-phase calculations:

```json
{
  "metadata": {
    "hydrophonokit_version": "2.7.0",
    "timestamp": "2026-05-22T01:39:16.123456",
    "formula": "Na2Ca2B6H24",
    "space_group": "Fm-3m",
    "n_atoms_primitive": 34,
    "density_g_cm3": 1.28
  },
  "thermodynamics": {
    "computed": true,
    "ZPE_kJ_mol": 619.0906,
    "validation": {
      "dulong_petit_limit_J_mol_K": 848.08,
      "third_law_check": "passed"
    },
    "at_300K": {
      "free_energy_kJ_mol": 512.456,
      "entropy_J_mol_K": 310.124,
      "heat_capacity_J_mol_K": 482.351
    }
  },
  "hydrogen_analysis": {
    "gravimetric_capacity_wt_percent": 7.82,
    "volumetric_capacity_g_L": 100.15,
    "decomposition_temperature_1bar_K": 382.4,
    "reaction_enthalpy_kJ_mol_H2": 42.15,
    "reaction_entropy_J_mol_K_H2": 130.5
  }
}
```

### 3. Publication-Ready Plots (`visualization/`)
The visualizer auto-generates premium-style vector diagrams (PDF/PNG) stored directly in your specified output directory:
- **`phonon_band_structure.png`**: Electronic/phonon band dispersion across high-symmetry k-points.
- **`phonon_dos_total.png`**: Total and atom-projected phonon density of states.
- **`cv_dulong_petit.png`**: Temperature-dependent heat capacity plotted against the classical Dulong-Petit limit.
- **`thermal_conductivity.png`**: Slack model temperature-dependent lattice thermal conductivity ($\kappa_L$).

---

## 📄 Copyright & License

Copyright (c) 2024-2026 **Attia A Gadallah**. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, modification, distribution, or transfer of this software, via any medium, is strictly prohibited.
