# HydroPhonoKit Phase 5: Hydrogen Storage Analytics — Detailed Implementation Plan

> **Version:** Target v2.6.0
> **Scope:** Comprehensive hydrogen storage material evaluation from first-principles phonon calculations
> **Target Users:** Materials scientists, hydrogen storage researchers, DOE program evaluators

---

## 🎯 Objective

Create a complete, publication-grade hydrogen storage analytics module that:

1. Computes **thermodynamics of dehydrogenation** (ΔH, ΔS, ΔG) from DFT + phonon calculations
2. Predicts **equilibrium desorption temperature** T_des
3. Evaluates **gravimetric and volumetric capacities** against DOE targets
4. Analyzes **hydrogen vibrational modes** (librational, bending, stretching)
5. Computes **Van't Hoff plots** for equilibrium H₂ pressure
6. Validates **hydrogen diffusion barriers** (optional)
7. Generates **comprehensive HTML reports** for material screening

---

## 📐 Scientific Foundation

### Thermodynamics of Hydrogen Storage

For a generic dehydrogenation reaction:
```
Metal-Hydride  →  Dehydrogenated Phase  +  n/2 H₂(g)
```

**Enthalpy of dehydrogenation:**
```
ΔH_des(T) = [E_DFT(dehyd) + F_phonon(dehyd,T)] - [E_DFT(hyd) + F_phonon(hyd,T)]
            + (n/2) × [E_DFT(H₂) + F_vib(H₂,T)]
```

**Entropy of dehydrogenation:**
```
ΔS_des(T) = S_vib(dehyd,T) - S_vib(hyd,T) - (n/2) × S_total(H₂,T)

Where S_total(H₂) = S_trans + S_rot + S_vib + S_elec
  - S_trans: translational entropy (Sackur-Tetrode equation)
  - S_rot: rotational entropy (rigid rotor)
  - S_vib: vibrational entropy (harmonic oscillator)
  - S_elec: electronic entropy (negligible at low T)
```

**Gibbs free energy:**
```
ΔG_des(T) = ΔH_des(T) - T × ΔS_des(T)
```

**Equilibrium desorption temperature:**
```
T_des = ΔH_des / ΔS_des   (at P_H₂ = 1 bar)
```

**Van't Hoff equation:**
```
ln(P_H₂ / P₀) = ΔH_des / (R × T) - ΔS_des / R

Where P₀ = 1 bar, R = 8.314 J/(mol·K)
```

**Gravimetric capacity:**
```
wt% = (n × M_H) / M_hydride × 100

Where M_H = 1.008 g/mol, M_hydride = molar mass of hydride
```

**Volumetric capacity:**
```
g/L = (n × M_H) / V_molar

Where V_molar = molar volume of hydride (cm³/mol)
```

---

## 🏗️ Module Architecture

### Files to Create

| File | Lines | Purpose |
|------|-------|---------|
| `h_storage.py` | ~650 | Main H-storage thermodynamics engine |
| `h2_molecule.py` | ~250 | H₂ gas-phase reference properties |
| `tests/test_h_storage.py` | ~200 | Comprehensive tests against known hydrides |

### Files to Modify

| File | Changes | Purpose |
|------|---------|---------|
| `cli.py` | +70 lines | `hydrophonokit h-storage` command |
| `physics.py` | +50 lines | H₂ molecular properties, Sackur-Tetrode, etc. |
| `__init__.py` | Version 2.5.0 → 2.6.0 | Export H-storage functions |
| `CHANGES.md` | +150 lines | Document all changes |

---

## 🔬 Detailed Component Specification

### 1. H₂ Molecule Reference Module (`h2_molecule.py`)

**Purpose:** Accurate gas-phase H₂ reference for thermodynamic calculations.

#### 1.1 H₂ Molecular Properties

```python
class H2Molecule:
    """Reference properties for isolated H₂ molecule.

    Experimental values (NIST CCCBDB):
        - Bond length: 0.7414 Å
        - Vibrational frequency: 4401 cm⁻¹ (131.9 THz)
        - Rotational constant: 60.85 cm⁻¹ (1.82 THz)
        - Dissociation energy: 4.52 eV
        - Zero-point energy: 0.27 eV
    """

    # Molecular constants
    bond_length = 0.7414          # Å
    vib_freq_cm = 4401.0          # cm⁻¹
    vib_freq_thz = 131.9          # THz
    rot_const_cm = 60.85          # cm⁻¹
    rot_const_thz = 1.82          # THz
    dissociation_energy = 4.52    # eV
    zpe = 0.5 * h * vib_freq    # ~0.27 eV

    # Symmetry number for rotational partition function
    symmetry_number = 2  # Homonuclear diatomic
```

#### 1.2 H₂ Partition Functions

```python
def h2_translational_entropy(T, P=1e5):
    """Sackur-Tetrode equation for H₂ translational entropy.

    S_trans = k_B × [ln(V/N × (2πmkT/h²)^(3/2)) + 5/2]

    Where V/N = kT/P (ideal gas)

    Args:
        T: Temperature (K)
        P: Pressure (Pa), default 1 bar

    Returns:
        S_trans in J/(mol·K)

    Reference: McQuarrie, Statistical Mechanics (1976).
    """
```

```python
def h2_rotational_entropy(T):
    """Rigid rotor entropy for H₂.

    S_rot = k_B × [ln(T / (σ × θ_rot)) + 1]

    Where:
        σ = 2 (symmetry number for H₂)
        θ_rot = h² / (8π²Ik_B) = rotational temperature

    For H₂: θ_rot = 87.6 K

    Args:
        T: Temperature (K)

    Returns:
        S_rot in J/(mol·K)
    """
```

```python
def h2_vibrational_entropy(T):
    """Harmonic oscillator vibrational entropy for H₂.

    S_vib = k_B × [x / (e^x - 1) - ln(1 - e^(-x))]

    Where x = hν / (k_B T)

    For H₂: ν = 131.9 THz → x(300K) = 21.2 → S_vib ≈ 0

    Args:
        T: Temperature (K)

    Returns:
        S_vib in J/(mol·K)
    """
```

```python
def h2_total_entropy(T, P=1e5):
    """Total entropy of H₂ gas.

    S_total = S_trans + S_rot + S_vib

    At 300K, 1 bar:
        S_trans ≈ 130.7 J/(mol·K)
        S_rot   ≈  29.0 J/(mol·K)
        S_vib   ≈   0.0 J/(mol·K)  (frozen at room T)
        Total   ≈ 130.7 J/(mol·K)  (cf. NIST: 130.68)

    Returns:
        S_total in J/(mol·K)
    """
```

#### 1.3 H₂ Enthalpy and Gibbs Free Energy

```python
def h2_enthalpy(T, P=1e5):
    """Enthalpy of H₂ gas relative to 0 K.

    H(T) = H_trans + H_rot + H_vib + ZPE

    H_trans = 5/2 × kT  (monatomic ideal gas: 3/2, diatomic: 5/2)
    H_rot = kT          (2 rotational DOF)
    H_vib = hν / (e^(hν/kT) - 1)  → 0 at low T

    Returns:
        H in J/mol
    """

def h2_gibbs(T, P=1e5):
    """Gibbs free energy of H₂ gas.

    G(T,P) = H(T) - T × S(T,P)

    Returns:
        G in J/mol
    """
```

---

### 2. H-Storage Thermodynamics Engine (`h_storage.py`)

#### 2.1 Data Structures

```python
class HydridePhase:
    """Properties of the hydrogenated phase."""
    formula: str             # e.g., "MgH2"
    e_dft: float             # DFT total energy (eV per formula unit)
    free_energy_T: np.array  # F_phonon(T) array (kJ/mol)
    entropy_T: np.array      # S_vib(T) array (J/mol·K)
    volume: float            # Volume per formula unit (Å³)
    n_h: int                 # Number of H atoms per formula unit

class DehydridePhase:
    """Properties of the dehydrogenated phase."""
    formula: str             # e.g., "Mg"
    e_dft: float             # DFT total energy (eV)
    free_energy_T: np.array  # F_phonon(T) (kJ/mol)
    entropy_T: np.array      # S_vib(T) (J/mol·K)
    volume: float            # Volume (Å³)
```

```python
class HStorageResult:
    """Complete hydrogen storage analysis results."""

    # Input materials
    hydride_formula: str          # "MgH2"
    dehydride_formula: str        # "Mg"
    n_h2: float                   # moles H2 released per formula unit

    # Thermodynamics
    delta_H_0K: float             # kJ/mol H2 (0 K enthalpy)
    delta_S_300K: float           # J/(mol·K) (300 K entropy)
    delta_G_T: np.array           # kJ/mol H2 vs T
    T_des_1bar: float             # K (desorption temp at 1 bar)
    T_des_01bar: float            # K (at 0.1 bar)
    T_des_5bar: float             # K (at 5 bar)

    # Capacities
    wt_percent: float             # gravimetric capacity
    g_per_liter: float            # volumetric capacity
    h_density: float              # kg H2 / m3

    # H-vibrational analysis
    h_mode_fractions: dict        # {lib, bend, stretch}
    h_peak_freq: dict             # {stretch_thz, stretch_cm}

    # DOE targets
    doe_gravimetric_target: float # 5.5 wt%
    doe_volumetric_target: float  # 40 g/L
    doe_status: dict              # {gravimetric: pass/fail, volumetric: pass/fail}

    # Van't Hoff data
    vant_hoff_temps: np.array     # K
    vant_hoff_pressures: np.array # bar
```

#### 2.2 Main Engine

```python
class HStorageAnalyzer:
    """Complete hydrogen storage thermodynamics analyzer.

    Workflow:
        1. Load hydride phase (DFT energy + phonon free energy)
        2. Load dehydride phase (DFT energy + phonon free energy)
        3. Compute ΔH, ΔS, ΔG vs T
        4. Find T_des from ΔG = 0
        5. Generate Van't Hoff plot
        6. Evaluate against DOE targets
        7. Analyze H vibrational modes
    """

    def __init__(self, hydride_data, dehydride_data, h2_ref=None):
        """
        Args:
            hydride_data: dict with {
                'formula': 'MgH2',
                'e_dft': -5.67,  # eV per formula unit
                'free_energy_T': [...],  # kJ/mol vs T
                'entropy_T': [...],  # J/(mol·K) vs T
                'volume': 30.2,  # Å³ per formula unit
                'n_h': 2,  # H atoms per formula unit
                'temperatures': [0, 10, ..., 1000],
            }
            dehydride_data: dict (same structure, n_h = 0)
            h2_ref: H2Molecule instance (optional, uses defaults)
        """

    def execute(self, t_min=0, t_max=1000, t_step=10) -> HStorageResult:
        """Run complete H-storage analysis."""
```

#### 2.3 Core Calculations

```python
def compute_dehydrogenation_enthalpy(self, hydride, dehydride, n_h2):
    """Compute ΔH_des(T) for dehydrogenation reaction.

    Hydride → Dehydride + n_H2/2 × H₂(g)

    ΔH(T) = [E_DFT(dehyd) + F_phonon(dehyd,T) - T×S_vib(dehyd,T)]
            - [E_DFT(hyd) + F_phonon(hyd,T) - T×S_vib(hyd,T)]
            + n_H2/2 × [E_DFT(H₂) + H_gas(H₂,T)]

    Args:
        hydride: HydridePhase
        dehydride: DehydridePhase
        n_h2: moles of H2 released per reaction

    Returns:
        ΔH vs T in kJ/mol H2
    """

def compute_dehydrogenation_entropy(self, hydride, dehydride, n_h2, T):
    """Compute ΔS_des(T).

    ΔS(T) = S_vib(dehyd,T) - S_vib(hyd,T) - n_H2/2 × S_total(H₂,T)

    Note: S_total(H₂) includes translational + rotational + vibrational.
    This is the dominant term (S_trans ~ 130 J/(mol·K) at 300K).

    Returns:
        ΔS in J/(mol H2·K)
    """

def compute_gibbs_free_energy(self, delta_H_T, delta_S_T, temperatures):
    """Compute ΔG_des(T) = ΔH(T) - T × ΔS(T).

    Returns:
        ΔG vs T in kJ/mol H2
    """

def find_desorption_temperature(self, delta_G_T, temperatures, P_H2=1.0):
    """Find T where ΔG(T) = 0 at given H2 pressure.

    From Van't Hoff: ln(P) = ΔH/(RT) - ΔS/R
    At P = 1 bar: T_des = ΔH / ΔS

    For non-1-bar:
    ln(P/1bar) = ΔH/(R×T_des) - ΔS/R
    → T_des = ΔH / (ΔS + R×ln(P))

    Args:
        delta_G_T: ΔG vs T array (kJ/mol H2)
        temperatures: T array (K)
        P_H2: H2 pressure (bar)

    Returns:
        T_des in K
    """

def generate_vant_hoff_plot(self, delta_H, delta_S, T_range=(300, 800)):
    """Generate Van't Hoff plot data.

    ln(P_H2) = ΔH/(RT) - ΔS/R

    Args:
        delta_H: kJ/mol H2 (assumed T-independent for V-H plot)
        delta_S: J/(mol H2·K)
        T_range: (T_min, T_max) in K

    Returns:
        T_array, P_H2_array (bar)
    """

def evaluate_doe_targets(self, wt_percent, g_per_liter):
    """Evaluate against DOE 2025 targets.

    DOE 2025 System Targets:
        - Gravimetric: ≥ 5.5 wt%
        - Volumetric: ≥ 40 g/L
        - Operating T: 300-500 K (for practical use)

    Returns:
        dict: {gravimetric: {value, target, pass},
               volumetric: {value, target, pass}}
    """
```

#### 2.4 Vibrational Analysis (H-specific)

```python
def analyze_hydrogen_modes(self, hydride_phonon):
    """Detailed H-mode analysis from phonon DOS.

    Separates H contribution into:
        - Librational (5-21 THz): hindered rotation of H2/BH4 units
        - Bending (21-50 THz): H-X-H bending
        - Stretching (50-100 THz): X-H stretching

    Also identifies:
        - Principal B-H/N-H/O-H stretch frequency
        - H-mode fraction (what % of total DOS is H)
        - Comparison with experimental IR/Raman frequencies

    Returns:
        dict: {lib_frac, bend_frac, stretch_frac,
               peak_stretch_thz, peak_stretch_cm,
               h_fraction_of_total_dos}
    """

def identify_hydride_type(self, h_stretch_freq_cm):
    """Identify hydride type from H stretch frequency.

    Reference frequencies (cm⁻¹):
        - Ionic hydrides (MgH2, CaH2): 1100-1400
        - Complex hydrides (NaAlH4): 1700-1850 (Al-H)
        - Borohydrides (LiBH4): 2200-2600 (B-H)
        - Amides (LiNH2): 3100-3500 (N-H)

    Args:
        h_stretch_freq_cm: principal H stretch frequency (cm⁻¹)

    Returns:
        str: 'ionic' | 'complex' | 'borohydride' | 'amide' | 'unknown'
    """
```

---

### 3. CLI Command Specification

```bash
# Basic usage: compare hydride and dehydride
hydrophonokit h-storage \
    --hydride /path/to/MgH2_workspace \
    --dehydride /path/to/Mg_workspace \
    --output mg_h2_results/

# With explicit DFT energies (if phonon not available)
hydrophonokit h-storage \
    --hydride-formula MgH2 \
    --hydride-energy -5.67 \
    --dehydride-formula Mg \
    --dehydride-energy -1.52 \
    --output mg_h2_simple/

# Full workflow with Van't Hoff and DOE targets
hydrophonokit h-storage \
    --hydride MgH2_dir/ \
    --dehydride Mg_dir/ \
    --h2-energy -6.77 \
    --n-h2 1 \
    --plot-vant-hoff \
    --doe-report \
    --output MgH2_analysis/
```

#### CLI Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--hydride` | path | Yes* | Directory with hydride phonon results |
| `--dehydride` | path | Yes* | Directory with dehydride phonon results |
| `--hydride-formula` | str | Yes | Chemical formula of hydride |
| `--dehydride-formula` | str | Yes | Chemical formula of dehydride |
| `--hydride-energy` | float | Yes | DFT energy of hydride (eV/f.u.) |
| `--dehydride-energy` | float | Yes | DFT energy of dehydride (eV/f.u.) |
| `--h2-energy` | float | No | DFT energy of H2 molecule (eV), default: -6.77 |
| `--n-h2` | float | No | Moles H2 released, default: computed from formulas |
| `--output` | path | No | Output directory, default: current dir |
| `--plot-vant-hoff` | flag | No | Generate Van't Hoff plot |
| `--doe-report` | flag | No | Include DOE target comparison |
| `--temp-range` | str | No | T range: "0-1000", default: "0-1000" |

---

### 4. Output Specification

#### 4.1 Console Output

```
============================================================
  Hydrogen Storage Analysis: MgH2 → Mg + H2
============================================================

  Reaction: MgH2 → Mg + 1 H2

  Thermodynamics:
    ΔH (0 K)     =  76.2 kJ/mol H2
    ΔS (300 K)   = 135.4 J/(mol H2·K)
    ΔG (300 K)   =  35.6 kJ/mol H2

  Desorption Temperatures:
    T_des (1 bar)   =  563 K  (290 °C)
    T_des (0.1 bar) =  512 K  (239 °C)
    T_des (5 bar)   =  632 K  (359 °C)

  Capacities:
    Gravimetric:  7.66 wt%  [✓ exceeds DOE 5.5 wt%]
    Volumetric:  110.0 g/L  [✓ exceeds DOE 40 g/L]

  Hydrogen Vibrational Analysis:
    H-mode decomposition:
      Librational:  12.3 %
      Bending:      18.7 %
      Stretching:   69.0 %
    Principal Mg-H stretch: 1245 cm⁻¹ (37.3 THz)
    Hydride type: Ionic hydride

  Van't Hoff Data:
    T (K)    P_H2 (bar)    ln(P)
    400      0.002         -6.21
    450      0.035         -3.35
    500      0.278         -1.28
    550      1.410          0.34
    600      5.150          1.64

  DOE 2025 Target Comparison:
    Gravimetric:  7.66 wt%  vs  5.5 wt%   [PASS ✓]
    Volumetric: 110.0 g/L   vs 40.0 g/L   [PASS ✓]
    T_des (1 bar): 563 K    vs 300-500 K  [WARNING: above range]

============================================================
```

#### 4.2 Generated Files

| File | Format | Content |
|------|--------|---------|
| `h_storage_report.html` | HTML | Full interactive report |
| `vant_hoff_plot.png` | PNG | Van't Hoff: ln(P) vs 1/T |
| `thermodynamics_plot.png` | PNG | ΔH, ΔS, ΔG vs T |
| `h_modes_plot.png` | PNG | H vibrational mode decomposition |
| `h_storage_data.json` | JSON | All numerical results |

---

### 5. Test Specification

#### 5.1 Test Cases Against Known Materials

| Material | Reaction | ΔH (kJ/mol H₂) | T_des (1 bar, K) | wt% | Reference |
|----------|----------|-----------------|------------------|-----|-----------|
| MgH₂ | MgH₂ → Mg + H₂ | 74-76 | 550-570 | 7.66 | Bogdanovic (2004) |
| LiBH₄ | LiBH₄ → LiH + B + 1.5H₂ | 67-75 | 600-650 | 18.5 | Züttel (2005) |
| NaAlH₄ | NaAlH₄ → Na₃AlH₆ + Al + H₂ | 37-47 | 400-450 | 5.6 | Bogdanovic (1997) |
| LaNi₅H₆ | LaNi₅H₆ → LaNi₅ + 3H₂ | 30-31 | 330-340 | 1.37 | Sandrock (1999) |
| LiNH₂ | 2LiNH₂ → Li₂NH + NH₃ | ~40 | 450-500 | 6.5 | Chen (2003) |

#### 5.2 Unit Tests

```python
class TestH2Molecule:
    def test_entropy_300K(self):
        """H2 entropy at 300K should be ~130.68 J/(mol·K) (NIST)."""

    def test_enthalpy_300K(self):
        """H2 enthalpy at 300K should be ~8.47 kJ/mol."""

class TestThermodynamics:
    def test_MgH2_enthalpy(self):
        """MgH2 ΔH should be 74-76 kJ/mol H2."""

    def test_MgH2_desorption_temp(self):
        """MgH2 T_des (1 bar) should be 550-570 K."""

    def test_entropy_dominance(self):
        """ΔS should be dominated by H2 gas entropy."""

class TestCapacities:
    def test_MgH2_gravimetric(self):
        """MgH2 gravimetric should be 7.66 wt%."""

    def test_MgH2_volumetric(self):
        """MgH2 volumetric should be ~110 g/L."""

class TestDOETargets:
    def test_MgH2_passes_gravimetric(self):
        """MgH2 should pass DOE gravimetric target."""

    def test_MgH2_passes_volumetric(self):
        """MgH2 should pass DOE volumetric target."""
```

---

### 6. Scientific References

| # | Reference | Topic |
|---|-----------|-------|
| 1 | Bogdanović et al., J. Alloys Compd. 382, 1 (2004) | MgH₂ thermodynamics |
| 2 | Züttel et al., Nature Mater. 4, 673 (2005) | LiBH₄ properties |
| 3 | Bogdanović et al., J. Alloys Compd. 253, 1 (1997) | NaAlH₄ doping |
| 4 | DOE Hydrogen Program Plan (2023) | System targets |
| 5 | NIST Chemistry WebBook | H₂ gas properties |
| 6 | Baricco et al., J. Alloys Compd. 509, S344 (2011) | Van't Hoff analysis |
| 7 | Nakamoto, IR/Raman Spectra (2009) | H-stretch frequencies |
| 8 | McQuarrie, Statistical Mechanics (1976) | Partition functions |

---

### 7. Implementation Order

| Sprint | Task | Estimated Lines | Dependencies |
|--------|------|----------------|--------------|
| 1 | H₂ molecule module | ~250 | `physics.py` constants |
| 2 | Thermodynamics engine core | ~350 | H₂ module |
| 3 | Van't Hoff & DOE targets | ~150 | Thermodynamics |
| 4 | CLI command | ~70 | Engine complete |
| 5 | Tests (5 known materials) | ~200 | All above |
| 6 | Documentation & HTML report | ~150 | All above |

**Total new code: ~1,170 lines**

---

### 8. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21 | Array operations |
| scipy | ≥1.8 | Root finding (T_des) |
| Existing HydroPhonoKit | v2.5.0+ | Phonon data loading |

**No new external dependencies required.**

---

*End of Plan — Phase 5: H-Storage Analytics*
