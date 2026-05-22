# HydroPhonoKit Scientific Features Roadmap

> **Version:** 2.2.0 → 3.0.0
> **Scope:** Advanced scientific capabilities for next-generation phonon analysis
> **Timeline:** Phased implementation with clear deliverables

---

## 🎯 Vision

Transform HydroPhonoKit from a phonon calculation workflow tool into a **comprehensive materials thermodynamics and dynamics analysis platform** with publication-grade results.

---

## 📊 Feature Categories

### A. Quasi-Harmonic Approximation (QHA)
### B. Anharmonic Phonon Calculations
### C. Elastic Constants & Sound Velocities
### D. Raman/IR Activity Prediction
### E. Advanced Hydrogen Storage Analytics
### F. Electron-Phonon Coupling
### G. Temperature-Dependent Phonons

---

## 🔬 Phase 1: Quasi-Harmonic Approximation (QHA)

**Goal:** Compute thermal expansion, C_p, and temperature-dependent properties from phonon calculations at multiple volumes.

**Scientific Foundation:**
- Harmonic approximation assumes fixed lattice → fails at high T
- QHA: phonon frequencies depend on volume ω(q,V)
- Free energy F(V,T) minimized at each T → equilibrium V(T)
- From V(T): thermal expansion α(T), bulk modulus B(T), C_p(T)

**Implementation:**

### 1.1 QHA Workflow (`qha.py`)

```
Input: phonon calculations at 5-9 volumes (±2% around equilibrium)
Output:
  - V(T): equilibrium volume vs temperature
  - α(T): volumetric thermal expansion coefficient
  - B(T): isothermal bulk modulus
  - C_p(T): heat capacity at constant pressure
  - Grüneisen parameter γ(T)
  - Plots: V vs T, α vs T, B vs T, C_p vs T
```

**Formulas:**

| Property | Formula | Unit |
|----------|---------|------|
| F(V,T) | E_DFT(V) + F_phonon(V,T) | kJ/mol |
| V_eq(T) | argmin_V F(V,T) | Å³ |
| α(T) | (1/V)(∂V/∂T)_P | K⁻¹ |
| B(T) | V(∂²F/∂V²)_T | GPa |
| C_p(T) | C_v(T) + TVα²B | J/(mol·K) |
| γ(V) | -(V/ω)(∂ω/∂V) | dimensionless |

**Dependencies:**
- Existing: phonopy, numpy
- New: scipy (for equation of state fitting, minimization)

**Files to Create:**
- `hydrophonokit/qha.py` (~400 lines)
- `hydrophonokit/eos.py` — Equation of state models (Birch-Murnaghan, Vinet, etc.) (~200 lines)

**Tests Required:**
- EOS fitting against known materials (Si, MgO)
- QHA results against literature (Al, Cu thermal expansion)

---

### 1.2 Equation of State Models (`eos.py`)

Implement multiple EOS for fitting E(V) data:

| EOS | Formula | Parameters |
|-----|---------|------------|
| Birch-Murnaghan (3rd) | E(V) = E₀ + (9V₀B₀/16){[(V₀/V)^(2/3)-1]³B' + [(V₀/V)^(2/3)-1]²[6-4(V₀/V)^(2/3)]} | E₀, V₀, B₀, B' |
| Vinet | E(V) = E₀ + (2B₀V₀/(B'-1)²)[-(1+x)e^(-bx) + 1] | E₀, V₀, B₀, B' |
| Murnaghan | E(V) = E₀ + B₀V/B'[(V₀/V)^B'/(B'-1) + 1] - V₀B₀/(B'-1) | E₀, V₀, B₀, B' |

Where x = 3(B'-1)/2 × [(V/V₀)^(1/3) - 1], b = 3(B'-1)/2

---

## 🔬 Phase 2: Anharmonic Phonon Calculations

**Goal:** Compute phonon lifetimes, linewidths, and thermal conductivity from phonon-phonon interactions.

**Scientific Foundation:**
- Harmonic approximation: phonons don't interact → infinite lifetime
- Real materials: 3-phonon and 4-phonon processes → finite linewidth Γ
- Phonon linewidth Γ → lifetime τ = 1/(2Γ)
- Thermal conductivity κ from Boltzmann transport equation

**Implementation:**

### 2.1 Third-Order Force Constants (`anharmonic.py`)

```
Input: Supercell with displaced atom pairs (requires ~200-500 VASP calculations)
Output:
  - Third-order IFCs (Φ_ijk)
  - Phonon linewidths Γ(q,ω,T)
  - Phonon lifetimes τ(q,ω,T)
  - Temperature-dependent frequency shifts Δω(T)
```

**Formulas:**

| Property | Formula | Notes |
|----------|---------|-------|
| Γ_λ(T) | (π/4ℏ²) × Σ_λλ' |V_λλ'λ''|²[(n'+n''+1)δ(ω-ω'-ω'') + ...] | 3-phonon scattering |
| τ_λ | 1/(2Γ_λ) | Phonon lifetime |
| Δω_λ(T) | Principal value integral of same expression | Frequency shift |

**Dependencies:**
- New: phono3py (phonopy extension for 3rd-order IFCs)

**Files to Create:**
- `hydrophonokit/anharmonic.py` (~350 lines)
- `hydrophonokit/thermal_conductivity.py` (~250 lines)

---

### 2.2 Thermal Conductivity (`thermal_conductivity.py`)

```
Input: phonon linewidths + group velocities
Output:
  - κ(T): lattice thermal conductivity vs temperature
  - κ_λ: mode-resolved thermal conductivity
  - Mean free path ℓ_λ = v_g,λ × τ_λ
  - Cumulative κ vs mean free path
  - Comparison with Slack model (high-T limit)
```

**Slack Model (analytic high-T limit):**
```
κ = (M θ_D³ δ) / (γ² T n^(2/3))

Where:
  M = average atomic mass
  θ_D = Debye temperature
  δ³ = volume per atom
  γ = Grüneisen parameter
  n = atoms per primitive cell
```

---

## 🔬 Phase 3: Elastic Constants from Phonons

**Goal:** Extract elastic constants, sound velocities, and mechanical properties from long-wavelength phonon dispersion.

**Scientific Foundation:**
- Acoustic branches near Γ: ω = v_s × |q|
- Sound velocity v_s relates to elastic constants C_ij
- For cubic crystals: v_LA = √(C₁₁/ρ), v_TA = √(C₄₄/ρ)
- From C_ij: bulk modulus B, shear modulus G, Young's modulus E, Poisson's ratio ν

**Implementation:**

### 3.1 Elastic Constants (`elastic.py`)

```
Input: Phonon band structure (fine q-mesh near Γ)
Output:
  - C_ij: elastic constants tensor (Voigt notation)
  - B: bulk modulus (Voigt, Reuss, VRH averages)
  - G: shear modulus
  - E: Young's modulus
  - ν: Poisson's ratio
  - v_LA, v_TA: longitudinal/transverse sound velocities
  - v_D: Debye velocity → Θ_D (Debye temperature)
  - Mechanical stability check (Born-Huang criteria)
```

**Formulas:**

| Property | Formula | Crystal System |
|----------|---------|----------------|
| v_LA | √(C₁₁/ρ) | Cubic |
| v_TA1 | √(C₄₄/ρ) | Cubic |
| v_TA2 | √((C₁₁-C₁₂)/2ρ) | Cubic |
| B_V | (C₁₁ + 2C₁₂)/3 | Cubic |
| G_V | (C₁₁ - C₁₂ + 3C₄₄)/5 | Cubic |
| E | 9BG/(3B+G) | All |
| ν | (3B-2G)/(6B+2G) | All |
| v_D | [1/3 (1/v_LA³ + 2/v_TA³)]^(-1/3) | All |

**Files to Create:**
- `hydrophonokit/elastic.py` (~300 lines)

---

## 🔬 Phase 4: Raman/IR Activity

**Goal:** Predict Raman and infrared spectra from phonon calculations and compare with experiment.

**Scientific Foundation:**
- IR activity: requires change in dipole moment → non-zero Born effective charges
- Raman activity: requires change in polarizability → non-zero Raman tensor
- Selection rules from crystal symmetry (group theory)
- Peak positions from phonon frequencies, intensities from derivatives

**Implementation:**

### 4.1 Spectroscopy Module (`spectroscopy.py`)

```
Input:
  - Phonon frequencies + eigenvectors
  - Born effective charges (for IR)
  - Dielectric tensor
Output:
  - IR spectrum: intensity vs wavenumber (cm⁻¹)
  - Raman spectrum: intensity vs wavenumber
  - Mode assignments (symmetry labels)
  - Comparison with experimental peak positions
```

**Formulas:**

| Spectrum | Intensity ∝ | Notes |
|----------|-------------|-------|
| IR | |Σ_i Z*_i · e_i|² / ω | e_i = eigenvector |
| Raman | |Σ_i ∂α/∂u_i · e_i|² | Requires polarizability derivatives |

**Files to Create:**
- `hydrophonokit/spectroscopy.py` (~350 lines)
- `hydrophonokit/symmetry.py` — Group theory & selection rules (~250 lines)

---

## 🔬 Phase 5: Advanced Hydrogen Storage Analytics

**Goal:** Comprehensive hydrogen storage material evaluation with DOE target comparison.

**Scientific Foundation:**
- Hydrogen storage capacity: gravimetric (wt%) and volumetric (g/L)
- Thermodynamics: ΔH, ΔG, ΔS of hydrogen absorption/desorption
- Vibrational contribution to entropy: ΔS_vib from phonons
- Desorption temperature: T_des = ΔH/ΔS
- Binding energy: E_bind from DFT + phonon corrections

**Implementation:**

### 5.1 H-Storage Module (`h_storage.py`)

```
Input:
  - Phonon calculations for hydride + dehydrogenated phase
  - H₂ molecule phonon (gas phase reference)
Output:
  - ΔH_des: enthalpy of dehydrogenation
  - ΔS_des: entropy of dehydrogenation (including vibrational)
  - T_des: equilibrium desorption temperature
  - wt%: gravimetric capacity
  - g/L: volumetric capacity
  - DOE target comparison (2025 targets)
  - Van't Hoff plot: ln(P_H2) vs 1/T
```

**Formulas:**

| Property | Formula | Notes |
|----------|---------|-------|
| ΔH_des | E_DFT(hydride) - E_DFT(dehydride) - n×E_DFT(H₂) | 0 K enthalpy |
| ΔS_vib | S_vib(hydride) - S_vib(dehydride) - n×S_vib(H₂) | From phonons |
| ΔG(T) | ΔH - TΔS | Gibbs free energy |
| T_des | ΔH/ΔS | At P_H2 = 1 bar |
| wt% | n×M_H / M_hydride × 100 | Gravimetric capacity |
| Van't Hoff | ln(P) = ΔH/RT - ΔS/R | Equilibrium pressure |

**Files to Create:**
- `hydrophonokit/h_storage.py` (~400 lines)
- Add to `physics.py`: H₂ molecular properties

---

## 🔬 Phase 6: Electron-Phonon Coupling

**Goal:** Estimate superconducting transition temperature from phonon properties.

**Scientific Foundation:**
- Electron-phonon coupling λ from Eliashberg function α²F(ω)
- McMillan formula: T_c = (θ_D/1.45) exp[-1.04(1+λ)/(λ-μ*(1+0.62λ))]
- Requires electronic structure (DOS at Fermi level) + phonon DOS

**Implementation:**

### 6.1 Superconductivity Module (`superconductivity.py`)

```
Input:
  - Phonon DOS
  - Electronic DOS at Fermi level (from VASP vasprun.xml)
  - Electron-phonon matrix elements (approximate)
Output:
  - λ: electron-phonon coupling constant
  - μ*: Coulomb pseudopotential (0.10-0.15 typical)
  - T_c: superconducting transition temperature
  - α²F(ω): Eliashberg spectral function
  - Comparison with experimental T_c
```

**McMillan Formula:**
```
T_c = (θ_D / 1.45) × exp[-1.04(1+λ) / (λ - μ*(1+0.62λ))]

Where:
  θ_D = Debye temperature
  λ = electron-phonon coupling
  μ* = Coulomb pseudopotential (0.10-0.15)
```

**Files to Create:**
- `hydrophonokit/superconductivity.py` (~200 lines)

---

## 📅 Implementation Timeline

### Sprint 1: QHA Foundation (Weeks 1-2)
- [ ] `physics.py`: Add Grüneisen functions, QHA formulas
- [ ] `eos.py`: Equation of state models (Birch-Murnaghan, Vinet)
- [ ] `qha.py`: QHA workflow engine
- [ ] Tests: EOS fitting validation, QHA against literature

### Sprint 2: Elastic Constants (Week 3)
- [ ] `elastic.py`: Sound velocity extraction, C_ij computation
- [ ] `physics.py`: Add elastic formulas, Born-Huang criteria
- [ ] Tests: Compare with known elastic constants (Si, Al, Cu)

### Sprint 3: Spectroscopy (Week 4)
- [ ] `symmetry.py`: Group theory, selection rules
- [ ] `spectroscopy.py`: IR/Raman spectrum generation
- [ ] Tests: Compare with experimental IR spectra

### Sprint 4: H-Storage Analytics (Week 5)
- [ ] `h_storage.py`: Dehydrogenation thermodynamics
- [ ] `physics.py`: H₂ molecule reference data
- [ ] Tests: Known hydrides (MgH₂, LiBH₄, NaAlH₄)

### Sprint 5: Anharmonicity (Weeks 6-7)
- [ ] `anharmonic.py`: Third-order IFC interface (phono3py)
- [ ] `thermal_conductivity.py`: κ(T) calculation
- [ ] Tests: Si, Ge thermal conductivity vs experiment

### Sprint 6: Superconductivity (Week 8)
- [ ] `superconductivity.py`: McMillan T_c estimation
- [ ] Integration tests: Full workflow
- [ ] Documentation: Complete scientific reference guide

---

## 📊 Expected Deliverables

### New Modules (8 files, ~3000 lines)

| Module | Lines | Phase |
|--------|-------|-------|
| `qha.py` | 400 | Sprint 1 |
| `eos.py` | 200 | Sprint 1 |
| `elastic.py` | 300 | Sprint 2 |
| `symmetry.py` | 250 | Sprint 3 |
| `spectroscopy.py` | 350 | Sprint 3 |
| `h_storage.py` | 400 | Sprint 4 |
| `anharmonic.py` | 350 | Sprint 5 |
| `thermal_conductivity.py` | 250 | Sprint 5 |
| `superconductivity.py` | 200 | Sprint 6 |
| `physics.py` (enhancements) | +200 | All sprints |
| **Total** | **~2900** | |

### CLI Commands

```bash
# New commands in hydrophonokit CLI
hydrophonokit qha --volumes dir1 dir2 dir3 --output qha_results
hydrophonokit elastic --source DIR
hydrophonokit raman --source DIR --ir --plot
hydrophonokit h-storage --hydride DIR1 --dehydro DIR2
hydrophonokit thermal-conductivity --workspace DIR
hydrophonokit superconductivity --source DIR
```

### Scientific References

Each module will include:
- Full mathematical derivations
- Literature validation cases
- CODATA 2018 constants
- Peer-reviewed paper citations

---

## 🔧 Technical Requirements

### New Dependencies

| Package | Version | Purpose | Optional? |
|---------|---------|---------|-----------|
| scipy | >=1.8 | EOS fitting, optimization | No (core) |
| phono3py | >=2.0 | Third-order IFCs | Yes (anharmonicity) |
| spglib | >=2.0 | Symmetry analysis | Already installed |

### Testing Requirements

- Unit tests for all mathematical functions
- Integration tests with known materials
- Coverage target: 80%+ for new modules
- Reference data comparison (Si, Al, MgH₂, etc.)

---

## 🎯 Success Criteria

1. **QHA**: Thermal expansion of Al matches experiment within 10%
2. **Elastic**: Sound velocities within 5% of measured values
3. **Spectroscopy**: IR peak positions within 20 cm⁻¹ of experiment
4. **H-Storage**: ΔH_des for MgH₂ within 5 kJ/mol H₂ of literature
5. **Thermal Conductivity**: Si κ(300K) within 20% of experiment
6. **Superconductivity**: T_c for Pb within 1K of experiment (6.2K)

---

## 📝 Notes

- All formulas verified against standard textbooks (Ashcroft & Mermin, Born & Huang)
- Unit consistency enforced throughout (THz, cm⁻¹, K, eV, J/mol conversions explicit)
- Error propagation included in all results
- Backward compatible: existing workflows unchanged

---

*End of Roadmap — HydroPhonoKit v3.0.0 Scientific Features*
