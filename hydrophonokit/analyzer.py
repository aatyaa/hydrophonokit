"""
=============================================================================
  HydroPhonoKit v2.1 -- Advanced Material Intelligence Engine
  
  A rigorous scientific analysis module that inspects VASP outputs and
  constructs a complete physical profile of the material, then generates
  optimized phonon calculation recommendations grounded in:
  
    - Quantitative force/stress convergence metrics
    - POTCAR basis-set validation (Pulay stress prevention)
    - Pymatgen-based electronic structure (bandgap extraction)
    - Coordination chemistry & molecular fragment detection
    - Spin-polarization and spin-orbit coupling assessment
    - Phonon readiness scoring (0-100)
    - Literature-calibrated expected frequency ranges
    - Computational cost modeling (N^3 DFT scaling)
    
  References:
    [1] Togo & Tanaka, Scr. Mater. 108, 1 (2015)  -- Phonopy
    [2] Baroni et al., Rev. Mod. Phys. 73, 515 (2001) -- DFPT
    [3] Parlinski et al., Phys. Rev. Lett. 78, 4063 (1997) -- Finite disp.
=============================================================================
"""
import os
import re
import math
import numpy as np
from phonopy.interface.vasp import read_vasp

# ============================================================================
# PERIODIC TABLE DATABASE
# ============================================================================
# symbol -> (Z, mass_amu, electronegativity_pauling, typical_oxidation)
ELEMENT_DB = {
    'H':  (1,   1.008, 2.20, [-1,1]),    'He': (2,   4.003, 0.00, [0]),
    'Li': (3,   6.941, 0.98, [1]),        'Be': (4,   9.012, 1.57, [2]),
    'B':  (5,  10.81,  2.04, [3]),        'C':  (6,  12.01,  2.55, [-4,4]),
    'N':  (7,  14.01,  3.04, [-3,3,5]),   'O':  (8,  16.00,  3.44, [-2]),
    'F':  (9,  19.00,  3.98, [-1]),       'Ne': (10, 20.18,  0.00, [0]),
    'Na': (11, 22.99,  0.93, [1]),        'Mg': (12, 24.31,  1.31, [2]),
    'Al': (13, 26.98,  1.61, [3]),        'Si': (14, 28.09,  1.90, [4,-4]),
    'P':  (15, 30.97,  2.19, [3,5,-3]),   'S':  (16, 32.07,  2.58, [-2,4,6]),
    'Cl': (17, 35.45,  3.16, [-1]),       'Ar': (18, 39.95,  0.00, [0]),
    'K':  (19, 39.10,  0.82, [1]),        'Ca': (20, 40.08,  1.00, [2]),
    'Sc': (21, 44.96,  1.36, [3]),        'Ti': (22, 47.87,  1.54, [2,3,4]),
    'V':  (23, 50.94,  1.63, [2,3,4,5]), 'Cr': (24, 52.00,  1.66, [2,3,6]),
    'Mn': (25, 54.94,  1.55, [2,3,4,7]), 'Fe': (26, 55.85,  1.83, [2,3]),
    'Co': (27, 58.93,  1.88, [2,3]),      'Ni': (28, 58.69,  1.91, [2]),
    'Cu': (29, 63.55,  1.90, [1,2]),      'Zn': (30, 65.38,  1.65, [2]),
    'Ga': (31, 69.72,  1.81, [3]),        'Ge': (32, 72.63,  2.01, [4]),
    'As': (33, 74.92,  2.18, [3,5,-3]),   'Se': (34, 78.97,  2.55, [-2,4,6]),
    'Br': (35, 79.90,  2.96, [-1]),       'Kr': (36, 83.80,  0.00, [0]),
    'Rb': (37, 85.47,  0.82, [1]),        'Sr': (38, 87.62,  0.95, [2]),
    'Y':  (39, 88.91,  1.22, [3]),        'Zr': (40, 91.22,  1.33, [4]),
    'Nb': (41, 92.91,  1.60, [3,5]),      'Mo': (42, 95.95,  2.16, [4,6]),
    'Tc': (43, 98.00,  1.90, [4,7]),      'Ru': (44, 101.07, 2.20, [3,4]),
    'Rh': (45, 102.91, 2.28, [3]),        'Pd': (46, 106.42, 2.20, [2,4]),
    'Ag': (47, 107.87, 1.93, [1]),        'Cd': (48, 112.41, 1.69, [2]),
    'In': (49, 114.82, 1.78, [3]),        'Sn': (50, 118.71, 1.96, [2,4]),
    'Sb': (51, 121.76, 2.05, [3,5,-3]),   'Te': (52, 127.60, 2.10, [-2,4,6]),
    'I':  (53, 126.90, 2.66, [-1]),       'Xe': (54, 131.29, 2.60, [0]),
    'Cs': (55, 132.91, 0.79, [1]),        'Ba': (56, 137.33, 0.89, [2]),
    'La': (57, 138.91, 1.10, [3]),        'Ce': (58, 140.12, 1.12, [3,4]),
    'Pr': (59, 140.91, 1.13, [3]),        'Nd': (60, 144.24, 1.14, [3]),
    'Sm': (62, 150.36, 1.17, [3]),        'Eu': (63, 151.96, 1.20, [2,3]),
    'Gd': (64, 157.25, 1.20, [3]),        'Tb': (65, 158.93, 1.22, [3]),
    'Dy': (66, 162.50, 1.23, [3]),        'Ho': (67, 164.93, 1.24, [3]),
    'Er': (68, 167.26, 1.24, [3]),        'Tm': (69, 168.93, 1.25, [3]),
    'Yb': (70, 173.05, 1.25, [2,3]),      'Lu': (71, 174.97, 1.27, [3]),
    'Hf': (72, 178.49, 1.30, [4]),        'Ta': (73, 180.95, 1.50, [5]),
    'W':  (74, 183.84, 2.36, [4,6]),      'Re': (75, 186.21, 1.90, [4,7]),
    'Os': (76, 190.23, 2.20, [4]),        'Ir': (77, 192.22, 2.20, [3,4]),
    'Pt': (78, 195.08, 2.28, [2,4]),      'Au': (79, 196.97, 2.54, [1,3]),
    'Hg': (80, 200.59, 2.00, [1,2]),      'Tl': (81, 204.38, 1.62, [1,3]),
    'Pb': (82, 207.20, 1.87, [2,4]),      'Bi': (83, 208.98, 2.02, [3,5]),
    'Po': (84, 208.98, 2.00, [-2,2,4]),   'At': (85, 209.99, 2.20, [-1,1]),
    'Rn': (86, 222.02, 0.00, [0]),        'Fr': (87, 223.02, 0.70, [1]),
    'Ra': (88, 226.03, 0.90, [2]),        'Ac': (89, 227.03, 1.10, [3]),
    'Th': (90, 232.04, 1.30, [4]),        'Pa': (91, 231.04, 1.50, [3,4,5]),
    'U':  (92, 238.03, 1.38, [3,4,5,6]), 'Np': (93, 237.05, 1.36, [3,4,5,6]),
    'Pu': (94, 244.06, 1.28, [3,4,5,6]), 'Am': (95, 243.06, 1.30, [2,3,4,5,6]),
    'Cm': (96, 247.07, 1.30, [3,4]),      'Bk': (97, 247.07, 1.30, [3,4]),
    'Cf': (98, 251.08, 1.30, [3]),        'Es': (99, 252.08, 1.30, [3]),
    'Fm': (100, 257.10, 1.30, [2,3]),
    'Pm': (61, 144.91, 1.13, [3]),
}

Z_TO_SYMBOL = {v[0]: k for k, v in ELEMENT_DB.items()}

# Classification sets
LIGHT_ATOMS     = {'H', 'He', 'Li', 'Be', 'B'}
MAGNETIC_ATOMS  = {'Fe', 'Co', 'Ni', 'Mn', 'Cr', 'V', 'Gd', 'Eu', 'Nd', 'Ce'}
HEAVY_SOC_ATOMS = {s for s, (z,_,_,_) in ELEMENT_DB.items() if z > 56}

# Known bond stretching frequency ranges (cm^-1) for validation
# Reference: Nakamoto, "Infrared and Raman Spectra of Inorganic Compounds"
BOND_FREQ_LIBRARY = {
    ('B','H'):  (2200, 2600, 'B-H stretch'),
    ('O','H'):  (3200, 3700, 'O-H stretch'),
    ('N','H'):  (3100, 3500, 'N-H stretch'),
    ('C','H'):  (2800, 3100, 'C-H stretch'),
    ('Si','O'): (900,  1100, 'Si-O stretch'),
    ('Al','O'): (600,  900,  'Al-O stretch'),
    ('Ti','O'): (400,  700,  'Ti-O stretch'),
    ('Zr','O'): (350,  650,  'Zr-O stretch'),
    ('Li','F'): (500,  700,  'Li-F stretch'),
    ('Na','Cl'):(200,  350,  'Na-Cl stretch'),
    ('Ca','F'): (350,  500,  'Ca-F stretch'),
    ('Mg','O'): (400,  700,  'Mg-O stretch'),
}

# ============================================================================
# MATERIAL PROFILE DATA CLASS
# ============================================================================
class MaterialProfile:
    """Complete scientific profile of a material for phonon planning."""
    def __init__(self):
        # ---- Identity ----
        self.formula = ""
        self.reduced_formula = ""
        self.elements = {}            # {symbol: count}
        self.n_atoms = 0
        self.space_group = ""
        self.point_group = ""
        self.crystal_system = ""
        self.n_symmetry_ops = 0
        
        # ---- Masses ----
        self.lightest_atom = ("", 999.0)
        self.heaviest_atom = ("", 0.0)
        self.mass_ratio = 1.0
        self.has_light_atoms = False
        self.avg_mass = 0.0
        
        # ---- Electronic ----
        self.is_insulator = True
        self.bandgap = None           # eV
        self.bandgap_source = "unknown"
        self.ismear_detected = 0
        self.efermi = None
        
        # ---- Magnetic / SOC ----
        self.is_magnetic = False
        self.ispin_detected = 1
        self.has_soc_candidates = False
        self.soc_elements = []
        
        # ---- Convergence (quantitative) ----
        self.converged = False
        self.max_force = None         # eV/A
        self.rms_force = None         # eV/A
        self.external_pressure = None # kBar
        self.pulay_stress = None      # kBar
        self.n_ionic_steps = 0
        self.final_energy = None      # eV
        
        # ---- POTCAR / Basis Set ----
        self.potcar_elements = []
        self.potcar_enmax = []        # ENMAX per element from POTCAR
        self.potcar_max_enmax = 0.0
        self.encut_ratio = 0.0        # ENCUT / max(ENMAX) -- should be >= 1.3
        
        # ---- Bonding ----
        self.min_bond_length = 999.0
        self.min_bond_pair = ("", "")
        self.has_vdw = False
        self.ivdw_detected = 0
        self.molecular_units = []
        self.coordination_env = {}     # {atom_idx: [(neighbor_sym, dist), ...]}
        self.bond_pairs_found = []     # [(sym_i, sym_j, avg_dist)]
        self.expected_freq_ranges = [] # from BOND_FREQ_LIBRARY
        self.ionicity_index = 0.0      # electronegativity difference metric
        
        # ---- Lattice ----
        self.lattice_params = (0, 0, 0)
        self.lattice_angles = (90, 90, 90)
        self.volume = 0.0
        self.density = 0.0            # g/cm^3
        
        # ---- VASP parameters (extracted) ----
        self.encut = 520.0
        self.kspacing = 0.25
        self.ediff = 1e-6
        self.ediffg = -0.01
        
        # ---- Recommendations ----
        self.rec_supercell = [2, 2, 2]
        self.rec_displacement = 0.01
        self.rec_born = True
        self.rec_ismear = 0
        self.rec_sigma = 0.05
        self.rec_vdw = False
        self.rec_ivdw = 0
        self.rec_kpoints_sc = [2, 2, 2]
        self.rec_born_time_hrs = 6
        self.rec_disp_time_hrs = 12
        self.rec_dos_mesh = [15, 15, 15]
        self.rec_ediff_phonon = 1e-8
        self.rec_algo = "Fast"            # VASP electronic minimization algorithm
        self.rec_isym = None              # VASP symmetry flag (None = VASP default)
        self.warnings = []
        self.notes = []
        self.n_disp_estimate = 0
        self.sc_atoms = 0
        self.phonon_readiness_score = 0   # 0-100
        
        # ---- Source INCAR detected values ----
        self.isym_detected = None         # ISYM from source INCAR
        self.algo_detected = "Fast"       # ALGO from source INCAR


# ============================================================================
# ADVANCED MATERIAL ANALYZER
# ============================================================================
class MaterialAnalyzer:
    """
    Rigorous multi-layer analysis of VASP outputs for phonon readiness.
    
    Analysis Pipeline:
      Layer 1: Crystallographic profiling (CONTCAR + spglib)
      Layer 2: INCAR parameter extraction
      Layer 3: POTCAR basis-set validation
      Layer 4: OUTCAR quantitative convergence (forces, stress, energy)
      Layer 5: Electronic structure (bandgap from vasprun.xml)
      Layer 6: Coordination chemistry & molecular fragment detection
      Layer 7: Spin & relativistic assessment
      Layer 8: Phonon readiness scoring
      Layer 9: Smart recommendation engine
    """
    
    def __init__(self, target_dir):
        self.target_dir = target_dir
        self.contcar = os.path.join(target_dir, 'CONTCAR')
        self.outcar  = os.path.join(target_dir, 'OUTCAR')
        self.incar   = os.path.join(target_dir, 'INCAR')
        self.potcar  = os.path.join(target_dir, 'POTCAR')
        self.vasprun = os.path.join(target_dir, 'vasprun.xml')
        self.profile = MaterialProfile()
        self._cell = None
        self._cart_pos = None
    
    def analyze(self):
        """Execute the full 9-layer analysis pipeline."""
        self._layer1_crystallography()
        self._layer2_incar()
        self._layer3_potcar()
        self._layer4_outcar()
        self._layer5_electronic()
        self._layer6_bonding()
        self._layer7_spin_soc()
        self._layer8_readiness_score()
        self._layer9_recommendations()
        return self.profile
    
    # ================================================================
    # LAYER 1: Crystallographic Profiling
    # ================================================================
    def _layer1_crystallography(self):
        """Parse CONTCAR for lattice, symmetry, composition, and mass analysis."""
        import spglib
        cell = read_vasp(self.contcar)
        p = self.profile
        self._cell = cell
        
        # Lattice parameters
        lat = cell.cell
        a = np.linalg.norm(lat[0])
        b = np.linalg.norm(lat[1])
        c = np.linalg.norm(lat[2])
        p.lattice_params = (a, b, c)
        
        # Lattice angles
        alpha = np.degrees(np.arccos(np.clip(
            np.dot(lat[1], lat[2]) / (b * c), -1, 1)))
        beta  = np.degrees(np.arccos(np.clip(
            np.dot(lat[0], lat[2]) / (a * c), -1, 1)))
        gamma = np.degrees(np.arccos(np.clip(
            np.dot(lat[0], lat[1]) / (a * b), -1, 1)))
        p.lattice_angles = (round(alpha, 2), round(beta, 2), round(gamma, 2))
        
        p.volume = abs(np.linalg.det(lat))
        
        # Symmetry analysis (attribute API for spglib >= 2.x)
        spg_data = spglib.get_symmetry_dataset(
            (cell.cell, cell.scaled_positions, cell.numbers), symprec=1e-5)
        if spg_data:
            # Use attribute interface (modern spglib), fallback to dict
            try:
                sg_int = spg_data.international
                sg_num = spg_data.number
                pg     = spg_data.pointgroup
                n_ops  = len(spg_data.rotations)
            except AttributeError:
                sg_int = spg_data['international']
                sg_num = spg_data['number']
                pg     = spg_data.get('pointgroup', '')
                n_ops  = len(spg_data.get('rotations', []))
            p.space_group = f"{sg_int} ({sg_num})"
            p.point_group = pg
            p.n_symmetry_ops = n_ops
            # Crystal system from space group number
            sgn = sg_num
            if   sgn <= 2:   p.crystal_system = "Triclinic"
            elif sgn <= 15:  p.crystal_system = "Monoclinic"
            elif sgn <= 74:  p.crystal_system = "Orthorhombic"
            elif sgn <= 142: p.crystal_system = "Tetragonal"
            elif sgn <= 167: p.crystal_system = "Trigonal"
            elif sgn <= 194: p.crystal_system = "Hexagonal"
            else:            p.crystal_system = "Cubic"
        else:
            p.space_group = "Unknown"
        
        # Composition
        p.n_atoms = len(cell.numbers)
        elem_counts = {}
        for z in cell.numbers:
            s = Z_TO_SYMBOL.get(z, f"Z{z}")
            elem_counts[s] = elem_counts.get(s, 0) + 1
        p.elements = elem_counts
        p.formula = "".join(f"{s}{c}" for s, c in p.elements.items())
        
        # Reduced formula (divide by GCD)
        from math import gcd
        from functools import reduce
        counts = list(p.elements.values())
        g = reduce(gcd, counts)
        p.reduced_formula = "".join(
            f"{s}{c//g if c//g > 1 else ''}" for s, c in p.elements.items())
        
        # Mass analysis
        total_mass = 0.0
        for sym, cnt in p.elements.items():
            if sym in ELEMENT_DB:
                m = ELEMENT_DB[sym][1]
                total_mass += m * cnt
                if m < p.lightest_atom[1]:
                    p.lightest_atom = (sym, m)
                if m > p.heaviest_atom[1]:
                    p.heaviest_atom = (sym, m)
        
        p.mass_ratio = p.heaviest_atom[1] / max(p.lightest_atom[1], 0.001)
        p.avg_mass = total_mass / max(p.n_atoms, 1)
        p.has_light_atoms = any(s in LIGHT_ATOMS for s in p.elements)
        
        # Density  (g/cm^3)
        # mass in amu, volume in A^3; 1 amu = 1.66054e-24 g, 1 A^3 = 1e-24 cm^3
        p.density = (total_mass * 1.66054) / max(p.volume, 0.001)
        
        # Cartesian positions for bonding analysis
        self._cart_pos = (cell.cell @ cell.scaled_positions.T).T
    
    # ================================================================
    # LAYER 2: INCAR Parameter Extraction
    # ================================================================
    def _layer2_incar(self):
        """Extract all phonon-relevant VASP parameters from INCAR."""
        p = self.profile
        if not os.path.exists(self.incar):
            p.warnings.append("INCAR not found")
            return
        
        with open(self.incar, 'r', encoding='utf-8', errors='replace') as f:
            txt = f.read()
        
        def _get(pattern, default, cast=float):
            m = re.search(pattern, txt)
            return cast(m.group(1)) if m else default
        
        p.encut    = _get(r'ENCUT\s*=\s*([0-9.]+)',    520.0)
        p.kspacing = _get(r'KSPACING\s*=\s*([0-9.]+)', 0.25)
        p.ediff    = _get(r'EDIFF\s*=\s*([0-9.Ee-]+)', 1e-6)
        p.ediffg   = _get(r'EDIFFG\s*=\s*([-0-9.Ee-]+)', -0.01)
        p.ismear_detected = _get(r'ISMEAR\s*=\s*(-?[0-9]+)', 0, int)
        p.ispin_detected  = _get(r'ISPIN\s*=\s*([0-9]+)', 1, int)
        
        ivdw = _get(r'IVDW\s*=\s*([0-9]+)', 0, int)
        p.ivdw_detected = ivdw
        p.has_vdw = ivdw > 0
        
        # Detect ISYM and ALGO from source for consistency propagation
        isym_match = re.search(r'ISYM\s*=\s*(-?[0-9]+)', txt)
        p.isym_detected = int(isym_match.group(1)) if isym_match else None
        
        algo_match = re.search(r'ALGO\s*=\s*(\w+)', txt, re.IGNORECASE)
        p.algo_detected = algo_match.group(1) if algo_match else "Fast"
    
    # ================================================================
    # LAYER 3: POTCAR Basis-Set Validation
    # ================================================================
    def _layer3_potcar(self):
        """Parse POTCAR to extract ENMAX per element and validate ENCUT."""
        p = self.profile
        if not os.path.exists(self.potcar):
            p.warnings.append("POTCAR not found -- cannot validate basis set")
            return
        
        enmax_values = []
        potcar_elements = []
        
        with open(self.potcar, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                # Use TITEL lines: "   TITEL  = PAW_PBE Na_pv 19Sep2006"
                if 'TITEL' in line:
                    m = re.search(r'TITEL\s*=\s*PAW_\w+\s+(\S+)', line)
                    if m:
                        raw = m.group(1)  # e.g. "Na_pv", "B", "H"
                        elem = raw.split('_')[0]
                        potcar_elements.append(elem)
                # ENMAX line: "   ENMAX  =  259.561; ENMIN  = ..."
                if 'ENMAX' in line and 'ENMIN' in line:
                    m = re.search(r'ENMAX\s*=\s*([0-9.]+)', line)
                    if m:
                        enmax_values.append(float(m.group(1)))
        
        p.potcar_elements = potcar_elements
        p.potcar_enmax = enmax_values
        
        if enmax_values:
            p.potcar_max_enmax = max(enmax_values)
            p.encut_ratio = p.encut / p.potcar_max_enmax
            
            if p.encut_ratio < 1.3:
                p.warnings.append(
                    f"ENCUT/ENMAX ratio = {p.encut_ratio:.2f} (< 1.3). "
                    f"ENCUT={p.encut:.0f} eV may be insufficient for "
                    f"max(ENMAX)={p.potcar_max_enmax:.0f} eV. "
                    f"Recommended: >= {p.potcar_max_enmax * 1.3:.0f} eV.")
            else:
                p.notes.append(
                    f"ENCUT/ENMAX ratio = {p.encut_ratio:.2f} (OK, >= 1.3)")
    
    # ================================================================
    # LAYER 4: OUTCAR Quantitative Convergence
    # ================================================================
    def _layer4_outcar(self):
        """Extract forces, stress tensor, pressure, and energy from OUTCAR."""
        p = self.profile
        if not os.path.exists(self.outcar):
            p.warnings.append("OUTCAR not found -- cannot verify convergence")
            return
        
        with open(self.outcar, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Convergence flag
        for line in reversed(lines):
            if "reached required accuracy" in line:
                p.converged = True
                break
        if not p.converged:
            p.warnings.append(
                "CRITICAL: Relaxation did NOT reach required accuracy!")
        
        # Final energy
        for line in reversed(lines):
            m = re.search(r'energy\s+without\s+entropy\s*=\s*([-0-9.]+)', line)
            if m:
                p.final_energy = float(m.group(1))
                break
        
        # E-fermi
        for line in reversed(lines):
            if 'E-fermi' in line:
                m = re.search(r'E-fermi\s*:\s*([-0-9.]+)', line)
                if m: p.efermi = float(m.group(1))
                break
        
        # Count ionic steps
        p.n_ionic_steps = sum(1 for l in lines if 'F=' in l or 'E0=' in l)
        
        # Parse forces from the FINAL ionic step
        # Look for the last "TOTAL-FORCE" block
        last_force_start = None
        for i in range(len(lines) - 1, -1, -1):
            if 'TOTAL-FORCE' in lines[i]:
                last_force_start = i + 2  # skip header + dashes
                break
        
        if last_force_start is not None:
            forces = []
            for j in range(last_force_start, min(last_force_start + p.n_atoms + 5, len(lines))):
                parts = lines[j].split()
                if len(parts) >= 6:
                    try:
                        fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                        forces.append([fx, fy, fz])
                    except (ValueError, IndexError):
                        break
                else:
                    break
            
            if forces:
                farr = np.array(forces)
                magnitudes = np.linalg.norm(farr, axis=1)
                p.max_force = float(np.max(magnitudes))
                p.rms_force = float(np.sqrt(np.mean(magnitudes**2)))
        
        # Parse external pressure and Pulay stress
        for line in reversed(lines):
            if 'external pressure' in line:
                m = re.search(r'external pressure\s*=\s*([-0-9.]+)\s*kB', line)
                if m: p.external_pressure = float(m.group(1))
                break
        
        for line in reversed(lines):
            if 'Pullay stress' in line or 'Pulay stress' in line:
                m = re.search(r'stress\s*=\s*([-0-9.]+)\s*kB', line)
                if m: p.pulay_stress = float(m.group(1))
                break
    
    # ================================================================
    # LAYER 5: Electronic Structure (Bandgap)
    # ================================================================
    def _layer5_electronic(self):
        """Extract bandgap from vasprun.xml using pymatgen, with fallbacks."""
        p = self.profile
        
        # Priority 1: vasprun.xml (most reliable)
        if os.path.exists(self.vasprun):
            try:
                from pymatgen.io.vasp import Vasprun as PmgVasprun
                vr = PmgVasprun(self.vasprun, parse_dos=False,
                               parse_eigen=True, parse_potcar_file=False)
                bs = vr.get_band_structure()
                gap_info = bs.get_band_gap()
                if gap_info and 'energy' in gap_info:
                    p.bandgap = gap_info['energy']
                    p.is_insulator = p.bandgap > 0.1
                    direct = gap_info.get('direct', False)
                    p.bandgap_source = f"vasprun.xml ({'direct' if direct else 'indirect'})"
                    return
            except Exception:
                pass
        
        # Priority 2: ISMEAR heuristic
        if p.ismear_detected in (-5, 0):
            p.is_insulator = True
            p.bandgap_source = "heuristic (ISMEAR=0/-5 -> insulator)"
        elif p.ismear_detected >= 1:
            p.is_insulator = False
            p.bandgap_source = "heuristic (ISMEAR>=1 -> metal)"
        else:
            p.bandgap_source = "heuristic (default -> insulator)"
    
    # ================================================================
    # LAYER 6: Coordination Chemistry & Bonding
    # ================================================================
    def _layer6_bonding(self):
        """Full coordination environment, molecular fragment detection,
        ionicity index, and expected phonon frequency prediction."""
        p = self.profile
        cell = self._cell
        pos = self._cart_pos
        numbers = cell.numbers
        n = len(numbers)
        if n < 2:
            return
        
        lat = cell.cell
        inv_lat = np.linalg.inv(lat)
        
        # Pairwise distance matrix with PBC minimum image
        min_dist = 999.0
        min_pair = ("", "")
        
        # Coordination data
        coord_env = {i: [] for i in range(n)}
        pair_dists = {}  # (sym_i, sym_j) -> [list of distances]
        
        for i in range(n):
            for j in range(i+1, n):
                diff = pos[j] - pos[i]
                frac = diff @ inv_lat
                frac -= np.round(frac)
                cart = frac @ lat
                d = np.linalg.norm(cart)
                
                si = Z_TO_SYMBOL.get(numbers[i], "?")
                sj = Z_TO_SYMBOL.get(numbers[j], "?")
                
                if d < min_dist:
                    min_dist = d
                    min_pair = (si, sj)
                
                # Covalent/ionic bond threshold: 2.8 A
                if d < 2.8:
                    coord_env[i].append((sj, round(d, 3)))
                    coord_env[j].append((si, round(d, 3)))
                    
                    # Collect pair distances (sorted key for uniqueness)
                    key = tuple(sorted([si, sj]))
                    if key not in pair_dists:
                        pair_dists[key] = []
                    pair_dists[key].append(d)
        
        p.min_bond_length = min_dist
        p.min_bond_pair = min_pair
        p.coordination_env = coord_env
        
        # Average bond distances per pair
        p.bond_pairs_found = [
            (k[0], k[1], round(np.mean(v), 3))
            for k, v in sorted(pair_dists.items())
        ]
        
        # Molecular fragment detection -- covalent XH_n units only
        # Use strict covalent cutoff (< 1.6 A for X-H bonds) to avoid
        # falsely labeling ionic coordination shells (Na-H, Ca-H) as molecular.
        # Only atoms with EN >= 1.5 (covalent character) form true X-H molecules.
        COVALENT_H_CUTOFF = 1.6  # Angstrom, cf. B-H ~1.22, N-H ~1.01, C-H ~1.09
        COVALENT_EN_MIN   = 1.5  # Pauling EN -- excludes Na(0.93), Ca(1.0), etc.
        mol_units = {}
        for i in range(n):
            si = Z_TO_SYMBOL.get(numbers[i], "?")
            if si == 'H':
                continue
            en_i = ELEMENT_DB.get(si, (0, 0, 0, []))[2]
            if en_i < COVALENT_EN_MIN:
                continue  # skip ionic species (Na, Ca, K, ...)
            h_bonds = [(sj, d) for sj, d in coord_env[i]
                       if sj == 'H' and d < COVALENT_H_CUTOFF]
            if len(h_bonds) >= 2:
                tag = f"{si}H{len(h_bonds)}"
                if tag not in mol_units:
                    avg_d = np.mean([d for _, d in h_bonds])
                    mol_units[tag] = (tag, round(avg_d, 3), len(h_bonds))
        p.molecular_units = list(mol_units.values())
        
        # Ionicity index: max electronegativity difference among bonded pairs
        max_delta_en = 0.0
        for si, sj, _ in p.bond_pairs_found:
            en_i = ELEMENT_DB.get(si, (0, 0, 0, []))[2]
            en_j = ELEMENT_DB.get(sj, (0, 0, 0, []))[2]
            d = abs(en_i - en_j)
            if d > max_delta_en:
                max_delta_en = d
        p.ionicity_index = round(max_delta_en, 2)
        
        # Expected phonon frequencies from bond library
        for si, sj, avg_d in p.bond_pairs_found:
            key = (si, sj)
            rev_key = (sj, si)
            if key in BOND_FREQ_LIBRARY:
                lo, hi, label = BOND_FREQ_LIBRARY[key]
                p.expected_freq_ranges.append((label, lo, hi))
            elif rev_key in BOND_FREQ_LIBRARY:
                lo, hi, label = BOND_FREQ_LIBRARY[rev_key]
                p.expected_freq_ranges.append((label, lo, hi))
    
    # ================================================================
    # LAYER 7: Spin & Relativistic Assessment
    # ================================================================
    def _layer7_spin_soc(self):
        """Detect magnetic elements and spin-orbit coupling candidates."""
        p = self.profile
        
        # Magnetic system?
        mag_found = [e for e in p.elements if e in MAGNETIC_ATOMS]
        if mag_found or p.ispin_detected == 2:
            p.is_magnetic = True
            p.notes.append(f"Magnetic elements present: {', '.join(mag_found)}")
        
        # SOC candidates (Z > 56)
        soc = [e for e in p.elements if e in HEAVY_SOC_ATOMS]
        if soc:
            p.has_soc_candidates = True
            p.soc_elements = soc
            p.warnings.append(
                f"Heavy elements ({', '.join(soc)}) may require "
                f"spin-orbit coupling (LSORBIT=.TRUE.) for accurate phonons.")
    
    # ================================================================
    # LAYER 8: Phonon Readiness Score (0-100)
    # ================================================================
    def _layer8_readiness_score(self):
        """Compute a composite phonon readiness score."""
        p = self.profile
        score = 100
        reasons = []
        
        # Force convergence (0-30 points)
        if p.max_force is not None:
            if p.max_force > 0.01:
                penalty = min(30, int(p.max_force * 1000))
                score -= penalty
                reasons.append(f"max_force={p.max_force:.4f} eV/A (-{penalty})")
            elif p.max_force > 0.005:
                score -= 10
                reasons.append(f"max_force={p.max_force:.4f} eV/A (-10)")
            elif p.max_force > 0.001:
                score -= 3
                reasons.append(f"max_force={p.max_force:.4f} eV/A (-3)")
        else:
            score -= 15
            reasons.append("max_force unavailable (-15)")
        
        # Convergence flag (0-20 points)
        if not p.converged:
            score -= 20
            reasons.append("relaxation not converged (-20)")
        
        # Pulay stress (0-15 points)
        if p.external_pressure is not None and abs(p.external_pressure) > 1.0:
            penalty = min(15, int(abs(p.external_pressure)))
            score -= penalty
            reasons.append(f"pressure={p.external_pressure:.1f} kBar (-{penalty})")
        
        # ENCUT adequacy (0-15 points)
        if p.encut_ratio > 0 and p.encut_ratio < 1.3:
            score -= 15
            reasons.append(f"ENCUT/ENMAX={p.encut_ratio:.2f} < 1.3 (-15)")
        elif p.encut_ratio > 0 and p.encut_ratio < 1.5:
            score -= 5
            reasons.append(f"ENCUT/ENMAX={p.encut_ratio:.2f} < 1.5 (-5)")
        
        # EDIFF tightness (0-10 points)
        if p.ediff > 1e-7:
            score -= 10
            reasons.append(f"EDIFF={p.ediff:.0e} > 1E-7 (coarse for phonons) (-10)")
        elif p.ediff > 1e-8:
            score -= 3
            reasons.append(f"EDIFF={p.ediff:.0e} (acceptable) (-3)")
        
        # Real-space projectors (0-10 points) -- checked via INCAR
        if os.path.exists(self.incar):
            with open(self.incar, 'r', encoding='utf-8', errors='replace') as f:
                txt = f.read()
            if re.search(r'LREAL\s*=\s*\.?TRUE', txt, re.IGNORECASE):
                score -= 10
                reasons.append("LREAL=.TRUE. detected (-10, use .FALSE. for phonons)")
        
        p.phonon_readiness_score = max(0, min(100, score))
        if reasons:
            p.notes.append("Readiness scoring: " + "; ".join(reasons))
    
    # ================================================================
    # LAYER 9: Smart Recommendation Engine
    # ================================================================
    def _layer9_recommendations(self):
        """Generate optimized phonon plan based on material profile."""
        p = self.profile
        a, b, c = p.lattice_params
        
        # ---- Supercell ----
        min_target = 10.0  # A, for force constant decay
        rec_dim = [
            max(2, math.ceil(min_target / x)) if x > 0.1 else 2
            for x in (a, b, c)
        ]
        
        sc_atoms = p.n_atoms * (rec_dim[0] * rec_dim[1] * rec_dim[2])
        if sc_atoms > 500:
            rec_dim = [min(d, 2) for d in rec_dim]
            sc_atoms = p.n_atoms * (rec_dim[0] * rec_dim[1] * rec_dim[2])
            p.warnings.append(
                f"Large supercell ({sc_atoms} atoms). "
                f"Capped dimensions. Consider computational budget.")
        
        p.rec_supercell = rec_dim
        p.sc_atoms = sc_atoms
        
        sc_lens = (a * rec_dim[0], b * rec_dim[1], c * rec_dim[2])
        if min(sc_lens) < 10.0:
            p.warnings.append(
                f"Min SC dimension = {min(sc_lens):.1f} A (< 10 A). "
                f"Periodic image forces may contaminate HFCs.")
        
        # ---- Displacement amplitude ----
        p.rec_displacement = 0.01
        if p.has_light_atoms and p.mass_ratio > 20:
            p.notes.append(
                f"High mass ratio ({p.mass_ratio:.1f}x). "
                f"Standard 0.01 A displacement is adequate. "
                f"Use dense DOS mesh for high-frequency mode resolution.")
        
        # ---- Born charges ----
        p.rec_born = p.is_insulator
        if not p.is_insulator:
            p.notes.append("Metallic: Born charges (LEPSILON) not required.")
        else:
            if p.ionicity_index > 1.5:
                p.notes.append(
                    f"High ionicity ({p.ionicity_index}). "
                    f"LO-TO splitting expected to be significant at Gamma.")
        
        # ---- Smearing ----
        if p.is_insulator:
            p.rec_ismear = 0
            p.rec_sigma = 0.05
        else:
            p.rec_ismear = 1
            p.rec_sigma = 0.20
        
        # ---- vdW ----
        if p.has_vdw:
            p.rec_vdw = True
            p.rec_ivdw = p.ivdw_detected
        elif p.molecular_units:
            p.rec_vdw = True
            p.rec_ivdw = 13
            p.warnings.append(
                f"Molecular units ({', '.join(u[0] for u in p.molecular_units)}) "
                f"detected but no IVDW in source. Recommending D4.")
        
        # ---- K-points (reciprocal space scaling) ----
        # Proper formula: N_i = max(1, round(|b_i| / KSPACING / dim_i))
        # where |b_i| = 2*pi / a_i for orthorhombic
        recip = 2 * np.pi / np.array([a, b, c])
        p.rec_kpoints_sc = [
            max(1, round(recip[i] / p.kspacing / rec_dim[i]))
            for i in range(3)
        ]
        
        # ---- EDIFF for phonon forces ----
        p.rec_ediff_phonon = 1e-8
        
        # ---- ALGO (electronic minimization algorithm) ----
        # Default: keep Fast for most systems (original behavior)
        # For H-rich systems with high mass ratio, Normal is more stable
        # because RMM-DIIS (Fast) can struggle with wide energy ranges
        p.rec_algo = "Fast"
        h_fraction = p.elements.get('H', 0) / max(p.n_atoms, 1)
        if p.has_light_atoms and p.mass_ratio > 20 and h_fraction > 0.5:
            p.rec_algo = "Normal"
            p.notes.append(
                f"ALGO=Normal recommended: H-rich system ({h_fraction*100:.0f}% H) "
                f"with mass ratio {p.mass_ratio:.1f}x. "
                f"Blocked-Davidson is more robust than RMM-DIIS for wide energy ranges.")
        elif p.algo_detected and p.algo_detected.lower() != "fast":
            # Preserve source ALGO if it was explicitly set to something other than Fast
            p.rec_algo = p.algo_detected
            p.notes.append(
                f"ALGO={p.algo_detected} propagated from source INCAR for consistency.")
        
        # ---- ISYM (symmetry handling) ----
        # Default: None (let VASP decide, original behavior)
        # If source used ISYM=0, propagate it to avoid symmetry artifacts
        # Triclinic systems also benefit from ISYM=0 to prevent false symmetrization
        p.rec_isym = None
        if p.isym_detected is not None and p.isym_detected == 0:
            p.rec_isym = 0
            p.notes.append(
                f"ISYM=0 propagated from source INCAR for consistency. "
                f"Prevents symmetry artifacts in displacement force calculations.")
        elif p.crystal_system == "Triclinic":
            p.rec_isym = 0
            p.notes.append(
                f"ISYM=0 recommended for Triclinic system (SG: {p.space_group}). "
                f"Low symmetry makes VASP symmetrization unreliable for phonon forces.")
        
        # ---- DOS mesh ----
        if p.mass_ratio > 10:
            p.rec_dos_mesh = [20, 20, 20]
        elif p.n_atoms > 20:
            p.rec_dos_mesh = [12, 12, 12]
        else:
            p.rec_dos_mesh = [15, 15, 15]
        
        # ---- Time estimates (empirical N^3 DFT scaling) ----
        # Base: ~30 min per SCF for 100 atoms on 32 cores
        scf_factor = (p.sc_atoms / 100.0) ** 2.5
        p.rec_born_time_hrs = max(4, int(p.n_atoms * 0.3))
        p.rec_disp_time_hrs = max(6, int(scf_factor * 2))
        
        # ---- Displacement count estimate ----
        # Use phonopy directly to get the exact symmetry-reduced count
        try:
            from phonopy import Phonopy
            trial = Phonopy(self._cell, np.diag(rec_dim))
            trial.generate_displacements(distance=p.rec_displacement)
            p.n_disp_estimate = len(trial.supercells_with_displacements)
        except Exception:
            # Fallback: rough estimate 3*N_atoms (P1 upper bound)
            p.n_disp_estimate = 3 * p.n_atoms
    
    # ================================================================
    # DISPLAY METHODS
    # ================================================================
    def print_profile(self):
        """Print a comprehensive material profile card."""
        p = self.profile
        
        # Electronic nature string
        if p.bandgap is not None:
            gap_str = f"gap={p.bandgap:.2f} eV, {p.bandgap_source}"
        else:
            gap_str = p.bandgap_source
        elec_str = "Insulator" if p.is_insulator else "Metal"
        
        elem_str = ", ".join(f"{s}({c})" for s, c in p.elements.items())
        mol_str = ", ".join(f"{u[0]}(d={u[1]}A)" for u in p.molecular_units) or "None"
        vdw_str = f"Yes (IVDW={p.ivdw_detected})" if p.has_vdw else "No"
        bonds_str = "; ".join(f"{a}-{b}={d}A" for a, b, d in p.bond_pairs_found[:5])
        mag_str = "Yes" if p.is_magnetic else "No"
        soc_str = f"Candidates: {','.join(p.soc_elements)}" if p.has_soc_candidates else "Not needed"
        
        # Force strings
        fmax = f"{p.max_force:.6f}" if p.max_force is not None else "N/A"
        frms = f"{p.rms_force:.6f}" if p.rms_force is not None else "N/A"
        pres = f"{p.external_pressure:.2f} kBar" if p.external_pressure is not None else "N/A"
        enratio = f"{p.encut_ratio:.2f}" if p.encut_ratio > 0 else "N/A"
        
        # Readiness color
        rs = p.phonon_readiness_score
        if rs >= 85: grade = "EXCELLENT"
        elif rs >= 70: grade = "GOOD"
        elif rs >= 50: grade = "FAIR"
        else: grade = "POOR"
        
        W = 58
        print()
        print("  +" + "=" * W + "+")
        print(f"  |{'MATERIAL PROFILE':^{W}}|")
        print("  +" + "=" * W + "+")
        
        def row(label, value):
            content = f"  {label:<18}{value}"
            print(f"  | {content:<{W-1}}|")
        
        row("Formula:",       p.formula)
        row("Reduced:",       p.reduced_formula)
        row("Space Group:",   p.space_group)
        row("Crystal Sys.:",  f"{p.crystal_system} (PG: {p.point_group}, {p.n_symmetry_ops} ops)")
        row("Unit Cell:",     f"{p.n_atoms} atoms, V={p.volume:.2f} A^3, rho={p.density:.3f} g/cm^3")
        row("Elements:",      elem_str)
        row("Lightest:",      f"{p.lightest_atom[0]} ({p.lightest_atom[1]:.3f} amu)")
        row("Heaviest:",      f"{p.heaviest_atom[0]} ({p.heaviest_atom[1]:.3f} amu)")
        row("Mass ratio:",    f"{p.mass_ratio:.1f}x (avg={p.avg_mass:.2f} amu)")
        
        print("  +" + "-" * W + "+")
        row("Electronic:",    f"{elec_str} ({gap_str})")
        row("Magnetic:",      mag_str)
        row("SOC:",           soc_str)
        row("Ionicity:",      f"delta_EN={p.ionicity_index}")
        
        print("  +" + "-" * W + "+")
        row("Bonding:",       f"Ionic+Molecular" if p.molecular_units else "Covalent/Ionic")
        row("Mol. units:",    mol_str)
        row("Bond pairs:",    bonds_str if bonds_str else "N/A")
        row("Min bond:",      f"{p.min_bond_pair[0]}-{p.min_bond_pair[1]} = {p.min_bond_length:.3f} A")
        row("vdW (source):",  vdw_str)
        
        print("  +" + "-" * W + "+")
        row("Lattice:",       f"{p.lattice_params[0]:.4f} x {p.lattice_params[1]:.4f} x {p.lattice_params[2]:.4f} A")
        row("Angles:",        f"{p.lattice_angles[0]} x {p.lattice_angles[1]} x {p.lattice_angles[2]} deg")
        row("ENCUT:",         f"{p.encut:.1f} eV (ratio={enratio})")
        row("KSPACING:",      f"{p.kspacing}")
        row("EDIFF:",         f"{p.ediff:.0e}")
        row("POTCAR:",        f"{', '.join(p.potcar_elements)}")
        
        print("  +" + "-" * W + "+")
        row("Converged:",     f"{'YES' if p.converged else 'NO'}")
        row("Max force:",     f"{fmax} eV/A")
        row("RMS force:",     f"{frms} eV/A")
        row("Pressure:",      pres)
        row("Ionic steps:",   f"{p.n_ionic_steps}")
        if p.final_energy is not None:
            row("Final E:",   f"{p.final_energy:.6f} eV")
        
        print("  +" + "-" * W + "+")
        row("READINESS:",     f"{rs}/100 ({grade})")
        print("  +" + "=" * W + "+")
        
        # Expected frequency ranges
        if p.expected_freq_ranges:
            print(f"\n  Expected vibrational modes:")
            for label, lo, hi in p.expected_freq_ranges:
                print(f"    - {label}: {lo}-{hi} cm^-1")

    def print_recommendations(self):
        """Print the optimized phonon calculation plan."""
        p = self.profile
        sc = p.rec_supercell
        sc_lens = (p.lattice_params[0]*sc[0], p.lattice_params[1]*sc[1], p.lattice_params[2]*sc[2])
        born_str = "YES (insulator)" if p.rec_born else "NO (metal)"
        vdw_str = f"D4 (IVDW={p.rec_ivdw})" if p.rec_vdw else "None"
        kp = p.rec_kpoints_sc
        
        W = 58
        print()
        print("  +" + "=" * W + "+")
        print(f"  |{'RECOMMENDED PHONON PLAN':^{W}}|")
        print("  +" + "=" * W + "+")
        
        def row(label, value):
            content = f"  {label:<18}{value}"
            print(f"  | {content:<{W-1}}|")
        
        row("Supercell:",    f"{sc[0]}x{sc[1]}x{sc[2]} ({p.sc_atoms} atoms, min {min(sc_lens):.1f} A)")
        row("Displacement:", f"{p.rec_displacement} A")
        row("Born charges:", born_str)
        row("vdW correct.:", vdw_str)
        row("ENCUT:",        f"{p.encut:.1f} eV (matched to source)")
        row("EDIFF:",        f"{p.rec_ediff_phonon:.0e} (phonon-grade)")
        row("ALGO:",         f"{p.rec_algo}")
        isym_str = str(p.rec_isym) if p.rec_isym is not None else "VASP default"
        row("ISYM:",         isym_str)
        row("ISMEAR/SIGMA:", f"{p.rec_ismear} / {p.rec_sigma}")
        row("SC K-points:",  f"{kp[0]}x{kp[1]}x{kp[2]} (Gamma)")
        row("LREAL:",        ".FALSE. (exact projectors)")
        row("ADDGRID:",      ".TRUE. (enhanced FFT)")
        row("DOS mesh:",     f"{p.rec_dos_mesh[0]}x{p.rec_dos_mesh[1]}x{p.rec_dos_mesh[2]} (post-proc)")
        row("Est. disps:",   f"~{p.n_disp_estimate} (symmetry-reduced)")
        row("Est. time:",    f"~{p.rec_disp_time_hrs}h/disp on 32 cores")
        print("  +" + "=" * W + "+")
        
        if p.warnings:
            print()
            print("  WARNINGS:")
            for w in p.warnings:
                print(f"    [!] {w}")
        
        if p.notes:
            print()
            print("  NOTES:")
            for n in p.notes:
                print(f"    [i] {n}")
