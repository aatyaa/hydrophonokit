"""
=============================================================================
  HydroPhonoKit v2.1 -- Scientific Post-Processing Engine
  
  Rigorous data extraction, force constant computation, and thermodynamic
  analysis. Implements strict sum rules, Born effective charge symmetrization,
  and LO-TO splitting. Replaces legacy single-use scripts.
=============================================================================
"""
import os
import sys
import warnings
import numpy as np

# NumPy 2.0+ compatibility: np.trapz was renamed to np.trapezoid
if hasattr(np, 'trapezoid'):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz

# Import rigorous physical constants and methods
from .physics import (
    THZ_TO_CM, BORN_FACTOR,
    R_GAS, H_PLANCK, K_BOLTZMANN, N_AVOGADRO, HBAR,
    bose_einstein, helmholtz_free_energy, phonon_entropy,
    heat_capacity_cv, dulong_petit_limit, zero_point_energy,
    check_dynamical_stability,
    H_MODE_RANGES, HYDRIDE_STRETCH_LIBRARY,
)

# Suppress only specific noisy warnings, not all warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='phonopy')

from phonopy import load as phonopy_load
from phonopy.interface.vasp import Vasprun

# Configure matplotlib backend: use Agg for non-interactive (file output)
# but respect user's environment if they've set MPLBACKEND
import os as _os
if not _os.environ.get('MPLBACKEND'):
    import matplotlib
    matplotlib.use('Agg')
del _os

import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.linewidth': 1.2, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

class PhononPostProcessor:
    """Master post-processing engine adapting to the MaterialProfile."""
    
    def __init__(self, workspace_dir, profile, output_dir=None):
        """
        Args:
            workspace_dir (str): Path to generated/run HydroPhonoKit workspace.
            profile (MaterialProfile): The scientific profile built by analyzer.py.
            output_dir (str, optional): Override the default output directory.
        """
        self.workspace_dir = workspace_dir
        self.profile = profile
        
        # Paths
        self.disp_yaml    = os.path.join(workspace_dir, 'phonopy_disp.yaml')
        self.born_outcar  = os.path.join(workspace_dir, '01_born', 'OUTCAR')
        self.disp_dir     = os.path.join(workspace_dir, '02_displacements')
        if output_dir:
            self.output_dir = output_dir
        else:
            import datetime
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = os.path.join(workspace_dir, f'phonon_results_{stamp}')
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.phonon       = None
        self.dielectric   = None
        self.born_charges = None
        self.force_sets   = None

    def execute_pipeline(self):
        """Execute the full post-processing pipeline."""
        print("\n" + "=" * 60)
        print(f"  PHONON POST-PROCESSING: {self.profile.formula}")
        print("=" * 60)
        
        self._phase1_data()
        self._phase2_force_constants()
        self._phase3_bands_dos()
        self._phase4_thermodynamics()
        
        if 'H' in self.profile.elements:
            self._phase5_h_analysis()

        self._plot_all()
        self._phase6_reporting()

        # Write BORN file to results
        if self.born_charges is not None:
            born_path = os.path.join(self.output_dir, 'BORN')
            with open(born_path, 'w') as f:
                f.write(f"{BORN_FACTOR:.5f}\n")
                d = self.dielectric
                f.write(f"{d[0,0]:.6f} {d[0,1]:.6f} {d[0,2]:.6f} "
                        f"{d[1,0]:.6f} {d[1,1]:.6f} {d[1,2]:.6f} "
                        f"{d[2,0]:.6f} {d[2,1]:.6f} {d[2,2]:.6f}\n")
                for z in self.born_charges:
                    f.write(f"{z[0,0]:.6f} {z[0,1]:.6f} {z[0,2]:.6f} "
                            f"{z[1,0]:.6f} {z[1,1]:.6f} {z[1,2]:.6f} "
                            f"{z[2,0]:.6f} {z[2,1]:.6f} {z[2,2]:.6f}\n")
            print(f"  --> Saved BORN to: {born_path}")

        print("\n[✓] Post-processing pipeline complete.")
        print(f"    Results saved to: {self.output_dir}")
        
    # ========================================================================
    # PHASE 1: DATA COLLECTION & RIGOROUS SYMMETRIZATION
    # ========================================================================
    def _phase1_data(self):
        """Load displacements, extract forces, and parse Born charges with sum rules."""
        print("\n[Phase 1] Data Collection & Precision Validations")
        
        # 1.1 Load phonopy object
        if not os.path.exists(self.disp_yaml):
            raise FileNotFoundError(f"Cannot find {self.disp_yaml}. Did generator run?")
        
        self.phonon = phonopy_load(self.disp_yaml)
        n_atoms_uc = len(self.phonon.unitcell.numbers)
        n_atoms_sc = len(self.phonon.supercell.numbers)
        print(f"  --> Loaded geometry: {n_atoms_uc} atoms in primitive cell.")
        print(f"  --> Supercell size: {n_atoms_sc} atoms.")

        # 1.2 Read Forces
        print("  --> Parsing forces from VASP vasprun.xml files ...")
        n_disp = len(self.phonon.supercells_with_displacements)
        force_sets = []
        
        for i in range(n_disp):
            disp_id = f"{i+1:03d}"
            vpath = os.path.join(self.disp_dir, f'disp-{disp_id}', 'vasprun.xml')
            if not os.path.exists(vpath):
                raise FileNotFoundError(f"Missing force data: {vpath}")
            
            vr = Vasprun(vpath)
            # Drift correction: VASP forces might have tiny residual drift.
            # phonopy will handle acoustic sum rule internally, but we can verify it.
            forces = vr.read_forces()
            force_sets.append(forces)
            
        self.phonon.forces = np.array(force_sets)
        print(f"  --> Successfully collected {n_disp} force sets.")

        # 1.3 Read Born Charges (if insulator)
        if self.profile.rec_born:
            if not os.path.exists(self.born_outcar):
                print("  [!] WARNING: Profile expects Born charges, but NO 01_born/OUTCAR found.")
                print("      Proceeding WITHOUT non-analytical corrections (LO-TO splitting disabled).")
                self.profile.rec_born = False
            else:
                self._extract_and_symmetrize_born()

    def _extract_and_symmetrize_born(self):
        """
        Extract dielectric tensor and Born effective charges.
        Rigorous application of Acoustic Sum Rule (ASR) to Born charges:
            Sum_i Z*_{i, a, b} = 0
        """
        print("  --> Parsing OUTCAR for Dielectric Tensor and Born Effective Charges ...")
        with open(self.born_outcar, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract Dielectric
        diel = np.zeros((3, 3))
        idx = content.find('MACROSCOPIC STATIC DIELECTRIC TENSOR (including')
        if idx == -1: idx = content.find('MACROSCOPIC STATIC DIELECTRIC TENSOR')
        if idx != -1:
            lines = content[idx:idx+500].split('\n')
            r = 0
            for line in lines[2:]:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        diel[r] = [float(parts[0]), float(parts[1]), float(parts[2])]
                        r += 1
                        if r >= 3: break
                    except ValueError:
                        break
        
        # Symmetrize Dielectric Tensor (ε_αβ = ε_βα)
        self.dielectric = 0.5 * (diel + diel.T)
        print(f"      Dielectric Trace (Tr/3): {np.trace(self.dielectric)/3.0:.4f}")

        # Extract Born Charges
        bc_list = []
        idx = content.find('BORN EFFECTIVE CHARGES')
        if idx != -1:
            lines = content[idx:].split('\n')
            i = 0
            while i < len(lines):
                if 'ion' in lines[i] and len(lines[i].split()) <= 3:
                    z_tensor = []
                    for j in range(1, 4):
                        if i + j < len(lines):
                            parts = lines[i+j].split()
                            if len(parts) >= 3:
                                try:
                                    z_tensor.append([float(parts[1]), float(parts[2]), float(parts[3])])
                                except ValueError: pass
                    if len(z_tensor) == 3:
                        bc_list.append(z_tensor)
                    i += 4
                else:
                    i += 1
                if len(bc_list) >= len(self.phonon.unitcell.numbers):
                    break
        
        if len(bc_list) == len(self.phonon.unitcell.numbers):
            Z = np.array(bc_list)
            
            # Application of Acoustic Sum Rule (ASR) for Born Charges
            # To ensure translation invariance, the sum of Born charges over all atoms must strictly be zero.
            z_sum = np.sum(Z, axis=0)
            initial_trace_err = np.trace(z_sum)/3.0
            print(f"      Born Charge Sum Rule Error (before correction): {initial_trace_err:.5f}")
            
            # Distribute the error equally
            correction = z_sum / len(Z)
            Z_corrected = Z - correction
            
            self.born_charges = Z_corrected
            new_err = np.trace(np.sum(self.born_charges, axis=0))/3.0
            print(f"      Born Charge Sum Rule Error (after correction):  {new_err:.5f}")
            
            # Register with phonopy
            nac_params = {
                'born': self.born_charges,
                'dielectric': self.dielectric,
                'factor': 14.399652,  # VASP internal constant: e^2 / (4π ε_0) in eV·A
            }
            self.phonon.nac_params = nac_params
            print("  --> Non-Analytical Corrections (NAC) initialized successfully.")
        else:
            print("  [!] WARNING: Failed to extract proper Born charges. Disabling NAC.")
            self.profile.rec_born = False

    # ========================================================================
    # PHASE 2: FORCE CONSTANTS (IFCs)
    # ========================================================================
    def _phase2_force_constants(self):
        """Construct Interatomic Force Constants (IFCs) using state-of-the-art methods."""
        print("\n[Phase 2] Force Constants & Acoustic Sum Rules")
        
        # We attempt to use symfc (Symmetry-Adapted Force Constants)
        # It rigorously enforces point-group/translational symmetry, which is critical
        # to rid the force constants of noise from grid-cutoff anomalies.
        used_symfc = False
        try:
            from symfc import Symfc
            print("  --> Using 'symfc' (Symmetry-Adapted Force Constants) engine ...")
            self.phonon.produce_force_constants(fc_calculator='symfc')
            
            # Sanity check: if symfc returns a zero matrix due to version/API issues
            if np.max(np.abs(self.phonon.force_constants)) < 1e-3:
                print("  [!] symfc returned suspicious near-zero IFCs. Falling back to default phonopy solver.")
                used_symfc = False
            else:
                used_symfc = True
                print("  --> symfc IFCs generated successfully.")
        except Exception as e:
            print(f"  [i] symfc not available or failed ({e}). Using standard phonopy solver...")

        if not used_symfc:
            self.phonon.produce_force_constants()
            print("  --> Standard Phonopy IFCs generated.")

        # Save the computed force constants
        fc_path = os.path.join(self.output_dir, 'FORCE_CONSTANTS')
        from phonopy.file_IO import write_FORCE_CONSTANTS
        write_FORCE_CONSTANTS(self.phonon.force_constants, fc_path)
        print(f"  --> Saved IFCs to: {fc_path}")

    # ========================================================================
    # PHASE 3: BAND STRUCTURE & PARTIAL DOS
    # ========================================================================
    def _phase3_bands_dos(self):
        """Compute band structure (with seekpath k-path) and partial DOS."""
        print("\n[Phase 3] Phonon Band Structure & Partial Density of States")
        
        # 3.1: Seekpath High-Symmetry K-Path
        try:
            import seekpath
            cell = (self.phonon.unitcell.cell, 
                    self.phonon.unitcell.scaled_positions,
                    self.phonon.unitcell.numbers)
            kpath = seekpath.get_path(cell)
            labels = []
            band_paths = []
            path_data = kpath['path']
            special_points = kpath['point_coords']
            
            for seg in path_data:
                start_label, end_label = seg
                band_paths.append([special_points[start_label], special_points[end_label]])
                labels.append([start_label.replace('GAMMA', 'Γ'), end_label.replace('GAMMA', 'Γ')])
            
            flat_labels = [labels[0][0]] + [l[1] for l in labels]
            print(f"  --> Identified High-Symmetry path: {' -> '.join(flat_labels)}")
            self.phonon.run_band_structure(band_paths, labels=[l for pair in labels for l in pair])
            used_auto = False
        except Exception as e:
            print(f"  [i] seekpath not installed or failed ({e}). Using phonopy auto band structure...")
            self.phonon.auto_band_structure(npoints=101)
            used_auto = True

        # Extract frequencies to check for dynamic instability
        band_dict = self.phonon.get_band_structure_dict()
        bands_freq = np.concatenate([f.flatten() for f in band_dict['frequencies']])
        min_freq = np.min(bands_freq)
        self.profile.notes.append(f"Min freq from band structure: {min_freq:.3f} THz")

        # Rigorous dynamical stability check using physics module
        stability_result = check_dynamical_stability(bands_freq)
        self.profile.notes.append(
            f"Dynamical stability: {'STABLE' if stability_result['stable'] else 'UNSTABLE'} "
            f"(min={min_freq:.3f} THz, n_imaginary={stability_result['n_imaginary']})")

        if stability_result['n_imaginary'] > 0:
            print(f"  [!] IMAGINARY FREQUENCIES DETECTED:")
            print(f"      Minimum frequency: {stability_result['min_freq']:.2f} THz")
            print(f"      Number of imaginary modes: {stability_result['n_imaginary']}")
            print(f"      Phase is dynamically UNSTABLE -- structural phase transition likely")
        else:
            print(f"  [✓] Phase dynamically stable at 0K (min freq: {min_freq:.2f} THz)")

        # Save exact band.yaml for future plotting integrations
        self.phonon.write_yaml_band_structure(filename=os.path.join(self.output_dir, 'band.yaml'))
        
        # 3.2: Partial Density of States (DOS)
        # For pDOS, use a mesh scaling similar to what Analyzer suggested
        mesh = self.profile.rec_dos_mesh
        print(f"  --> Computing pDOS on {mesh[0]}x{mesh[1]}x{mesh[2]} Q-point mesh...")
        
        # Use Gaussian smearing (tetrahedron method fails for unstable / non-gamma cells easily)
        self.phonon.run_mesh(mesh, is_mesh_symmetry=False, with_eigenvectors=True)
        self.phonon.run_projected_dos(freq_pitch=0.1, use_tetrahedron_method=False, sigma=0.2)
        pdos_dict = self.phonon.projected_dos
        
        # Determine atom index mappings from unit cell
        sym_to_idx = {}
        for i, z in enumerate(self.phonon.unitcell.numbers):
            from .analyzer import Z_TO_SYMBOL
            sym = Z_TO_SYMBOL.get(z, f"Z{z}")
            if sym not in sym_to_idx: sym_to_idx[sym] = []
            sym_to_idx[sym].append(i)
            
        print("  --> Partial DOS calculated successfully.")
        
        # Make data accessible to plotting/analysis
        self._band_dict = band_dict
        self._pdos_data = (pdos_dict.frequency_points, pdos_dict.projected_dos, sym_to_idx)
        self.min_freq = min_freq

    # ========================================================================
    # PHASE 4: THERMODYNAMICS
    # ========================================================================
    def _phase4_thermodynamics(self):
        """Compute F(T), S(T), Cv(T) with harmonic approximation.

        Uses rigorous statistical mechanics formulas:
          - F(T) = kT * Sum ln[2 sinh(hbar*omega / 2kT)]  (Helmholtz free energy)
          - S(T) = k * Sum [(n+1)ln(n+1) - n*ln(n)]       (entropy)
          - C_v(T) = k * Sum x^2 / (4*sinh^2(x/2))        (heat capacity)

        Validates against:
          - Zero-Point Energy at T=0
          - Dulong-Petit limit at high T: C_v -> 3N*R
          - Third law of thermodynamics: S -> 0 as T -> 0

        Reference: Born & Huang, Dynamical Theory of Crystal Lattices (1954);
                   Baroni et al., Rev. Mod. Phys. 73, 515 (2001).
        """
        print("\n[Phase 4] Thermodynamic Integrals (Harmonic Approximation)")

        # Compute phonon DOS on dense mesh for accurate thermodynamics
        mesh = self.profile.rec_dos_mesh
        print(f"  --> Computing thermodynamics on {mesh[0]}x{mesh[1]}x{mesh[2]} Q-point mesh...")
        self.phonon.run_mesh(mesh)
        self.phonon.run_thermal_properties(t_min=0, t_max=1000, t_step=10)

        tp = self.phonon.get_thermal_properties_dict()
        self._thermo_data = tp

        temps = tp['temperatures']
        free_energy = tp['free_energy']       # kJ/mol
        entropy = tp['entropy']               # J/(mol.K)
        heat_capacity = tp['heat_capacity']   # J/(mol.K)

        # Zero-point energy (ZPE) at T=0
        # Phonopy reports F(T=0) which equals ZPE in harmonic approximation
        zpe = free_energy[0]
        self.profile.notes.append(f"ZPE: {zpe:.3f} kJ/mol")
        print(f"  --> Zero-Point Energy (ZPE): {zpe:.3f} kJ/mol")

        # Validate Third Law: S(T=0) should be 0
        if abs(entropy[0]) > 0.01:
            print(f"  [!] WARNING: S(T=0) = {entropy[0]:.4f} J/(mol·K) -- should be ~0 (Third Law)")
        else:
            print(f"  [✓] Third Law validated: S(T=0) = {entropy[0]:.4e} J/(mol·K)")

        # 300K Data
        idx_300 = int(np.argmin(np.abs(temps - 300)))
        print(f"  --> Thermodynamics @ 300 K:")
        print(f"      F   = {free_energy[idx_300]:.3f} kJ/mol")
        print(f"      S   = {entropy[idx_300]:.3f} J/(mol·K)")
        print(f"      C_v = {heat_capacity[idx_300]:.3f} J/(mol·K)")

        # Validate Dulong-Petit Limit: C_v -> 3N*R at high T
        n_atoms = len(self.phonon.unitcell.numbers)
        dp_limit = dulong_petit_limit(n_atoms)
        cv_1000k = heat_capacity[-1]
        dp_error = abs(cv_1000k - dp_limit) / dp_limit * 100

        print(f"  --> Dulong-Petit Limit Check (3N*R):")
        print(f"      Theoretical limit : {dp_limit:.1f} J/(mol·K)  (N={n_atoms})")
        print(f"      Calculated C_v@1000K: {cv_1000k:.1f} J/(mol·K)")
        print(f"      Deviation: {dp_error:.1f}%")

        if dp_error > 5:
            print(f"  [!] WARNING: C_v at 1000K deviates {dp_error:.1f}% from Dulong-Petit.")
            print(f"      Material may have optical modes not yet converged, or T_max is insufficient.")

        # Store validation results
        self._thermo_validations = {
            'third_law_ok': abs(entropy[0]) < 0.01,
            'dulong_petit_error_pct': dp_error,
            'dulong_petit_limit': dp_limit,
            'cv_at_1000k': cv_1000k,
        }

        # Save raw data with extended metadata
        raw_path = os.path.join(self.output_dir, 'thermodynamic_properties.dat')
        with open(raw_path, 'w') as f:
            f.write("# HydroPhonoKit v2.2 -- Thermodynamic Properties (Harmonic Approximation)\n")
            f.write("# Reference: Born & Huang, Dynamical Theory of Crystal Lattices (1954)\n")
            f.write("#\n")
            f.write(f"# Material: {self.profile.formula}\n")
            f.write(f"# Atoms per cell: {n_atoms}\n")
            f.write(f"# Dulong-Petit limit: {dp_limit:.2f} J/(mol·K)\n")
            f.write(f"# ZPE: {zpe:.3f} kJ/mol\n")
            f.write("#\n")
            f.write("# T(K)   F(kJ/mol)   S(J/mol.K)   Cv(J/mol.K)\n")
            for t, ffe, s, cv in zip(temps, free_energy, entropy, heat_capacity):
                f.write(f"{t:6.1f}   {ffe:10.4f}   {s:10.4f}   {cv:10.4f}\n")
        print(f"  --> Saved Integrals to: {raw_path}")

    # ========================================================================
    # PHASE 5: MOLECULAR SPECIES ANALYSIS (HYDROGEN STORAGE)
    # ========================================================================
    def _phase5_h_analysis(self):
        """Quantitatively separate H modes into Librational, Bending, and Stretching fractions.

        Uses rigorous integration boundaries based on metal hydride spectroscopy.
        Reference: Nakamoto, IR/Raman Spectra of Inorganic Compounds;
                   Bogdanovic et al., J. Alloys Compd. (2004).
        """
        print("\n[Phase 5] Hydrogen Storage Phonon Analysis")

        freq, pdos, sym_to_idx = self._pdos_data

        h_dos = np.zeros_like(freq)
        for idx in sym_to_idx.get('H', []):
            if idx < pdos.shape[0]:
                h_dos += pdos[idx]

        freq_cm = freq * THZ_TO_CM

        # Use scientifically defined integration boundaries from physics module
        lib_range = H_MODE_RANGES['librational']    # (5.0, 20.9) THz
        bend_range = H_MODE_RANGES['bending']       # (20.9, 50.0) THz
        stretch_range = H_MODE_RANGES['stretching'] # (50.0, 100.0) THz

        librational = (freq > lib_range[0]) & (freq < lib_range[1])
        bending = (freq > bend_range[0]) & (freq < bend_range[1])
        stretching = (freq > stretch_range[0]) & (freq < stretch_range[1])

        h_lib = _trapz(h_dos[librational], freq[librational]) if np.any(librational) else 0.0
        h_bend = _trapz(h_dos[bending], freq[bending]) if np.any(bending) else 0.0
        h_stretch = _trapz(h_dos[stretching], freq[stretching]) if np.any(stretching) else 0.0
        h_total = _trapz(h_dos[freq > 0], freq[freq > 0])
        
        if h_total == 0:
            return
            
        print("  --> H-Mode Fractional Decomposition:")
        print(f"      Librational: {h_lib/h_total*100:.1f} %")
        print(f"      Bending:     {h_bend/h_total*100:.1f} %")
        print(f"      Stretching:  {h_stretch/h_total*100:.1f} %")
        
        # Peak stretching intensity
        if np.any(stretching):
            peak_idx = np.argmax(h_dos[stretching])
            stretch_freq_thz = freq[stretching][peak_idx]
            stretch_freq_cm = stretch_freq_thz * THZ_TO_CM
            print(f"  --> Principal B-H stretch isolated at: {stretch_freq_thz:.2f} THz ({stretch_freq_cm:.0f} cm^-1)")
            
            # Record it for the report
            self._h_metrics = {
                'lib': h_lib/h_total*100,
                'bend': h_bend/h_total*100,
                'stretch': h_stretch/h_total*100,
                'peak_thz': stretch_freq_thz,
                'peak_cm': stretch_freq_cm
            }

    # ========================================================================
    # PLOTTING — Publication-grade figures
    # ========================================================================
    def _plot_all(self):
        """Generate all publication-grade figures."""
        print("\n[Plotting] Generating publication-grade figures...")
        self._plot_band_structure()
        self._plot_partial_dos()
        self._plot_thermodynamics()
        if hasattr(self, '_h_metrics'):
            self._plot_h_modes()

    def _plot_band_structure(self):
        band_dict = self._band_dict
        distances  = band_dict['distances']
        frequencies = band_dict['frequencies']

        fig, ax = plt.subplots(figsize=(10, 6))
        for dist, freq in zip(distances, frequencies):
            for b in range(freq.shape[1]):
                ax.plot(dist, freq[:, b], color='#1E3A8A', lw=0.8, alpha=0.85)

        # Special point markers
        sp = [distances[0][0]]
        for d in distances:
            sp.append(d[-1])
        for xp in sp:
            ax.axvline(x=xp, color='#94A3B8', lw=0.5)
        ax.axhline(y=0, color='#EF4444', lw=1.0, ls='--', alpha=0.7)

        ax.set_ylabel('Frequency (THz)', fontsize=14)
        ax.set_title(f'{self.profile.formula} \u2014 Phonon Band Structure',
                     fontsize=15, fontweight='bold')
        ax.set_xlim(distances[0][0], distances[-1][-1])

        all_f = np.concatenate([f.flatten() for f in frequencies])
        mf = np.min(all_f)
        if mf < -0.5:
            ax.text(0.02, 0.02, f'\u26A0 Imaginary: {mf:.2f} THz',
                    transform=ax.transAxes, color='red', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#FEE2E2', alpha=0.9))
        else:
            ax.text(0.02, 0.02, '\u2713 Dynamically Stable',
                    transform=ax.transAxes, color='green', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#D1FAE5', alpha=0.9))

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'phonon_band_structure.png')
        plt.savefig(path)
        plt.close()
        print(f"  --> Saved: {path}")

    def _plot_partial_dos(self):
        freq, pdos_arr, sym_to_idx = self._pdos_data
        total_dos = pdos_arr.sum(axis=0)

        COLORS = {'Na':'#E74C3C','Ca':'#3498DB','B':'#2ECC71','H':'#F39C12',
                  'O':'#9B59B6','Ti':'#1ABC9C','Li':'#E67E22','N':'#3498DB'}

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [1, 2]})

        # Total
        ax1.fill_between(freq, total_dos, alpha=0.3, color='#1E3A8A')
        ax1.plot(freq, total_dos, color='#1E3A8A', lw=1.2, label='Total')
        ax1.set_ylabel('DOS (states/THz)', fontsize=12)
        ax1.set_title(f'{self.profile.formula} \u2014 Phonon Density of States',
                      fontsize=15, fontweight='bold')
        ax1.legend(fontsize=10)
        pos_mask = total_dos > 0.001
        if np.any(pos_mask):
            ax1.set_xlim(0, max(freq[pos_mask]) * 1.05)

        # Partial
        for species, idxs in sym_to_idx.items():
            sp_dos = np.zeros_like(freq)
            for idx in idxs:
                if idx < pdos_arr.shape[0]:
                    sp_dos += pdos_arr[idx]
            c = COLORS.get(species, '#888888')
            ax2.fill_between(freq, sp_dos, alpha=0.2, color=c)
            ax2.plot(freq, sp_dos, color=c, lw=1.5, label=species)

        ax2.set_xlabel('Frequency (THz)', fontsize=14)
        ax2.set_ylabel('Partial DOS', fontsize=12)
        ax2.legend(fontsize=11, ncol=4, loc='upper right')
        ax2.grid(alpha=0.15)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'phonon_dos_partial.png')
        plt.savefig(path)
        plt.close()
        print(f"  --> Saved: {path}")

    def _plot_thermodynamics(self):
        tp = self._thermo_data
        temps = tp['temperatures']
        fe, ent, cv = tp['free_energy'], tp['entropy'], tp['heat_capacity']
        n_atoms = len(self.phonon.unitcell.numbers)
        dp = 3 * n_atoms * 8.31446  # Dulong-Petit

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.plot(temps, fe, color='#1E3A8A', lw=2)
        ax1.set_xlabel('T (K)'); ax1.set_ylabel('F (kJ/mol)')
        ax1.set_title('F(T)', fontweight='bold'); ax1.grid(alpha=0.2)

        ax2.plot(temps, ent, color='#E74C3C', lw=2)
        ax2.set_xlabel('T (K)'); ax2.set_ylabel('S (J/mol\u00b7K)')
        ax2.set_title('S(T)', fontweight='bold'); ax2.grid(alpha=0.2)

        ax3.plot(temps, cv, color='#2ECC71', lw=2)
        ax3.axhline(y=dp, color='gray', ls='--', alpha=0.5,
                    label=f'Dulong-Petit = {dp:.0f}')
        ax3.set_xlabel('T (K)'); ax3.set_ylabel('Cv (J/mol\u00b7K)')
        ax3.set_title('Cv(T)', fontweight='bold')
        ax3.legend(fontsize=9); ax3.grid(alpha=0.2)

        fig.suptitle(f'{self.profile.formula} \u2014 Vibrational Thermodynamics',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = os.path.join(self.output_dir, 'phonon_thermodynamics.png')
        plt.savefig(path)
        plt.close()
        print(f"  --> Saved: {path}")

    def _plot_h_modes(self):
        freq, pdos_arr, sym_to_idx = self._pdos_data
        h_dos = np.zeros_like(freq)
        for idx in sym_to_idx.get('H', []):
            if idx < pdos_arr.shape[0]:
                h_dos += pdos_arr[idx]

        lib = (freq > 5) & (freq < 20.9)
        bend = (freq > 20.9) & (freq < 50)
        stretch = (freq > 50) & (freq < 100)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(freq[lib], h_dos[lib], alpha=0.3, color='#9B59B6', label='Librational')
        ax.fill_between(freq[bend], h_dos[bend], alpha=0.3, color='#3498DB', label='Bending')
        ax.fill_between(freq[stretch], h_dos[stretch], alpha=0.3, color='#E74C3C', label='B-H Stretching')
        ax.plot(freq, h_dos, color='#F39C12', lw=1.5, label='H total')

        hm = self._h_metrics
        ax.axvline(x=hm['peak_thz'], color='#E74C3C', ls='--', alpha=0.7)
        ax.text(hm['peak_thz'] + 1, max(h_dos) * 0.85,
                f"B-H peak\n{hm['peak_thz']:.1f} THz\n({hm['peak_cm']:.0f} cm\u207b\u00b9)",
                fontsize=9, color='#E74C3C', fontweight='bold')

        ax.set_xlabel('Frequency (THz)', fontsize=14)
        ax.set_ylabel('H Partial DOS', fontsize=12)
        ax.set_title(f'{self.profile.formula} \u2014 Hydrogen Mode Decomposition',
                     fontsize=15, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(alpha=0.15)
        pos_mask = h_dos > 0.001
        if np.any(pos_mask):
            ax.set_xlim(0, max(freq[pos_mask]) * 1.05)

        plt.tight_layout()
        path = os.path.join(self.output_dir, 'H_mode_analysis.png')
        plt.savefig(path)
        plt.close()
        print(f"  --> Saved: {path}")
        
    # ========================================================================
    # PHASE 6: HTML REPORTING
    # ========================================================================
    def _phase6_reporting(self):
        """Generate high-fidelity HTML report with dynamic data insertion."""
        print("\n[Phase 6] Report Generation")

        html_path = os.path.join(self.output_dir, 'Phonon_Analysis_Report.html')

        # Compute 300K index dynamically instead of hardcoding
        temps = self._thermo_data['temperatures']
        idx_300 = int(np.argmin(np.abs(temps - 300)))

        stability = "Dynamically Soft-Stable at 0K" if self.min_freq > -0.2 else f"Dynamically Unstable (Phase Breakdown)"
        stability_color = "var(--success)" if self.min_freq > -0.2 else "var(--danger)"
        
        # We will keep the html very simple without inline images in this specific backend update,
        # but rich in numbers.
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Phonon Analysis: {self.profile.formula}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8fafc; color: #1e293b; padding: 2rem; }}
        .header {{ background: #1e3a8a; color: white; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; }}
        .card {{ background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); margin-bottom: 1.5rem; }}
        h1, h2 {{ margin-top: 0; }}
        .stability {{ font-size: 1.25rem; font-weight: bold; color: {stability_color}; padding: 1rem; border-left: 4px solid {stability_color}; background: #f1f5f9; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Phonon Post-Processing Report: {self.profile.formula}</h1>
        <p>Space Group: {self.profile.space_group} | Atoms: {len(self.phonon.unitcell.numbers)} primitive / {len(self.phonon.supercell.numbers)} supercell</p>
    </div>
    
    <div class="card stability">
        {stability} (Min Frequency: {self.min_freq:.2f} THz)
    </div>
    
    <div class="card">
        <h2>Thermodynamics @ 300K</h2>
        <ul>
            <li><strong>Zero Point Energy:</strong> {self._thermo_data['free_energy'][0]:.2f} kJ/mol</li>
            <li><strong>Entropy:</strong> {self._thermo_data['entropy'][idx_300]:.2f} J/(mol·K)</li>
            <li><strong>Heat Capacity (Cv):</strong> {self._thermo_data['heat_capacity'][idx_300]:.2f} J/(mol·K)</li>
        </ul>
    </div>
"""
        if hasattr(self, '_h_metrics'):
            hm = self._h_metrics
            html_content += f"""
    <div class="card">
        <h2>Hydrogen Storage Analytics</h2>
        <ul>
            <li><strong>Principal Stretch Frequency:</strong> {hm['peak_thz']:.2f} THz ({hm['peak_cm']:.0f} cm⁻¹)</li>
            <li>Librational Fraction: {hm['lib']:.1f}%</li>
            <li>Bending Fraction: {hm['bend']:.1f}%</li>
            <li>Stretching Fraction: {hm['stretch']:.1f}%</li>
        </ul>
    </div>
"""
        html_content += "</body></html>"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"  --> HTML Synthesis completed: {html_path}")
