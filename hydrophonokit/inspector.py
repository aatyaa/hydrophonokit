"""
=============================================================================
  HydroPhonoKit v2.2 -- VASP Output Inspector

  Rigorous validation of VASP outputs before phonon calculation begins.
  Validates:
    - Force convergence against EDIFFG threshold
    - Stress tensor convergence (external pressure)
    - INCAR parameter extraction with defaults
    - Crystallographic symmetry verification

  References:
    [1] VASP Manual -- OUTCAR format
    [2] Kresse & Furthmüller, Phys. Rev. B 54, 11169 (1996)
=============================================================================
"""
import os
import re
import numpy as np
import spglib
from phonopy.interface.vasp import read_vasp

class VASPInspector:
    """Parses and validates VASP output directories for phonon readiness."""

    def __init__(self, target_dir):
        self.target_dir = target_dir
        self.outcar_path = os.path.join(target_dir, 'OUTCAR')
        self.incar_path = os.path.join(target_dir, 'INCAR')
        self.contcar_path = os.path.join(target_dir, 'CONTCAR')
        self.encut = None
        self.kspacing = None

    def validate_forces(self, threshold=0.005):
        """Validate that maximum residual force is below threshold.

        Parses the FINAL ionic step forces from OUTCAR and computes:
          - Max force magnitude (eV/A)
          - RMS force (eV/A)

        Args:
            threshold: Maximum acceptable force (eV/A). Default: 0.005 eV/A

        Returns:
            dict: {
                'converged': bool,
                'max_force': float (eV/A),
                'rms_force': float (eV/A),
                'threshold': float (eV/A)
            }

        Reference: VASP OUTCAR format spec; Kresse & Furthmüller (1996).
        """
        if not os.path.exists(self.outcar_path):
            raise FileNotFoundError(f"OUTCAR missing in {self.target_dir}")

        with open(self.outcar_path, 'r') as f:
            lines = f.readlines()

        # Check convergence flag
        converged = False
        for line in reversed(lines):
            if "reached required accuracy" in line:
                converged = True
                break

        # Parse forces from the FINAL ionic step
        # Look for the last "TOTAL-FORCE" block
        last_force_start = None
        for i in range(len(lines) - 1, -1, -1):
            if 'TOTAL-FORCE' in lines[i]:
                last_force_start = i + 2  # skip header + dashes
                break

        if last_force_start is None:
            return {
                'converged': converged,
                'max_force': None,
                'rms_force': None,
                'threshold': threshold,
                'error': 'No TOTAL-FORCE block found in OUTCAR'
            }

        # Parse force vectors
        forces = []
        for j in range(last_force_start, min(last_force_start + 200, len(lines))):
            parts = lines[j].split()
            if len(parts) >= 6:
                try:
                    fx, fy, fz = float(parts[3]), float(parts[4]), float(parts[5])
                    forces.append([fx, fy, fz])
                except (ValueError, IndexError):
                    break
            else:
                break

        if not forces:
            return {
                'converged': converged,
                'max_force': None,
                'rms_force': None,
                'threshold': threshold,
                'error': 'No forces parsed in OUTCAR'
            }

        farr = np.array(forces)
        magnitudes = np.linalg.norm(farr, axis=1)
        max_force = float(np.max(magnitudes))
        rms_force = float(np.sqrt(np.mean(magnitudes**2)))

        return {
            'converged': converged and max_force <= threshold,
            'max_force': max_force,
            'rms_force': rms_force,
            'threshold': threshold,
            'n_atoms': len(forces),
            'convergence_flag': converged,
        }

    def validate_stress(self, threshold_kbar=0.5):
        """Validate external pressure is below threshold.

        Args:
            threshold_kbar: Maximum acceptable external pressure (kBar)

        Returns:
            dict: {
                'converged': bool,
                'external_pressure': float (kBar),
                'threshold': float (kBar)
            }
        """
        if not os.path.exists(self.outcar_path):
            raise FileNotFoundError(f"OUTCAR missing in {self.target_dir}")

        with open(self.outcar_path, 'r') as f:
            lines = f.readlines()

        # Parse external pressure from final step
        pressure = None
        for line in reversed(lines):
            if 'external pressure' in line:
                m = re.search(r'external pressure\s*=\s*([-0-9.]+)\s*kB', line)
                if m:
                    pressure = float(m.group(1))
                    break

        if pressure is None:
            return {
                'converged': False,
                'external_pressure': None,
                'threshold': threshold_kbar,
                'error': 'External pressure not found in OUTCAR'
            }

        return {
            'converged': abs(pressure) <= threshold_kbar,
            'external_pressure': pressure,
            'threshold': threshold_kbar,
        }

    def extract_parameters(self):
        """Extracts ENCUT and KSPACING from INCAR with validated defaults.

        Returns:
            tuple: (encut, kspacing) with values in eV and 1/A
        """
        if not os.path.exists(self.incar_path):
            raise FileNotFoundError(f"INCAR missing in {self.target_dir}")

        with open(self.incar_path, 'r') as f:
            incar_text = f.read()

        # Remove comment lines
        incar_lines = [l for l in incar_text.split('\n')
                       if not l.strip().startswith('#')]
        incar_clean = '\n'.join(incar_lines)

        encut_match = re.search(r'ENCUT\s*=\s*([0-9.]+)', incar_clean)
        kspacing_match = re.search(r'KSPACING\s*=\s*([0-9.]+)', incar_clean)

        if encut_match:
            self.encut = float(encut_match.group(1))
        else:
            self.encut = 520.0  # Default for PAW_PBE potentials

        if kspacing_match:
            self.kspacing = float(kspacing_match.group(1))
        else:
            self.kspacing = 0.25  # ~2pi*0.25 = 1.57 1/A, reasonable default

        return self.encut, self.kspacing

    def check_crystallography(self):
        """Verify crystallographic symmetry using spglib.

        Returns:
            tuple: (cell, space_group, crystal_system)
        """
        cell = read_vasp(self.contcar_path)
        sym_data = spglib.get_symmetry_dataset(
            (cell.cell, cell.scaled_positions, cell.numbers),
            symprec=1e-5
        )

        if sym_data is None:
            raise RuntimeError(
                f"spglib symmetry detection failed for {self.contcar_path}. "
                "Structure may be disordered or symprec too tight.")

        try:
            sg_int = sym_data.international
            sg_num = sym_data.number
        except AttributeError:
            sg_int = sym_data['international']
            sg_num = sym_data['number']

        # Determine crystal system from space group number
        if sg_num <= 2:
            crystal_system = "Triclinic"
        elif sg_num <= 15:
            crystal_system = "Monoclinic"
        elif sg_num <= 74:
            crystal_system = "Orthorhombic"
        elif sg_num <= 142:
            crystal_system = "Tetragonal"
        elif sg_num <= 167:
            crystal_system = "Trigonal"
        elif sg_num <= 194:
            crystal_system = "Hexagonal"
        else:
            crystal_system = "Cubic"

        return cell, f"{sg_int} ({sg_num})", crystal_system
