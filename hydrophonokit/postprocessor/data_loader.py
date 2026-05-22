"""
=============================================================================
  HydroPhonoKit Postprocessor — Data Loader

  Loads phonon calculation data from workspace directories:
    - phonopy_disp.yaml (displacement geometry)
    - vasprun.xml files (forces from VASP)
    - OUTCAR (Born effective charges, dielectric tensor)

  Scientific Foundation:
    Force extraction from DFT calculations requires careful parsing of
    VASP output files. The forces must be extracted from the final
    ionic step and validated against convergence criteria.

    Born effective charges (Z*) describe the change in polarization
    with respect to atomic displacement:
      Z*_{i,αβ} = Ω/ε × ∂P_α/∂u_{i,β}

    These are essential for Non-Analytical Corrections (NAC) which
    account for the macroscopic electric field created by long-wavelength
    LO modes, causing LO-TO splitting.

  References:
    [1] Gonze & Lee, Phys. Rev. B 55, 10355 (1997) -- DFPT Born charges
    [2] VASP Manual -- OUTCAR format specification
    [3] Phonopy Documentation -- phonopy_disp.yaml schema
=============================================================================
"""
import os
import warnings
import numpy as np
from typing import Dict, Optional, Tuple

from phonopy import load as phonopy_load
from phonopy.interface.vasp import Vasprun

from ..physics import BORN_FACTOR


class DataLoader:
    """Loads and validates phonon calculation data from workspace.

    This class handles:
      1. Loading phonopy_disp.yaml (geometry + displacements)
      2. Parsing vasprun.xml files for force sets
      3. Extracting Born charges and dielectric tensor from OUTCAR
      4. Applying Acoustic Sum Rule (ASR) to Born charges
    """

    def __init__(self, workspace_dir: str, profile):
        """
        Args:
            workspace_dir: Path to HydroPhonoKit workspace
            profile: MaterialProfile from analyzer
        """
        self.workspace_dir = workspace_dir
        self.profile = profile

        # Paths
        self.disp_yaml = os.path.join(workspace_dir, 'phonopy_disp.yaml')
        self.born_outcar = os.path.join(workspace_dir, '01_born', 'OUTCAR')
        self.disp_dir = os.path.join(workspace_dir, '02_displacements')

        # Loaded data
        self.phonon = None
        self.dielectric = None
        self.born_charges = None
        self.force_sets = None

    def load(self) -> Dict:
        """Execute full data loading pipeline.

        Returns:
            dict: {
                'phonon': Phonopy object with forces loaded,
                'born_charges': array (n_atoms, 3, 3) or None,
                'dielectric': array (3, 3) or None,
                'nac_applied': bool,
                'n_displacements': int,
                'n_atoms_uc': int,
                'n_atoms_sc': int,
            }
        """
        print("\n[Phase 1] Data Collection & Precision Validations")

        # 1.1 Load phonopy object
        self._load_phonopy()

        # 1.2 Read Forces
        self._load_forces()

        # 1.3 Read Born Charges (if insulator)
        nac_applied = False
        if self.profile.rec_born:
            nac_applied = self._load_born_charges()

        return {
            'phonon': self.phonon,
            'born_charges': self.born_charges,
            'dielectric': self.dielectric,
            'nac_applied': nac_applied,
            'n_displacements': len(self.phonon.supercells_with_displacements),
            'n_atoms_uc': len(self.phonon.unitcell.numbers),
            'n_atoms_sc': len(self.phonon.supercell.numbers),
        }

    def _load_phonopy(self):
        """Load phonopy_disp.yaml and validate geometry."""
        if not os.path.exists(self.disp_yaml):
            raise FileNotFoundError(
                f"Cannot find {self.disp_yaml}. "
                f"Did the workspace generator run?"
            )

        self.phonon = phonopy_load(self.disp_yaml)
        n_atoms_uc = len(self.phonon.unitcell.numbers)
        n_atoms_sc = len(self.phonon.supercell.numbers)

        print(f"  --> Loaded geometry: {n_atoms_uc} atoms in primitive cell.")
        print(f"  --> Supercell size: {n_atoms_sc} atoms.")

    def _load_forces(self):
        """Parse vasprun.xml files for force sets.

        Forces are extracted from the final ionic step of each
        displacement calculation. VASP outputs forces in the
        'varray' section of vasprun.xml under 'forces'.
        """
        print("  --> Parsing forces from VASP vasprun.xml files ...")

        n_disp = len(self.phonon.supercells_with_displacements)
        force_sets = []

        for i in range(n_disp):
            disp_id = f"{i+1:03d}"
            vpath = os.path.join(self.disp_dir, f'disp-{disp_id}', 'vasprun.xml')

            if not os.path.exists(vpath):
                raise FileNotFoundError(
                    f"Missing force data: {vpath}\n"
                    f"  Expected {n_disp} displacement calculations."
                )

            try:
                vr = Vasprun(vpath)
                forces = vr.read_forces()
                force_sets.append(forces)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse forces from {vpath}: {e}\n"
                    f"  Check that VASP completed successfully."
                ) from e

        self.phonon.forces = np.array(force_sets)
        self.force_sets = self.phonon.forces
        print(f"  --> Successfully collected {n_disp} force sets.")

    def _load_born_charges(self) -> bool:
        """Extract and symmetrize Born effective charges.

        Born charges are extracted from the DFPT calculation in
        01_born/OUTCAR. The Acoustic Sum Rule (ASR) is applied
        to ensure translational invariance:
            Σ_i Z*_{i,αβ} = 0

        Returns:
            bool: True if NAC was successfully applied
        """
        if not os.path.exists(self.born_outcar):
            print("  [!] WARNING: Profile expects Born charges, but NO 01_born/OUTCAR found.")
            print("      Proceeding WITHOUT non-analytical corrections (LO-TO splitting disabled).")
            self.profile.rec_born = False
            return False

        print("  --> Parsing OUTCAR for Dielectric Tensor and Born Effective Charges ...")

        with open(self.born_outcar, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Extract Dielectric Tensor
        diel = self._extract_dielectric_tensor(content)
        if diel is None:
            print("  [!] WARNING: Failed to extract dielectric tensor. Disabling NAC.")
            self.profile.rec_born = False
            return False

        # Symmetrize Dielectric Tensor (ε_αβ = ε_βα)
        self.dielectric = 0.5 * (diel + diel.T)
        print(f"      Dielectric Trace (Tr/3): {np.trace(self.dielectric)/3.0:.4f}")

        # Extract Born Charges
        bc_list = self._extract_born_charges(content)

        if bc_list is None or len(bc_list) != len(self.phonon.unitcell.numbers):
            print("  [!] WARNING: Failed to extract proper Born charges. Disabling NAC.")
            self.profile.rec_born = False
            return False

        Z = np.array(bc_list)

        # Application of Acoustic Sum Rule (ASR) for Born Charges
        # To ensure translation invariance, the sum of Born charges
        # over all atoms must strictly be zero.
        z_sum = np.sum(Z, axis=0)
        initial_trace_err = np.trace(z_sum) / 3.0
        print(f"      Born Charge Sum Rule Error (before correction): {initial_trace_err:.5f}")

        # Distribute the error equally among all atoms
        correction = z_sum / len(Z)
        Z_corrected = Z - correction

        self.born_charges = Z_corrected
        new_err = np.trace(np.sum(self.born_charges, axis=0)) / 3.0
        print(f"      Born Charge Sum Rule Error (after correction):  {new_err:.5f}")

        # Register with phonopy
        nac_params = {
            'born': self.born_charges,
            'dielectric': self.dielectric,
            'factor': BORN_FACTOR,  # VASP internal constant: e²/(4πε₀) in eV·Å
        }
        self.phonon.nac_params = nac_params
        print("  --> Non-Analytical Corrections (NAC) initialized successfully.")

        return True

    def _extract_dielectric_tensor(self, content: str) -> Optional[np.ndarray]:
        """Extract macroscopic static dielectric tensor from OUTCAR.

        Searches for:
          "MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects)"
        or:
          "MACROSCOPIC STATIC DIELECTRIC TENSOR"

        Returns:
            3x3 array or None if not found
        """
        diel = np.zeros((3, 3))

        # Try with local field effects first
        idx = content.find('MACROSCOPIC STATIC DIELECTRIC TENSOR (including')
        if idx == -1:
            idx = content.find('MACROSCOPIC STATIC DIELECTRIC TENSOR')

        if idx == -1:
            print("  [!] Dielectric tensor not found in OUTCAR.")
            return None

        # Parse the 3x3 matrix
        lines = content[idx:idx+500].split('\n')
        r = 0
        for line in lines[2:]:  # Skip header lines
            parts = line.split()
            if len(parts) >= 3:
                try:
                    diel[r] = [float(parts[0]), float(parts[1]), float(parts[2])]
                    r += 1
                    if r >= 3:
                        break
                except ValueError:
                    break

        if r < 3:
            print("  [!] Failed to parse complete dielectric tensor.")
            return None

        return diel

    def _extract_born_charges(self, content: str) -> Optional[list]:
        """Extract Born effective charge tensors from OUTCAR.

        Searches for "BORN EFFECTIVE CHARGES" section and parses
        the 3x3 tensor for each ion.

        Returns:
            List of 3x3 arrays, one per atom, or None if failed
        """
        bc_list = []
        idx = content.find('BORN EFFECTIVE CHARGES')

        if idx == -1:
            print("  [!] Born effective charges section not found in OUTCAR.")
            return None

        lines = content[idx:].split('\n')
        i = 0
        n_atoms = len(self.phonon.unitcell.numbers)

        while i < len(lines):
            # Look for ion header line (e.g., " ion    1 ")
            if 'ion' in lines[i] and len(lines[i].split()) <= 3:
                z_tensor = []
                for j in range(1, 4):
                    if i + j < len(lines):
                        parts = lines[i + j].split()
                        if len(parts) >= 3:
                            try:
                                z_tensor.append([
                                    float(parts[1]),
                                    float(parts[2]),
                                    float(parts[3])
                                ])
                            except ValueError:
                                pass

                if len(z_tensor) == 3:
                    bc_list.append(z_tensor)

                i += 4  # Skip ion header + 3 tensor rows
            else:
                i += 1

            if len(bc_list) >= n_atoms:
                break

        if len(bc_list) < n_atoms:
            print(f"  [!] Extracted {len(bc_list)} Born tensors, expected {n_atoms}.")
            return None

        return bc_list
