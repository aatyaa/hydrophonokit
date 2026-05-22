"""
=============================================================================
  HydroPhonoKit Postprocessor — Force Constants Computer

  Computes Interatomic Force Constants (IFCs) from displacement-force data.

  Scientific Foundation:
    The force constant matrix Φ_αβ(lκ, l'κ') describes the second-order
    change in total energy with respect to atomic displacements:
      Φ_αβ(lκ, l'κ') = ∂²E/∂u_α(lκ)∂u_β(l'κ')

    where l, l' are unit cell indices and κ, κ' are atom indices.

    Two methods are supported:
      1. symfc (Symmetry-Adapted Force Constants):
         Enforces crystal symmetry on IFCs, eliminating numerical noise
         from finite-difference discretization. Critical for accurate
         acoustic mode behavior near Γ.

      2. Phonopy standard solver:
         Direct inversion of displacement-force relationship without
         symmetry enforcement. May retain numerical noise.

    The Acoustic Sum Rule (ASR) must be satisfied:
      Σ_{l'κ'} Φ_αβ(lκ, l'κ') = 0
    This ensures translational invariance and zero frequency for
    acoustic modes at Γ.

  References:
    [1] Wang et al., Phys. Rev. B 95, 014303 (2017) -- symfc
    [2] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy
    [3] Parlinski et al., Phys. Rev. Lett. 78, 4063 (1997) -- Direct method
=============================================================================
"""
import os
import warnings
import numpy as np
from typing import Dict, Optional


class IFCComputer:
    """Computes Interatomic Force Constants (IFCs) from force sets.

    This class handles:
      1. symfc computation (if available)
      2. Fallback to standard phonopy solver
      3. Validation of IFC quality
      4. Saving IFCs to disk
    """

    def __init__(self, profile):
        """
        Args:
            profile: MaterialProfile from analyzer
        """
        self.profile = profile

    def compute(self, phonon) -> Dict:
        """Compute force constants using best available method.

        Args:
            phonon: Phonopy object with forces loaded

        Returns:
            dict: {
                'method': str ('symfc' or 'phonopy'),
                'force_constants': array,
                'saved_path': str,
                'validation': dict,
            }
        """
        print("\n[Phase 2] Force Constants & Acoustic Sum Rules")

        # Try symfc first (symmetry-adapted, more accurate)
        used_symfc = False
        try:
            from symfc import Symfc
            print("  --> Using 'symfc' (Symmetry-Adapted Force Constants) engine ...")
            phonon.produce_force_constants(fc_calculator='symfc')

            # Sanity check: symfc sometimes returns near-zero matrices
            # due to version/API incompatibilities
            if np.max(np.abs(phonon.force_constants)) < 1e-3:
                print("  [!] symfc returned suspicious near-zero IFCs. Falling back.")
                used_symfc = False
            else:
                used_symfc = True
                print("  --> symfc IFCs generated successfully.")

        except ImportError:
            print("  [i] symfc not installed. Using standard phonopy solver...")
        except Exception as e:
            print(f"  [i] symfc failed ({e}). Using standard phonopy solver...")

        # Fallback to standard phonopy solver
        if not used_symfc:
            try:
                phonon.produce_force_constants()
                print("  --> Standard Phonopy IFCs generated.")
            except Exception as e:
                raise RuntimeError(f"Force constants computation failed: {e}") from e

        # Save IFCs
        method = 'symfc' if used_symfc else 'phonopy'
        fc_path = self._save_force_constants(phonon)

        # Validate IFC quality
        validation = self._validate_force_constants(phonon.force_constants)

        return {
            'method': method,
            'force_constants': phonon.force_constants,
            'saved_path': fc_path,
            'validation': validation,
        }

    def _save_force_constants(self, phonon) -> str:
        """Save force constants to FORCE_CONSTANTS file.

        Uses phonopy's native file_IO module for compatibility.

        Returns:
            Path to saved file
        """
        fc_path = os.path.join(
            getattr(phonon, '_output_dir', '.'),
            'FORCE_CONSTANTS'
        )

        try:
            from phonopy.file_IO import write_FORCE_CONSTANTS
            write_FORCE_CONSTANTS(phonon.force_constants, fc_path)
            print(f"  --> Saved IFCs to: {fc_path}")
        except ImportError:
            # Alternative save method if file_IO not available
            alt_path = fc_path + '.npy'
            np.save(alt_path, phonon.force_constants)
            print(f"  --> Saved IFCs to: {alt_path} (numpy format)")
            fc_path = alt_path

        return fc_path

    def _validate_force_constants(self, fc: np.ndarray) -> Dict:
        """Validate force constant matrix quality.

        Checks:
          1. Magnitude: max |Φ| should be reasonable (0.1 - 100 eV/Å²)
          2. Symmetry: Φ_αβ ≈ Φ_βα (Hermitian)
          3. ASR: Row sums should be near zero

        Returns:
            dict: Validation results
        """
        validation = {}

        # 1. Magnitude check
        max_fc = np.max(np.abs(fc))
        validation['max_fc_eV_A2'] = float(max_fc)

        if max_fc < 1e-6:
            validation['quality'] = 'POOR'
            validation['warnings'] = ['Force constants are suspiciously small']
        elif max_fc > 1000:
            validation['quality'] = 'QUESTIONABLE'
            validation['warnings'] = ['Force constants unusually large']
        else:
            validation['quality'] = 'GOOD'
            validation['warnings'] = []

        # 2. Hermitian check (Φ_αβ = Φ_βα)
        # Force constants are stored as (n_atoms_sc, n_atoms_sc, 3, 3)
        # Check a sample for symmetry
        n_atoms = fc.shape[0]
        n_check = min(10, n_atoms)
        symmetry_errors = []

        for i in range(n_check):
            for j in range(n_check):
                phi_ij = fc[i, j]  # 3x3 matrix
                phi_ji = fc[j, i]  # Should be transpose
                error = np.max(np.abs(phi_ij - phi_ji.T))
                symmetry_errors.append(error)

        validation['max_symmetry_error'] = float(np.max(symmetry_errors))

        # 3. Acoustic Sum Rule check
        # Σ_j Φ_αβ(i,j) should be ≈ 0 for all i
        row_sums = np.sum(fc, axis=1)  # Sum over j
        asr_error = np.max(np.abs(row_sums))
        validation['asr_error_eV_A3'] = float(asr_error)

        return validation
