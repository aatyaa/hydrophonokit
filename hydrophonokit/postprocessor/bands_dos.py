"""
=============================================================================
  HydroPhonoKit Postprocessor — Band Structure & DOS Computer

  Computes phonon band structure and partial density of states.

  Scientific Foundation:
    The phonon band structure ω(q) along high-symmetry q-paths reveals
    dynamical stability and phonon dispersion. The partial DOS g_i(ω)
    projects the total DOS onto individual atoms, enabling element-specific
    analysis of vibrational properties.

    Band structure is computed along high-symmetry paths determined by
    seekpath (based on the Brillouin zone of the crystal structure).

    Partial DOS:
      g_i(ω) = Σ_qj |e_i(qj)|² δ(ω - ω(qj))
    where e_i(qj) is the eigenvector component for atom i in mode (q,j).

  References:
    [1] Hinuma et al., Comput. Mater. Sci. 140, 2017 -- seekpath
    [2] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy bands/DOS
=============================================================================
"""
import os
import warnings
import numpy as np
from typing import Dict, Optional, Tuple


class BandsDOSComputer:
    """Computes band structure and partial density of states.

    This class handles:
      1. High-symmetry q-path generation (seekpath)
      2. Band structure computation
      3. Partial DOS with eigenvectors
      4. Dynamical stability validation
    """

    def __init__(self, profile):
        """
        Args:
            profile: MaterialProfile from analyzer
        """
        self.profile = profile

    def compute(self, phonon) -> Dict:
        """Compute band structure and partial DOS.

        Args:
            phonon: Phonopy object with force constants computed

        Returns:
            dict: {
                'band_dict': dict (phonopy band structure data),
                'pdos_data': tuple (freq, pdos, sym_to_idx),
                'min_freq': float (THz),
                'stability': dict (stability analysis),
                'k_path': str (high-symmetry path),
            }
        """
        print("\n[Phase 3] Phonon Band Structure & Partial Density of States")

        # 3.1: Band Structure
        band_dict, k_path = self._compute_band_structure(phonon)

        # Extract frequencies and validate stability
        bands_freq = np.concatenate([f.flatten() for f in band_dict['frequencies']])
        min_freq = np.min(bands_freq)

        # Stability check
        from ..physics import check_dynamical_stability
        stability = check_dynamical_stability(bands_freq)

        self._report_stability(stability)

        # Save band.yaml
        band_yaml_path = os.path.join(
            getattr(phonon, '_output_dir', '.'),
            'band.yaml'
        )
        phonon.write_yaml_band_structure(filename=band_yaml_path)

        # 3.2: Partial DOS
        pdos_data = self._compute_partial_dos(phonon)

        return {
            'band_dict': band_dict,
            'pdos_data': pdos_data,
            'min_freq': min_freq,
            'stability': stability,
            'k_path': k_path,
        }

    def _compute_band_structure(self, phonon):
        """Compute phonon band structure along high-symmetry path.

        Uses seekpath for automatic path generation, falls back to
        phonopy's automatic band structure if seekpath unavailable.

        Returns:
            (band_dict, k_path_string)
        """
        k_path = "auto-generated"

        try:
            import seekpath
            cell = (
                phonon.unitcell.cell,
                phonon.unitcell.scaled_positions,
                phonon.unitcell.numbers
            )
            kpath = seekpath.get_path(cell)

            labels = []
            band_paths = []
            path_data = kpath['path']
            special_points = kpath['point_coords']

            for seg in path_data:
                start_label, end_label = seg
                band_paths.append([
                    special_points[start_label],
                    special_points[end_label]
                ])
                labels.append([
                    start_label.replace('GAMMA', 'Γ'),
                    end_label.replace('GAMMA', 'Γ')
                ])

            flat_labels = [labels[0][0]] + [l[1] for l in labels]
            k_path = ' → '.join(flat_labels)
            print(f"  --> Identified High-Symmetry path: {k_path}")

            phonon.run_band_structure(
                band_paths,
                labels=[l for pair in labels for l in pair]
            )

        except ImportError:
            print("  [i] seekpath not installed. Using automatic band structure...")
            try:
                phonon.auto_band_structure(npoints=101, with_eigenvectors=True)
            except Exception:
                # Fallback: generate simple band structure without seekpath
                phonon.auto_band_structure(npoints=101)
        except Exception as e:
            print(f"  [i] seekpath failed ({e}). Using automatic band structure...")
            try:
                phonon.auto_band_structure(npoints=101, with_eigenvectors=True)
            except Exception:
                phonon.auto_band_structure(npoints=101)

        return phonon.get_band_structure_dict(), k_path

    def _compute_partial_dos(self, phonon):
        """Compute partial density of states with eigenvectors.

        Uses Gaussian smearing for DOS projection. The tetrahedron
        method is disabled as it fails for unstable/non-Γ-centered cells.

        Returns:
            (freq_array, pdos_array, sym_to_idx_dict)
        """
        mesh = self.profile.rec_dos_mesh
        print(f"  --> Computing pDOS on {mesh[0]}x{mesh[1]}x{mesh[2]} Q-point mesh...")

        # Run DOS mesh with eigenvectors
        phonon.run_mesh(mesh, is_mesh_symmetry=False, with_eigenvectors=True)
        phonon.run_projected_dos(
            freq_pitch=0.1,
            use_tetrahedron_method=False,
            sigma=0.2
        )

        pdos_dict = phonon.projected_dos

        # Map atom indices to element symbols
        from ..analyzer import Z_TO_SYMBOL
        sym_to_idx = {}
        for i, z in enumerate(phonon.unitcell.numbers):
            sym = Z_TO_SYMBOL.get(z, f"Z{z}")
            if sym not in sym_to_idx:
                sym_to_idx[sym] = []
            sym_to_idx[sym].append(i)

        print("  --> Partial DOS calculated successfully.")

        return (
            pdos_dict.frequency_points,
            pdos_dict.projected_dos,
            sym_to_idx
        )

    def _report_stability(self, stability: Dict):
        """Report dynamical stability analysis results."""
        if stability['n_imaginary'] > 0:
            print(f"  [!] IMAGINARY FREQUENCIES DETECTED:")
            print(f"      Minimum frequency: {stability['min_freq']:.2f} THz")
            print(f"      Number of imaginary modes: {stability['n_imaginary']}")
            print(f"      Phase is dynamically UNSTABLE -- structural phase transition likely")
        else:
            print(f"  [STABLE] Phase dynamically stable at 0K (min freq: {stability['min_freq']:.2f} THz)")
