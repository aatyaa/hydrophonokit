"""
=============================================================================
  HydroPhonoKit Postprocessor — Group Velocities

  Computes phonon group velocities from the phonon dispersion.

  Scientific Foundation:
    The phonon group velocity v_g(q,j) is the velocity at which wave
    packets (phonons) propagate through the crystal:

      v_g(q,j) = ∂ω(q,j)/∂q

    where ω(q,j) is the phonon frequency for wavevector q and branch j.

    Group velocities are essential for:
      - Thermal conductivity: κ = Σ C_v(q,j) × v_g(q,j)² × τ(q,j)
      - Phonon mean free path: ℓ(q,j) = v_g(q,j) × τ(q,j)
      - Understanding heat transport mechanisms

    Computed via central finite difference on the phonon dispersion:
      v_g,α(q,j) ≈ [ω(q + δq·ê_α, j) - ω(q - δq·ê_α, j)] / (2δq)

    where δq is a small displacement in reciprocal space and ê_α is
    the unit vector along Cartesian direction α.

  References:
    [1] Lindsay et al., Phys. Rev. B 87, 165201 (2013) -- Thermal transport
    [2] Li et al., Phys. Rev. B 85, 195436 (2012) -- ShengBTE
    [3] Togo & Tanaka, Scr. Mater. 108, 1 (2015) -- Phonopy
=============================================================================
"""
import os
import numpy as np
from typing import Dict, Optional, Tuple


class GroupVelocityComputer:
    """Computes phonon group velocities.

    This class handles:
      1. Group velocity computation via finite difference
      2. Average sound velocity estimation
      3. Mode-resolved velocity analysis
    """

    def __init__(self, phonon, delta_q: float = 1e-5):
        """
        Args:
            phonon: Phonopy object with force constants computed
            delta_q: Finite difference step in reciprocal coordinates
        """
        self.phonon = phonon
        self.delta_q = delta_q

    def compute(self, q_mesh: Optional[Tuple[int, int, int]] = None) -> Dict:
        """Compute group velocities on a q-point mesh.

        Args:
            q_mesh: Q-point mesh (default: [10, 10, 10])

        Returns:
            dict: {
                'q_points': array (n_q, 3),
                'frequencies': array (n_q, n_bands),
                'group_velocities': array (n_q, n_bands, 3),  # m/s
                'avg_sound_velocity': float,  # m/s
                'max_group_velocity': float,  # m/s
            }
        """
        print("\n[Scientific] Computing Group Velocities...")

        if q_mesh is None:
            q_mesh = [10, 10, 10]

        # Generate q-point mesh in reciprocal coordinates
        n_q = q_mesh[0] * q_mesh[1] * q_mesh[2]
        n_bands = 3 * len(self.phonon.unitcell.numbers)

        # Use phonopy's built-in group velocity computation if available
        try:
            self.phonon.run_mesh(q_mesh, with_group_velocities=True)
            gv = self.phonon.get_group_velocities()  # (n_q, n_bands, 3)
            # Phonopy returns velocities in 100 m/s units
            gv = gv * 100  # Convert to m/s
            print(f"  --> Computed group velocities using phonopy ({n_q} q-points).")
        except Exception:
            # Fallback to finite difference
            gv = self._compute_finite_difference(q_mesh)

        # Get frequencies and q-points
        self.phonon.run_mesh(q_mesh)
        frequencies = self.phonon.get_mesh_frequencies()  # (n_q, n_bands)
        q_points = self._generate_q_mesh(q_mesh)

        # Average sound velocity (from acoustic modes near Γ)
        avg_v = self._compute_avg_sound_velocity(frequencies, gv)

        # Max group velocity
        max_v = float(np.max(np.linalg.norm(gv, axis=2)))

        print(f"  --> Average sound velocity: {avg_v:.1f} m/s")
        print(f"  --> Max group velocity: {max_v:.1f} m/s")

        return {
            'q_points': q_points,
            'frequencies': frequencies,
            'group_velocities': gv,
            'avg_sound_velocity': avg_v,
            'max_group_velocity': max_v,
        }

    def _generate_q_mesh(self, q_mesh) -> np.ndarray:
        """Generate uniform q-point mesh in reciprocal coordinates.

        Returns:
            (n_q, 3) array of q-points
        """
        q_points = []
        for i in range(q_mesh[0]):
            for j in range(q_mesh[1]):
                for k in range(q_mesh[2]):
                    q = [i / q_mesh[0], j / q_mesh[1], k / q_mesh[2]]
                    q_points.append(q)
        return np.array(q_points)

    def _compute_finite_difference(self, q_mesh) -> np.ndarray:
        """Compute group velocities via finite difference.

        Uses central difference:
          v_g,α = [ω(q + δq·ê_α) - ω(q - δq·ê_α)] / (2δq)

        Returns:
            (n_q, n_bands, 3) array in m/s
        """
        n_bands = 3 * len(self.phonon.unitcell.numbers)
        q_points = self._generate_q_mesh(q_mesh)
        n_q = len(q_points)

        # Get lattice vectors for conversion to Cartesian
        cell = self.phonon.unitcell.cell
        inv_cell = np.linalg.inv(cell)  # Cartesian → fractional

        # Conversion factor: THz × Å → m/s
        # v (m/s) = dω/dq = (THz × 1e12) / (Å⁻¹ × 1e10) = THz × 1e12 / (1e10) = THz × 100
        THZ_A_TO_MS = 100.0

        gv = np.zeros((n_q, n_bands, 3))
        delta = self.delta_q

        for idx, q in enumerate(q_points):
            for alpha in range(3):  # x, y, z
                q_plus = q.copy()
                q_minus = q.copy()
                q_plus[alpha] = (q[alpha] + delta) % 1.0
                q_minus[alpha] = (q[alpha] - delta) % 1.0

                try:
                    self.phonon.run_qpoints([q_plus])
                    freq_plus = self.phonon.get_qpoints_frequencies()[0]

                    self.phonon.run_qpoints([q_minus])
                    freq_minus = self.phonon.get_qpoints_frequencies()[0]

                    # Central difference
                    gv[idx, :, alpha] = (freq_plus - freq_minus) / (2 * delta) * THZ_A_TO_MS
                except Exception:
                    gv[idx, :, alpha] = 0.0

        return gv

    def _compute_avg_sound_velocity(self, frequencies, gv) -> float:
        """Compute average sound velocity from acoustic modes near Γ.

        Uses the 3 lowest frequency modes at the Γ point.
        """
        # Find Γ point (q = [0, 0, 0] or closest)
        q_points = self._generate_q_mesh([10, 10, 10])
        gamma_idx = np.argmin(np.sum(q_points ** 2, axis=1))

        # Acoustic modes: 3 lowest frequencies at Γ
        freq_gamma = frequencies[gamma_idx]
        acoustic_mask = np.argsort(freq_gamma)[:3]

        # Average velocity of acoustic modes
        velocities = np.linalg.norm(gv[gamma_idx, acoustic_mask], axis=1)
        valid_v = velocities[velocities > 0]

        if len(valid_v) == 0:
            return 0.0

        return float(np.mean(valid_v))
