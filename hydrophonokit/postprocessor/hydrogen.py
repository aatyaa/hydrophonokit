"""
=============================================================================
  HydroPhonoKit Postprocessor — Hydrogen Storage Analysis

  Analyzes hydrogen vibrational modes in hydride materials.

  Scientific Foundation:
    Hydrogen in solids exhibits characteristic vibrational modes:
      - Librational (5-21 THz, ~167-700 cm⁻¹): hindered rotation of H₂/BH₄
      - Bending (21-50 THz, ~700-1668 cm⁻¹): H-X-H bending modes
      - Stretching (50-100 THz, ~1668-3336 cm⁻¹): X-H stretching

    These modes are diagnostic of the hydrogen bonding environment:
      - Ionic hydrides (MgH₂, CaH₂): M-H stretch ~1100-1400 cm⁻¹
      - Complex hydrides (NaAlH₄): Al-H stretch ~1700-1850 cm⁻¹
      - Borohydrides (LiBH₄): B-H stretch ~2200-2600 cm⁻¹
      - Amides (LiNH₂): N-H stretch ~3100-3500 cm⁻¹

    The mode decomposition provides insight into hydrogen dynamics and
    is essential for understanding hydrogen storage behavior.

  References:
    [1] Nakamoto, IR/Raman Spectra of Inorganic Compounds (2009)
    [2] Bogdanovic et al., J. Alloys Compd. 382, 1 (2004) -- MgH₂
    [3] Züttel et al., Nature Mater. 4, 673 (2005) -- LiBH₄
=============================================================================
"""
import os
import numpy as np
from typing import Dict, Optional

from ..physics import THZ_TO_CM, H_MODE_RANGES


class HydrogenAnalyzer:
    """Analyzes hydrogen vibrational modes.

    This class handles:
      1. H partial DOS extraction from pDOS data
      2. Mode decomposition (librational/bending/stretching)
      3. Peak frequency identification
      4. Hydride type classification
    """

    # Hydride type classification by stretch frequency (cm⁻¹)
    HYDRIDE_TYPES = {
        'ionic': (1100, 1400),
        'complex_alh': (1700, 1850),
        'borohydride': (2200, 2600),
        'amide': (3100, 3500),
        'hydrocarbon': (2800, 3100),
    }

    def __init__(self, profile):
        """
        Args:
            profile: MaterialProfile from analyzer
        """
        self.profile = profile

    def analyze(self, bands_dos_data: Dict) -> Optional[Dict]:
        """Perform hydrogen mode analysis.

        Args:
            bands_dos_data: Output from BandsDOSComputer.compute()

        Returns:
            dict: Analysis results, or None if no H present or no pDOS data
        """
        if 'H' not in self.profile.elements:
            return None

        print("\n[Phase 5] Hydrogen Storage Phonon Analysis")

        if bands_dos_data is None:
            print("  [!] No band/DOS data available for H analysis.")
            return None

        pdos_data = bands_dos_data.get('pdos_data')
        if pdos_data is None:
            print("  [!] No pDOS data available for H analysis.")
            return None

        freq, pdos, sym_to_idx = pdos_data

        # Extract H partial DOS
        h_dos = self._extract_h_dos(pdos, sym_to_idx)
        if h_dos is None:
            print("  [!] No H atoms found in pDOS.")
            return None

        # Mode decomposition
        freq_cm = freq * THZ_TO_CM
        decomposition = self._decompose_modes(h_dos, freq)

        # Peak identification
        peak_info = self._find_peak_stretching(h_dos, freq)

        # Hydride classification
        hydride_type = 'unknown'
        if peak_info and peak_info['freq_cm'] > 0:
            hydride_type = self._classify_hydride(peak_info['freq_cm'])

        # Report results
        self._report_results(decomposition, peak_info, hydride_type)

        return {
            'decomposition': decomposition,
            'peak_stretching': peak_info,
            'hydride_type': hydride_type,
            'h_dos': h_dos,
            'freq': freq,
        }

    def _extract_h_dos(self, pdos, sym_to_idx) -> Optional[np.ndarray]:
        """Extract hydrogen partial DOS from total pDOS."""
        h_indices = sym_to_idx.get('H', [])
        if not h_indices:
            return None

        h_dos = np.zeros_like(pdos.sum(axis=0))
        for idx in h_indices:
            if idx < pdos.shape[0]:
                h_dos += pdos[idx]

        return h_dos

    def _decompose_modes(self, h_dos: np.ndarray, freq: np.ndarray) -> Dict:
        """Decompose H DOS into librational/bending/stretching fractions."""
        lib_range = H_MODE_RANGES['librational']
        bend_range = H_MODE_RANGES['bending']
        stretch_range = H_MODE_RANGES['stretching']

        lib_mask = (freq > lib_range[0]) & (freq < lib_range[1])
        bend_mask = (freq > bend_range[0]) & (freq < bend_range[1])
        stretch_mask = (freq > stretch_range[0]) & (freq < stretch_range[1])

        # Use trapezoidal integration
        if hasattr(np, 'trapezoid'):
            trapz = np.trapezoid
        else:
            trapz = np.trapz

        h_lib = trapz(h_dos[lib_mask], freq[lib_mask]) if np.any(lib_mask) else 0.0
        h_bend = trapz(h_dos[bend_mask], freq[bend_mask]) if np.any(bend_mask) else 0.0
        h_stretch = trapz(h_dos[stretch_mask], freq[stretch_mask]) if np.any(stretch_mask) else 0.0
        h_total = trapz(h_dos[freq > 0], freq[freq > 0])

        if h_total == 0:
            return {'lib': 0, 'bend': 0, 'stretch': 0, 'total': 0}

        return {
            'lib': h_lib / h_total * 100,
            'bend': h_bend / h_total * 100,
            'stretch': h_stretch / h_total * 100,
            'total': h_total,
        }

    def _find_peak_stretching(self, h_dos: np.ndarray, freq: np.ndarray) -> Optional[Dict]:
        """Find the principal stretching peak frequency."""
        stretch_range = H_MODE_RANGES['stretching']
        stretch_mask = (freq > stretch_range[0]) & (freq < stretch_range[1])

        if not np.any(stretch_mask):
            return None

        peak_idx = np.argmax(h_dos[stretch_mask])
        peak_thz = float(freq[stretch_mask][peak_idx])
        peak_cm = peak_thz * THZ_TO_CM

        return {
            'freq_thz': peak_thz,
            'freq_cm': peak_cm,
        }

    def _classify_hydride(self, stretch_freq_cm: float) -> str:
        """Classify hydride type from stretching frequency."""
        for hydride_type, (low, high) in self.HYDRIDE_TYPES.items():
            if low <= stretch_freq_cm <= high:
                return hydride_type
        return 'unknown'

    def _report_results(self, decomposition: Dict, peak_info: Dict, hydride_type: str):
        """Print analysis results to console."""
        print("  --> H-Mode Fractional Decomposition:")
        print(f"      Librational: {decomposition['lib']:.1f} %")
        print(f"      Bending:     {decomposition['bend']:.1f} %")
        print(f"      Stretching:  {decomposition['stretch']:.1f} %")

        if peak_info:
            print(f"  --> Principal stretch isolated at: {peak_info['freq_thz']:.2f} THz ({peak_info['freq_cm']:.0f} cm^-1)")
            print(f"  --> Hydride type: {hydride_type}")
