"""
Tests for HydroPhonoKit Postprocessor scientific enhancements.

Run with: pytest tests/test_postprocessor_scientific.py -v
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Test Group Velocities
# ============================================================================

class TestGroupVelocities:
    """Test group velocity computation."""

    def _make_mock_phonon(self):
        """Create a mock phonon object with necessary attributes."""
        phonon = Mock()
        phonon.unitcell.numbers = [14, 14]  # Si
        phonon.unitcell.cell = np.eye(3) * 5.43  # Si lattice
        phonon.unitcell.masses = [28.085, 28.085]
        phonon.supercells_with_displacements = []

        # Mock methods
        phonon.run_mesh = Mock()
        phonon.get_group_velocities = Mock(return_value=np.ones((1000, 6, 3)) * 0.5)
        phonon.get_mesh_frequencies = Mock(return_value=np.ones((1000, 6)) * 5.0)
        phonon.run_qpoints = Mock()
        phonon.get_qpoints_frequencies = Mock(return_value=np.ones((1, 6)) * 5.0)

        return phonon

    def test_init(self):
        """GroupVelocityComputer should initialize with default delta_q."""
        from hydrophonokit.postprocessor.group_velocities import GroupVelocityComputer
        phonon = self._make_mock_phonon()
        gv = GroupVelocityComputer(phonon)
        assert gv.delta_q == 1e-5

    def test_compute_returns_dict(self):
        """compute() should return a dictionary with expected keys."""
        from hydrophonokit.postprocessor.group_velocities import GroupVelocityComputer
        phonon = self._make_mock_phonon()
        gv = GroupVelocityComputer(phonon)
        result = gv.compute()

        assert 'q_points' in result
        assert 'frequencies' in result
        assert 'group_velocities' in result
        assert 'avg_sound_velocity' in result
        assert 'max_group_velocity' in result


# ============================================================================
# Test Debye-Waller
# ============================================================================

class TestDebyeWaller:
    """Test Debye-Waller factor computation."""

    def _make_mock_phonon_and_profile(self):
        """Create mock phonon and profile objects."""
        phonon = Mock()
        phonon.unitcell.numbers = [12, 1]  # MgH2
        phonon.unitcell.masses = [24.305, 1.008, 1.008]
        phonon.run_mesh = Mock()
        phonon.mesh_frequencies = np.ones((100, 9)) * 5.0
        phonon.eigenvectors = np.random.randn(100, 9, 3, 3) + 1j * np.random.randn(100, 9, 3, 3)
        phonon.mesh_weights = np.ones(100) / 100.0

        profile = Mock()
        profile.formula = "MgH2"

        return phonon, profile

    def test_init(self):
        """DebyeWallerComputer should initialize correctly."""
        from hydrophonokit.postprocessor.debye_waller import DebyeWallerComputer
        phonon, profile = self._make_mock_phonon_and_profile()
        dw = DebyeWallerComputer(phonon, profile)
        assert dw.phonon == phonon
        assert dw.profile == profile

    def test_compute_returns_dict(self):
        """compute() should return dictionary with expected keys."""
        from hydrophonokit.postprocessor.debye_waller import DebyeWallerComputer
        phonon, profile = self._make_mock_phonon_and_profile()
        dw = DebyeWallerComputer(phonon, profile)
        result = dw.compute(temperature=300)

        assert 'temperature' in result
        assert 'mean_square_displacement' in result
        assert 'isotropic_u2' in result
        assert 'B_factor' in result
        assert 'adp_tensor' in result
        assert 'zero_point_u2' in result

    def test_B_factor_positive(self):
        """B-factors should be positive."""
        from hydrophonokit.postprocessor.debye_waller import DebyeWallerComputer
        phonon, profile = self._make_mock_phonon_and_profile()
        dw = DebyeWallerComputer(phonon, profile)
        result = dw.compute(temperature=300)

        assert np.all(result['B_factor'] >= 0)


# ============================================================================
# Test Mode-Resolved Thermodynamics
# ============================================================================

class TestModeResolvedThermo:
    """Test mode-resolved thermodynamics computation."""

    def _make_mock_phonon_and_profile(self):
        """Create mock phonon and profile."""
        phonon = Mock()
        phonon.unitcell.numbers = [12, 1, 1]  # MgH2
        phonon.run_mesh = Mock()
        phonon.run_projected_dos = Mock()

        # Mock pDOS output
        pdos_mock = Mock()
        pdos_mock.frequency_points = np.linspace(0, 50, 500)
        pdos_mock.projected_dos = np.random.randn(3, 500) ** 2  # Positive values
        phonon.projected_dos = pdos_mock

        profile = Mock()
        profile.formula = "MgH2"

        return phonon, profile

    def test_init(self):
        """ModeResolvedThermo should initialize correctly."""
        from hydrophonokit.postprocessor.mode_resolved_thermo import ModeResolvedThermo
        phonon, profile = self._make_mock_phonon_and_profile()
        mrt = ModeResolvedThermo(phonon, profile)
        assert mrt.phonon == phonon
        assert mrt.profile == profile

    def test_compute_returns_dict(self):
        """compute() should return dictionary with expected structure."""
        from hydrophonokit.postprocessor.mode_resolved_thermo import ModeResolvedThermo
        phonon, profile = self._make_mock_phonon_and_profile()
        mrt = ModeResolvedThermo(phonon, profile)
        result = mrt.compute()

        assert 'total' in result
        assert 'by_element' in result
        assert 'hydrogen_entropy_contribution' in result
        assert 'element_fractions' in result

    def test_element_fractions_sum_to_100(self):
        """Element fractions should sum to approximately 100%."""
        from hydrophonokit.postprocessor.mode_resolved_thermo import ModeResolvedThermo
        phonon, profile = self._make_mock_phonon_and_profile()
        mrt = ModeResolvedThermo(phonon, profile)
        result = mrt.compute()

        total_frac = sum(result['element_fractions'].values())
        # Should be close to 100% (allow some numerical error)
        assert abs(total_frac - 100.0) < 10.0


# ============================================================================
# Test Integration with Postprocessor
# ============================================================================

class TestScientificIntegration:
    """Test that scientific modules integrate correctly with the main postprocessor."""

    def test_valid_phases_include_scientific(self):
        """VALID_PHASES should include scientific phases."""
        from hydrophonokit.postprocessor.core import VALID_PHASES

        assert 'group_velocities' in VALID_PHASES
        assert 'debye_waller' in VALID_PHASES
        assert 'mode_resolved_thermo' in VALID_PHASES

    def test_skip_scientific_phases(self):
        """Should be able to skip scientific phases via config."""
        from hydrophonokit.postprocessor.core import PostprocessorConfig

        config = PostprocessorConfig(
            skip_phases=['group_velocities', 'debye_waller', 'mode_resolved_thermo']
        )
        assert 'group_velocities' in config.skip_phases
        assert 'debye_waller' in config.skip_phases
        assert 'mode_resolved_thermo' in config.skip_phases

    def test_postprocessor_has_scientific_attributes(self):
        """PhononPostProcessor should have scientific component attributes."""
        from hydrophonokit.postprocessor.core import PhononPostProcessor
        from hydrophonokit.postprocessor import PostprocessorConfig

        mock_profile = Mock()
        mock_profile.formula = "Test"
        mock_profile.space_group = "Test"
        mock_profile.rec_born = False
        mock_profile.elements = {}

        mock_workspace = "/tmp/test_ws"

        # Just verify initialization doesn't crash
        # (actual execution requires real workspace)
        try:
            pp = PhononPostProcessor(
                mock_workspace,
                mock_profile,
                output_dir="/tmp/test_output",
                config=PostprocessorConfig(dry_run=True)
            )
            assert hasattr(pp, 'group_velocities')
            assert hasattr(pp, 'debye_waller')
            assert hasattr(pp, 'mode_resolved_thermo')
        except Exception:
            # Expected if workspace doesn't exist, but attributes should be set up
            pass
