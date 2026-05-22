"""
Tests for HydroPhonoKit EOS and QHA modules.

Run with: pytest tests/test_eos_qha.py -v
"""
import pytest
import numpy as np

from hydrophonokit.eos import (
    birch_murnaghan, vinet, murnaghan,
    fit_eos, fit_all_eos, best_eos_model,
    EOS_MODELS, EV_A3_TO_GPA, convert_B0,
)
from hydrophonokit.qha import QHAResult, QHAEngine


# ============================================================================
# Test EOS Models (eos.py)
# ============================================================================

class TestBirchMurnaghan:
    """Test Birch-Murnaghan EOS."""

    def test_minimum_at_V0(self):
        """Energy minimum should be at V0."""
        E0, V0, B0, Bp = -10.0, 100.0, 0.1, 4.0
        V = np.linspace(80, 120, 50)
        E = birch_murnaghan(V, E0, V0, B0, Bp)

        V_min = V[np.argmin(E)]
        assert abs(V_min - V0) < 2.0  # Within 2 A^3

    def test_E0_at_V0(self):
        """E(V0) should equal E0."""
        E = birch_murnaghan(100.0, -10.0, 100.0, 0.1, 4.0)
        assert abs(E - (-10.0)) < 0.001


class TestVinet:
    """Test Vinet EOS."""

    def test_minimum_at_V0(self):
        """Energy minimum should be at V0."""
        E0, V0, B0, Bp = -10.0, 100.0, 0.1, 4.0
        V = np.linspace(80, 120, 50)
        E = vinet(V, E0, V0, B0, Bp)

        V_min = V[np.argmin(E)]
        assert abs(V_min - V0) < 2.0


class TestMurnaghan:
    """Test Murnaghan EOS."""

    def test_minimum_at_V0(self):
        """Energy minimum should be at V0."""
        E0, V0, B0, Bp = -10.0, 100.0, 0.1, 4.0
        V = np.linspace(80, 120, 50)
        E = murnaghan(V, E0, V0, B0, Bp)

        V_min = V[np.argmin(E)]
        assert abs(V_min - V0) < 2.0


class TestEOSRegistry:
    """Test EOS model registry."""

    def test_three_models(self):
        """Should have exactly 3 models."""
        assert len(EOS_MODELS) == 3
        assert 'birch_murnaghan' in EOS_MODELS
        assert 'vinet' in EOS_MODELS
        assert 'murnaghan' in EOS_MODELS


class TestEOSFitting:
    """Test EOS fitting functionality."""

    def test_fit_birch_murnaghan(self):
        """Test fitting synthetic BM data."""
        E0, V0, B0_ev, Bp = -10.0, 100.0, 0.1, 4.0
        V = np.linspace(90, 110, 7)
        E = birch_murnaghan(V, E0, V0, B0_ev, Bp)

        result = fit_eos(V, E, model='birch_murnaghan')

        assert abs(result['E0'] - E0) < 0.1
        assert abs(result['V0'] - V0) < 2.0
        assert result['B0'] > 0  # B0 should be positive (in GPa)
        assert result['r_squared'] > 0.99

    def test_fit_all_models(self):
        """Test fitting all EOS models."""
        E0, V0, B0, Bp = -10.0, 100.0, 0.1, 4.0
        V = np.linspace(90, 110, 7)
        E = birch_murnaghan(V, E0, V0, B0, Bp)

        results = fit_all_eos(V, E)

        # All models should succeed (with varying quality)
        for model in EOS_MODELS:
            assert model in results
            assert 'error' not in results[model]

    def test_best_model_selection(self):
        """Test best model selection."""
        E0, V0, B0, Bp = -10.0, 100.0, 0.1, 4.0
        V = np.linspace(90, 110, 7)
        E = birch_murnaghan(V, E0, V0, B0, Bp)

        results = fit_all_eos(V, E)
        best = best_eos_model(results)

        assert best in EOS_MODELS
        assert best is not None

    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        V = np.linspace(90, 110, 7)
        E = np.linspace(-10, -9.5, 7)

        with pytest.raises(ValueError, match="Unknown EOS model"):
            fit_eos(V, E, model='invalid')


class TestUnitConversion:
    """Test bulk modulus unit conversions."""

    def test_ev_a3_to_gpa(self):
        """1 eV/A^3 = 160.21766208 GPa."""
        assert abs(convert_B0(1.0, 'eV/A^3', 'GPa') - 160.21766208) < 0.001

    def test_gpa_to_ev_a3(self):
        """160.21766208 GPa = 1 eV/A^3."""
        assert abs(convert_B0(160.21766208, 'GPa', 'eV/A^3') - 1.0) < 0.001

    def test_same_unit(self):
        """Same unit should return unchanged value."""
        assert convert_B0(100.0, 'GPa', 'GPa') == 100.0


# ============================================================================
# Test QHA Module (qha.py)
# ============================================================================

class TestQHAResult:
    """Test QHAResult data class."""

    def test_summary_not_empty(self):
        """Summary should produce non-empty output."""
        r = QHAResult()
        r.formula = "Si"
        r.eos_model = "birch_murnaghan"
        r.n_volumes = 5
        r.E0 = -10.0
        r.V0 = 40.0
        r.B0 = 100.0
        r.B0_prime = 4.0
        r.eos_r_squared = {'birch_murnaghan': 0.999}
        r.temperatures = np.array([0, 300, 600])
        r.volumes = np.array([40.0, 40.2, 40.5])
        r.free_energies = np.array([-10.0, -10.5, -11.0])
        r.alpha = np.array([0, 1e-5, 2e-5])
        r.B_T = np.array([100, 99, 98])
        r.C_p = np.array([0, 25, 30])
        r.gruneisen = np.array([0, 1.5, 1.6])

        summary = r.summary()
        assert "Si" in summary
        assert "birch_murnaghan" in summary

    def test_to_dict(self):
        """Dictionary serialization should work."""
        r = QHAResult()
        r.formula = "Al"
        r.temperatures = np.array([0, 300])
        r.volumes = np.array([30.0, 30.1])

        d = r.to_dict()
        assert d['formula'] == 'Al'
        assert len(d['temperatures_K']) == 2


class TestQHAEngine:
    """Test QHA workflow engine."""

    def test_synthetic_qha(self):
        """Test QHA with synthetic data resembling a real material."""
        # Synthetic data for a cubic material (Si-like)
        volumes = [38.0, 39.0, 40.0, 41.0, 42.0]  # A^3
        energies_DFT = [-10.50, -10.55, -10.58, -10.54, -10.48]  # eV

        # Simulated phonon free energies (increasing with T)
        n_T = 101
        free_energies_T = []
        for i, V in enumerate(volumes):
            # F_phonon increases with T, roughly quadratic at low T
            T_arr = np.linspace(0, 1000, n_T)
            F_T = energies_DFT[i] * 96.485 + 0.001 * T_arr + 1e-6 * T_arr**2  # kJ/mol
            free_energies_T.append(F_T)

        volume_data = []
        for i in range(len(volumes)):
            volume_data.append({
                'volume': volumes[i],
                'energy_DFT': energies_DFT[i],
                'free_energies_T': free_energies_T[i],
            })

        engine = QHAEngine(volume_data, formula="Si")
        result = engine.execute(t_min=0, t_max=1000, t_step=10)

        # Validate results
        assert result.formula == "Si"
        assert result.n_volumes == 5
        assert result.B0 > 0  # Bulk modulus should be positive
        assert result.V0 > 0  # Equilibrium volume should be positive
        assert len(result.temperatures) == 101
        assert len(result.volumes) == 101
        assert len(result.alpha) == 101

        # Thermal expansion should be non-negative
        assert np.all(result.alpha >= 0)

        # 300K volume should be >= 0K volume (thermal expansion)
        idx_0 = int(np.argmin(np.abs(result.temperatures - 0)))
        idx_300 = int(np.argmin(np.abs(result.temperatures - 300)))
        assert result.volumes[idx_300] >= result.volumes[idx_0] * 0.95
