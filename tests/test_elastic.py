"""
Tests for HydroPhonoKit elastic constants module.

Run with: pytest tests/test_elastic.py -v
"""
import pytest
import numpy as np

from hydrophonokit.physics import (
    sound_velocity_to_elastic_constant_cubic,
    compute_vrh_moduli,
    youngs_modulus_from_BG,
    poisson_ratio_from_BG,
    debye_temperature_from_sound_velocity,
    check_mechanical_stability_cubic,
)
from hydrophonokit.elastic import ElasticResult, ElasticConstantsExtractor


# ============================================================================
# Test Physical Constants (physics.py)
# ============================================================================

class TestSoundVelocityToElastic:
    """Test conversion of sound velocities to elastic constants."""

    def test_aluminum_typical(self):
        """Test with typical Al values.

        Literature: C11 ~ 107 GPa, C12 ~ 61 GPa, C44 ~ 28 GPa
        """
        v_LA = 6420   # m/s (typical for Al)
        v_TA1 = 3040  # m/s
        v_TA2 = 2870  # m/s
        rho = 2700    # kg/m^3

        result = sound_velocity_to_elastic_constant_cubic(v_LA, v_TA1, v_TA2, rho)

        assert result['C11'] > 0
        assert result['C12'] > 0
        assert result['C44'] > 0
        # C11 should be largest
        assert result['C11'] > result['C12']
        assert result['C11'] > result['C44']

    def test_zero_velocity(self):
        """Test that zero velocities give zero elastic constants."""
        result = sound_velocity_to_elastic_constant_cubic(0, 0, 0, 2700)
        assert result['C11'] == 0
        assert result['C44'] == 0

    def test_c12_less_than_c11(self):
        """Verify C12 < C11 for physical materials."""
        v_LA = 5000
        v_TA1 = 2500
        v_TA2 = 2300
        rho = 5000

        result = sound_velocity_to_elastic_constant_cubic(v_LA, v_TA1, v_TA2, rho)
        assert result['C12'] < result['C11']


class TestVRHModuli:
    """Test Voigt-Reuss-Hill modulus averaging."""

    def test_cubic_moduli(self):
        """Test VRH averaging for cubic crystal."""
        C11, C12, C44 = 100, 50, 30  # GPa

        result = compute_vrh_moduli(C11, C12, C44)

        # B_V = (C11 + 2*C12) / 3 = (100 + 100) / 3 = 66.67
        assert abs(result['B_V'] - 66.67) < 0.1
        # G_V = (C11 - C12 + 3*C44) / 5 = (50 + 90) / 5 = 28
        assert abs(result['G_V'] - 28.0) < 0.1
        # VRH average should be between Voigt and Reuss
        assert result['G_R'] <= result['G_VRH'] <= result['G_V']

    def test_zero_moduli(self):
        """Test with zero elastic constants."""
        result = compute_vrh_moduli(0, 0, 0)
        assert result['B_VRH'] == 0
        assert result['G_VRH'] == 0


class TestYoungsModulus:
    """Test Young's modulus computation."""

    def test_typical_steel(self):
        """Test with typical steel values.

        Steel: B ~ 160 GPa, G ~ 80 GPa -> E ~ 210 GPa
        """
        E = youngs_modulus_from_BG(160, 80)
        assert 200 < E < 220  # ~210 GPa

    def test_zero_shear(self):
        """Test with zero shear modulus (fluid)."""
        E = youngs_modulus_from_BG(100, 0)
        assert E == 0


class TestPoissonRatio:
    """Test Poisson's ratio computation."""

    def test_incompressible(self):
        """Incompressible material (G << B) -> nu -> 0.5."""
        nu = poisson_ratio_from_BG(1000, 1)
        assert nu > 0.49

    def test_typical_metal(self):
        """Typical metal: nu ~ 0.33."""
        nu = poisson_ratio_from_BG(100, 40)
        assert 0.25 < nu < 0.40

    def test_auxetic(self):
        """Auxetic material (G > 1.5B) -> negative nu."""
        nu = poisson_ratio_from_BG(50, 100)
        assert nu < 0


class TestDebyeTemperature:
    """Test Debye temperature from sound velocity."""

    def test_aluminum(self):
        """Aluminum: Theta_D ~ 428 K."""
        v_avg = 3600  # m/s (average for Al)
        rho = 2700    # kg/m^3
        M = 0.02698   # kg/mol (Al molar mass)

        theta_D = debye_temperature_from_sound_velocity(v_avg, rho, M)
        assert 350 < theta_D < 500  # Reasonable range for Al

    def test_zero_velocity(self):
        """Zero velocity should give zero Debye temperature."""
        theta_D = debye_temperature_from_sound_velocity(0, 2700, 0.027)
        assert theta_D == 0


class TestMechanicalStability:
    """Test Born-Huang mechanical stability criteria."""

    def test_stable_cubic(self):
        """Test with stable cubic material (Al-like)."""
        C11, C12, C44 = 107, 61, 28
        result = check_mechanical_stability_cubic(C11, C12, C44)

        assert result['stable'] is True
        assert all(c[1] for c in result['criteria'])

    def test_unstable_c11(self):
        """Negative C11 -> unstable."""
        result = check_mechanical_stability_cubic(-10, 50, 28)
        assert result['stable'] is False

    def test_unstable_c44(self):
        """Negative C44 -> unstable."""
        result = check_mechanical_stability_cubic(107, 61, -5)
        assert result['stable'] is False

    def test_unstable_c12(self):
        """|C12| > C11 -> unstable."""
        result = check_mechanical_stability_cubic(100, 150, 30)
        assert result['stable'] is False


# ============================================================================
# Test Elastic Module (elastic.py)
# ============================================================================

class TestElasticResult:
    """Test ElasticResult data class."""

    def test_summary_not_empty(self):
        """Test that summary produces non-empty output."""
        result = ElasticResult()
        result.formula = "Test"
        result.crystal_system = "cubic"
        summary = result.summary()
        assert "Test" in summary
        assert "cubic" in summary.lower()

    def test_to_dict(self):
        """Test dictionary serialization."""
        result = ElasticResult()
        result.formula = "Si"
        result.crystal_system = "cubic"
        result.density_g_cm3 = 2.33

        d = result.to_dict()
        assert d['formula'] == 'Si'
        assert d['crystal_system'] == 'cubic'
        assert d['density_g_cm3'] == 2.33
        assert isinstance(d['elastic_constants_GPa'], list)
