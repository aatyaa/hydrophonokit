"""
Tests for HydroPhonoKit anharmonic and thermal conductivity modules.

Run with: pytest tests/test_anharmonic.py -v
"""
import pytest
import numpy as np

from hydrophonokit.physics import bose_einstein
from hydrophonokit.anharmonic import (
    AnharmonicResult, AnharmonicCalculator, slack_thermal_conductivity,
)


# ============================================================================
# Test Bose-Einstein (from physics.py, used by anharmonic)
# ============================================================================

class TestBoseEinstein:
    """Test Bose-Einstein occupation function."""

    def test_zero_temperature(self):
        """At T=0, occupation should be 0."""
        assert bose_einstein(5.0, 0) == 0.0

    def test_high_temperature(self):
        """At high T, n ≈ kT/(hbar*omega) (classical limit)."""
        w = 5.0  # THz
        T = 3000  # Very high T
        n = bose_einstein(w, T)
        # Classical: n ≈ kT/(hbar*omega) = T / (48.0 * w)
        expected = T / (48.0 * w)
        assert n > expected * 0.5  # Within factor of 2

    def test_positive_occupation(self):
        """Occupation should always be positive."""
        assert bose_einstein(10.0, 300) > 0


# ============================================================================
# Test Slack Model
# ============================================================================

class TestSlackModel:
    """Test Slack model for thermal conductivity."""

    def test_si_typical(self):
        """Silicon at 300K: κ ~ 150 W/(m·K).

        Si: θ_D=645K, M=28amu, δ=2.72Å, γ=1.0, n=2
        """
        kappa = slack_thermal_conductivity(
            theta_D=645, M=28.0, delta=2.72, gamma=1.0, T=300, n=2
        )
        # Slack model gives order-of-magnitude estimate
        assert 50 < kappa < 500

    def test_temperature_dependence(self):
        """κ should decrease with T (high-T limit: κ ∝ 1/T)."""
        kappa_300 = slack_thermal_conductivity(645, 28, 2.72, 1.0, 300, 2)
        kappa_600 = slack_thermal_conductivity(645, 28, 2.72, 1.0, 600, 2)

        # κ(600) should be ~ half of κ(300)
        assert kappa_600 < kappa_300
        assert abs(kappa_600 / kappa_300 - 0.5) < 0.1

    def test_zero_temperature(self):
        """At T=0, model should return 0."""
        kappa = slack_thermal_conductivity(645, 28, 2.72, 1.0, 0, 2)
        assert kappa == 0.0

    def test_zero_gruneisen(self):
        """With γ=0 (no anharmonicity), should return 0 (divergent κ)."""
        kappa = slack_thermal_conductivity(645, 28, 2.72, 0.0, 300, 2)
        assert kappa == 0.0

    def test_debye_dependence(self):
        """Higher θ_D should give higher κ (θ_D³ dependence)."""
        kappa_low = slack_thermal_conductivity(300, 28, 2.72, 1.0, 300, 2)
        kappa_high = slack_thermal_conductivity(600, 28, 2.72, 1.0, 300, 2)

        assert kappa_high > kappa_low
        # θ_D³ dependence: (600/300)³ = 8x
        assert kappa_high / kappa_low > 5  # At least 5x


# ============================================================================
# Test AnharmonicResult
# ============================================================================

class TestAnharmonicResult:
    """Test AnharmonicResult data class."""

    def test_summary_not_empty(self):
        """Summary should produce non-empty output."""
        r = AnharmonicResult()
        r.formula = "Si"
        r.n_qpoints = 100
        r.n_bands = 6
        r.temperatures = np.array([0, 300, 600])
        r.frequencies = np.ones((100, 6)) * 5.0
        r.lifetimes = np.ones((100, 6, 3)) * 10.0  # 10 ps
        r.lifetimes_inv = np.ones((100, 6, 3)) * 0.05

        summary = r.summary()
        assert "Si" in summary
        assert "100" in summary  # n_qpoints

    def test_to_dict(self):
        """Dictionary serialization should work."""
        r = AnharmonicResult()
        r.formula = "Si"
        r.n_qpoints = 10
        r.n_bands = 3
        r.temperatures = np.array([0, 300])
        r.lifetimes = np.ones((10, 3, 2)) * 5.0

        d = r.to_dict()
        assert d['formula'] == 'Si'
        assert d['n_qpoints'] == 10
        assert len(d['lifetimes_ps']) == 10
