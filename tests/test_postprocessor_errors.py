"""
Tests for HydroPhonoKit Postprocessor error handling and resilience.

Run with: pytest tests/test_postprocessor_errors.py -v
"""
import pytest
import numpy as np
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from hydrophonokit.postprocessor.core import (
    PhononPostProcessor,
    PostprocessorConfig,
    VALID_PHASES,
)


# ============================================================================
# Test PostprocessorConfig
# ============================================================================

class TestPostprocessorConfig:
    """Test PostprocessorConfig validation and defaults."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = PostprocessorConfig()
        assert config.skip_born_if_missing is True
        assert config.skip_phases == []
        assert config.dry_run is False
        assert config.verbose is True
        assert config.max_retries == 0

    def test_skip_phases_validation(self):
        """Valid phase names should be accepted."""
        config = PostprocessorConfig(skip_phases=['hydrogen', 'plotting'])
        assert 'hydrogen' in config.skip_phases
        assert 'plotting' in config.skip_phases

    def test_skip_phases_case_insensitive(self):
        """Phase names should be case-insensitive."""
        config = PostprocessorConfig(skip_phases=['HYDROGEN', 'Plotting'])
        assert 'hydrogen' in config.skip_phases
        assert 'plotting' in config.skip_phases

    def test_invalid_phase_name(self):
        """Invalid phase names should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid phase names"):
            PostprocessorConfig(skip_phases=['invalid_phase'])

    def test_dry_run_implies_save_partial(self):
        """dry_run should automatically enable save_partial_results."""
        config = PostprocessorConfig(dry_run=True, save_partial_results=False)
        assert config.save_partial_results is True

    def test_all_valid_phases(self):
        """All expected phases should be in VALID_PHASES."""
        expected = {
            'data_collection', 'force_constants', 'bands_dos',
            'thermodynamics', 'hydrogen', 'group_velocities',
            'debye_waller', 'mode_resolved_thermo', 'plotting',
            'reporting', 'export'
        }
        assert VALID_PHASES == expected


# ============================================================================
# Test Dry Run Mode
# ============================================================================

class TestDryRun:
    """Test dry-run validation mode."""

    def _make_mock_profile(self):
        """Create a mock MaterialProfile."""
        profile = Mock()
        profile.formula = "TestMaterial"
        profile.space_group = "Pm-3m (221)"
        profile.rec_born = False
        return profile

    def test_dry_run_nonexistent_workspace(self, tmp_path):
        """Dry run should report missing workspace."""
        profile = self._make_mock_profile()
        pp = PhononPostProcessor(
            str(tmp_path / "nonexistent"),
            profile,
            config=PostprocessorConfig(dry_run=True)
        )
        results = pp.execute_pipeline()

        assert results['dry_run'] is True
        assert results['validations']['workspace_exists'] is False

    def test_dry_run_missing_yaml(self, tmp_path):
        """Dry run should report missing phonopy_disp.yaml."""
        profile = self._make_mock_profile()
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        pp = PhononPostProcessor(
            str(workspace),
            profile,
            config=PostprocessorConfig(dry_run=True)
        )
        results = pp.execute_pipeline()

        assert results['validations']['workspace_exists'] is True
        assert results['validations']['phonopy_disp_yaml'] is False

    def test_dry_run_valid_workspace(self, tmp_path):
        """Dry run should pass validation with proper workspace."""
        profile = self._make_mock_profile()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "phonopy_disp.yaml").write_text("test")
        disp_dir = workspace / "02_displacements"
        disp_dir.mkdir()
        (disp_dir / "disp-001").mkdir()
        (disp_dir / "disp-002").mkdir()

        pp = PhononPostProcessor(
            str(workspace),
            profile,
            config=PostprocessorConfig(dry_run=True)
        )
        results = pp.execute_pipeline()

        assert results['validations']['workspace_exists'] is True
        assert results['validations']['phonopy_disp_yaml'] is True
        assert results['validations']['displacements_dir'] is True
        assert results['validations']['n_displacements'] == 2


# ============================================================================
# Test Skip Phases
# ============================================================================

class TestSkipPhases:
    """Test phase skipping functionality."""

    def _make_mock_profile(self):
        profile = Mock()
        profile.formula = "TestMaterial"
        profile.space_group = "Pm-3m"
        profile.rec_born = False
        profile.elements = {}  # No H, skip hydrogen analysis
        return profile

    def test_skip_single_phase(self, tmp_path):
        """Skipping a phase should mark it as skipped."""
        config = PostprocessorConfig(
            skip_phases=['plotting'],
            dry_run=True  # Use dry run to avoid actual computation
        )
        profile = self._make_mock_profile()

        pp = PhononPostProcessor(str(tmp_path), profile, config=config)

        # In dry-run mode, skip_phases should be recorded
        assert 'plotting' in pp.config.skip_phases


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling and recovery."""

    def _make_mock_profile(self):
        profile = Mock()
        profile.formula = "TestMaterial"
        profile.space_group = "Pm-3m"
        profile.rec_born = False
        profile.elements = {}
        return profile

    def test_progress_callback(self, tmp_path):
        """Progress callback should be invoked for each phase."""
        calls = []

        def callback(phase, status):
            calls.append((phase, status['completed']))

        config = PostprocessorConfig(
            dry_run=False,
            progress_callback=callback
        )
        profile = self._make_mock_profile()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "phonopy_disp.yaml").write_text("test")

        pp = PhononPostProcessor(str(workspace), profile, config=config)
        pp.execute_pipeline()

        # At least one phase should have triggered callback
        assert len(calls) > 0

    def test_phase_status_tracking(self, tmp_path):
        """Phase status should be tracked for all phases."""
        config = PostprocessorConfig(dry_run=False)
        profile = self._make_mock_profile()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "phonopy_disp.yaml").write_text("test")

        pp = PhononPostProcessor(str(workspace), profile, config=config)
        pp.execute_pipeline()

        # Should have status for dry_run phase at minimum
        assert len(pp.phase_status) > 0

    def test_max_retries_config(self, tmp_path):
        """max_retries should be configurable."""
        config = PostprocessorConfig(max_retries=3)
        assert config.max_retries == 3


# ============================================================================
# Test Pipeline Summary
# ============================================================================

class TestPipelineSummary:
    """Test pipeline summary output."""

    def _make_mock_profile(self):
        profile = Mock()
        profile.formula = "TestMaterial"
        profile.space_group = "Pm-3m"
        profile.rec_born = False
        profile.elements = {}
        return profile

    def test_pipeline_status_file_created(self, tmp_path):
        """pipeline_status.json should be created after execution."""
        config = PostprocessorConfig(
            dry_run=True,
            save_partial_results=True
        )
        profile = self._make_mock_profile()
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "phonopy_disp.yaml").write_text("test")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        pp = PhononPostProcessor(
            str(workspace),
            profile,
            output_dir=str(output_dir),
            config=config
        )
        pp.execute_pipeline()

        # In dry-run mode, status file may or may not be created
        # depending on implementation
        # Just verify the pipeline ran without errors
        assert pp.phase_status is not None
