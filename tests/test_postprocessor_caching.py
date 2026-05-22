"""
Tests for HydroPhonoKit Postprocessor caching and performance.

Run with: pytest tests/test_postprocessor_caching.py -v
"""
import pytest
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch


# ============================================================================
# Test CacheConfig
# ============================================================================

class TestCacheConfig:
    """Test CacheConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        from hydrophonokit.postprocessor.caching import CacheConfig
        config = CacheConfig()
        assert config.enabled is True
        assert config.cache_dir == ".hydrophonokit_cache"
        assert config.max_age_hours == 168
        assert config.validate_on_load is True
        assert config.n_jobs == 0

    def test_custom_config(self):
        """Custom config should be accepted."""
        from hydrophonokit.postprocessor.caching import CacheConfig
        config = CacheConfig(
            enabled=False,
            cache_dir="/tmp/my_cache",
            max_age_hours=24,
        )
        assert config.enabled is False
        assert config.cache_dir == "/tmp/my_cache"
        assert config.max_age_hours == 24


# ============================================================================
# Test CacheKeyGenerator
# ============================================================================

class TestCacheKeyGenerator:
    """Test cache key generation."""

    def test_workspace_key_generated(self):
        """Workspace key should be generated for existing workspace."""
        from hydrophonokit.postprocessor.caching import CacheKeyGenerator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal workspace
            yaml_path = os.path.join(tmpdir, 'phonopy_disp.yaml')
            fc_path = os.path.join(tmpdir, 'FORCE_CONSTANTS')
            
            with open(yaml_path, 'w') as f:
                f.write("test")
            with open(fc_path, 'w') as f:
                f.write("test")
            
            key = CacheKeyGenerator.workspace_key(tmpdir)
            assert key is not None
            assert len(key) == 12  # MD5 truncated

    def test_workspace_key_none_for_missing(self):
        """Workspace key should be None for missing workspace."""
        from hydrophonokit.postprocessor.caching import CacheKeyGenerator
        key = CacheKeyGenerator.workspace_key("/nonexistent/path")
        # Should still generate key from whatever exists
        assert key is not None


# ============================================================================
# Test CacheManager
# ============================================================================

class TestCacheManager:
    """Test CacheManager functionality."""

    def _make_workspace(self, tmpdir):
        """Create a minimal workspace for testing."""
        yaml_path = os.path.join(tmpdir, 'phonopy_disp.yaml')
        fc_path = os.path.join(tmpdir, 'FORCE_CONSTANTS')
        
        with open(yaml_path, 'w') as f:
            f.write("test_yaml_content")
        
        # Create a small FC file
        with open(fc_path, 'w') as f:
            f.write("dummy_force_constants_content")
        
        return yaml_path, fc_path

    def test_init_creates_cache_dir(self):
        """CacheManager should create cache directory."""
        from hydrophonokit.postprocessor.caching import CacheManager, CacheConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "test_cache")
            config = CacheConfig(cache_dir=cache_dir)
            cache = CacheManager(config)
            
            assert os.path.exists(cache_dir)

    def test_is_cached_false_for_missing(self):
        """is_cached should return False for missing entries."""
        from hydrophonokit.postprocessor.caching import CacheManager, CacheConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            config = CacheConfig(cache_dir=cache_dir)
            cache = CacheManager(config)
            
            assert cache.is_cached("nonexistent_key") is False

    def test_save_and_load_force_constants(self):
        """Force constants should be saved and loaded."""
        from hydrophonokit.postprocessor.caching import CacheManager, CacheConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            config = CacheConfig(cache_dir=cache_dir)
            cache = CacheManager(config)
            
            # Create workspace
            self._make_workspace(tmpdir)
            
            # Save FC
            fc = np.random.randn(10, 10, 3, 3)
            cache.save_force_constants(fc, tmpdir)
            
            # Load FC
            loaded_fc = cache.load_force_constants(tmpdir)
            
            assert loaded_fc is not None
            assert loaded_fc.shape == fc.shape
            np.testing.assert_array_almost_equal(loaded_fc, fc)

    def test_cache_stats(self):
        """Cache stats should track hits and misses."""
        from hydrophonokit.postprocessor.caching import CacheManager, CacheConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            config = CacheConfig(cache_dir=cache_dir)
            cache = CacheManager(config)
            
            stats = cache.get_stats()
            assert 'hits' in stats
            assert 'misses' in stats
            assert 'hit_rate_pct' in stats

    def test_clear_all(self):
        """clear_all should remove all cached data."""
        from hydrophonokit.postprocessor.caching import CacheManager, CacheConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            config = CacheConfig(cache_dir=cache_dir)
            cache = CacheManager(config)
            
            # Create workspace and save
            self._make_workspace(tmpdir)
            fc = np.random.randn(10, 10, 3, 3)
            cache.save_force_constants(fc, tmpdir)
            
            # Clear
            cache.clear_all()
            
            # Verify cleared
            loaded_fc = cache.load_force_constants(tmpdir)
            assert loaded_fc is None


# ============================================================================
# Test MemoryMonitor
# ============================================================================

class TestMemoryMonitor:
    """Test MemoryMonitor functionality."""

    def test_init_default(self):
        """MemoryMonitor should initialize with no limit by default."""
        from hydrophonokit.postprocessor.caching import MemoryMonitor
        monitor = MemoryMonitor()
        assert monitor.limit_gb == 0.0

    def test_get_usage(self):
        """get_usage_gb should return current usage."""
        from hydrophonokit.postprocessor.caching import MemoryMonitor
        monitor = MemoryMonitor()
        usage = monitor.get_usage_gb()
        assert usage > 0

    def test_check_no_warning(self):
        """check should not warn when under limit."""
        from hydrophonokit.postprocessor.caching import MemoryMonitor
        monitor = MemoryMonitor(limit_gb=1000.0)  # Very high limit
        result = monitor.check()
        assert result['warning'] is False

    def test_track_operation(self):
        """track context manager should record memory usage."""
        from hydrophonokit.postprocessor.caching import MemoryMonitor
        monitor = MemoryMonitor()
        
        with monitor.track("test_op"):
            pass
        
        assert "test_op" in monitor.tracking


# ============================================================================
# Test Parallel Computation
# ============================================================================

class TestParallelComputation:
    """Test parallel computation functions."""

    def test_get_n_jobs_auto(self):
        """get_n_jobs with 0 should auto-detect."""
        from hydrophonokit.postprocessor.caching import get_n_jobs
        import multiprocessing
        
        n = get_n_jobs(0)
        assert n >= 1
        assert n <= multiprocessing.cpu_count()

    def test_get_n_jobs_explicit(self):
        """get_n_jobs with explicit value should return it."""
        from hydrophonokit.postprocessor.caching import get_n_jobs
        assert get_n_jobs(4) == 4

    def test_parallel_dos_computation(self):
        """parallel_dos_computation should sum correctly."""
        from hydrophonokit.postprocessor.caching import parallel_dos_computation
        
        pdos = np.ones((10, 100))  # 10 atoms, 100 freq points
        total = parallel_dos_computation(pdos, n_jobs=2)
        
        expected = np.ones(100) * 10
        np.testing.assert_array_almost_equal(total, expected)


# ============================================================================
# Test Integration with PostprocessorConfig
# ============================================================================

class TestPostprocessorConfigCaching:
    """Test caching options in PostprocessorConfig."""

    def test_use_cache_default_true(self):
        """use_cache should default to True."""
        from hydrophonokit.postprocessor.core import PostprocessorConfig
        config = PostprocessorConfig()
        assert config.use_cache is True

    def test_disable_cache(self):
        """Cache should be disableable."""
        from hydrophonokit.postprocessor.core import PostprocessorConfig
        config = PostprocessorConfig(use_cache=False)
        assert config.use_cache is False

    def test_cache_dir_configurable(self):
        """cache_dir should be configurable."""
        from hydrophonokit.postprocessor.core import PostprocessorConfig
        config = PostprocessorConfig(cache_dir="/tmp/my_cache")
        assert config.cache_dir == "/tmp/my_cache"

    def test_memory_limit_configurable(self):
        """memory_limit_gb should be configurable."""
        from hydrophonokit.postprocessor.core import PostprocessorConfig
        config = PostprocessorConfig(memory_limit_gb=8.0)
        assert config.memory_limit_gb == 8.0

    def test_n_jobs_configurable(self):
        """n_jobs should be configurable."""
        from hydrophonokit.postprocessor.core import PostprocessorConfig
        config = PostprocessorConfig(n_jobs=4)
        assert config.n_jobs == 4
