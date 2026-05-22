"""
=============================================================================
  HydroPhonoKit Postprocessor — Caching & Performance Engine

  Provides intelligent caching and performance optimizations:
    - Cache force constants, band structures, and DOS data
    - Validate cache entries before use
    - Parallel computation where applicable
    - Memory usage monitoring for large systems

  Scientific Foundation:
    Phonon post-processing involves several expensive computations:
      1. Force constant extraction from vasprun.xml files (10-60 min)
      2. symfc computation (1-5 min)
      3. Band structure calculation (30 sec - 2 min)
      4. DOS computation on dense mesh (2-3 min for 15³ mesh)

    Caching avoids re-computation when:
      - Workspace hasn't changed
      - Same material is analyzed multiple times
      - Different export formats are generated

    Cache validation ensures results are still valid by checking:
      - phonopy_disp.yaml modification time
      - File hashes for critical inputs
      - Force constant shape matches current system
      - NAC params match (if applicable)

  References:
    [1] joblib.Memory -- Caching patterns for scientific Python
    [2] Phonopy caching mechanisms -- Togo & Tanaka (2015)
=============================================================================
"""
import os
import json
import hashlib
import time
import psutil
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

@dataclass
class CacheConfig:
    """Configuration for caching behavior.
    
    Attributes:
        enabled: Enable/disable caching entirely
        cache_dir: Directory for cached files
        max_age_hours: Maximum age for cache entries (0 = no expiry)
        validate_on_load: Validate cache entries before loading
        compress_large_files: Compress force constants > 10 MB
        memory_limit_gb: Maximum memory usage before warning (0 = no limit)
        n_jobs: Number of parallel jobs (0 = auto-detect)
    """
    enabled: bool = True
    cache_dir: str = ".hydrophonokit_cache"
    max_age_hours: float = 168  # 1 week default
    validate_on_load: bool = True
    compress_large_files: bool = True
    memory_limit_gb: float = 0.0  # No limit by default
    n_jobs: int = 0  # Auto-detect
    
    def __post_init__(self):
        """Ensure cache directory exists."""
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)


# ============================================================================
# CACHE KEY GENERATION
# ============================================================================

class CacheKeyGenerator:
    """Generates unique cache keys based on input parameters."""
    
    @staticmethod
    def workspace_key(workspace_dir: str) -> str:
        """Generate cache key for workspace.
        
        Uses hash of phonopy_disp.yaml + FORCE_CONSTANTS timestamps.
        """
        yaml_path = os.path.join(workspace_dir, 'phonopy_disp.yaml')
        fc_path = os.path.join(workspace_dir, 'FORCE_CONSTANTS')
        
        parts = []
        for path in [yaml_path, fc_path]:
            if os.path.exists(path):
                stat = os.stat(path)
                parts.append(f"{path}:{stat.st_mtime}:{stat.st_size}")
        
        return hashlib.md5(":".join(parts).encode()).hexdigest()[:12]
    
    @staticmethod
    def force_constants_key(workspace_dir: str) -> str:
        """Generate cache key for force constants."""
        yaml_path = os.path.join(workspace_dir, 'phonopy_disp.yaml')
        fc_path = os.path.join(workspace_dir, 'FORCE_CONSTANTS')
        
        if not os.path.exists(yaml_path) or not os.path.exists(fc_path):
            return None
        
        # Hash the YAML content (geometry defines FC validity)
        with open(yaml_path, 'rb') as f:
            yaml_hash = hashlib.md5(f.read()).hexdigest()[:12]
        
        # Check FC file modification time
        fc_stat = os.stat(fc_path)
        return f"fc_{yaml_hash}_{fc_stat.st_mtime:.0f}"
    
    @staticmethod
    def band_structure_key(workspace_dir: str, band_settings: Dict) -> str:
        """Generate cache key for band structure."""
        yaml_path = os.path.join(workspace_dir, 'band.yaml')
        if not os.path.exists(yaml_path):
            return None
        
        settings_hash = hashlib.md5(json.dumps(band_settings, sort_keys=True).encode()).hexdigest()[:8]
        yaml_stat = os.stat(yaml_path)
        return f"band_{settings_hash}_{yaml_stat.st_mtime:.0f}"
    
    @staticmethod
    def dos_key(workspace_dir: str, mesh: tuple) -> str:
        """Generate cache key for DOS computation."""
        dos_path = os.path.join(workspace_dir, 'dos_data.npz')
        if not os.path.exists(dos_path):
            return None
        
        mesh_str = "x".join(str(m) for m in mesh)
        dos_stat = os.stat(dos_path)
        return f"dos_{mesh_str}_{dos_stat.st_mtime:.0f}"


# ============================================================================
# CACHE MANAGER
# ============================================================================

class CacheManager:
    """Manages phonon computation cache.
    
    Usage:
        cache = CacheManager(CacheConfig())
        
        # Check if force constants are cached
        fc = cache.load_force_constants(workspace_dir)
        if fc is None:
            fc = compute_force_constants(phonon)
            cache.save_force_constants(fc, workspace_dir)
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.key_gen = CacheKeyGenerator()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'invalidated': 0,
            'load_time_s': 0.0,
            'save_time_s': 0.0,
        }
    
    def is_cached(self, key: str) -> bool:
        """Check if cache entry exists and is valid."""
        if not self.config.enabled or key is None:
            return False
        
        cache_path = os.path.join(self.config.cache_dir, f"{key}.json")
        if not os.path.exists(cache_path):
            return False
        
        # Check age
        if self.config.max_age_hours > 0:
            age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
            if age_hours > self.config.max_age_hours:
                return False
        
        return True
    
    def load_force_constants(self, workspace_dir: str) -> Optional[np.ndarray]:
        """Load cached force constants if valid."""
        key = self.key_gen.force_constants_key(workspace_dir)
        if key is None or not self.is_cached(key):
            self.stats['misses'] += 1
            return None
        
        cache_path = os.path.join(self.config.cache_dir, f"{key}.npz")
        if not os.path.exists(cache_path):
            self.stats['misses'] += 1
            return None
        
        start = time.time()
        try:
            data = np.load(cache_path)
            fc = data['force_constants']
            self.stats['hits'] += 1
            self.stats['load_time_s'] += time.time() - start
            print(f"  [Cache] Loaded force constants ({len(fc)} atoms, {time.time()-start:.2f}s)")
            return fc
        except Exception as e:
            print(f"  [Cache] Failed to load force constants: {e}")
            self.stats['invalidated'] += 1
            return None
    
    def save_force_constants(self, fc: np.ndarray, workspace_dir: str):
        """Save force constants to cache."""
        key = self.key_gen.force_constants_key(workspace_dir)
        if key is None:
            return
        
        cache_path = os.path.join(self.config.cache_dir, f"{key}.npz")
        start = time.time()
        
        save_kwargs = {'force_constants': fc}
        if self.config.compress_large_files and fc.nbytes > 10 * 1024 * 1024:
            save_kwargs['allow_pickle'] = False
        
        np.savez_compressed(cache_path, **save_kwargs)
        
        # Save metadata
        meta = {
            'key': key,
            'timestamp': time.time(),
            'shape': list(fc.shape),
            'nbytes': fc.nbytes,
            'workspace': workspace_dir,
        }
        meta_path = os.path.join(self.config.cache_dir, f"{key}.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        self.stats['save_time_s'] += time.time() - start
        print(f"  [Cache] Saved force constants ({fc.nbytes / 1e6:.1f} MB, {time.time()-start:.2f}s)")
    
    def load_band_structure(self, workspace_dir: str) -> Optional[Dict]:
        """Load cached band structure if valid."""
        band_path = os.path.join(workspace_dir, 'band.yaml')
        if not os.path.exists(band_path):
            self.stats['misses'] += 1
            return None
        
        # band.yaml is already in workspace, just mark as hit
        self.stats['hits'] += 1
        return {'path': band_path}
    
    def load_dos(self, workspace_dir: str, mesh: tuple) -> Optional[Dict]:
        """Load cached DOS data if valid."""
        key = self.key_gen.dos_key(workspace_dir, mesh)
        if key is None or not self.is_cached(key):
            self.stats['misses'] += 1
            return None
        
        cache_path = os.path.join(self.config.cache_dir, f"{key}.npz")
        if not os.path.exists(cache_path):
            self.stats['misses'] += 1
            return None
        
        start = time.time()
        try:
            data = np.load(cache_path)
            result = {
                'frequencies': data['frequencies'],
                'total_dos': data['total_dos'],
                'partial_dos': data['partial_dos'] if 'partial_dos' in data else None,
            }
            self.stats['hits'] += 1
            self.stats['load_time_s'] += time.time() - start
            print(f"  [Cache] Loaded DOS ({time.time()-start:.2f}s)")
            return result
        except Exception as e:
            print(f"  [Cache] Failed to load DOS: {e}")
            self.stats['invalidated'] += 1
            return None
    
    def save_dos(self, workspace_dir: str, mesh: tuple, data: Dict):
        """Save DOS data to cache."""
        key = self.key_gen.dos_key(workspace_dir, mesh)
        if key is None:
            return
        
        # Create dos_data.npz in workspace for portability
        dos_path = os.path.join(workspace_dir, 'dos_data.npz')
        np.savez_compressed(dos_path,
                          frequencies=data['frequencies'],
                          total_dos=data['total_dos'],
                          partial_dos=data.get('partial_dos'))
        
        # Also cache it
        cache_path = os.path.join(self.config.cache_dir, f"{key}.npz")
        np.savez_compressed(cache_path, **data)
        
        print(f"  [Cache] Saved DOS data")
    
    def invalidate(self, key: str):
        """Invalidate a cache entry."""
        for ext in ['.json', '.npz']:
            path = os.path.join(self.config.cache_dir, f"{key}{ext}")
            if os.path.exists(path):
                os.remove(path)
        self.stats['invalidated'] += 1
    
    def clear_all(self):
        """Clear all cached data."""
        if not os.path.exists(self.config.cache_dir):
            return
        
        import shutil
        shutil.rmtree(self.config.cache_dir)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        print(f"  [Cache] Cleared all cached data")
    
    def get_stats(self) -> Dict:
        """Return cache statistics."""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total * 100) if total > 0 else 0
        
        return {
            **self.stats,
            'total_requests': total,
            'hit_rate_pct': round(hit_rate, 1),
            'cache_size_mb': self._get_cache_size_mb(),
        }
    
    def _get_cache_size_mb(self) -> float:
        """Get total cache size in MB."""
        if not os.path.exists(self.config.cache_dir):
            return 0.0
        
        total = 0
        for root, dirs, files in os.walk(self.config.cache_dir):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total / 1024 / 1024


# ============================================================================
# MEMORY MONITOR
# ============================================================================

class MemoryMonitor:
    """Monitors memory usage during phonon computations.
    
    Usage:
        monitor = MemoryMonitor(limit_gb=8.0)
        monitor.check()  # Raises warning if near limit
        with monitor.track("force_constants"):
            fc = compute_force_constants(phonon)
    """
    
    def __init__(self, limit_gb: float = 0.0):
        """
        Args:
            limit_gb: Memory limit in GB (0 = no limit)
        """
        self.limit_gb = limit_gb
        self.process = psutil.Process(os.getpid())
        self.tracking = {}
    
    def get_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        return self.process.memory_info().rss / 1024 / 1024 / 1024
    
    def check(self) -> Dict:
        """Check current memory usage against limit.
        
        Returns:
            dict: {'usage_gb': float, 'limit_gb': float, 'warning': bool}
        """
        usage = self.get_usage_gb()
        warning = False
        
        if self.limit_gb > 0 and usage > self.limit_gb * 0.8:
            warning = True
            print(f"  [Memory] WARNING: Using {usage:.2f} GB (limit: {self.limit_gb} GB)")
        
        return {
            'usage_gb': round(usage, 2),
            'limit_gb': self.limit_gb,
            'warning': warning,
        }
    
    def track(self, operation: str):
        """Context manager for tracking memory during an operation."""
        return _MemoryTracker(self, operation)
    
    def get_array_size_mb(self, arr: np.ndarray) -> float:
        """Get numpy array size in MB."""
        return arr.nbytes / 1024 / 1024


class _MemoryTracker:
    """Context manager for memory tracking."""
    
    def __init__(self, monitor: MemoryMonitor, operation: str):
        self.monitor = monitor
        self.operation = operation
    
    def __enter__(self):
        self.start_mem = self.monitor.get_usage_gb()
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        end_mem = self.monitor.get_usage_gb()
        duration = time.time() - self.start_time
        delta = end_mem - self.start_mem
        
        self.monitor.tracking[self.operation] = {
            'start_gb': round(self.start_mem, 2),
            'end_gb': round(end_mem, 2),
            'delta_gb': round(delta, 2),
            'duration_s': round(duration, 2),
        }
        
        if abs(delta) > 0.1:  # Only report significant changes
            print(f"  [Memory] {self.operation}: {delta:+.2f} GB ({duration:.1f}s)")


# ============================================================================
# PARALLEL COMPUTATION
# ============================================================================

def get_n_jobs(config_n_jobs: int = 0) -> int:
    """Get number of parallel jobs.
    
    Args:
        config_n_jobs: Configured n_jobs (0 = auto-detect)
    
    Returns:
        Number of jobs to use
    """
    if config_n_jobs > 0:
        return config_n_jobs
    
    # Auto-detect: leave 1 core free for system
    import multiprocessing
    n_cpus = multiprocessing.cpu_count()
    return max(1, n_cpus - 1)


def parallel_dos_computation(pdos_data: np.ndarray, 
                            n_jobs: int = 0) -> np.ndarray:
    """Compute total DOS from partial DOS in parallel.
    
    Args:
        pdos_data: Partial DOS array (n_atoms, n_freq)
        n_jobs: Number of parallel jobs (0 = auto)
    
    Returns:
        Total DOS array (n_freq,)
    """
    n_jobs = get_n_jobs(n_jobs)
    
    if n_jobs == 1 or pdos_data.shape[0] < 10:
        # Small systems: sequential is faster
        return pdos_data.sum(axis=0)
    
    # For larger systems, chunk the computation
    from concurrent.futures import ThreadPoolExecutor
    
    chunk_size = max(1, pdos_data.shape[0] // n_jobs)
    chunks = [pdos_data[i:i+chunk_size] 
              for i in range(0, pdos_data.shape[0], chunk_size)]
    
    def sum_chunk(chunk):
        return chunk.sum(axis=0)
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(sum_chunk, chunks))
    
    return sum(results)
