"""
=============================================================================
  HydroPhonoKit v2.7 -- HPC Profile System

  Built-in SLURM configurations for known HPC platforms.
  Eliminates the need for manual post-generation customization scripts.

  Usage:
    hydrophonokit build --source ./relax --hpc bridges2
    hydrophonokit build --source ./relax --hpc roar
    hydrophonokit build --source ./relax --hpc custom --hpc-config cluster.yaml

  Each profile defines partition, cores, modules, MPI launcher, and VASP
  executable. Profiles are validated against a required schema before use.

  Adding a new profile:
    Add an entry to HPC_PROFILES dict with the required fields.
    No code changes needed elsewhere -- generator picks it up automatically.
=============================================================================
"""
import os
import copy
from typing import Dict, Optional


# ============================================================================
# PROFILE SCHEMA
# ============================================================================

REQUIRED_FIELDS = [
    'name', 'partition', 'cores_per_node', 'nodes',
    'modules', 'mpi_command', 'vasp_executable',
]

OPTIONAL_FIELDS = {
    'memory': None,              # e.g. "256GB" -- None = omit from SLURM
    'module_purge': True,        # module purge before loading
    'module_use': None,          # custom module path (module use ...)
    'extra_env': {},             # extra environment variables
    'born_walltime': '12:00:00',
    'disp_walltime': '06:00:00',
    'email': None,               # fallback to SLURM_EMAIL env var
    'account': None,             # --account for allocation charging
    'qos': None,                 # --qos if required
    'constraint': None,          # --constraint (e.g. "cache" on Perlmutter)
}


# ============================================================================
# BUILT-IN PROFILES
# ============================================================================

HPC_PROFILES: Dict[str, dict] = {

    'bridges2': {
        'name':            'PSC Bridges-2',
        'partition':       'RM',
        'cores_per_node':  128,
        'nodes':           1,
        'memory':          None,       # RM nodes have 256GB, no need to specify
        'module_purge':    True,
        'module_use':      None,
        'modules':         ['intel-oneapi', 'hdf5/1.12.0-intel20.4', 'VASP/6.4.3-intel'],
        'mpi_command':     'mpirun -np $SLURM_NTASKS',
        'vasp_executable': 'vasp_std',
        'extra_env':       {'OMP_NUM_THREADS': '1'},
        'born_walltime':   '12:00:00',
        'disp_walltime':   '06:00:00',
        'email':           None,  # Will use SLURM_EMAIL env var or --email flag
        'account':         None,
    },

    'roar': {
        'name':            'PSU ICDS ROAR Collab',
        'partition':       'open',
        'cores_per_node':  32,
        'nodes':           1,
        'memory':          '64GB',
        'module_purge':    True,
        'module_use':      '/storage/icds/RISE/sw8/modules',
        'modules':         ['vasp/vasp-6.5.1bml'],
        'mpi_command':     'srun -n {total_tasks}',
        'vasp_executable': 'vasp_std',
        'extra_env':       {'OMP_NUM_THREADS': '1'},
        'born_walltime':   '24:00:00',
        'disp_walltime':   '12:00:00',
        'email':           None,
        'account':         None,
    },

    'generic': {
        'name':            'Generic SLURM Cluster',
        'partition':       'compute',
        'cores_per_node':  32,
        'nodes':           1,
        'memory':          '64GB',
        'module_purge':    True,
        'module_use':      None,
        'modules':         ['vasp'],
        'mpi_command':     'srun -n {total_tasks}',
        'vasp_executable': 'vasp_std',
        'extra_env':       {'OMP_NUM_THREADS': '1'},
        'born_walltime':   '24:00:00',
        'disp_walltime':   '12:00:00',
        'email':           None,
        'account':         None,
    },
}


# ============================================================================
# PROFILE LOADER
# ============================================================================

def get_profile(name: str) -> dict:
    """Get a built-in HPC profile by name.

    Args:
        name: Profile name (case-insensitive). One of: bridges2, roar, generic

    Returns:
        dict: Complete profile with all fields populated (defaults applied)

    Raises:
        ValueError: If profile name is unknown
    """
    key = name.lower().strip()
    if key not in HPC_PROFILES:
        available = ', '.join(sorted(HPC_PROFILES.keys()))
        raise ValueError(
            f"Unknown HPC profile: '{name}'\n"
            f"Available profiles: {available}\n"
            f"Use --hpc-config <file.yaml> for custom clusters."
        )

    profile = copy.deepcopy(HPC_PROFILES[key])

    # Apply defaults for optional fields
    for field, default in OPTIONAL_FIELDS.items():
        if field not in profile:
            profile[field] = default

    # Email fallback: env var
    if profile['email'] is None:
        profile['email'] = os.environ.get('SLURM_EMAIL')

    return profile


def load_custom_profile(yaml_path: str) -> dict:
    """Load an HPC profile from a YAML file.

    Expected format:
        name: My Cluster
        partition: gpu
        cores_per_node: 64
        nodes: 1
        modules:
          - vasp/6.4.1
        mpi_command: mpirun -np $SLURM_NTASKS
        vasp_executable: vasp_std
        born_walltime: "12:00:00"
        disp_walltime: "08:00:00"

    Args:
        yaml_path: Path to YAML config file

    Returns:
        dict: Validated profile

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required fields are missing
    """
    import yaml

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"HPC config file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        profile = yaml.safe_load(f)

    # Validate required fields
    missing = [f for f in REQUIRED_FIELDS if f not in profile]
    if missing:
        raise ValueError(
            f"HPC config missing required fields: {', '.join(missing)}\n"
            f"Required: {', '.join(REQUIRED_FIELDS)}"
        )

    # Apply defaults for optional fields
    for field, default in OPTIONAL_FIELDS.items():
        if field not in profile:
            profile[field] = default

    # Email fallback
    if profile.get('email') is None:
        profile['email'] = os.environ.get('SLURM_EMAIL')

    return profile


def list_profiles() -> str:
    """Return formatted string listing all available HPC profiles."""
    lines = ["Available HPC Profiles:", "=" * 50]
    for key, p in sorted(HPC_PROFILES.items()):
        lines.append(f"  {key:12s}  {p['name']}")
        lines.append(f"               {p['cores_per_node']} cores/node, "
                      f"partition={p['partition']}")
        lines.append(f"               modules: {', '.join(p['modules'])}")
        lines.append("")
    lines.append("  Use --hpc <name> to select a profile.")
    lines.append("  Use --hpc custom --hpc-config <file.yaml> for custom clusters.")
    return "\n".join(lines)


# ============================================================================
# SLURM SCRIPT GENERATOR (profile-aware)
# ============================================================================

def make_slurm_from_profile(profile: dict, job_name: str,
                             walltime: str, job_prefix: str = "") -> str:
    """Generate a SLURM batch script from an HPC profile.

    Args:
        profile: HPC profile dict (from get_profile or load_custom_profile)
        job_name: SLURM job name
        walltime: Wall time string (HH:MM:SS)
        job_prefix: Optional prefix for echo messages

    Returns:
        str: Complete SLURM script content
    """
    total_tasks = profile['cores_per_node'] * profile['nodes']
    lines = ['#!/bin/bash']

    # SLURM headers
    lines.append(f'#SBATCH --job-name="{job_name}"')
    lines.append(f'#SBATCH --partition={profile["partition"]}')
    lines.append(f'#SBATCH --nodes={profile["nodes"]}')
    lines.append(f'#SBATCH --ntasks-per-node={profile["cores_per_node"]}')
    if profile.get('memory'):
        lines.append(f'#SBATCH --mem={profile["memory"]}')
    lines.append(f'#SBATCH --time={walltime}')
    lines.append('#SBATCH --output=%x_%j.out')
    lines.append('#SBATCH --error=%x_%j.err')
    if profile.get('account'):
        lines.append(f'#SBATCH --account={profile["account"]}')
    if profile.get('qos'):
        lines.append(f'#SBATCH --qos={profile["qos"]}')
    if profile.get('constraint'):
        lines.append(f'#SBATCH --constraint={profile["constraint"]}')
    if profile.get('email'):
        lines.append('#SBATCH --mail-type=ALL')
        lines.append(f'#SBATCH --mail-user={profile["email"]}')

    lines.append('')

    # Module loading
    if profile.get('module_purge', True):
        lines.append('module purge')
    if profile.get('module_use'):
        lines.append(f'module use {profile["module_use"]}')
    for mod in profile.get('modules', []):
        lines.append(f'module load {mod}')

    lines.append('')

    # Environment variables
    for key, val in profile.get('extra_env', {}).items():
        lines.append(f'export {key}={val}')

    lines.append('')

    # Job info echo
    desc = job_prefix or job_name
    lines.append(f'echo "Starting: {desc}"')
    lines.append('echo "Directory: $(pwd)"')
    lines.append('echo "Date: $(date)"')
    lines.append('')

    # MPI execution
    mpi_cmd = profile['mpi_command']
    if '{total_tasks}' in mpi_cmd:
        mpi_cmd = mpi_cmd.format(total_tasks=total_tasks)
    lines.append(f'{mpi_cmd} {profile["vasp_executable"]}')

    lines.append('')
    lines.append('echo "Completed at: $(date)"')

    return '\n'.join(lines) + '\n'
