import os

def get_incar_born(encut, kspacing, ismear=0, sigma=0.05, ivdw=0, isym=None, algo="Fast"):
    vdw_block = f"\n# ===== Van der Waals =====\nIVDW    = {ivdw}\n" if ivdw > 0 else ""
    isym_block = f"ISYM    = {isym}\n" if isym is not None else ""
    return f"""\
System = DFPT_Born_Charges_Auto
# ===== Electronic Parameters =====
PREC    = Accurate
ALGO    = {algo}
EDIFF   = 1E-08        
ENCUT   = {encut}
ISMEAR  = {ismear}            
SIGMA   = {sigma}
LASPH   = .TRUE.
{isym_block}
# ===== No Ionic Relaxation =====
IBRION  = -1
NSW     = 0

# ===== DFPT: Born Effective Charges & Dielectric Tensor =====
LEPSILON = .TRUE.       

# ===== Force Precision =====
LREAL   = .FALSE.       
ADDGRID = .TRUE.        
{vdw_block}
# ===== Output =====
LWAVE   = .FALSE.
LCHARG  = .FALSE.

# ===== Parallelization =====
NCORE   = 4
KSPACING = {kspacing}         
"""

def get_incar_force(encut, kspacing=None, ismear=0, sigma=0.05, ivdw=0, isym=None, algo="Fast"):
    vdw_block = f"\n# ===== Van der Waals =====\nIVDW    = {ivdw}\n" if ivdw > 0 else ""
    kspacing_line = f"\n# ===== K-Point Mesh =====\nKSPACING = {kspacing}\n" if kspacing else ""
    isym_block = f"ISYM    = {isym}\n" if isym is not None else ""
    return f"""\
System = Phonon_Force_Auto
# ===== Electronic Parameters =====
PREC    = Accurate
ALGO    = {algo}
EDIFF   = 1E-08
ENCUT   = {encut}
ISMEAR  = {ismear}
SIGMA   = {sigma}
LASPH   = .TRUE.
{isym_block}
# ===== No Ionic Relaxation (Single Point) =====
IBRION  = -1
NSW     = 0

# ===== Force Precision (Publication-Grade) =====
LREAL   = .FALSE.
ADDGRID = .TRUE.
{vdw_block}{kspacing_line}# ===== Output =====
LWAVE   = .FALSE.
LCHARG  = .FALSE.

# ===== Parallelization =====
NCORE   = 4
"""

def get_kpoints_supercell(dim):
    return f"""\
Gamma-centered mesh
0
Gamma
{dim[0]} {dim[1]} {dim[2]}
0 0 0
"""

def make_slurm_script(job_name, time_hours, email=None, extra_cd="",
                      nodes=1, tasks_per_node=32, mem="64GB",
                      vasp_module="vasp/vasp-6.5.1bml",
                      vasp_exec="vasp_std",
                      module_path=None):
    """Generate a SLURM batch script for VASP phonon calculations.

    Args:
        job_name: SLURM job name
        time_hours: Wall time in hours
        email: Email for notifications (default: SLURM_EMAIL env var or None)
        extra_cd: Additional cd command before execution
        nodes: Number of nodes
        tasks_per_node: MPI tasks per node
        mem: Memory allocation
        vasp_module: VASP module to load
        vasp_exec: VASP executable name
        module_path: Custom module path (default: none for portability)
    """
    email = email or os.environ.get("SLURM_EMAIL")
    email_block = ""
    if email:
        email_block = f"""#SBATCH --mail-user={email}
#SBATCH --mail-type=ALL
"""

    module_path_line = f"module use {module_path}\n" if module_path else ""
    cd_line = f"cd {extra_cd}\n" if extra_cd else ""
    total_tasks = nodes * tasks_per_node

    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --mem={mem}
#SBATCH --time={int(time_hours):02d}:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
{email_block}
module purge
{module_path_line}module load {vasp_module}

ulimit -s unlimited
export OMP_NUM_THREADS=1

{cd_line}srun -n {total_tasks} {vasp_exec}
"""
