# HPC execution

The full public-data tiers are intended for an HPC or long-running workstation. The implementation is CPU-oriented; a GPU is not required by the compact calibration head.

Clone the repository, create the environment, place public data outside Git, and submit the included SLURM job:

```bash
git clone https://github.com/Devchandrasen/fedtwin-cryptid.git
cd fedtwin-cryptid
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
sbatch hpc_run_full.slurm
```

Override storage locations without editing the script:

```bash
PROJECT_DIR=$PWD DATA_DIR=/scratch/$USER/fedtwin-data RUN_DIR=/scratch/$USER/fedtwin-runs sbatch hpc_run_full.slurm
```

The bundled job runs the synthetic feature-level tier. For VCSL ISC or FMA, use the corresponding `run_benchmark.py --tier` option and pass the local data paths shown by `python run_benchmark.py --help`.

If the cluster does not use SLURM, run the command body from `hpc_run_full.slurm` inside the available scheduler. Keep generated results in scratch or another non-repository directory.
