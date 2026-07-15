#!/bin/bash
#SBATCH --job-name=fedtwin_full
#SBATCH --output=logs/fedtwin_full_%j.out
#SBATCH --error=logs/fedtwin_full_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/fedtwin-cryptid}"
DATA_DIR="${DATA_DIR:-/scratch/$USER/fedtwin-cryptid/data}"
RUN_DIR="${RUN_DIR:-/scratch/$USER/fedtwin-cryptid/runs}"

cd "$PROJECT_DIR"
mkdir -p logs "$DATA_DIR" "$RUN_DIR"

python -m pip install --upgrade pip
python -m pip install -e ".[all]"

python run_benchmark.py \
  --tier tier1 \
  --data-dir "$DATA_DIR" \
  --run-dir "$RUN_DIR" \
  --clients 10 \
  --difficulty 1.6 \
  --seeds 31 37 41 \
  --federated-methods centralized local fedavg fedprox secureagg_sim quantized_transport_proxy \
  --modalities video audio multimodal \
  --ledger-modes none hashchain \
  --output-tag tier1_full
