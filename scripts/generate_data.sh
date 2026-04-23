#!/bin/bash
#SBATCH --job-name=dla_gen
#SBATCH --account=pi_machta_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --output=logs/dla_gen_%A_%a.out
#SBATCH --error=logs/dla_gen_%A_%a.err
#SBATCH --array=0-49
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#
# Generate DLA training data in parallel via SLURM job array.
# Each array task generates a batch of clusters.
#
# Total: 50 tasks x 100 clusters/task = 5,000 clusters
#
# Submit from the dla_project/ directory:
#   sbatch scripts/generate_data.sh

set -euo pipefail

# ---- Configuration ----
MIN_PARTICLES=2000      # min particles per cluster
MAX_PARTICLES=8000      # max particles per cluster (randomized per image)
IMG_SIZE=256            # image resolution
CLUSTERS_PER_TASK=100   # clusters per array task
OUTPUT_DIR="data/output"
SIMULATOR="simulator/dla_sim"

# Each task gets a unique seed range
BASE_SEED=42
TASK_SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID * CLUSTERS_PER_TASK))

# ---- Setup ----
mkdir -p logs
echo "=== DLA Data Generation ==="
echo "Task ID:    ${SLURM_ARRAY_TASK_ID}"
echo "Particles:  random [${MIN_PARTICLES}, ${MAX_PARTICLES}]"
echo "Clusters:   ${CLUSTERS_PER_TASK}"
echo "Seed:       ${TASK_SEED}"
echo "Output:     ${OUTPUT_DIR}"

# Build simulator if needed
if [ ! -f "${SIMULATOR}" ]; then
    echo "Building simulator..."
    make -C simulator
fi

# ---- Run ----
./${SIMULATOR} \
    --min_particles ${MIN_PARTICLES} \
    --max_particles ${MAX_PARTICLES} \
    --size ${IMG_SIZE} \
    --count ${CLUSTERS_PER_TASK} \
    --outdir ${OUTPUT_DIR} \
    --seed ${TASK_SEED} \
    --prefix "dla_t${SLURM_ARRAY_TASK_ID}" \
    --verbose

echo "Task ${SLURM_ARRAY_TASK_ID} complete."
