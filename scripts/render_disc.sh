#!/bin/bash
#SBATCH --job-name=dla_render
#SBATCH --account=pi_machta_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --output=logs/render_%j.out
#SBATCH --error=logs/render_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

python model/render_disc.py \
    --particle_dir data/fixed_n1000_raw_v2/particles \
    --output_dir data/fixed_n1000_disc2/images \
    --metadata_dir data/fixed_n1000_disc2/metadata \
    --image_size 256 \
    --scale 1.0 \
    --disc_radius 2
