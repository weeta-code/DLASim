#!/bin/bash
#SBATCH --job-name=dla_render_clean
#SBATCH --account=pi_machta_umass_edu
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --output=logs/render_clean_%j.out
#SBATCH --error=logs/render_clean_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Clean rendering: scale=2, disc_radius=1 (clear visibility, no blobs)
python model/render_disc.py \
    --particle_dir data/fixed_n1000_raw_v2/particles \
    --output_dir data/fixed_n1000_clean/images \
    --metadata_dir data/fixed_n1000_clean/metadata \
    --image_size 512 \
    --scale 2.0 \
    --disc_radius 1
