#!/usr/bin/bash
#
# Create the virtual environment on the login node.
# Run this ONCE before submitting any SLURM jobs.
#
# Usage:
#   bash setup.sh
#

set -euo pipefail

echo "Creating venv..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

echo ""
echo "Setup complete. You can now submit jobs:"
echo "  JOBID=\$(sbatch --parsable run_array.sh)"
echo "  sbatch --dependency=afterok:\$JOBID run_plot.sh"
