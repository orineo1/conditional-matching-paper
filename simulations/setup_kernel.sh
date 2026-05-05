#!/bin/bash
# Run this ONCE on the HUJI cluster login node (or an interactive session)
# to install simulation dependencies and register the Jupyter kernel.
#
# Usage:
#   bash simulations/setup_kernel.sh
#
# After this runs, open Jupyter via the OOD portal and select
# the kernel named "Simulations (conditional-matching)" from the dropdown.

set -e

ENV_PATH="/sci/labs/orzuk/ori_m/dps_env"

echo "=== Activating environment: $ENV_PATH ==="
source "$ENV_PATH/bin/activate"

echo "=== Installing simulation requirements ==="
pip install -r "$(dirname "$0")/requirements.txt" -q

echo "=== Installing ipykernel (needed to register the kernel) ==="
pip install ipykernel -q

echo "=== Registering Jupyter kernel ==="
python -m ipykernel install \
    --user \
    --name "sim-cond-matching" \
    --display-name "Simulations (conditional-matching)"

echo ""
echo "✅ Done! Kernel 'Simulations (conditional-matching)' is now registered."
echo "   Open Jupyter via the OOD portal and select it from the kernel dropdown."
echo ""
echo "   Portal: https://portal2.cs.huji.ac.il"
echo "   (or the portal listed on the wiki page)"
