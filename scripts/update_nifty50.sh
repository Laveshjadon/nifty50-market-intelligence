#!/bin/bash
# update_nifty50.sh
# Cron wrapper for updating Nifty 50 weights periodically.

# Set the project directory (adjust if necessary)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate virtual environment if applicable
# source "${PROJECT_DIR}/.venv/bin/activate"

# Export any necessary environment variables
export PYTHONPATH="${PROJECT_DIR}"

echo "[$(date)] Starting Nifty 50 weights update..."
python3 "${PROJECT_DIR}/src/update_nifty50_weights.py"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Update completed successfully."
else
    echo "[$(date)] Update failed. Please check the application logs."
    # Optional: curl -X POST "https://api.slack.com/webhook" -d '{"text": "Nifty 50 Weight Update Failed!"}'
fi

exit $EXIT_CODE
