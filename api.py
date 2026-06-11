import secrets
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security, status
from fastapi.responses import PlainTextResponse
from fastapi.security import APIKeyHeader
import json
import yaml
from pathlib import Path
from generate_7day_report import main as generate_report
from src.settings import get_settings
from src.database_manager import get_db

app = FastAPI(title="Nifty 50 Intelligence API", version="1.0")

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
REPORT_LOCK_ID = 123456789


def load_config() -> dict[str, Any]:
    """Load non-secret project configuration from config.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json_safe(path_str: str) -> dict[str, Any]:
    """Read a project-relative JSON file, returning an empty dict if absent."""
    path = PROJECT_ROOT / path_str
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_generate_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> None:
    """Validate the API key required for report-generation jobs."""
    expected_api_key = get_settings().api_key
    if not expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NIFTY_API_KEY is not configured on the server.",
        )

    if api_key is None or not secrets.compare_digest(api_key, expected_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


def require_report_lock():
    """Dependency to acquire a distributed Postgres lock. Kept open during background task."""
    db = get_db()
    with db.advisory_lock(REPORT_LOCK_ID) as acquired:
        if not acquired:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Report generation is already in progress. Please try again later."
            )
        yield


@app.get("/")
def read_root() -> dict[str, str]:
    """Return API health status."""
    return {"status": "Nifty 50 API is running"}

@app.post("/generate")
def trigger_generation(
    background_tasks: BackgroundTasks,
    _: None = Depends(require_generate_api_key),
    __: None = Depends(require_report_lock),
) -> dict[str, str]:
    """Trigger 7-day report generation in the background."""
    background_tasks.add_task(generate_report)
    return {"message": "Report generation started in the background. Check back in a few minutes."}

@app.get("/report/markdown", response_class=PlainTextResponse)
def get_report_markdown() -> str:
    """Return the raw Markdown text of the latest generated report."""
    config = load_config()
    report_path = PROJECT_ROOT / config.get("report", {}).get("output_path", "output/report.md")

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not generated yet.")

    return report_path.read_text(encoding="utf-8")

@app.get("/report/data")
def get_report_data() -> dict[str, Any]:
    """Return the structured JSON data used to build the report."""
    config = load_config()
    try:
        signals = load_json_safe(config["signals"]["output_path"])
        model_pred = load_json_safe(config["modeling"]["output_latest_prediction_path"])
        model_metrics = load_json_safe(config["modeling"]["output_metrics_path"])
        news_summary = load_json_safe(config["news"]["output_summary_json"])

        return {
            "signals": signals,
            "model_prediction": model_pred,
            "model_metrics": model_metrics,
            "news_summary": news_summary
        }
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Configuration error: missing key {e}") from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
