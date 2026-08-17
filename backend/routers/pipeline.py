"""
POST /api/run-pipeline — same subprocess.run(["scripts/run_prototype.py"])
call the Streamlit "Run Pipeline" button used, ported as-is.
"""
import sys
import subprocess

from fastapi import APIRouter

router = APIRouter()


@router.post("/run-pipeline")
def run_pipeline():
    result = subprocess.run(
        [sys.executable, "scripts/run_prototype.py"], capture_output=True, text=True
    )
    if result.returncode == 0:
        return {"success": True}
    else:
        with open("logs/pipeline_error.log", "w") as f:
            f.write(result.stderr)
        return {"success": False, "logPath": "logs/pipeline_error.log"}
