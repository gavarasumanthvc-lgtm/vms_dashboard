import os
import tempfile

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from audit_store import AuditStore
from vms_sop_analyzer import VMSSOPAnalyzer

app = FastAPI(title="VMS SOP Analyzer API", version="1.0.0")
store = AuditStore()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "vms-sop-analyzer",
        "version": "1.0.0",
    }


@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    process_type: str = Form("Return"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    if process_type not in {"Return", "Forward"}:
        raise HTTPException(status_code=400, detail="process_type must be 'Return' or 'Forward'.")

    temp_path = None
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        analyzer = VMSSOPAnalyzer(process_type=process_type)
        result = analyzer.analyze(temp_path)
        record_id = store.save(file.filename, process_type, result)
        result["record_id"] = record_id
        return {"status": "ok", "result": result}
    except Exception as exc:  # pragma: no cover - defensive production guard
        raise HTTPException(status_code=400, detail=f"Unable to analyze video: {str(exc)}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/results")
def list_results(limit: int = 20):
    return {"results": store.list_recent(limit=max(1, min(limit, 100)))}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
