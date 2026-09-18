# VMS SOP Analyzer

This project provides a computer-vision SOP review tool for logistics inspection videos.

## Components

- `vms_sop_analyzer.py`: core analysis engine
- `app.py`: Streamlit demo UI
- `server.py`: FastAPI backend for other applications
- `audit_store.py`: SQLite-based result storage for review tracking

## Run the API locally

```bash
pip install -r requirements.txt
python server.py
```

Then open:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Example upload

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@sample_video.mp4" \
  -F "process_type=Return"
```

## Run the Streamlit UI

```bash
streamlit run app.py
```

## Production note

This project is configured as a review-first system. Videos with uncertain detections are marked as `review` rather than being auto-approved.
