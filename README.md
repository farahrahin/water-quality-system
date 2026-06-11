# PAIP Water Quality System v6

OCR-based water quality logbook digitization system for PAIP (Pengurusan Air Pahang).

## Environment Variables (set in Railway dashboard)

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (from Supabase) |
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_ANON_KEY` | Supabase anon/public key |
| `SUPABASE_JWT_SECRET` | Supabase JWT secret |
| `ROBOFLOW_API_KEY` | Roboflow API key for table detection |

## Files

- `main_v6.py` — FastAPI backend with calibrated OCR pipeline
- `index_v6.html` — Frontend UI
- `Dockerfile` — Uses CPU-only PyTorch to reduce memory usage
- `requirements_app.txt` — App dependencies (torch installed separately)
