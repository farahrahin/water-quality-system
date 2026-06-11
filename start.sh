#!/bin/sh
exec uvicorn main_v6:app --host=0.0.0.0 --port=${PORT:-8000}
