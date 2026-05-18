import os
import uvicorn

port = int(os.environ.get("PORT", 8000))

uvicorn.run(
    "main_v5:app",
    host="0.0.0.0",
    port=port
)
