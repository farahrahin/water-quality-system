import os
import uvicorn

port = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    uvicorn.run(
        "main_v5:app",
        host="0.0.0.0",
        port=port
    )
