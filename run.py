import uvicorn

if __name__ == "__main__":
    uvicorn.run("main_v5:app", host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 8000)))
