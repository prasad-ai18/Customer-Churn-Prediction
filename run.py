"""Quick launcher script for local development."""
import uvicorn

if __name__ == "__main__":
    print("Starting Customer Churn Intelligence Platform on http://127.0.0.1:8000 ...")
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)
