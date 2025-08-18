from fastapi import FastAPI
from api.endpoints import router  # Import the router containing API endpoints
import uvicorn  # ASGI server used to run FastAPI applications

# Create an instance of the FastAPI application with a custom title
app = FastAPI(title="Anomaly Detection API")

# Register the router with the main application
app.include_router(router)

def main():
    """
    Launches the FastAPI application using Uvicorn.

    - host: "0.0.0.0" allows external access to the server
    - port: 8000 is the default port for the API
    - reload: True enables auto-reloading during development
    """
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Entry point for running the application directly
if __name__ == "__main__":
    main()

