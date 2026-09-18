# api/server.py
# Thin FastAPI entry point. All endpoints live in api.routes.
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router, init_engines, verifier_engine

logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Agent Live Call API")

# Enable CORS for the Streamlit web UI (same machine / LAN client).
# Note: wildcard origins must NOT send credentials, so allow_credentials is off.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Re-exported for backward compatibility with ui/app.py
__all__ = ["app", "init_engines", "verifier_engine"]