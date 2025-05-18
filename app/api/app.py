"""Main FastAPI application factory."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
from app.api.endpoints.mongo_endpoints import router as mongo_router
from app.api.endpoints.streaming_endpoints import router as streaming_router
from app.api.endpoints.agent_endpoints import router as agent_router
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MongoDB Integration for Unified Agentic AI Query Platform",
        description="A service that allows AI agents to interact with MongoDB through a Model Context Protocol",
        version="0.1.0",
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(mongo_router, prefix="/mcp/mongo", tags=["MongoDB MCP"])
    app.include_router(streaming_router, prefix="/mcp/mongo/stream", tags=["MongoDB Streaming"])
    app.include_router(agent_router, prefix="/agent", tags=["AI Agent"])
    
    # Mount static files - find the correct path to the static directory
    # Get the directory where this script is located
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent  # Go up to project root
    static_dir = project_root / "app" / "static"
    
    logger.info(f"Static directory path: {static_dir}")
    
    if static_dir.exists():
        logger.info(f"Mounting static files from: {static_dir}")
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    else:
        logger.error(f"Static directory not found at: {static_dir}")
    
    @app.get("/", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok", "message": "MongoDB MCP server is running"}
        
    @app.on_event("startup")
    async def startup_event():
        """Startup event handler."""
        logger.info("MongoDB MCP server is starting up...")
        
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown event handler."""
        logger.info("MongoDB MCP server is shutting down...")
        # Close MongoDB connections if needed
        
    return app