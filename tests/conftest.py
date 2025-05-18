"""Test configuration and fixtures."""
import pytest
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from app.api.app import create_app
from fastapi.testclient import TestClient

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def app():
    """Create a test app."""
    return create_app()

@pytest.fixture
def client(app):
    """Create a test client."""
    with TestClient(app) as client:
        yield client

@pytest.fixture
async def test_mongodb_client():
    """Create a test MongoDB client."""
    # Use a test database
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    test_db = client["test_agentic_ai_platform"]
    
    # Clean database before tests
    collections = await test_db.list_collection_names()
    for collection in collections:
        await test_db[collection].drop()
        
    yield test_db
    
    # Clean up after tests
    collections = await test_db.list_collection_names()
    for collection in collections:
        await test_db[collection].drop()
