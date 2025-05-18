"""MongoDB client service."""
from motor.motor_asyncio import AsyncIOMotorClient
from app.config.settings import MONGODB_URI
from typing import Optional, Dict, Any

# Global client instance
_mongodb_client: Optional[AsyncIOMotorClient] = None

def get_mongodb_client() -> AsyncIOMotorClient:
    """Get or create the global MongoDB client."""
    global _mongodb_client
    
    if _mongodb_client is None:
        _mongodb_client = AsyncIOMotorClient(MONGODB_URI)
    
    return _mongodb_client

async def get_database(db_name: str = None):
    """
    Get MongoDB database as a dependency.
    
    Args:
        db_name: The name of the database to connect to.
               If None, uses the default database from settings.
    """
    client = get_mongodb_client()
    
    if not db_name:
        from app.config.settings import MONGODB_DB_NAME
        db_name = MONGODB_DB_NAME
        
    return client[db_name]

# Create a connection pool for multiple databases
_database_connections: Dict[str, Any] = {}

async def get_database_connection(db_name: str):
    """
    Get a cached database connection for the specified database name.
    This helps optimize repeated connections to the same database.
    """
    global _database_connections
    
    if db_name not in _database_connections:
        client = get_mongodb_client()
        _database_connections[db_name] = client[db_name]
        
    return _database_connections[db_name]