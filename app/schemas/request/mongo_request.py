"""MongoDB request schemas."""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class MongoBaseRequest(BaseModel):
    """Base request schema for MongoDB operations."""
    db_name: str = Field(..., description="The name of the database to connect to")

class MongoFindRequest(MongoBaseRequest):
    """Request schema for MongoDB find operation."""
    collection_name: str = Field(..., description="The name of the collection to query")
    filter: Dict[str, Any] = Field(default_factory=dict, description="MongoDB filter query document")
    projection: Optional[Dict[str, Any]] = Field(None, description="MongoDB projection document")
    sort: Optional[List[Dict[str, int]]] = Field(None, description="MongoDB sort criteria")
    skip: Optional[int] = Field(0, ge=0, description="Number of documents to skip")
    limit: Optional[int] = Field(100, ge=1, le=1000, description="Maximum number of documents to return")

class MongoSchemaRequest(MongoBaseRequest):
    """Request schema for MongoDB schema operation."""
    collection_name: Optional[str] = Field(None, description="Optional name of the collection to get schema for")