"""MongoDB response schemas."""
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class CollectionSchema(BaseModel):
    """Schema of a MongoDB collection."""
    collection_name: str
    fields: Dict[str, str]  # Field name -> data type
    sample_size: int
    documents_sampled: int

class MongoSchemaResponse(BaseModel):
    """Response schema for MongoDB schema operation."""
    collections: List[CollectionSchema]
    database_name: str

class MongoFindResponse(BaseModel):
    """Response schema for MongoDB find operation."""
    results: List[Dict[str, Any]]
    count: int
    total_count: Optional[int] = None  # Total count regardless of limit
    database_name: str
    collection_name: str