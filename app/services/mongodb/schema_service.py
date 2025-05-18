"""MongoDB schema service."""
from typing import Dict, List, Any, Optional
from app.config.settings import MONGODB_SCHEMA_SAMPLE_SIZE
import bson
from datetime import datetime
from bson import ObjectId

def get_bson_type_name(value: Any) -> str:
    """
    Get a human-readable type name for a BSON value.
    
    Args:
        value: The value to get the type for
        
    Returns:
        A string representing the BSON type
    """
    if isinstance(value, str):
        return "string"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "double"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, dict):
        return "document"
    elif isinstance(value, ObjectId):
        return "objectId"
    elif isinstance(value, datetime):
        return "date"
    elif isinstance(value, bson.Binary):
        return "binary"
    elif value is None:
        return "null"
    else:
        # Default fallback
        return type(value).__name__

async def infer_collection_schema(db, collection_name: str) -> Dict[str, Any]:
    """
    Infer the schema of a MongoDB collection by sampling documents.
    
    Args:
        db: MongoDB database client
        collection_name: Name of the collection to infer schema for
        
    Returns:
        Dict containing collection name, field types, and sample size
    """
    sample_size = MONGODB_SCHEMA_SAMPLE_SIZE
    field_types = {}
    documents_sampled = 0
    
    # Get the collection
    collection = db[collection_name]
    
    # Check if collection exists and has documents
    count = await collection.count_documents({})
    if count == 0:
        return {
            "collection_name": collection_name,
            "fields": {},
            "sample_size": 0,
            "documents_sampled": 0
        }
    
    # Sample documents to infer schema
    cursor = collection.find().limit(sample_size)
    
    # Process each document to infer field types
    async for doc in cursor:
        documents_sampled += 1
        await process_document(doc, field_types)
    
    return {
        "collection_name": collection_name,
        "fields": field_types,
        "sample_size": sample_size,
        "documents_sampled": documents_sampled
    }

async def process_document(doc: Dict[str, Any], field_types: Dict[str, Any], prefix: str = ""):
    """
    Process a document to extract field types.
    
    Args:
        doc: The document to process
        field_types: Dictionary to update with field types
        prefix: Prefix for nested document fields
    """
    for field_name, value in doc.items():
        full_field_name = f"{prefix}{field_name}" if prefix else field_name
        
        # Get the type of the value
        field_type = get_bson_type_name(value)
        
        # Handle nested documents
        if isinstance(value, dict) and field_name != '_id':
            # Add this as a document type
            if full_field_name not in field_types:
                field_types[full_field_name] = field_type
            
            # Recursively process nested document
            await process_document(value, field_types, f"{full_field_name}.")
            continue
            
        # Handle arrays - check first element for type
        if isinstance(value, list) and value:
            array_type = f"array<{get_bson_type_name(value[0])}>"
            
            # If all elements are of same type, use that specific array type
            all_same_type = all(isinstance(item, type(value[0])) for item in value)
            if not all_same_type:
                array_type = "array<mixed>"
                
            # For arrays of documents, we could recursively process them
            # but for simplicity we'll just mark them as document arrays
            if isinstance(value[0], dict):
                array_type = "array<document>"
                
            field_type = array_type
        
        # If we haven't seen this field before, add it
        if full_field_name not in field_types:
            field_types[full_field_name] = field_type
        # If we have seen this field but the type is different, mark as "mixed"
        elif field_types[full_field_name] != field_type:
            field_types[full_field_name] = "mixed"

async def list_collections(db) -> List[str]:
    """
    List all collections in the database.
    
    Args:
        db: MongoDB database client
        
    Returns:
        List of collection names
    """
    return await db.list_collection_names()

async def get_database_schema(db, collection_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Get schema for all collections or a specific collection.
    
    Args:
        db: MongoDB database client
        collection_filter: Optional collection name to filter by
        
    Returns:
        Dict containing database schema information
    """
    collections = []
    
    if collection_filter:
        # Get schema for a specific collection
        schema = await infer_collection_schema(db, collection_filter)
        collections.append(schema)
    else:
        # Get schema for all collections
        collection_names = await list_collections(db)
        for collection_name in collection_names:
            schema = await infer_collection_schema(db, collection_name)
            collections.append(schema)
    
    return {
        "collections": collections,
        "database_name": db.name
    }