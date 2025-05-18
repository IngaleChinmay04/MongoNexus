"""MongoDB MCP endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from app.services.mongodb.client import get_database
from app.schemas.request.mongo_request import MongoFindRequest, MongoSchemaRequest
from app.schemas.response.mongo_response import MongoSchemaResponse, MongoFindResponse, CollectionSchema
from app.services.mongodb.schema_service import get_database_schema
from app.services.mongodb.query_service import find_documents, count_documents
from typing import List, Dict, Any, Optional

router = APIRouter()

@router.post("/schema", response_model=MongoSchemaResponse)
async def get_mongo_schema(request: MongoSchemaRequest):
    """
    Get the schema of MongoDB collections by sampling documents.
    
    - If collection_name is provided, returns schema for just that collection
    - Otherwise returns schema for all collections in the database
    """
    try:
        # Connect to the specified database
        db = await get_database(request.db_name)
        
        # Get schema
        schema_data = await get_database_schema(db, request.collection_name)
        
        return MongoSchemaResponse(**schema_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get schema: {str(e)}"
        )

@router.post("/find", response_model=MongoFindResponse)
async def find_mongo_documents(request: MongoFindRequest):
    """
    Find documents in a MongoDB collection.
    
    - collection_name: The name of the collection to query
    - filter: MongoDB filter query document
    - projection: Optional MongoDB projection document
    - sort: Optional sort criteria
    - skip: Optional number of documents to skip
    - limit: Optional maximum number of documents to return
    """
    try:
        # Connect to the specified database
        db = await get_database(request.db_name)
        
        # Execute the query
        results = await find_documents(
            db,
            request.collection_name,
            request.filter,
            request.projection,
            request.sort,
            request.skip,
            request.limit
        )
        
        # Get the total count of matching documents (without skip/limit)
        total_count = await count_documents(db, request.collection_name, request.filter)
        
        return MongoFindResponse(
            results=results,
            count=len(results),
            total_count=total_count,
            database_name=request.db_name,
            collection_name=request.collection_name
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query documents: {str(e)}"
        )