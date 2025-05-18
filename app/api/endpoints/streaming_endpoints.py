"""Streaming endpoints for MongoDB data."""
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from app.services.mongodb.client import get_database
from app.schemas.request.mongo_request import MongoFindRequest
from app.services.mongodb.query_service import count_documents
from app.services.streaming.sse_service import stream_mongo_results, document_generator, format_sse_event
from typing import List, Dict, Any, Optional

router = APIRouter()

@router.post("/find")
async def stream_mongo_find(request: MongoFindRequest, batch_size: int = Query(10, ge=1, le=100)):
    """
    Stream documents from a MongoDB collection using Server-Sent Events.
    
    - db_name: The name of the database to connect to
    - collection_name: The name of the collection to query
    - filter: MongoDB filter query document
    - projection: Optional MongoDB projection document
    - sort: Optional sort criteria
    - skip: Optional number of documents to skip
    - limit: Optional maximum number of documents to return
    - batch_size: Number of documents to include in each SSE event
    """
    try:
        # Connect to the specified database
        db = await get_database(request.db_name)
        
        # Get the total count (for progress tracking)
        total_count = await count_documents(db, request.collection_name, request.filter)
        
        # Create a document generator
        doc_gen = document_generator(
            db,
            request.collection_name,
            request.filter,
            request.projection,
            request.sort,
            request.skip,
            request.limit
        )
        
        # Create streaming response
        async def event_generator():
            # Send initial metadata as first event
            metadata = {
                "status": "started",
                "total_count": total_count,
                "database": request.db_name,
                "collection": request.collection_name
            }
            yield await format_sse_event(data=metadata, event="metadata")
            
            # Stream results
            async for event_text in stream_mongo_results(doc_gen, batch_size):
                yield event_text
        
        return StreamingResponse(
            content=event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable buffering in Nginx
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stream documents: {str(e)}"
        )

@router.post("/aggregate")
async def stream_mongo_aggregate(request: Request, batch_size: int = Query(10, ge=1, le=100)):
    """
    Stream results of a MongoDB aggregation pipeline using Server-Sent Events.
    
    Request body:
    - db_name: The name of the database to connect to
    - collection_name: The name of the collection to query
    - pipeline: MongoDB aggregation pipeline
    - batch_size: Number of documents to include in each SSE event
    """
    try:
        # Parse request body
        data = await request.json()
        db_name = data.get("db_name")
        collection_name = data.get("collection_name")
        pipeline = data.get("pipeline", [])
        
        if not db_name or not collection_name:
            raise HTTPException(
                status_code=400,
                detail="db_name and collection_name are required"
            )
            
        if not isinstance(pipeline, list):
            raise HTTPException(
                status_code=400,
                detail="pipeline must be an array of aggregation stages"
            )
        
        # Connect to the specified database
        db = await get_database(db_name)
        
        # Get the collection
        collection = db[collection_name]
        
        # Create an async generator for aggregation results
        async def agg_generator():
            from app.services.mongodb.query_service import serialize_mongo_doc
            
            # Execute the aggregation
            cursor = collection.aggregate(pipeline)
            
            # Yield documents
            async for doc in cursor:
                serialized_doc = serialize_mongo_doc(doc)
                yield serialized_doc
        
        # Create streaming response
        async def event_generator():
            # Send initial metadata
            metadata = {
                "status": "started",
                "database": db_name,
                "collection": collection_name
            }
            yield await format_sse_event(data=metadata, event="metadata")
            
            # Stream results
            async for event_text in stream_mongo_results(agg_generator(), batch_size):
                yield event_text
        
        return StreamingResponse(
            content=event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable buffering in Nginx
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stream aggregation results: {str(e)}"
        )