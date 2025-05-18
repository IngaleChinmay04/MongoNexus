"""Server-Sent Events (SSE) service."""
from fastapi import Response
from typing import AsyncGenerator, Any, Dict, List, Optional, Union, AsyncIterable
import json
import asyncio
from app.utils.bson_helpers import BSONEncoder

class SSEResponse(Response):
    """Server-Sent Events response class."""
    media_type = "text/event-stream"

    def __init__(self, content: Any = None, **kwargs):
        """
        Initialize SSEResponse with default headers for SSE.
        
        Args:
            content: Content to stream
            **kwargs: Additional keyword arguments for Response
        """
        kwargs.setdefault("headers", {})
        kwargs["headers"].update({
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in Nginx
        })
        super().__init__(content=content, **kwargs)

async def format_sse_event(
    data: Any, 
    event: Optional[str] = None, 
    id: Optional[str] = None, 
    retry: Optional[int] = None
) -> str:
    """
    Format a Server-Sent Event message.
    
    Args:
        data: The data to send
        event: Optional event type
        id: Optional event ID
        retry: Optional retry interval in milliseconds
    
    Returns:
        Formatted SSE message
    """
    message = []
    
    if id is not None:
        message.append(f"id: {id}")
        
    if event is not None:
        message.append(f"event: {event}")
        
    if retry is not None:
        message.append(f"retry: {retry}")
    
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, cls=BSONEncoder)
    else:
        data_str = str(data)
        
    for line in data_str.splitlines():
        message.append(f"data: {line}")
    
    # End with a blank line to signal the end of the event
    message.append("")
    message.append("")
    
    return "\n".join(message)

async def stream_mongo_results(
    results_generator: AsyncIterable[Dict[str, Any]], 
    batch_size: int = 10, 
    delay: float = 0.1
) -> AsyncGenerator[str, None]:
    """
    Stream MongoDB results as SSE events.
    
    Args:
        results_generator: Async generator producing MongoDB documents
        batch_size: Number of documents to batch in each event
        delay: Delay between batches in seconds
    
    Yields:
        SSE formatted events
    """
    batch = []
    count = 0
    
    try:
        async for doc in results_generator:
            batch.append(doc)
            count += 1
            
            # Send a batch when it reaches the specified size
            if len(batch) >= batch_size:
                event_data = {
                    "batch": batch,
                    "count": count,
                    "batch_size": len(batch),
                    "status": "streaming"
                }
                
                yield await format_sse_event(
                    data=event_data,
                    event="batch"
                )
                
                # Clear the batch for the next chunk
                batch = []
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(delay)
        
        # Send any remaining documents in the final batch
        if batch:
            event_data = {
                "batch": batch,
                "count": count,
                "batch_size": len(batch),
                "status": "streaming"
            }
            
            yield await format_sse_event(
                data=event_data,
                event="batch"
            )
        
        # Send completion event
        yield await format_sse_event(
            data={"status": "complete", "total_count": count},
            event="complete"
        )
        
    except Exception as e:
        # Send error event
        error_data = {"status": "error", "message": str(e)}
        yield await format_sse_event(
            data=error_data,
            event="error"
        )

async def document_generator(
    db, 
    collection_name: str,
    filter_query: Dict[str, Any],
    projection: Optional[Dict[str, Any]] = None,
    sort: Optional[List[Dict[str, int]]] = None,
    skip: int = 0,
    limit: Optional[int] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate MongoDB documents from a query.
    
    Args:
        db: MongoDB database client
        collection_name: Name of the collection to query
        filter_query: MongoDB filter query document
        projection: Optional MongoDB projection document
        sort: Optional sort criteria
        skip: Optional number of documents to skip
        limit: Optional maximum number of documents to return
        
    Yields:
        MongoDB documents
    """
    from app.services.mongodb.query_service import parse_query_object_ids, serialize_mongo_doc
    
    # Get the collection
    collection = db[collection_name]
    
    # Parse ObjectIds in filter_query if they're in string format
    filter_query = await parse_query_object_ids(filter_query)
    
    # Prepare find options
    find_options = {}
    
    if projection:
        find_options["projection"] = projection
        
    # Get cursor
    cursor = collection.find(filter_query, **find_options)
    
    # Apply skip and limit
    if skip > 0:
        cursor = cursor.skip(skip)
    
    if limit:
        cursor = cursor.limit(limit)
    
    # Apply sort if provided
    if sort:
        # Convert sort list of dicts to list of tuples
        sort_list = []
        for sort_item in sort:
            for field, direction in sort_item.items():
                sort_list.append((field, direction))
        
        cursor = cursor.sort(sort_list)
    
    # Yield documents
    async for doc in cursor:
        serialized_doc = serialize_mongo_doc(doc)
        yield serialized_doc