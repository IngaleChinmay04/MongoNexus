"""Agent interaction endpoints."""
from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, Query, Depends, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from app.services.query_service import MongoDBQueryService
try:
    from app.services.graph_rag.graph_processor import GraphRAGProcessor
    GRAPH_RAG_AVAILABLE = True
except ImportError:
    GRAPH_RAG_AVAILABLE = False
    
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import asyncio
import os
from pathlib import Path
import traceback
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class QueryRequest(BaseModel):
    """Request schema for natural language queries."""
    db_name: str
    query: str
    stream: bool = False
    use_graph_rag: bool = True  # Enable/disable Graph RAG enhancement

class QueryResponse(BaseModel):
    """Response schema for natural language queries."""
    explanation: str
    results: List[Dict[str, Any]]
    mongo_query: Dict[str, Any]

@router.get("/interface", response_class=HTMLResponse)
async def get_agent_interface():
    """
    Get the agent interface HTML page.
    """
    # Read the HTML file content
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent  # Go up to project root
    html_path = project_root / "app" / "static" / "agent_interface.html"
    
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Agent interface HTML file not found")
    
    with open(html_path, "r", encoding="utf-8") as file:
        html_content = file.read()
    
    return HTMLResponse(content=html_content)

@router.post("/query")
async def process_query(request: QueryRequest):
    """
    Process a natural language query against a MongoDB database.
    
    The query is interpreted using advanced NLP techniques, converted to MongoDB operations,
    and the results are explained in natural language.
    """
    processor = None
    
    try:
        # Log the received request
        logger.info(f"Processing query for database: {request.db_name}")
        logger.info(f"Query text: {request.query}")
        logger.info(f"Using Graph RAG: {request.use_graph_rag}")
        
        # Determine whether to use Graph RAG
        use_graph_rag = request.use_graph_rag and GRAPH_RAG_AVAILABLE
        
        if use_graph_rag:
            try:
                # Use Graph RAG processor for enhanced understanding
                from app.config.settings import API_PORT
                processor = GraphRAGProcessor(request.db_name, api_port=API_PORT)
                
                # Initialize the processor
                if await processor.initialize():
                    # Process the query
                    explanation, query_params, results = await processor.process_query(request.query)
                else:
                    raise ValueError("Failed to initialize the Graph RAG processor")
            except Exception as rag_error:
                # Log the error
                logger.error(f"Graph RAG processing failed: {str(rag_error)}")
                logger.error(traceback.format_exc())
                
                # Fall back to schema-aware processor
                logger.info("Falling back to standard query processor")
                processor = MongoDBQueryService(request.db_name)
                explanation, query_params, results = await processor.process_nl_query(request.query)
        else:
            # Use schema-aware processor
            processor = MongoDBQueryService(request.db_name)
            explanation, query_params, results = await processor.process_nl_query(request.query)
        
        # Remove metadata from query params for response
        cleaned_query_params = {k: v for k, v in query_params.items() if not k.startswith('_')}
        
        # Check for complex query processing
        is_complex = query_params.get("_meta", {}).get("complex_query", False)
        
        # Check if this was a count query
        is_count = query_params.get("_meta", {}).get("intent") == "count"
        
        if is_complex:
            # For complex queries, format the response with insights
            complex_results = results.get("complex_reasoning", {})
            
            return {
                "explanation": explanation,
                "results": results.get("initial_results", {}).get("results", []),
                "mongo_query": {
                    "db_name": request.db_name,
                    "collection_name": cleaned_query_params.get("collection_name", ""),
                    "filter": cleaned_query_params.get("filter", {}),
                    "total_count": results.get("initial_results", {}).get("total_count", 0),
                    "is_complex_query": True,
                    "collections_queried": complex_results.get("collections_queried", [])
                }
            }
        elif is_count:
            # For count queries, include the count
            total_count = results.get("count", results.get("total_count", len(results.get("results", []))))
            
            return {
                "explanation": explanation,
                "results": results.get("results", []),
                "mongo_query": {
                    "db_name": request.db_name,
                    "collection_name": cleaned_query_params.get("collection_name", ""),
                    "filter": cleaned_query_params.get("filter", {}),
                    "total_count": total_count,
                    "count": total_count
                }
            }
        else:
            # Standard query response
            return {
                "explanation": explanation,
                "results": results.get("results", []),
                "mongo_query": {
                    "db_name": request.db_name,
                    "collection_name": cleaned_query_params.get("collection_name", ""),
                    "filter": cleaned_query_params.get("filter", {}),
                    "total_count": results.get("total_count", len(results.get("results", [])))
                }
            }
    except Exception as e:
        # Log the full traceback for debugging
        error_traceback = traceback.format_exc()
        logger.error(f"Error processing query: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        
        # Provide a helpful error message
        return {
            "explanation": f"An error occurred while processing your query: {str(e)}. Please try rephrasing your question with more specific details.",
            "results": [],
            "mongo_query": {
                "db_name": request.db_name,
                "collection_name": None,
                "filter": {},
                "total_count": 0,
                "error": str(e)
            }
        }
    finally:
        # Close the processor client
        if processor:
            await processor.close()