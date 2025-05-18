"""Executor Agent for MongoDB queries."""
from typing import Dict, List, Any, Optional, Tuple, Union, AsyncGenerator
import httpx
import json
import asyncio
import logging
from app.services.agents.llm_service import LLMService
from app.config.settings import API_HOST, API_PORT

logger = logging.getLogger(__name__)

class MongoDBExecutorAgent:
    """
    Agent for executing MongoDB operations based on natural language queries.
    This agent handles:
    1. Understanding the collection schema
    2. Converting natural language to MongoDB queries
    3. Executing queries against the MCP server
    4. Interpreting results
    """
    
    def __init__(self, db_name: str, base_url: Optional[str] = None):
        """
        Initialize the MongoDB Executor Agent.
        
        Args:
            db_name: The name of the database to query
            base_url: Optional base URL for the MCP server
        """
        self.db_name = db_name
        # Since we're running in the same process, we can use localhost
        self.base_url = base_url or f"http://localhost:{API_PORT}/mcp/mongo"
        logger.info(f"Initializing MongoDB Executor Agent with base_url: {self.base_url}")
        
        # Initialize HTTP client with proper timeout and limits
        self.client = httpx.AsyncClient(timeout=30.0)  # 30 second timeout
        self.collection_schemas = {}
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def get_database_schema(self) -> Dict[str, Any]:
        """
        Fetch the schema of all collections in the database.
        
        Returns:
            Database schema information
        """
        try:
            logger.info(f"Fetching schema for database {self.db_name} from {self.base_url}/schema")
            response = await self.client.post(
                f"{self.base_url}/schema",
                json={"db_name": self.db_name},
                timeout=30.0  # Explicit timeout
            )
            
            response.raise_for_status()
            schema_data = response.json()
            
            # Cache the schema information
            self.collection_schemas = {
                collection["collection_name"]: collection["fields"]
                for collection in schema_data.get("collections", [])
            }
            
            logger.info(f"Successfully retrieved schema with {len(schema_data.get('collections', []))} collections")
            return self.collection_schemas
            
        except httpx.ConnectError as e:
            logger.error(f"Connection error when getting schema: {str(e)}")
            # Provide a minimal schema as fallback
            return {"error": f"Connection error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error getting schema: {str(e)}", exc_info=True)
            return {"error": f"Schema error: {str(e)}"}
    
    async def execute_query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a MongoDB query via the MCP server.
        
        Args:
            query_params: MongoDB query parameters
            
        Returns:
            Query results
        """
        # Ensure db_name is included
        if "db_name" not in query_params:
            query_params["db_name"] = self.db_name
            
        try:
            logger.info(f"Executing query against {self.base_url}/find: {json.dumps(query_params)}")
            response = await self.client.post(
                f"{self.base_url}/find",
                json=query_params,
                timeout=30.0
            )
            
            response.raise_for_status()
            results = response.json()
            logger.info(f"Query executed successfully, received {len(results.get('results', []))} results")
            return results
            
        except httpx.ConnectError as e:
            logger.error(f"Connection error executing query: {str(e)}")
            # Return empty results as fallback
            return {
                "results": [],
                "collection_name": query_params.get("collection_name", ""),
                "total_count": 0,
                "error": f"Connection error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}", exc_info=True)
            return {
                "results": [],
                "collection_name": query_params.get("collection_name", ""),
                "total_count": 0,
                "error": f"Query error: {str(e)}"
            }
    
    async def process_query(self, user_query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a natural language query from start to finish.
        
        Args:
            user_query: The user's natural language query
            
        Returns:
            Tuple of (explanation, raw_results)
        """
        try:
            # Step 1: Get database schema
            schema_data = await self.get_database_schema()
            if "error" in schema_data:
                return f"Error retrieving schema: {schema_data['error']}", {"results": []}
            
            # If we have no collections, return an error message
            if not schema_data:
                return "No collections found in the database. Please check your database name and connection.", {"results": []}
            
            # Step 2: Parse the user's query into MongoDB parameters
            logger.info("Parsing user query with Groq LLM")
            mongo_params = await LLMService.parse_user_query(user_query, schema_data)
            logger.info(f"Query parsed into MongoDB parameters: {json.dumps(mongo_params)}")
            
            # Step 3: Execute the query against the MCP server
            query_params = {
                "db_name": self.db_name,
                "collection_name": mongo_params["collection_name"],
                "filter": mongo_params.get("filter", {}),
                "projection": mongo_params.get("projection"),
                "sort": mongo_params.get("sort"),
                "skip": mongo_params.get("skip", 0),
                "limit": mongo_params.get("limit", 10)
            }
            
            results = await self.execute_query(query_params)
            
            # Step 4: Generate a natural language explanation of the results
            logger.info("Generating explanation with Groq LLM")
            explanation = await LLMService.generate_explanation(
                user_query, 
                results.get("results", []),
                mongo_params["collection_name"]
            )
            
            # Include the mongo_params in the results for transparency
            results["mongo_params"] = mongo_params
            
            return explanation, results
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}", exc_info=True)
            return f"Error processing query: {str(e)}", {"results": []}