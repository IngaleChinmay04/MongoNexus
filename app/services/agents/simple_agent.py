"""Simple agent for MongoDB queries (no LLM)."""
from typing import Dict, List, Any, Optional, Tuple
import httpx
import json
import re
import logging

logger = logging.getLogger(__name__)

class SimpleMongoDBAgent:
    """
    A simplified agent for MongoDB queries that doesn't use an LLM.
    This is useful for testing the API endpoints without requiring a Groq API key.
    """
    
    def __init__(self, db_name: str, base_url: Optional[str] = None):
        """Initialize the agent."""
        self.db_name = db_name
        self.base_url = base_url or "http://localhost:8000/mcp/mongo"
        self.client = httpx.AsyncClient()
        self.collections = {}
        
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
        
    async def get_schema(self) -> Dict[str, Any]:
        """Get the database schema."""
        response = await self.client.post(
            f"{self.base_url}/schema",
            json={"db_name": self.db_name}
        )
        
        schema_data = response.json()
        
        # Store collection names for later use
        self.collections = {
            collection["collection_name"]: collection["fields"]
            for collection in schema_data.get("collections", [])
        }
        
        return schema_data
    
    async def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query into MongoDB query parameters.
        This is a very simple implementation that looks for collection names in the query.
        """
        # Get schema if not already available
        if not self.collections:
            await self.get_schema()
            
        # Default to the first collection if we have any
        collection_name = next(iter(self.collections.keys())) if self.collections else "unknown"
        
        # Look for collection names in the query
        query_lower = query.lower()
        for coll_name in self.collections:
            if coll_name.lower() in query_lower:
                collection_name = coll_name
                break
        
        # Create a simple filter based on the query
        filter_query = {}
        
        # Look for simple conditions in the query
        # This is very basic and just for demonstration
        field_match = re.search(r'where\s+(\w+)\s*=\s*["\'](.*?)["\']', query, re.IGNORECASE)
        if field_match:
            field_name, value = field_match.groups()
            filter_query[field_name] = value
            
        # Return MongoDB query parameters
        return {
            "db_name": self.db_name,
            "collection_name": collection_name,
            "filter": filter_query,
            "limit": 10
        }
    
    async def execute_query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a MongoDB query."""
        response = await self.client.post(
            f"{self.base_url}/find",
            json=query_params
        )
        
        return response.json()
    
    async def process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Process a natural language query from start to finish."""
        # Parse the query
        mongo_params = await self.parse_query(query)
        
        # Execute the query
        results = await self.execute_query(mongo_params)
        
        # Generate a simple explanation
        result_count = len(results.get("results", []))
        explanation = f"Found {result_count} results in the '{mongo_params['collection_name']}' collection"
        if mongo_params['filter']:
            filter_conditions = ", ".join([f"{k}='{v}'" for k, v in mongo_params['filter'].items()])
            explanation += f" where {filter_conditions}"
        explanation += "."
        
        return explanation, results