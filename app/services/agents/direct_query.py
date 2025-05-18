"""Direct query handler for MongoDB that bypasses LLM for common query types."""
from typing import Dict, List, Any, Optional, Tuple, Union
import httpx
import json
import logging
import re

logger = logging.getLogger(__name__)

class DirectQueryHandler:
    """
    Direct query handler for MongoDB that parses specific query patterns
    without relying on an LLM. This ensures accuracy for common query types.
    """
    
    def __init__(self, db_name: str, base_url: Optional[str] = None, api_port: int = 8000):
        """
        Initialize the direct query handler.
        
        Args:
            db_name: The name of the database to query
            base_url: Optional base URL for the MCP server
            api_port: API port number (default: 8000)
        """
        self.db_name = db_name
        self.base_url = base_url or f"http://localhost:{api_port}/mcp/mongo"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.collections = []
        
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def get_collections(self) -> List[str]:
        """
        Get a list of all collections in the database.
        
        Returns:
            List of collection names
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/schema",
                json={"db_name": self.db_name},
                timeout=30.0
            )
            
            response.raise_for_status()
            schema_data = response.json()
            
            collections = [
                collection["collection_name"]
                for collection in schema_data.get("collections", [])
            ]
            
            logger.info(f"Found collections: {collections}")
            self.collections = collections
            return collections
        except Exception as e:
            logger.error(f"Error getting collections: {str(e)}")
            return []
    
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
            logger.info(f"Executing direct query: {json.dumps(query_params)}")
            response = await self.client.post(
                f"{self.base_url}/find",
                json=query_params,
                timeout=30.0
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error executing direct query: {str(e)}")
            return {
                "results": [],
                "collection_name": query_params.get("collection_name", ""),
                "total_count": 0,
                "error": str(e)
            }
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query into MongoDB query parameters.
        Uses a structured approach to extract collection name and conditions.
        
        Args:
            query: The natural language query
            
        Returns:
            MongoDB query parameters or None if parsing fails
        """
        # Try to parse the query
        try:
            logger.info(f"Attempting to parse query: {query}")
            
            # First, try to find the collection name
            collection_pattern = re.compile(
                r'in\s+(?:the\s+)?(\w+)(?:\s+collection)?',
                re.IGNORECASE
            )
            
            collection_match = collection_pattern.search(query)
            
            if not collection_match:
                logger.warning("Could not find collection name in query")
                return None
                
            collection_name = collection_match.group(1)
            logger.info(f"Found collection name: {collection_name}")
            
            # Next, try to extract the field and value
            # Remove the collection part from the query to avoid matching it as part of the value
            query_wo_collection = query.replace(collection_match.group(0), "").strip()
            
            # Try different field-value patterns
            field_patterns = [
                # Pattern for "with field value"
                r'with\s+(\w+)\s+([^,]+?)(?:,|\s+in\s+|$)',
                # Pattern for "where field is/= value"
                r'where\s+(\w+)\s+(?:is|=|==)\s+["\']?([^"\']+?)["\']?(?:,|\s+in\s+|$)',
                # Pattern for "field is/= value"
                r'(\w+)\s+(?:is|=|==)\s+["\']?([^"\']+?)["\']?(?:,|\s+in\s+|$)'
            ]
            
            field_name = None
            field_value = None
            
            for pattern in field_patterns:
                field_match = re.search(pattern, query_wo_collection, re.IGNORECASE)
                if field_match:
                    field_name, field_value = field_match.groups()
                    field_value = field_value.strip()
                    logger.info(f"Found field {field_name}={field_value}")
                    break
            
            # Build filter
            filter_query = {}
            if field_name and field_value:
                filter_query[field_name] = field_value
            
            # Build the MongoDB query parameters
            return {
                "db_name": self.db_name,
                "collection_name": collection_name,
                "filter": filter_query,
                "limit": 100
            }
            
        except Exception as e:
            logger.error(f"Error parsing query: {str(e)}", exc_info=True)
            return None
    
    async def process_direct_query(self, query: str) -> Tuple[str, Dict[str, Any], bool]:
        """
        Process a natural language query directly.
        
        Args:
            query: The natural language query
            
        Returns:
            Tuple of (explanation, results, was_handled)
        """
        # First, get available collections if we don't have them
        if not self.collections:
            await self.get_collections()
        
        # Parse the query into MongoDB parameters
        query_params = self.parse_query(query)
        
        if not query_params:
            return "", {}, False
        
        collection_name = query_params["collection_name"]
        
        # Check if the collection exists in our available collections
        if collection_name not in self.collections:
            # Try to find a case-insensitive match
            for coll in self.collections:
                if coll.lower() == collection_name.lower():
                    collection_name = coll
                    query_params["collection_name"] = coll
                    break
            else:
                logger.warning(f"Collection '{collection_name}' not found in available collections")
                return f"Could not find collection '{collection_name}'. Available collections are: {', '.join(self.collections)}", {}, True
        
        # Execute the query
        results = await self.execute_query(query_params)
        
        # Generate explanation
        count = len(results.get("results", []))
        
        explanation = f"Found {count} document(s) in the {collection_name} collection"
        
        if query_params["filter"]:
            filter_desc = ", ".join(f"{field} is '{value}'" for field, value in query_params["filter"].items())
            explanation += f" where {filter_desc}"
            
        explanation += "."
        
        if "username" in query_params["filter"] or "fullName" in query_params["filter"] or "name" in query_params["filter"]:
            # Special handling for user queries
            if count == 0:
                explanation += f" No users were found with the specified criteria in the {collection_name} collection."
            else:
                explanation += f" {count} user(s) matched your criteria in the {collection_name} collection."
        
        return explanation, results, True