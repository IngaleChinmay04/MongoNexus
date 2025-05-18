"""Query service for processing natural language queries to MongoDB."""
import httpx
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from app.services.schema_aware_processor import SchemaAwareProcessor, QueryIntent

logger = logging.getLogger(__name__)

class MongoDBQueryService:
    """
    Service for processing natural language queries to MongoDB.
    Handles the full pipeline from language processing to query execution.
    """
    
    def __init__(self, db_name: str, base_url: Optional[str] = None, api_port: int = 8000):
        """
        Initialize the MongoDB Query Service.
        
        Args:
            db_name: The database name
            base_url: Optional base URL for the MCP server
            api_port: API port number (default: 8000)
        """
        self.db_name = db_name
        self.base_url = base_url or f"http://localhost:{api_port}/mcp/mongo"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.schema = {}
        self.processor = None
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def get_schema(self) -> Dict[str, Any]:
        """
        Get the database schema.
        
        Returns:
            Database schema
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/schema",
                json={"db_name": self.db_name},
                timeout=30.0
            )
            
            response.raise_for_status()
            schema_data = response.json()
            
            # Process the schema into a more usable format
            # Convert from [{collection_name: X, fields: {...}}, ...] to {collection_name: fields, ...}
            self.schema = {}
            for collection in schema_data.get("collections", []):
                self.schema[collection["collection_name"]] = collection["fields"]
                
            logger.info(f"Retrieved schema with {len(self.schema)} collections")
            return self.schema
            
        except Exception as e:
            logger.error(f"Error getting schema: {str(e)}", exc_info=True)
            return {}
    
    async def execute_query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a MongoDB query.
        
        Args:
            query_params: MongoDB query parameters
            
        Returns:
            Query results
        """
        # Remove metadata from the query params before sending
        params_to_send = {k: v for k, v in query_params.items() if not k.startswith('_')}
        
        # For count operations, we still need to return documents
        is_count = query_params.get("_meta", {}).get("intent") == QueryIntent.COUNT
        if is_count:
            # For count queries, use a reasonable limit that will likely return all documents
            # But we should be cautious not to overload the system
            params_to_send["limit"] = 1000
        
        try:
            logger.info(f"Executing query: {json.dumps(params_to_send)}")
            response = await self.client.post(
                f"{self.base_url}/find",
                json=params_to_send,
                timeout=30.0
            )
            
            response.raise_for_status()
            results = response.json()
            
            # For count queries, we care about the total_count
            if is_count:
                # If total_count is not provided, use the length of results
                total_count = results.get("total_count", len(results.get("results", [])))
                results["count"] = total_count
                
            return results
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}", exc_info=True)
            return {
                "results": [],
                "error": str(e),
                "collection_name": query_params.get("collection_name", ""),
                "total_count": 0
            }
    
    def generate_explanation(self, query: str, query_params: Dict[str, Any], results: Dict[str, Any]) -> str:
        """
        Generate a natural language explanation of the query results.
        
        Args:
            query: The original query
            query_params: The query parameters
            results: The query results
            
        Returns:
            Natural language explanation
        """
        intent = query_params.get("_meta", {}).get("intent", "unknown")
        collection_name = query_params.get("collection_name", "unknown")
        filter_conditions = query_params.get("filter", {})
        
        # For count queries, we should use the total_count or count
        if intent == QueryIntent.COUNT:
            result_count = results.get("count", results.get("total_count", len(results.get("results", []))))
            explanation = f"There are {result_count} document(s) in the {collection_name} collection"
        else:
            result_count = results.get("total_count", len(results.get("results", [])))
            explanation = f"Found {result_count} document(s) in the {collection_name} collection"
            
        # Add filter description
        if filter_conditions:
            if "$or" in filter_conditions:
                # Handle $or conditions more naturally
                or_conditions = filter_conditions["$or"]
                field_values = []
                
                for condition in or_conditions:
                    for field, value in condition.items():
                        if isinstance(value, dict) and "$in" in value:
                            field_values.append(f"{field} contains '{value['$in'][0]}'")
                        else:
                            field_values.append(f"{field} is '{value}'")
                
                explanation += f" where {' or '.join(field_values)}"
            else:
                filter_desc = []
                for field, value in filter_conditions.items():
                    if isinstance(value, dict) and "$in" in value:
                        filter_desc.append(f"{field} contains '{value['$in'][0]}'")
                    else:
                        filter_desc.append(f"{field} is '{value}'")
                
                explanation += f" where {', '.join(filter_desc)}"
            
        explanation += "."
        
        # Add additional context based on results
        if result_count == 0 and intent != QueryIntent.COUNT:
            explanation += f" No matching documents were found in the {collection_name} collection with your criteria."
        elif intent != QueryIntent.COUNT and 1 <= result_count <= 3:
            # For small result sets, describe the results briefly
            explanation += " Here are the details of the matching document(s)."
        elif intent != QueryIntent.COUNT and result_count > 3:
            # For larger result sets, summarize
            explanation += f" Showing {min(len(results.get('results', [])), query_params.get('limit', 10))} out of {result_count} matching documents."
            
        return explanation
    
    async def process_nl_query(self, query: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Process a natural language query from start to finish.
        
        Args:
            query: The natural language query
            
        Returns:
            Tuple of (explanation, query_params, results)
        """
        # Ensure we have schema information
        if not self.schema:
            await self.get_schema()
            
        # Initialize the processor with the schema
        self.processor = SchemaAwareProcessor(self.db_name, self.schema)
        
        # Process the query
        query_params = self.processor.process_query(query)
        
        # Check for errors
        if "error" in query_params:
            return query_params["error"], query_params, {"results": [], "error": query_params["error"]}
            
        # Execute the query
        results = await self.execute_query(query_params)
        
        # Generate explanation
        explanation = self.generate_explanation(query, query_params, results)
        
        return explanation, query_params, results