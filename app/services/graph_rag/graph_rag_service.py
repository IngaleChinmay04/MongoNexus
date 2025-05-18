"""
Graph RAG Service for enhanced natural language query processing.
Coordinates knowledge graph, vector store, and MongoDB operations.
"""
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import httpx
import json
from app.services.graph_rag.knowledge_graph import KnowledgeGraph
from app.services.graph_rag.vector_store import VectorStore
from app.config.settings import VECTOR_STORE_PATH

logger = logging.getLogger(__name__)

class GraphRAGService:
    """
    Graph RAG Service that enhances query understanding through graph-based
    retrieval augmented generation.
    """
    
    def __init__(self, db_name: str, base_url: Optional[str] = None, api_port: int = 8000):
        """
        Initialize the Graph RAG Service.
        
        Args:
            db_name: Database name
            base_url: Optional base URL for the MCP server
            api_port: API port number (default: 8000)
        """
        self.db_name = db_name
        self.base_url = base_url or f"http://localhost:{api_port}/mcp/mongo"
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Initialize knowledge graph and vector store
        self.kg = KnowledgeGraph()
        self.vs = VectorStore()
        
        # Store database schema
        self.schema = {}
        
    async def close(self):
        """Close connections and resources."""
        await self.client.aclose()
        self.kg.close()
        
    async def initialize_graph_rag(self) -> bool:
        """
        Initialize the graph RAG system with schema information and sample data.
        
        Returns:
            True if successful, False otherwise
        """
        # Step 1: Get database schema
        schema = await self.get_schema()
        if not schema:
            logger.error("Failed to get database schema")
            return False
            
        # Step 2: Create knowledge graph
        logger.info("Creating knowledge graph from schema")
        kg_success = self.kg.create_schema_graph(self.db_name, schema)
        
        # Step 3: Create vector embeddings
        logger.info("Creating vector embeddings from schema")
        self.vs.create_index()
        self.vs.add_schema_embeddings(self.db_name, schema)
        
        # Step 4: Add example data to improve understanding
        await self.add_example_data()
        
        # Step 5: Save vector store for future use
        if VECTOR_STORE_PATH:
            os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
            self.vs.save(VECTOR_STORE_PATH)
        
        return kg_success
    
    async def get_schema(self) -> Dict[str, Any]:
        """
        Get the database schema.
        
        Returns:
            Database schema (collection -> fields mapping)
        """
        try:
            response = await self.client.post(
                f"{self.base_url}/schema",
                json={"db_name": self.db_name},
                timeout=30.0
            )
            
            response.raise_for_status()
            schema_data = response.json()
            
            # Convert from [{collection_name: X, fields: {...}}, ...] to {collection_name: fields, ...}
            self.schema = {}
            for collection in schema_data.get("collections", []):
                self.schema[collection["collection_name"]] = collection["fields"]
                
            logger.info(f"Retrieved schema with {len(self.schema)} collections")
            return self.schema
            
        except Exception as e:
            logger.error(f"Error getting schema: {str(e)}", exc_info=True)
            return {}
    
    async def add_example_data(self):
        """
        Add example data from each collection to the knowledge graph and vector store.
        """
        for collection_name in self.schema:
            try:
                # Get sample documents
                response = await self.client.post(
                    f"{self.base_url}/find",
                    json={
                        "db_name": self.db_name,
                        "collection_name": collection_name,
                        "limit": 10  # Limit to 10 examples per collection
                    },
                    timeout=30.0
                )
                
                response.raise_for_status()
                results = response.json()
                examples = results.get("results", [])
                
                if examples:
                    # Add to knowledge graph
                    self.kg.add_entity_examples(collection_name, examples)
                    
                    # Add to vector store
                    self.vs.add_example_embeddings(collection_name, examples)
                    
                    logger.info(f"Added {len(examples)} example documents for {collection_name}")
                    
            except Exception as e:
                logger.error(f"Error adding examples for {collection_name}: {str(e)}")
    
    def enhance_query_understanding(self, query: str, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance query understanding using knowledge graph and vector store.
        
        Args:
            query: Natural language query
            query_params: Initial MongoDB query parameters
            
        Returns:
            Enhanced query parameters
        """
        enhanced_params = query_params.copy()
        
        try:
            # Step 1: Get vector search suggestions
            vector_suggestions = self.vs.get_query_suggestions(query)
            
            # Step 2: Get knowledge graph suggestions
            kg_suggestions = self.kg.get_query_suggestions(query)
            
            # Step 3: Combine suggestions to improve collection detection
            if not enhanced_params.get("collection_name") and vector_suggestions.get("collections"):
                enhanced_params["collection_name"] = vector_suggestions["collections"][0]
                
            # Step 4: Enhance filter conditions
            if enhanced_params.get("collection_name"):
                collection = enhanced_params["collection_name"]
                
                # If fields were detected but not in filter, consider adding them
                suggested_fields = [f["field"] for f in vector_suggestions.get("fields", []) 
                                   if f["collection"] == collection]
                
                # Extract potential values from the query using field suggestions
                if suggested_fields and not enhanced_params.get("filter"):
                    enhanced_params["filter"] = self._extract_field_values_from_query(
                        query, collection, suggested_fields
                    )
                    
            # Step 5: Add metadata about the enhancement
            if "_meta" not in enhanced_params:
                enhanced_params["_meta"] = {}
                
            enhanced_params["_meta"]["enhanced"] = True
            enhanced_params["_meta"]["vector_suggestions"] = vector_suggestions
            enhanced_params["_meta"]["kg_suggestions"] = kg_suggestions
            
            return enhanced_params
            
        except Exception as e:
            logger.error(f"Error enhancing query understanding: {str(e)}", exc_info=True)
            return query_params
    
    def _extract_field_values_from_query(self, query: str, collection: str, suggested_fields: List[str]) -> Dict[str, Any]:
        """
        Extract field values from the query based on suggested fields.
        
        Args:
            query: Natural language query
            collection: Collection name
            suggested_fields: List of suggested fields
            
        Returns:
            Filter conditions
        """
        filters = {}
        
        # Get field suggestions from the knowledge graph
        for field in suggested_fields:
            # Look for patterns like "<field> <operator> <value>"
            patterns = [
                rf"{field}\s+(?:is|equals|=)\s+(\w+)",
                rf"{field}\s+(\w+)",
                rf"with\s+{field}\s+(\w+)",
                rf"with\s+{field}\s+(?:is|equals|=)\s+(\w+)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    filters[field] = value
                    break
                    
        return filters
    
    async def execute_enhanced_query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an enhanced MongoDB query.
        
        Args:
            query_params: Enhanced MongoDB query parameters
            
        Returns:
            Query results
        """
        # Remove metadata from the query params before sending
        params_to_send = {k: v for k, v in query_params.items() if not k.startswith('_')}
        
        try:
            logger.info(f"Executing enhanced query: {json.dumps(params_to_send)}")
            response = await self.client.post(
                f"{self.base_url}/find",
                json=params_to_send,
                timeout=30.0
            )
            
            response.raise_for_status()
            results = response.json()
            
            # Add enhancement metadata to results
            if "_meta" in query_params:
                results["_meta"] = query_params["_meta"]
                
            return results
            
        except Exception as e:
            logger.error(f"Error executing enhanced query: {str(e)}", exc_info=True)
            return {
                "results": [],
                "error": str(e),
                "collection_name": query_params.get("collection_name", ""),
                "total_count": 0
            }
    
    def generate_enhanced_explanation(self, query: str, query_params: Dict[str, Any], results: Dict[str, Any]) -> str:
        """
        Generate an enhanced natural language explanation of the query results.
        
        Args:
            query: Original query
            query_params: Enhanced query parameters
            results: Query results
            
        Returns:
            Enhanced natural language explanation
        """
        collection_name = query_params.get("collection_name", "unknown")
        filter_conditions = query_params.get("filter", {})
        result_count = results.get("total_count", len(results.get("results", [])))
        
        # Generate basic explanation
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
        
        # Add context from the knowledge graph and vector store
        meta = query_params.get("_meta", {})
        
        if meta.get("vector_suggestions", {}).get("templates"):
            templates = meta["vector_suggestions"]["templates"]
            if templates:
                explanation += f" Query matched the pattern: \"{templates[0]}\""
                
        # Add additional context based on results
        if result_count == 0:
            explanation += f" No matching documents were found in the {collection_name} collection with your criteria."
            
            # Suggest alternative fields or collections
            kg_suggestions = meta.get("kg_suggestions", {})
            if kg_suggestions.get("fields"):
                alternative_fields = [f"{f['field']} in {f['collection']}" 
                                     for f in kg_suggestions["fields"][:2]
                                     if f["collection"] != collection_name]
                if alternative_fields:
                    explanation += f" You might want to try querying {', '.join(alternative_fields)}."
                    
        elif 1 <= result_count <= 3:
            # For small result sets, describe the results briefly
            explanation += " Here are the details of the matching document(s)."
        else:
            # For larger result sets, summarize
            explanation += f" Showing {min(len(results.get('results', [])), query_params.get('limit', 10))} out of {result_count} matching documents."
            
        return explanation
    
    async def process_nl_query(self, query: str, initial_params: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Process a natural language query with Graph RAG enhancement.
        
        Args:
            query: Natural language query
            initial_params: Initial MongoDB query parameters
            
        Returns:
            Tuple of (explanation, enhanced_params, results)
        """
        # Step 1: Enhance query understanding
        enhanced_params = self.enhance_query_understanding(query, initial_params)
        
        # Step 2: Execute the enhanced query
        results = await self.execute_enhanced_query(enhanced_params)
        
        # Step 3: Generate enhanced explanation
        explanation = self.generate_enhanced_explanation(query, enhanced_params, results)
        
        return explanation, enhanced_params, results