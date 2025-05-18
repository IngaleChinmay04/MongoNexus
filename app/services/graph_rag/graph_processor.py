"""
Graph RAG Query Processor for enhanced NL to MongoDB queries.
"""
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import httpx
import json
import asyncio
import traceback
from app.services.schema_aware_processor import SchemaAwareProcessor

# Import conditionally to handle potential import errors
try:
    from app.services.graph_rag.graph_rag_service import GraphRAGService
    from app.services.graph_rag.multi_hop_agent import MultiHopAgent
    GRAPH_RAG_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GRAPH_RAG_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Graph RAG dependencies not available, some features will be disabled")

logger = logging.getLogger(__name__)

class GraphRAGProcessor:
    """
    Graph RAG Query Processor that combines schema-aware processing with
    graph-based retrieval augmented generation for enhanced query understanding.
    """
    
    def __init__(self, db_name: str, base_url: Optional[str] = None, api_port: int = 8000):
        """
        Initialize the Graph RAG Processor.
        
        Args:
            db_name: Database name
            base_url: Optional base URL for the MCP server
            api_port: API port number (default: 8000)
        """
        self.db_name = db_name
        self.base_url = base_url or f"http://localhost:{api_port}/mcp/mongo"
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Initialize components
        self.schema = {}
        self.schema_processor = None
        self.graph_rag_service = None
        self.multi_hop_agent = None
        
        # Query complexity classification
        self.is_complex_query = False
        
        # Track initialization status
        self._graph_rag_initialized = False
        self._schema_processor_initialized = False
        self._multi_hop_initialized = False
        
    async def initialize(self) -> bool:
        """
        Initialize the Graph RAG processor.
        
        Returns:
            True if initialization was successful
        """
        # Step 1: Get schema
        schema = await self.get_schema()
        
        if not schema:
            logger.error("Failed to get database schema")
            return False
            
        # Step 2: Initialize schema processor
        try:
            self.schema_processor = SchemaAwareProcessor(self.db_name, schema)
            self._schema_processor_initialized = True
            logger.info("Schema-aware processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize schema processor: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
        # Step 3: Initialize Graph RAG service
        if GRAPH_RAG_AVAILABLE:
            try:
                self.graph_rag_service = GraphRAGService(self.db_name, self.base_url)
                init_success = await self.graph_rag_service.initialize_graph_rag()
                self._graph_rag_initialized = init_success
                if init_success:
                    logger.info("Graph RAG service initialized successfully")
                else:
                    logger.warning("Graph RAG service initialization returned False")
            except Exception as e:
                logger.error(f"Failed to initialize Graph RAG service: {str(e)}")
                logger.error(traceback.format_exc())
                self._graph_rag_initialized = False
        else:
            logger.warning("Graph RAG dependencies not available, skipping service initialization")
            self._graph_rag_initialized = False
        
        # Step 4: Initialize multi-hop agent
        if GRAPH_RAG_AVAILABLE:
            try:
                self.multi_hop_agent = MultiHopAgent(self.db_name, schema)
                self._multi_hop_initialized = True  # This will always succeed now as we don't try to initialize LangGraph
                logger.info("Multi-hop agent initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize multi-hop agent: {str(e)}")
                logger.error(traceback.format_exc())
                self._multi_hop_initialized = False
        else:
            logger.warning("Graph RAG dependencies not available, skipping multi-hop agent initialization")
            self._multi_hop_initialized = False
        
        # Return True as long as at least schema processor is initialized
        return self._schema_processor_initialized
    
    async def close(self):
        """Close connections and resources."""
        await self.client.aclose()
        
        if self.graph_rag_service:
            await self.graph_rag_service.close()
            
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
    
    def classify_query_complexity(self, query: str) -> bool:
        """
        Determine if a query requires complex (multi-hop) reasoning.
        
        Args:
            query: Natural language query
            
        Returns:
            True if complex, False if simple
        """
        # Check for indicators of complex queries
        complex_indicators = [
            # Multiple collections mentioned
            lambda q: sum(1 for coll in self.schema if coll.lower() in q.lower()) > 1,
            
            # Join/relationship indicators
            lambda q: any(term in q.lower() for term in [
                "join", "related", "relationship", "between", "connect", 
                "linked", "associated", "together with"
            ]),
            
            # Multi-step reasoning indicators
            lambda q: any(term in q.lower() for term in [
                "and then", "after that", "followed by", "subsequently", 
                "next", "first", "second", "finally"
            ]),
            
            # Comparative queries
            lambda q: any(term in q.lower() for term in [
                "compare", "more than", "less than", "greater", "highest", 
                "lowest", "maximum", "minimum", "average", "most", "least"
            ]),
            
            # Temporal reasoning
            lambda q: any(term in q.lower() for term in [
                "before", "after", "during", "when", "while", 
                "since", "until", "latest", "newest", "oldest"
            ])
        ]
        
        # Check if any complexity indicators apply
        is_complex = any(indicator(query) for indicator in complex_indicators)
        logger.info(f"Query complexity classification: {'Complex' if is_complex else 'Simple'}")
        
        return is_complex
    
    async def process_query(self, query: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Process a natural language query using Graph RAG.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (explanation, query_params, results)
        """
        # Check if we have at least the schema processor initialized
        if not self._schema_processor_initialized:
            logger.error("Cannot process query: schema processor not initialized")
            error_msg = "Processing system not properly initialized. Please try again later."
            return error_msg, {"error": error_msg}, {"error": error_msg}
        
        # Step 1: Determine query complexity
        self.is_complex_query = self.classify_query_complexity(query)
        
        if self.is_complex_query and self._multi_hop_initialized:
            # Handle complex query with multi-hop reasoning
            logger.info(f"Processing complex query: {query}")
            return await self.process_complex_query(query)
        elif self._graph_rag_initialized:
            # Handle simple query with enhanced understanding
            logger.info(f"Processing simple query with Graph RAG: {query}")
            return await self.process_enhanced_query(query)
        else:
            # Fallback to basic schema-aware processing
            logger.info(f"Processing query with schema-aware processor: {query}")
            return await self.process_simple_query(query)
    
    async def process_simple_query(self, query: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Process a simple query using only schema-aware processing.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (explanation, query_params, results)
        """
        # Get query parameters from schema processor
        query_params = self.schema_processor.process_query(query)
        
        # Check for errors
        if "error" in query_params:
            return (
                f"Error processing query: {query_params['error']}",
                query_params,
                {"error": query_params['error'], "results": []}
            )
            
        # Execute the query
        results = await self.execute_query(query_params)
        
        # Generate explanation
        explanation = self.generate_explanation(query, query_params, results)
        
        return explanation, query_params, results
    
    async def process_enhanced_query(self, query: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Process a query with graph-enhanced understanding.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (explanation, query_params, results)
        """
        # Get initial query parameters from schema processor
        initial_params = self.schema_processor.process_query(query)
        
        # Check for errors
        if "error" in initial_params:
            return (
                f"Error processing query: {initial_params['error']}",
                initial_params,
                {"error": initial_params['error'], "results": []}
            )
            
        try:
            # Enhance with Graph RAG
            return await self.graph_rag_service.process_nl_query(query, initial_params)
        except Exception as e:
            logger.error(f"Error in Graph RAG processing: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fall back to simple query processing
            return await self.process_simple_query(query)
    
    async def process_complex_query(self, query: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Process a complex query using multi-hop reasoning.
        
        Args:
            query: Natural language query
            
        Returns:
            Tuple of (explanation, query_params, results)
        """
        # Get initial query parameters from schema processor
        initial_params = self.schema_processor.process_query(query)
        
        try:
            # Execute initial query to get starting data
            initial_results = await self.execute_query(initial_params)
            
            # Use multi-hop agent with initial results
            multi_hop_results = await self.multi_hop_agent.process_complex_query(query, {
                initial_params["collection_name"]: initial_results
            })
            
            # Combine the results and explanations
            combined_params = initial_params.copy()
            combined_params["_meta"] = combined_params.get("_meta", {})
            combined_params["_meta"]["complex_query"] = True
            combined_params["_meta"]["thoughts"] = multi_hop_results.get("thoughts", [])
            
            # Return the agent's response as the explanation
            return (
                multi_hop_results.get("response", ""),
                combined_params,
                {"initial_results": initial_results, "complex_reasoning": multi_hop_results}
            )
        except Exception as e:
            logger.error(f"Error processing complex query: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Fall back to simple query processing
            return await self.process_simple_query(query)
    
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
        is_count = query_params.get("_meta", {}).get("intent") == "count"
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
            logger.error(f"Error executing query: {str(e)}")
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
            query: Original query
            query_params: Query parameters
            results: Query results
            
        Returns:
            Natural language explanation
        """
        intent = query_params.get("_meta", {}).get("intent", "unknown")
        collection_name = query_params.get("collection_name", "unknown")
        filter_conditions = query_params.get("filter", {})
        
        # For count queries, we should use the total_count or count
        if intent == "count":
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
        if result_count == 0 and intent != "count":
            explanation += f" No matching documents were found in the {collection_name} collection with your criteria."
        elif intent != "count" and 1 <= result_count <= 3:
            # For small result sets, describe the results briefly
            explanation += " Here are the details of the matching document(s)."
        elif intent != "count" and result_count > 3:
            # For larger result sets, summarize
            explanation += f" Showing {min(len(results.get('results', [])), query_params.get('limit', 10))} out of {result_count} matching documents."
            
        return explanation