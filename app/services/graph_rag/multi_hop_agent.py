"""
Multi-hop Reasoning Agent using LangGraph for orchestrating complex queries.
"""
import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Type, Union
import json
import re
import traceback

logger = logging.getLogger(__name__)

# Check if langgraph is available
try:
    import langgraph
    
    # Try to determine what version of API we need to use
    try:
        from langgraph.graph import StateGraph, END
        LANGGRAPH_AVAILABLE = True
        logger.info("LangGraph is available")
    except ImportError:
        try:
            from langgraph.graph.graph import StateGraph, END
            LANGGRAPH_AVAILABLE = True
            logger.info("Using older LangGraph API")
        except ImportError:
            LANGGRAPH_AVAILABLE = False
            logger.warning("LangGraph modules not found")
            END = None
            StateGraph = None
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available")
    END = None
    StateGraph = None

from app.services.llm.llm_service import LLMService

class MultiHopAgent:
    """
    Multi-hop reasoning agent that uses LangGraph to coordinate complex query processing
    that may require joining information across collections or multiple reasoning steps.
    """
    
    def __init__(self, db_name: str, schema: Dict[str, Any]):
        """
        Initialize the multi-hop agent.
        
        Args:
            db_name: Database name
            schema: Database schema
        """
        self.db_name = db_name
        self.schema = schema
        self.llm = LLMService()
        self.workflow = None
        
        # We won't even try to build the workflow - we'll use the direct LLM fallback
        # This is because LangGraph integration is proving challenging and not critical
        logger.info("Using direct LLM calls for multi-hop reasoning instead of LangGraph")
    
    async def _analyze_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the user query to understand what collections and fields are needed.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        query = state.get("query", "")
        schema_info = self._format_schema_info()
        
        prompt = f"""
        You are a database query analyzer. Given a natural language query and database schema, 
        analyze what collections and fields are needed to answer the query.
        
        Database name: {self.db_name}
        
        Database schema:
        {schema_info}
        
        User query: "{query}"
        
        Task: Analyze what collections and fields are relevant to this query. Think about:
        1. Which collections contain information needed to answer the query?
        2. What fields in these collections are relevant?
        3. Are there relationships between collections that need to be considered?
        4. Does this query require aggregation, counting, or other operations?
        """
        
        try:
            # Get analysis from LLM
            analysis_text = await self.llm.generate_response(prompt)
            
            # Update state
            thoughts = state.get("thoughts", [])
            thoughts.append({
                "step": "analyze_query", 
                "analysis": analysis_text
            })
            
            # Extract collections from analysis
            collections_to_query = []
            for collection in self.schema:
                if collection.lower() in analysis_text.lower():
                    collections_to_query.append(collection)
                    
            if not collections_to_query and self.schema:
                # Default to first collection if none detected
                collections_to_query = [next(iter(self.schema.keys()))]
                
            # Return updated state
            return {
                **state,
                "thoughts": thoughts,
                "collections_to_query": collections_to_query,
                "query_results": state.get("query_results", {})
            }
        except Exception as e:
            logger.error(f"Error in analyze_query: {str(e)}")
            collections_to_query = list(self.schema.keys())[:1]  # Just use first collection
            thoughts = state.get("thoughts", [])
            thoughts.append({
                "step": "analyze_query", 
                "error": str(e)
            })
            
            # Return updated state with error
            return {
                **state,
                "thoughts": thoughts,
                "collections_to_query": collections_to_query,
                "query_results": state.get("query_results", {})
            }
    
    def _format_schema_info(self) -> str:
        """
        Format schema information for LLM prompts.
        
        Returns:
            Formatted schema information
        """
        formatted = ""
        for collection_name, fields in self.schema.items():
            formatted += f"Collection: {collection_name}\n"
            formatted += "Fields:\n"
            for field_name, field_type in fields.items():
                formatted += f"  - {field_name}: {field_type}\n"
            formatted += "\n"
            
        return formatted
    
    def _format_collection_info(self, collection: str) -> str:
        """
        Format collection information for LLM prompts.
        
        Args:
            collection: Collection name
            
        Returns:
            Formatted collection information
        """
        formatted = f"Collection: {collection}\n"
        formatted += "Fields:\n"
        
        if collection in self.schema:
            for field_name, field_type in self.schema[collection].items():
                formatted += f"  - {field_name}: {field_type}\n"
                
        return formatted
    
    def _summarize_results(self, results: Dict[str, Any]) -> str:
        """
        Summarize query results for LLM consumption.
        
        Args:
            results: Query results
            
        Returns:
            Summarized results
        """
        result_docs = results.get("results", [])
        total_count = results.get("total_count", len(result_docs))
        
        summary = f"Total documents: {total_count}\n"
        
        if result_docs:
            summary += "Sample documents:\n"
            for i, doc in enumerate(result_docs[:3]):  # Limit to 3 samples
                summary += f"Document {i+1}:\n"
                for key, value in doc.items():
                    # Truncate long values
                    str_value = str(value)
                    if len(str_value) > 100:
                        str_value = str_value[:100] + "..."
                    summary += f"  {key}: {str_value}\n"
                summary += "\n"
                
        return summary
    
    async def process_complex_query(self, query: str, initial_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a complex query using the LLM directly, without LangGraph.
        
        Args:
            query: User query
            initial_results: Optional initial query results
            
        Returns:
            Processing results including final response
        """
        try:
            # Format initial results for the prompt
            results_summary = ""
            if initial_results:
                for collection, results in initial_results.items():
                    summary = self._summarize_results(results)
                    results_summary += f"Results from {collection}:\n{summary}\n\n"
            
            # Prepare a comprehensive prompt that guides the LLM through the multi-hop reasoning process
            prompt = f"""You are a database query expert specialized in multi-hop reasoning.

Database name: {self.db_name}

Database schema:
{self._format_schema_info()}

User query: "{query}"

{f'Initial query results:\n{results_summary}' if results_summary else 'No initial results yet.'}

Please follow these steps to analyze this query:

1. First analyze which collections in the database are relevant for answering this query.
2. Identify what fields from these collections are needed.
3. Consider any relationships between collections that might be required for a complete answer.
4. Based on the available information and results, formulate a comprehensive answer.
5. If the results don't contain enough information, suggest what additional queries might be helpful.

Provide your detailed response that directly answers the user's question based on the available data.
Include relevant details from the query results if available.
"""
            
            # Get response from LLM
            response_text = await self.llm.generate_response(prompt)
            
            # Build result object with response
            return {
                "response": response_text,
                "collections_queried": list((initial_results or {}).keys()),
                "thoughts": [{
                    "step": "direct_llm_reasoning",
                    "analysis": "Used direct LLM reasoning instead of LangGraph workflow"
                }]
            }
            
        except Exception as e:
            logger.error(f"Error in direct LLM processing: {str(e)}")
            return {
                "response": f"I encountered a challenge processing your complex query. Let me share what I do know about your request: {query}",
                "error": str(e),
                "collections_queried": list((initial_results or {}).keys())
            }