"""LLM service using Groq."""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import logging
import traceback
from app.config.settings import GROQ_API_KEY, GROQ_MODEL_NAME

logger = logging.getLogger(__name__)

# Initialize Groq LLM
try:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME,
    )
    logger.info(f"Initialized Groq LLM with model: {GROQ_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize Groq LLM: {str(e)}")
    llm = None

class LLMService:
    """Service for interacting with LLMs."""
    
    @staticmethod
    async def generate_response(query: str, system_prompt: str = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            query: The user's query
            system_prompt: Optional system prompt to give context
            
        Returns:
            The LLM's response
        """
        if not llm:
            return "LLM service is not available. Please check your API key configuration."
            
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
            
        messages.append(HumanMessage(content=query))
        
        try:
            response = await llm.ainvoke(messages)
            return response.content
        except Exception as e:
            error_message = f"Error generating LLM response: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            return error_message
    
    @staticmethod
    async def parse_user_query(
        query: str, 
        schema_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse a user's natural language query into a structured MongoDB query.
        
        Args:
            query: The user's natural language query
            schema_info: Schema information with collection information
            
        Returns:
            Structured MongoDB query parameters
        """
        if not llm:
            # Return a default query if LLM is not available
            collections = list(schema_info.keys()) if schema_info else []
            default_collection = collections[0] if collections else "unknown_collection"
            return {
                "collection_name": default_collection,
                "filter": {},
                "limit": 10
            }
            
        system_prompt = """
        You are a database query assistant that converts natural language queries into MongoDB query parameters.
        Your task is to parse the user's query and extract relevant information to construct a MongoDB query.
        
        Be extremely literal and precise in your interpretation of the query. Only include filters that are explicitly mentioned.
        Pay close attention to collection names mentioned in the query.
        
        Return your response as a JSON object with the following structure:
        {
            "collection_name": "name of the collection to query",
            "filter": {}, // MongoDB filter criteria
            "projection": {}, // fields to include/exclude (optional)
            "sort": [], // sort criteria (optional)
            "limit": 10 // return 10 results by default
        }
        
        Available collections and their schemas:
        """
        
        # Add schema information in a clear format
        for collection_name, fields in schema_info.items():
            system_prompt += f"\n\n{collection_name} collection fields:"
            for field_name, field_type in fields.items():
                system_prompt += f"\n- {field_name}: {field_type}"
        
        try:
            # Direct prompt approach for more control
            prompt = f"""
            Based on the user query: "{query}"
            
            Generate a MongoDB query that would satisfy this request.
            Be extremely literal - only include filters that are explicitly mentioned.
            If a specific collection is mentioned, use that collection.
            
            Return only valid JSON in this exact format:
            {{
                "collection_name": "the_collection_name",
                "filter": {{
                    // filters based on the query
                }},
                "projection": {{
                    // fields to include/exclude if specified
                }},
                "limit": 10
            }}
            """
            
            response = await LLMService.generate_response(prompt, system_prompt)
            
            # Extract JSON from the response
            try:
                # Find JSON content between ```json and ``` if present
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0].strip()
                else:
                    json_str = response.strip()
                
                query_params = json.loads(json_str)
                
                # Ensure required fields are present
                if "collection_name" not in query_params:
                    query_params["collection_name"] = list(schema_info.keys())[0] if schema_info else "users"
                if "filter" not in query_params:
                    query_params["filter"] = {}
                if "limit" not in query_params:
                    query_params["limit"] = 10
                    
                return query_params
            except Exception as e:
                logger.error(f"Error parsing JSON from LLM response: {str(e)}")
                logger.error(f"Raw response: {response}")
                
                # Default query when parsing fails
                return {
                    "collection_name": list(schema_info.keys())[0] if schema_info else "users",
                    "filter": {},
                    "limit": 10
                }
                
        except Exception as e:
            error_message = f"Error parsing user query: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            
            # Return a default query on error
            collections = list(schema_info.keys()) if schema_info else []
            default_collection = collections[0] if collections else "users"
            return {
                "collection_name": default_collection,
                "filter": {},
                "limit": 10,
                "error": error_message
            }
    
    @staticmethod
    async def generate_explanation(
        query: str,
        results: List[Dict[str, Any]],
        collection_name: str
    ) -> str:
        """
        Generate a natural language explanation of query results.
        
        Args:
            query: The original user query
            results: The MongoDB query results
            collection_name: The name of the collection queried
            
        Returns:
            Natural language explanation of the results
        """
        if not llm:
            return f"Found {len(results)} results in the {collection_name} collection."
            
        system_prompt = """
        You are a helpful assistant that explains database query results in natural language.
        Given a user's query and the results from a MongoDB database, provide a clear and concise explanation
        of the findings. Be extremely precise and factual in your explanation.
        
        1. Always mention which collection was queried and how many results were found.
        2. If there are no results, clearly state that no documents were found.
        3. If there are results, summarize what was found.
        4. Answer the user's original query directly.
        5. Do not make assumptions or provide information not evident in the results.
        """
        
        human_prompt = f"""
        Original user query: "{query}"
        
        Collection queried: {collection_name}
        
        Number of results: {len(results)}
        
        Results:
        ```
        {json.dumps(results[:5], indent=2)}
        ```
        
        {f"...and {len(results) - 5} more results." if len(results) > 5 else ""}
        
        Please provide a clear, factual explanation of these results that directly addresses the user's original query.
        """
        
        try:
            response = await LLMService.generate_response(human_prompt, system_prompt)
            return response
        except Exception as e:
            error_message = f"Error generating explanation: {str(e)}"
            logger.error(error_message)
            logger.error(traceback.format_exc())
            return f"Found {len(results)} results in the {collection_name} collection matching your query."