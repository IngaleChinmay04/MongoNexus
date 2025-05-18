"""LLM service for generating text responses and reasoning."""
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import httpx
import json
import asyncio
from app.config.settings import GROQ_API_KEY, GROQ_MODEL_NAME

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for interacting with Large Language Models.
    Supports Qwen via Groq's API.
    """
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize the LLM service.
        
        Args:
            api_key: API key for Groq
            model_name: Model name to use (default to Qwen if available)
        """
        self.api_key = api_key or GROQ_API_KEY
        
        # Default to Qwen model if no model specified
        if not model_name:
            # Check if a Qwen model is specified in settings
            if GROQ_MODEL_NAME and "qwen" in GROQ_MODEL_NAME.lower():
                self.model_name = GROQ_MODEL_NAME
            else:
                # Use Qwen as default
                self.model_name = "qwen2-72b-instruct"  # Qwen 2.5 via Groq
        else:
            self.model_name = model_name
            
        # Log which model we're using
        logger.info(f"Using LLM model: {self.model_name}")
        
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def generate_response(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt
            temperature: Temperature for generation (0-1)
            
        Returns:
            Generated response
        """
        if not self.api_key:
            logger.warning("No API key provided, returning fallback response")
            return f"I need an API key to answer complex queries. For now, I'll try to help with what I know.\n\nYour query was about: {prompt[:100]}..."
            
        try:
            # Call Groq API with Qwen model
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Enhance the prompt for Qwen to encourage better structured reasoning
            enhanced_prompt = self._format_prompt_for_qwen(prompt)
            
            payload = {
                "messages": [{"role": "user", "content": enhanced_prompt}],
                "model": self.model_name,
                "temperature": temperature,
                "max_tokens": 2048  # Allow more tokens for comprehensive answers
            }
            
            response = await self.client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return f"I encountered an error while generating a response: {str(e)}. Let's try a different approach."
    
    def _format_prompt_for_qwen(self, prompt: str) -> str:
        """
        Format the prompt to get the best performance from Qwen.
        Qwen performs better with structured prompts that clearly define the task.
        
        Args:
            prompt: Original prompt
            
        Returns:
            Enhanced prompt for Qwen
        """
        # Check if this appears to be a database query task
        if any(keyword in prompt.lower() for keyword in ["database", "schema", "query", "mongodb", "collection", "field"]):
            return f"""You are a database query expert with deep knowledge of MongoDB.
            
Task: Please analyze the following request and provide a clear, accurate response.

{prompt}

When generating database queries or analyzing schemas:
1. Be precise with field names and syntax
2. Consider the schema carefully when suggesting queries
3. Provide JSON-formatted outputs when requested
4. Structure your reasoning clearly with step-by-step explanations

Your response:"""
        else:
            # For general prompts, just return as is
            return prompt
    
    async def extract_structured_data(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from text based on a schema.
        Qwen excels at structured extraction tasks.
        
        Args:
            text: Input text
            schema: Schema definition
            
        Returns:
            Structured data
        """
        schema_str = json.dumps(schema, indent=2)
        
        prompt = f"""You are a data extraction expert.

Task: Extract structured data from the text below according to this exact schema:

{schema_str}

Text to extract from:
{text}

Important instructions:
1. Return ONLY valid JSON that follows the schema exactly
2. Do not include ANY explanations or text outside the JSON
3. Make sure all field names match the schema exactly
4. If a field cannot be extracted, use null or an empty value based on the type

JSON output:"""
        
        response = await self.generate_response(prompt, temperature=0.1)
        
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                return json.loads(response)
        except Exception as e:
            logger.error(f"Error parsing structured data: {str(e)}", exc_info=True)
            return {}