"""
Schema-Aware Natural Language Query Processor.
Provides a more robust, schema-agnostic approach to query processing.
"""

import re
import logging
import difflib
import string
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

class QueryIntent:
    """Represents different query intents."""
    COUNT = "count"
    FIND = "find"
    AGGREGATE = "aggregate"
    DISTINCT = "distinct"
    UNKNOWN = "unknown"

class SchemaAwareProcessor:
    """
    A processor that analyzes database schemas and uses them to interpret
    natural language queries in a more robust way.
    """
    
    def __init__(self, db_name: str, schema: Dict[str, Any] = None):
        """
        Initialize the processor with database schema information.
        
        Args:
            db_name: Database name
            schema: Database schema (collection -> fields mapping)
        """
        self.db_name = db_name
        self.schema = schema or {}
        
        # Process schema information for faster lookup
        self.collection_fields = {}
        self.field_types = {}
        self.array_fields = set()
        
        for collection, fields in self.schema.items():
            self.collection_fields[collection] = set(fields.keys())
            for field, field_type in fields.items():
                self.field_types[(collection, field)] = field_type
                if isinstance(field_type, str) and "array" in field_type.lower():
                    self.array_fields.add((collection, field))
    
    def _is_string_field(self, collection: str, field_name: str) -> bool:
        """Helper to determine if a field is likely a string type."""
        field_type = self.field_types.get((collection, field_name))
        if isinstance(field_type, str):
            # Assuming types like "string", "text", "varchar". Case-insensitive check.
            ft_lower = field_type.lower()
            return "string" in ft_lower or \
                   "text" in ft_lower or \
                   "char" in ft_lower
        return False

    def _is_string_array_field(self, collection: str, field_name: str) -> bool:
        """Helper to determine if a field is likely an array of strings."""
        if (collection, field_name) not in self.array_fields:
            return False
        field_type = self.field_types.get((collection, field_name))
        if isinstance(field_type, str):
            # Example: "array of string", "list of text"
            # Check for "string", "text", "char" within the array type description
            ft_lower = field_type.lower()
            return "string" in ft_lower or \
                   "text" in ft_lower or \
                   "char" in ft_lower
        # If type info is not a string (e.g. a list or dict from a more structured schema)
        # and it's an array, we might need more sophisticated checks or make assumptions.
        # For now, rely on string descriptors.
        return False

    def preprocess_query(self, query: str) -> str:
        """
        Normalize and clean the query text.
        
        Args:
            query: Original query text
            
        Returns:
            Cleaned query text
        """
        # Convert to lowercase
        query = query.lower()
        
        # Replace multiple spaces with single space
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Remove punctuation that's not meaningful
        for punct in ".,;:!?":
            query = query.replace(punct, " ")
        
        # Normalize again after punctuation removal
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def find_collection(self, query: str) -> Tuple[str, float]:
        """
        Find the most likely collection being referred to in the query.
        Uses a combination of exact matching and fuzzy matching.
        
        Args:
            query: Preprocessed query text
            
        Returns:
            Tuple of (collection_name, confidence)
        """
        if not self.schema:
            return None, 0.0
            
        # Look for explicit collection references first
        collection_patterns = [
            r'in\s+(?:the\s+)?([a-z0-9_]+)(?:\s+collection)?',
            r'from\s+(?:the\s+)?([a-z0-9_]+)(?:\s+collection)?',
            r'of\s+(?:the\s+)?([a-z0-9_]+)(?:\s+collection)?',
        ]
        
        for pattern in collection_patterns:
            match = re.search(pattern, query)
            if match:
                collection_candidate = match.group(1)
                # Check if this exact collection exists
                if collection_candidate in self.schema:
                    return collection_candidate, 0.9
                
                # Try fuzzy matching
                matches = difflib.get_close_matches(
                    collection_candidate, 
                    self.schema.keys(),
                    n=1, 
                    cutoff=0.6
                )
                if matches:
                    return matches[0], 0.8
        
        # Look for collection names as individual words in the query
        words = set(query.split())
        for collection in self.schema:
            if collection.lower() in words:
                return collection, 0.7
                
            # Check plural form
            if collection.lower() + 's' in words:
                return collection, 0.65
                
            # Check singular form
            if collection.lower().endswith('s') and collection.lower()[:-1] in words:
                return collection, 0.65
        
        # Use contextual clues to infer the most relevant collection
        # Approach: Look for field names mentioned in the query
        collection_scores = {}
        
        for collection, fields in self.collection_fields.items():
            # Count how many field names from this collection appear in the query
            field_mentions = 0
            for field in fields:
                if field.lower() in query:
                    field_mentions += 1
                    
            if field_mentions > 0:
                collection_scores[collection] = field_mentions / len(fields)
        
        if collection_scores:
            best_collection = max(collection_scores.items(), key=lambda x: x[1])
            return best_collection[0], 0.5 * best_collection[1]
        
        # If we still haven't found a collection, use heuristics based on query content
        if any(word in query for word in ['user', 'users', 'person', 'people']):
            if 'users' in self.schema:
                return 'users', 0.4
                
        # Last resort: return the first collection with low confidence
        return next(iter(self.schema.keys())), 0.2
    
    def determine_query_intent(self, query: str) -> str:
        """
        Determine the intent of the query (count, find, etc.)
        
        Args:
            query: Preprocessed query text
            
        Returns:
            Query intent from QueryIntent class
        """
        # Count-related keywords
        count_keywords = [
            'how many', 'count', 'total', 'number of', 'sum', 
            'tally', 'quantity', 'amount'
        ]
        
        # Count patterns
        count_patterns = [
            r'(?:what|how)\s+(?:is|are)\s+(?:the\s+)?(?:total|number)',
            r'how\s+many\s+',
            r'count\s+(?:the\s+)?(?:number|total)?',
        ]
        
        # Check for count intent
        if any(keyword in query for keyword in count_keywords):
            return QueryIntent.COUNT
        
        if any(re.search(pattern, query) for pattern in count_patterns):
            return QueryIntent.COUNT
            
        # Find patterns
        find_patterns = [
            r'(?:show|find|list|get|give me|display)',
            r'(?:what|who|which)\s+(?:are|is)',
        ]
        
        if any(re.search(pattern, query) for pattern in find_patterns):
            return QueryIntent.FIND
            
        # Aggregation patterns
        agg_patterns = [
            r'(?:average|avg|mean|median|sum|group|grouped)',
            r'(?:group\s+by|grouped\s+by)',
        ]
        
        if any(re.search(pattern, query) for pattern in agg_patterns):
            return QueryIntent.AGGREGATE
            
        # Distinct patterns
        if any(word in query for word in ['distinct', 'unique', 'different']):
            return QueryIntent.DISTINCT
            
        # Default to find
        return QueryIntent.FIND
    
    def fuzzy_match_field(self, term: str, collection: str) -> Tuple[str, float]:
        """
        Find the field in the collection that best matches the given term.
        
        Args:
            term: Term to match
            collection: Collection name
            
        Returns:
            Tuple of (field_name, confidence)
        """
        if collection not in self.schema:
            return None, 0
            
        fields = list(self.collection_fields[collection])
        
        # Clean the term
        term = term.strip().lower()
        
        # First check for exact match
        if term in fields:
            return term, 1.0
            
        # Check if term appears as substring of any field
        for field in fields:
            if term in field.lower():
                return field, 0.9
                
            if field.lower() in term:
                return field, 0.7
        
        # Try fuzzy matching
        matches = difflib.get_close_matches(
            term,
            fields,
            n=1,
            cutoff=0.6
        )
        
        if matches:
            return matches[0], 0.6
            
        # No good match found
        return None, 0
    
    def extract_value_from_text(self, text: str, ignore_words=None) -> str:
        """
        Extract a clean value from text by removing specified words and extra whitespace.
        
        Args:
            text: The text to clean
            ignore_words: Words to remove from the text
            
        Returns:
            Cleaned value
        """
        if ignore_words is None:
            ignore_words = []
            
        # Convert to lowercase
        value = text.lower()
        
        # Remove words to ignore
        for word in ignore_words:
            # Make sure we're removing whole words with word boundaries
            value = re.sub(r'\b' + re.escape(word) + r'\b', '', value)
            
        # Remove any extra whitespace
        value = re.sub(r'\s+', ' ', value).strip()
        
        return value
    
    def extract_conditions(self, query: str, collection: str) -> Dict[str, Any]:
        """
        Extract filter conditions from the query.
        
        Args:
            query: Preprocessed query text
            collection: Collection name
            
        Returns:
            MongoDB filter conditions
        """
        if collection not in self.schema:
            return {}
            
        filter_conditions = {}
        
        # CASE 1 & CASE 2 (Interest and Name-based specific patterns) are removed for generalization.
        # The system will now rely on the generic CASE 3 for all field-value extractions.

        # CASE 3: Try to match generic field = value patterns
        condition_patterns = [
            # "field is value"
            r'(\w+)\s+is\s+([a-zA-Z0-9_\s]+)(?:\s+in|\s*$)',
            # "field = value"
            r'(\w+)\s+=\s+([a-zA-Z0-9_\s]+)(?:\s+in|\s*$)',
            # "field: value"
            r'(\w+):\s+([a-zA-Z0-9_\s]+)(?:\s+in|\s*$)',
            # "with field value"
            r'with\s+(\w+)\s+([a-zA-Z0-9_\s]+)(?:\s+in|\s*$)',
        ]
        
        for pattern in condition_patterns:
            for match in re.finditer(pattern, query):
                field_term, value = match.groups()
                field_term = field_term.strip()
                value = value.strip()
                
                # Skip if this seems to be a collection reference
                if "collection" in value:
                    continue
                    
                # Clean up the value - remove common connecting words
                value = self.extract_value_from_text(value, ['as', 'is', '=', ':', 'of', 'in', 'on'])
                
                field_name, confidence = self.fuzzy_match_field(field_term, collection)
                
                if field_name and confidence >= 0.6:
                    # The specialized logic for "name" splitting has been removed.
                    # All conditions now follow this generic path.

                    if (collection, field_name) in self.array_fields:
                        if isinstance(value, str) and self._is_string_array_field(collection, field_name):
                            filter_conditions[field_name] = {'$regex': f'^{re.escape(value)}$', '$options': 'i'}
                        else: 
                            filter_conditions[field_name] = {"$in": [value]}
                    elif self._is_string_field(collection, field_name): 
                        if ' ' in value: 
                            filter_conditions[field_name] = {'$regex': f'{re.escape(value)}', '$options': 'i'}
                        else:
                            filter_conditions[field_name] = {'$regex': f'^{re.escape(value)}$', '$options': 'i'}
                    else: 
                        filter_conditions[field_name] = value
        
        return filter_conditions
    
    def determine_limit(self, query: str, intent: str) -> int:
        """
        Determine the limit for the query results.
        
        Args:
            query: Preprocessed query text
            intent: Query intent
            
        Returns:
            Result limit
        """
        # Check for explicit limits
        limit_patterns = [
            r'(?:limit|top|first)\s+(\d+)',
            r'(\d+)\s+(?:results|entries|documents|records)',
        ]
        
        for pattern in limit_patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
        
        # Default limits based on intent
        if intent == QueryIntent.COUNT:
            return 1000  # Higher limit for counting
        elif intent == QueryIntent.FIND:
            return 20  # Default for find queries
        else:
            return 50  # Default for other operations
    
    def determine_sort(self, query: str, collection: str) -> Optional[List[Tuple[str, int]]]:
        """
        Determine the sort criteria for the query.
        
        Args:
            query: Preprocessed query text
            collection: Collection name
            
        Returns:
            List of (field, direction) tuples or None
        """
        if collection not in self.schema:
            return None
        
        # Look for sort indications
        sort_patterns = [
            r'(?:sort|order)\s+by\s+(\w+)\s+(asc|ascending|desc|descending)',
            r'(?:sort|order)\s+by\s+(\w+)',
            r'in\s+(\w+)\s+(?:asc|ascending|desc|descending)\s+order',
        ]
        
        for pattern in sort_patterns:
            match = re.search(pattern, query)
            if match:
                groups = match.groups()
                field_term = groups[0]
                
                field_name, confidence = self.fuzzy_match_field(field_term, collection)
                
                if field_name and confidence >= 0.6:
                    direction = 1  # Default ascending
                    if len(groups) > 1 and groups[1] and groups[1].startswith(('desc', 'descending')):
                        direction = -1
                    
                    return [(field_name, direction)]
        
        return None
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query into MongoDB query parameters.
        
        Args:
            query: Natural language query
            
        Returns:
            MongoDB query parameters
        """
        # Step 1: Preprocess the query
        processed_query = self.preprocess_query(query)
        
        # Step 2: Determine the query intent
        intent = self.determine_query_intent(processed_query)
        
        # Step 3: Find the collection being queried
        collection, collection_confidence = self.find_collection(processed_query)
        
        if not collection:
            return {
                "error": "Could not determine which collection to query. Please specify a collection name in your query."
            }
        
        # Step 4: Extract filter conditions
        filter_query = self.extract_conditions(processed_query, collection)
        
        # Step 5: Determine the limit
        limit = self.determine_limit(processed_query, intent)
        
        # Step 6: Determine sort criteria
        sort = self.determine_sort(processed_query, collection)
        
        # Build the MongoDB query parameters
        query_params = {
            "db_name": self.db_name,
            "collection_name": collection,
            "filter": filter_query,
            "limit": limit
        }
        
        if sort:
            query_params["sort"] = sort
        
        # Add metadata
        query_params["_meta"] = {
            "intent": intent,
            "collection_confidence": collection_confidence,
            "original_query": query,
            "processed_query": processed_query
        }
        
        logger.info(f"Processed query into parameters: {query_params}")
        return query_params