"""
Schema-Aware Natural Language Query Processor.
Provides a robust, schema-agnostic approach to query processing.
"""

import re
import logging
import difflib
from typing import Dict, List, Any, Optional, Tuple, Set

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
        self.name_fields = set()  # Track potential name fields
        self.interest_fields = set()  # Track potential interest fields
        
        for collection, fields in self.schema.items():
            self.collection_fields[collection] = set(fields.keys())
            
            for field, field_type in fields.items():
                self.field_types[(collection, field)] = field_type
                
                # Detect array fields
                if isinstance(field_type, str) and "array" in field_type.lower():
                    self.array_fields.add((collection, field))
                
                # Detect potential name fields
                field_lower = field.lower()
                if "name" in field_lower or "username" in field_lower or "user" == field_lower:
                    self.name_fields.add((collection, field))
                
                # Detect potential interest fields
                if field_lower in ["interests", "tags", "skills", "hobbies", "preferences"]:
                    self.interest_fields.add((collection, field))
    
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
        for keyword in count_keywords:
            if keyword in query:
                return QueryIntent.COUNT
        
        for pattern in count_patterns:
            if re.search(pattern, query):
                return QueryIntent.COUNT
            
        # Find patterns
        find_patterns = [
            r'(?:show|find|list|get|give me|display)',
            r'(?:what|who|which)\s+(?:are|is)',
        ]
        
        for pattern in find_patterns:
            if re.search(pattern, query):
                return QueryIntent.FIND
            
        # Aggregation patterns
        agg_patterns = [
            r'(?:average|avg|mean|median|sum|group|grouped)',
            r'(?:group\s+by|grouped\s+by)',
        ]
        
        for pattern in agg_patterns:
            if re.search(pattern, query):
                return QueryIntent.AGGREGATE
            
        # Distinct patterns
        if any(word in query for word in ['distinct', 'unique', 'different']):
            return QueryIntent.DISTINCT
            
        # Default to find
        return QueryIntent.FIND
    
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
    
    def find_best_field_for_attribute(self, collection: str, attribute_type: str, value: str) -> Tuple[str, float]:
        """
        Find the best field for a given attribute type (e.g., 'name', 'interest').
        
        Args:
            collection: Collection name
            attribute_type: Type of attribute (e.g., 'name', 'interest')
            value: The value for the attribute
            
        Returns:
            Tuple of (field_name, confidence)
        """
        if collection not in self.schema:
            return None, 0
            
        # Determine attribute-specific patterns
        if attribute_type == "name":
            # Try fields we've identified as name fields
            name_fields_in_collection = [field for coll, field in self.name_fields if coll == collection]
            for field in name_fields_in_collection:
                return field, 0.9
                
            # Try common name fields as fallback
            common_name_fields = ["fullName", "name", "username", "displayName", "firstName", "lastName"]
            for field in common_name_fields:
                if field in self.schema[collection]:
                    return field, 0.8
                    
        elif attribute_type == "interest":
            # Try fields we've identified as interest fields
            interest_fields_in_collection = [field for coll, field in self.interest_fields if coll == collection]
            for field in interest_fields_in_collection:
                return field, 0.9
                
            # Try common interest-related fields as fallback
            common_interest_fields = ["interests", "tags", "skills", "hobbies", "preferences"]
            for field in common_interest_fields:
                if field in self.schema[collection]:
                    return field, 0.8
        
        # If no direct match, try fuzzy matching
        matches = difflib.get_close_matches(
            attribute_type,
            list(self.schema[collection].keys()),
            n=1,
            cutoff=0.6
        )
        
        if matches:
            return matches[0], 0.7
            
        return None, 0
    
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
            
        fields = list(self.schema[collection].keys())
        
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
        
        # CASE 1: Interest/tag-based queries - handle special case
        interest_patterns = [
            r'with\s+interests?\s+(?:of|in|on)?\s+([a-zA-Z0-9_]+)',
            r'interests?\s+(?:of|in|on)?\s+([a-zA-Z0-9_]+)',
            r'interested\s+(?:in|on)?\s+([a-zA-Z0-9_]+)',
            r'like\s+([a-zA-Z0-9_]+)',
            r'know\s+([a-zA-Z0-9_]+)',
        ]
        
        for pattern in interest_patterns:
            match = re.search(pattern, query)
            if match:
                interest_value = match.group(1).strip()
                
                # Find the best interest field
                interest_field, confidence = self.find_best_field_for_attribute(collection, "interest", interest_value)
                
                if interest_field and confidence >= 0.6:
                    # Check if it's an array field
                    if (collection, interest_field) in self.array_fields:
                        filter_conditions[interest_field] = {"$in": [interest_value]}
                    else:
                        filter_conditions[interest_field] = interest_value
                    return filter_conditions
        
        # CASE 2: Name-based queries - handle special case
        name_patterns = [
            r'with\s+(?:name|fullname|full\s+name|username|display\s+name)\s+(?:as|is|=|:)?\s+([a-zA-Z0-9_\s]+)(?:\s+in|\s*$)',
            r'(?:name|fullname|full\s+name|username|display\s+name)\s+(?:as|is|=|:)?\s+([a-zA-Z0-9_\s]+)(?:\s+in|\s*$)',
            r'named\s+([a-zA-Z0-9_\s]+)(?:\s+in|\s*$)',
            r'called\s+([a-zA-Z0-9_\s]+)(?:\s+in|\s*$)',
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, query)
            if match:
                name_value = match.group(1).strip()
                
                # Remove words like 'as', 'is' if they somehow got captured
                name_value = self.extract_value_from_text(name_value, ['as', 'is', '=', ':'])
                
                # Find the best name field
                name_field, confidence = self.find_best_field_for_attribute(collection, "name", name_value)
                
                if name_field and confidence >= 0.6:
                    filter_conditions[name_field] = name_value
                    return filter_conditions
                    
                # If no specific name field found, try multiple fields
                if collection == "users":  # Special handling for users collection
                    name_fields = []
                    for field in self.schema[collection]:
                        if "name" in field.lower():
                            name_fields.append(field)
                    
                    if name_fields:
                        if len(name_fields) == 1:
                            filter_conditions[name_fields[0]] = name_value
                        else:
                            # Use $or for multiple name fields
                            filter_conditions["$or"] = [
                                {field: name_value} for field in name_fields
                            ]
                        return filter_conditions
        
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
                
                # Find the best matching field for this term
                field_name, confidence = self.fuzzy_match_field(field_term, collection)
                
                if field_name and confidence >= 0.6:
                    # Check if this is an array field and adjust the query accordingly
                    if (collection, field_name) in self.array_fields:
                        filter_conditions[field_name] = {"$in": [value]}
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