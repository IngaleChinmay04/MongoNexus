"""
Natural Language Query Processor for MongoDB.
This module handles parsing natural language into MongoDB queries.
"""
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import string
from collections import defaultdict

logger = logging.getLogger(__name__)

class QueryIntent:
    """Enum-like class to represent different query intents."""
    COUNT = "count"
    FIND = "find"
    AGGREGATE = "aggregate"
    DISTINCT = "distinct"
    UNKNOWN = "unknown"

class FieldMatchScore:
    """Score constants for field matching."""
    EXACT_MATCH = 10
    PARTIAL_MATCH = 5
    SEMANTIC_MATCH = 3
    FUZZY_MATCH = 2
    NO_MATCH = 0

class NLQueryProcessor:
    """
    Natural Language Query Processor that converts English queries into MongoDB operations.
    Uses schema information to make informed decisions about fields and collections.
    """
    
    # Common semantic mappings between natural language terms and database fields
    FIELD_SYNONYMS = {
        "name": ["fullname", "full name", "full_name", "username", "user name", "user_name", "alias"],
        "email": ["mail", "email address", "e-mail", "emailaddress", "mail address"],
        "age": ["years", "years old"],
        "id": ["identifier", "_id", "uuid", "userid", "uid"],
        "date": ["timestamp", "time", "datetime", "created", "created at", "createdat"],
        "user": ["users", "person", "people", "account", "accounts", "profile", "profiles"],
        "total": ["count", "number", "sum", "amount"],
        "active": ["enabled", "status", "state"],
        "interest": ["interests", "hobby", "hobbies", "likes", "preference", "preferences"],
        # Add more semantic mappings as needed
    }
    
    # Collection name to avoid matching as filters
    COLLECTION_NAMES = ["users", "posts", "comments", "products", "orders"]
    
    # Words that indicate a count operation
    COUNT_INDICATORS = {
        "count", "total", "number", "how many", "amount", "sum", "tally"
    }
    
    # Common entity types used to interpret sentences
    ENTITY_TYPES = {
        "users": ["user", "person", "account", "profile", "people", "member", "members"],
        "interests": ["interest", "hobby", "hobbies", "likes", "preference", "preferences"]
    }
    
    # Words that indicate filtering operations
    FILTER_OPERATORS = {
        "equal": "=",
        "equals": "=",
        "is": "=",
        "exactly": "=",
        "greater than": ">",
        "more than": ">",
        "larger than": ">",
        "higher than": ">",
        "above": ">",
        "over": ">",
        "less than": "<",
        "smaller than": "<",
        "lower than": "<",
        "below": "<",
        "under": "<",
        "at least": ">=",
        "greater than or equal": ">=",
        "at most": "<=",
        "less than or equal": "<=",
        "not equal": "!=",
        "different from": "!=",
        "other than": "!=",
        "between": "range",
        "in range": "range",
        "from": "range",
        # Add more operators as needed
    }
    
    def __init__(self, db_name: str, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize the NL Query Processor.
        
        Args:
            db_name: The database name
            schema: Optional schema information
        """
        self.db_name = db_name
        self.schema = schema or {}
        
        # Add known collection names to avoid matching as filters
        if schema:
            self.COLLECTION_NAMES = list(schema.keys())
        
        # Create reverse mappings for field synonyms
        self.reverse_field_map = {}
        for field, synonyms in self.FIELD_SYNONYMS.items():
            for synonym in synonyms:
                self.reverse_field_map[synonym] = field
                
        # Cache field information for each collection
        self.collection_fields = {}
        for collection, fields in self.schema.items():
            self.collection_fields[collection] = set(fields.keys())
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the query text to normalize it.
        
        Args:
            query: The original query
            
        Returns:
            Normalized query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Remove punctuation that's not meaningful for queries
        for punct in ".,;:!?":
            query = query.replace(punct, " ")
        
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def identify_search_target(self, query: str) -> Tuple[str, str, float]:
        """
        Identify the target collection and entity being searched.
        
        Args:
            query: The preprocessed query string
            
        Returns:
            Tuple of (collection_name, entity_type, confidence)
        """
        # Try to infer what the user is searching for based on entity types
        words = set(query.split())
        
        # Default to users if it seems like a person query
        for entity_term in self.ENTITY_TYPES["users"]:
            if entity_term in words:
                if "users" in self.schema:
                    return "users", "user", 0.8
                
        # If the query has terms like 'interest', it's likely about user interests
        for interest_term in self.ENTITY_TYPES["interests"]:
            if interest_term in words:
                if "users" in self.schema and "interests" in self.collection_fields.get("users", set()):
                    return "users", "interest", 0.75
                    
        # Default to first collection
        if self.schema:
            first_collection = next(iter(self.schema.keys()))
            return first_collection, "unknown", 0.1
            
        return "", "unknown", 0.0
    
    def extract_collection_name(self, query: str) -> Tuple[Optional[str], float]:
        """
        Extract collection name from the query with confidence score.
        
        Args:
            query: The query text
            
        Returns:
            Tuple of (collection_name, confidence_score)
        """
        # Check if this looks like a query about interests
        if re.search(r'(?:interest|hobby|like)\s+(?:in|of|is)\s+(\w+)', query, re.IGNORECASE):
            if "users" in self.schema:
                return "users", 0.9
                
        # Pattern for explicitly mentioned collection
        collection_patterns = [
            r'in\s+(?:the\s+)?([a-zA-Z0-9_]+)(?:\s+collection)?',  # in users collection
            r'from\s+(?:the\s+)?([a-zA-Z0-9_]+)(?:\s+collection)?',  # from users collection
            r'of\s+(?:the\s+)?([a-zA-Z0-9_]+)(?:\s+collection)?',   # of users collection
            r'(?:among|across)\s+(?:the\s+)?([a-zA-Z0-9_]+)'        # among users
        ]
        
        for pattern in collection_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                collection_name = match.group(1).lower()
                # Verify this is actually a collection, not a field value
                if collection_name in self.schema:
                    logger.info(f"Extracted collection name from query with pattern: {collection_name}")
                    return collection_name, 0.9  # High confidence for explicit mentions
        
        # If no explicit collection, look for collection-like terms in the query
        words = query.split()
        collections = set(self.schema.keys())
        
        for word in words:
            word = word.lower()
            # Check for exact collection name
            if word in collections:
                # Make sure this isn't a field value by checking context
                if self.is_likely_collection_reference(query, word):
                    logger.info(f"Found collection name as word in query: {word}")
                    return word, 0.8
                
            # Check for plural/singular forms
            if word.endswith('s') and word[:-1] in collections:
                if self.is_likely_collection_reference(query, word[:-1]):
                    logger.info(f"Found singular form of collection in query: {word[:-1]}")
                    return word[:-1], 0.7
                
            if not word.endswith('s') and word + 's' in collections:
                if self.is_likely_collection_reference(query, word + 's'):
                    logger.info(f"Found plural form of collection in query: {word + 's'}")
                    return word + 's', 0.7
                
        # Look for words that might refer to a collection
        # For queries about users, default to users collection
        if any(user_term in query for user_term in self.ENTITY_TYPES["users"]):
            if "users" in collections:
                logger.info(f"Defaulting to users collection for user-related query")
                return "users", 0.6
                
        # If query mentions interest/hobby/preference, it's probably about users
        if any(interest_term in query for interest_term in self.ENTITY_TYPES["interests"]):
            if "users" in collections and "interests" in self.collection_fields.get("users", set()):
                logger.info(f"Defaulting to users collection for interest-related query")
                return "users", 0.65
        
        # If still no match, try to infer from content
        potential_collections = []
        
        for coll_name in collections:
            # Simple heuristic: If query contains words in field names of this collection
            if coll_name in self.schema:
                collection_fields = self.schema[coll_name].keys()
                field_words = set(" ".join(collection_fields).split())
                word_overlap = len(set(words) & field_words)
                
                if word_overlap > 0:
                    potential_collections.append((coll_name, word_overlap / len(collection_fields)))
        
        if potential_collections:
            # Sort by overlap score
            potential_collections.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Inferred collection from content overlap: {potential_collections[0][0]}")
            return potential_collections[0][0], 0.5 * potential_collections[0][1]  # Lower confidence for inferred collections
        
        # Fall back to first collection with a very low confidence
        if collections:
            first_collection = next(iter(collections))
            logger.info(f"Falling back to first available collection: {first_collection}")
            return first_collection, 0.1
            
        return None, 0.0
        
    def is_likely_collection_reference(self, query: str, term: str) -> bool:
        """
        Check if a term is likely referring to a collection rather than a filter value.
        
        Args:
            query: The full query
            term: The term to check
            
        Returns:
            True if likely a collection reference
        """
        # If term is preceded by prepositions like "in", "from", "of", it's likely a collection
        collection_indicators = [
            rf'\b(?:in|from|of|among|across)\s+(?:the\s+)?{term}\b',
            rf'{term}(?:\s+collection)',
        ]
        
        for pattern in collection_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                return True
                
        # If preceded by "with" or "where", it's likely a field value, not a collection
        value_indicators = [
            rf'(?:with|where)\s+\w+\s+(?:is|=|:)\s+{term}\b',
            rf'(?:with|where)\s+{term}\s+',
            rf'(?:has|have|having)\s+{term}\b',
        ]
        
        for pattern in value_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                return False
                
        # Default to true for actual collection names
        return term in self.schema
    
    def determine_query_intent(self, query: str) -> str:
        """
        Determine the intent of the query (count, find, etc.).
        
        Args:
            query: The query text
            
        Returns:
            Query intent
        """
        query = query.lower()
        
        # Specific pattern for total/count questions that are common
        count_patterns = [
            r"(?:what|how)\s+(?:is|are)\s+(?:the\s+)?(?:total|number)\s+(?:of|in|for)\s+",
            r"(?:how\s+many)\s+(?:\w+\s+)?(?:are|is|do|does)\s+",
            r"(?:count|show|get|find|display)\s+(?:the\s+)?(?:total|number|count)\s+(?:of|for|in)\s+",
        ]
        
        if any(re.search(pattern, query) for pattern in count_patterns):
            logger.info("Detected COUNT intent from specific pattern")
            return QueryIntent.COUNT
        
        # Check for count words
        if any(indicator in query for indicator in self.COUNT_INDICATORS):
            logger.info("Detected COUNT intent from indicators")
            return QueryIntent.COUNT
            
        # Check for common find patterns
        find_patterns = [
            r"(?:find|show|list|get|display|give me|return)",
            r"(?:what|who|which)"
        ]
        
        if any(re.search(pattern, query) for pattern in find_patterns):
            logger.info("Detected FIND intent")
            return QueryIntent.FIND
            
        # Check for aggregation patterns
        agg_patterns = [
            r"(?:average|avg|mean|median|mode|sum|total|calculate)",
            r"(?:group by|grouped by|for each)"
        ]
        
        if any(re.search(pattern, query) for pattern in agg_patterns):
            logger.info("Detected AGGREGATE intent")
            return QueryIntent.AGGREGATE
            
        # Check for distinct patterns
        if re.search(r"(?:distinct|unique|different)", query):
            logger.info("Detected DISTINCT intent")
            return QueryIntent.DISTINCT
            
        # Default to find
        logger.info("No specific intent detected, defaulting to FIND")
        return QueryIntent.FIND
    
    def find_matching_fields(self, term: str, collection_name: str) -> List[Tuple[str, float]]:
        """
        Find fields in the collection that match the given term.
        
        Args:
            term: The term to match
            collection_name: The collection to search in
            
        Returns:
            List of (field_name, score) tuples
        """
        if collection_name not in self.schema:
            return []
            
        collection_fields = self.schema[collection_name]
        matches = []
        
        # Clean term
        term = term.lower().strip()
        
        # 1. Check for exact matches
        if term in collection_fields:
            matches.append((term, FieldMatchScore.EXACT_MATCH))
            
        # 2. Check for field names that contain the term
        for field in collection_fields:
            if field.lower() == term:
                if (field, FieldMatchScore.EXACT_MATCH) not in matches:
                    matches.append((field, FieldMatchScore.EXACT_MATCH))
            elif term in field.lower():
                matches.append((field, FieldMatchScore.PARTIAL_MATCH))
                
        # 3. Check semantic matches using synonyms
        if term in self.FIELD_SYNONYMS:
            synonyms = self.FIELD_SYNONYMS[term]
            for field in collection_fields:
                if field.lower() in synonyms:
                    matches.append((field, FieldMatchScore.SEMANTIC_MATCH))
                    
        # 4. Check reverse mapping of synonyms
        if term in self.reverse_field_map:
            canonical_term = self.reverse_field_map[term]
            if canonical_term in collection_fields:
                matches.append((canonical_term, FieldMatchScore.SEMANTIC_MATCH))
                
        # 5. Try fuzzy matching (simple case of checking word parts)
        if not matches:
            term_parts = set(term.split('_'))
            for field in collection_fields:
                field_parts = set(field.lower().split('_'))
                if term_parts & field_parts:  # If there's any intersection
                    matches.append((field, FieldMatchScore.FUZZY_MATCH))
                    
        # Sort matches by score and return
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def extract_conditions(self, query: str, collection_name: str) -> Dict[str, Any]:
        """
        Extract filter conditions from the query.
        
        Args:
            query: The query text
            collection_name: The collection name
            
        Returns:
            MongoDB filter conditions
        """
        if collection_name not in self.schema:
            return {}
            
        filter_query = {}
        query = query.lower()
        
        # Special case for interests in the users collection
        interest_match = re.search(r'(?:interest|hobby|like)\s+(?:in|of|is)\s+(\w+)', query, re.IGNORECASE)
        if interest_match and collection_name == "users" and "interests" in self.collection_fields.get("users", set()):
            interest_value = interest_match.group(1).strip()
            if interest_value not in self.COLLECTION_NAMES:  # Make sure we're not matching a collection
                logger.info(f"Found interest condition: {interest_value}")
                # Return array contains query for interests
                return {"interests": {"$in": [interest_value]}}
        
        # Special case for name in the users collection
        name_match = re.search(r'(?:with|where)?\s+(?:name|fullname|username)\s+(?:is|=|:|of)?\s+([^,]+?)(?:,|\s+(?:and|or)|\s+in\s+|$)', query, re.IGNORECASE)
        if name_match and collection_name == "users":
            name_value = name_match.group(1).strip()
            
            # Check if the name actually contains "in {collection}"
            if "in " in name_value:
                name_value = name_value.split("in ")[0].strip()
                
            logger.info(f"Found name condition: {name_value}")
            
            # Try with various name fields that might exist in users collection
            name_fields = []
            for field in ["fullName", "name", "username"]:
                if field in self.collection_fields.get("users", set()):
                    name_fields.append(field)
                    
            if name_fields:
                if len(name_fields) == 1:
                    # If only one name field exists, use it
                    return {name_fields[0]: name_value}
                else:
                    # If multiple name fields exist, use $or
                    return {"$or": [{field: name_value} for field in name_fields]}
        
        # Look for patterns like "with field value" or "where field is value"
        condition_patterns = [
            r'(?:with|where|having)\s+(\w+)\s+(?:is|=|==|equals?|:)\s+["\']?([^"\']+?)["\']?(?:,|\s+(?:and|or)|\s+in\s+|$)',
            r'(?:with|where|having)\s+(\w+)\s+([^,]+?)(?:,|\s+(?:and|or)|\s+in\s+|$)',
            r'(\w+)\s+(?:is|=|==|equals?|:)\s+["\']?([^"\']+?)["\']?(?:,|\s+(?:and|or)|\s+in\s+|$)'
        ]
        
        for pattern in condition_patterns:
            for match in re.finditer(pattern, query):
                field_term, value = match.groups()
                field_term = field_term.strip()
                value = value.strip()
                
                # Skip if this is part of "in the collection" phrase
                if "collection" in value:
                    continue
                    
                # Skip if this field or value is a collection name
                if field_term in self.COLLECTION_NAMES or value in self.COLLECTION_NAMES:
                    continue
                    
                # Find matching fields in the collection
                matching_fields = self.find_matching_fields(field_term, collection_name)
                
                if matching_fields:
                    field_name = matching_fields[0][0]  # Use highest scoring field
                    filter_query[field_name] = value
                    
        return filter_query
    
    def determine_limit(self, query: str) -> int:
        """
        Determine the limit from the query.
        
        Args:
            query: The query text
            
        Returns:
            Limit value
        """
        # Look for patterns like "top 10" or "limit 5"
        limit_patterns = [
            r'(?:top|first|limit)\s+(\d+)',
            r'(\d+)\s+(?:records|results|entries|documents)',
        ]
        
        for pattern in limit_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    pass
                    
        # Default limits based on intent
        if self.determine_query_intent(query) == QueryIntent.COUNT:
            return 1000  # We'll need documents to count them
        else:
            return 10  # Default limit for other queries
    
    def determine_sort(self, query: str, collection_name: str) -> Optional[List[Tuple[str, int]]]:
        """
        Determine the sort order from the query.
        
        Args:
            query: The query text
            collection_name: The collection name
            
        Returns:
            List of (field, direction) tuples or None
        """
        if collection_name not in self.schema:
            return None
            
        query = query.lower()
        
        # Look for sort patterns
        sort_patterns = [
            r'(?:sort|order)\s+by\s+(\w+)\s+(asc|ascending|desc|descending)',
            r'(?:sort|order)\s+by\s+(\w+)',
            r'(?:in|by)\s+(\w+)\s+(?:asc|ascending|desc|descending)(?:\s+order)?',
        ]
        
        for pattern in sort_patterns:
            match = re.search(pattern, query)
            if match:
                groups = match.groups()
                field_term = groups[0]
                
                # Determine direction
                direction = 1  # Default ascending
                if len(groups) > 1:
                    direction_term = groups[1] if len(groups) > 1 else ""
                    if direction_term and direction_term.startswith(("desc", "reverse")):
                        direction = -1
                
                # Find matching fields in the collection
                matching_fields = self.find_matching_fields(field_term, collection_name)
                
                if matching_fields:
                    field_name = matching_fields[0][0]  # Use highest scoring field
                    return [(field_name, direction)]
                    
        return None
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query into MongoDB query parameters.
        
        Args:
            query: The natural language query
            
        Returns:
            MongoDB query parameters
        """
        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Determine query intent
        intent = self.determine_query_intent(processed_query)
        
        # Extract collection name
        collection_name, collection_confidence = self.extract_collection_name(processed_query)
        
        if not collection_name:
            logger.warning("Could not determine collection name")
            return {
                "error": "Could not determine which collection to query. Please specify a collection name in your query."
            }
            
        # Extract conditions
        filter_query = self.extract_conditions(processed_query, collection_name)
        
        # Determine limit
        limit = self.determine_limit(processed_query)
        
        # Determine sort
        sort = self.determine_sort(processed_query, collection_name)
        
        # Build MongoDB query parameters
        query_params = {
            "db_name": self.db_name,
            "collection_name": collection_name,
            "filter": filter_query,
            "limit": limit
        }
        
        if sort:
            query_params["sort"] = sort
            
        # Add metadata about the processing
        query_params["_meta"] = {
            "intent": intent,
            "collection_confidence": collection_confidence,
            "original_query": query,
            "processed_query": processed_query
        }
        
        logger.info(f"Processed query into parameters: {query_params}")
        return query_params