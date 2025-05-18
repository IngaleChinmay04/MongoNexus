"""
Field analyzer for MongoDB collections.
"""
import logging
import re
from typing import Dict, List, Any, Set, Tuple, Optional

logger = logging.getLogger(__name__)

def identify_text_fields(schema: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Identify which fields in each collection are likely text fields that should be
    searched case-insensitively.
    
    Args:
        schema: Database schema (collection -> fields mapping)
        
    Returns:
        Dict mapping collections to list of text fields
    """
    text_fields = {}
    
    for collection, fields in schema.items():
        collection_text_fields = []
        
        for field_name, field_type in fields.items():
            # Check if field is likely a text field based on type or name
            is_text_field = False
            
            # Check type indicators
            if isinstance(field_type, str):
                type_lower = field_type.lower()
                is_text_field = any(text_type in type_lower for text_type in [
                    'string', 'text', 'char', 'varchar'
                ])
            
            # Check field name indicators
            if not is_text_field:
                field_lower = field_name.lower()
                is_text_field = any(name_indicator in field_lower for name_indicator in [
                    'name', 'title', 'description', 'text', 'address', 
                    'email', 'phone', 'city', 'state', 'country', 'tag',
                    'interest', 'skill', 'bio', 'comment', 'message'
                ])
                
            if is_text_field:
                collection_text_fields.append(field_name)
                
        text_fields[collection] = collection_text_fields
    
    return text_fields

def analyze_string_fields_for_case(schema: Dict[str, Dict[str, Any]], sample_docs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, str]]:
    """
    Analyze string fields to determine if they are case-sensitive or case-insensitive.
    
    Args:
        schema: Database schema
        sample_docs: Sample documents for each collection
        
    Returns:
        Dict mapping collections to field case sensitivity information
    """
    case_sensitivity = {}
    
    for collection, docs in sample_docs.items():
        if collection not in schema:
            continue
            
        collection_info = {}
        text_fields = identify_text_fields({collection: schema[collection]})[collection]
        
        for field in text_fields:
            # Check field values in sample docs
            values = []
            for doc in docs:
                if field in doc and isinstance(doc[field], str):
                    values.append(doc[field])
                    
            if values:
                # Check if all values are uppercase
                all_uppercase = all(v == v.upper() for v in values if v)
                
                # Check if all values are lowercase
                all_lowercase = all(v == v.lower() for v in values if v)
                
                if all_uppercase:
                    collection_info[field] = "uppercase"
                elif all_lowercase:
                    collection_info[field] = "lowercase"
                else:
                    collection_info[field] = "mixed_case"
            else:
                collection_info[field] = "unknown"
                
        case_sensitivity[collection] = collection_info
        
    return case_sensitivity

def analyze_schema_fields(schema: Dict[str, Dict[str, Any]], sample_docs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Perform a more detailed analysis of database schema fields.
    
    Args:
        schema: Database schema (collection -> fields mapping)
        sample_docs: Sample documents for each collection
        
    Returns:
        Dict with detailed field information for each collection
    """
    field_analysis = {}
    
    for collection, fields in schema.items():
        collection_analysis = {}
        
        # Get samples for this collection
        samples = sample_docs.get(collection, [])
        
        for field_name, declared_type in fields.items():
            field_info = {
                "declared_type": declared_type,
                "inferred_type": "unknown",
                "nullable": True,
                "unique_values": set(),
                "sample_values": [],
                "case_sensitivity": "unknown",
                "array_field": False,
                "value_pattern": None
            }
            
            # Analyze sample values
            values = []
            for doc in samples:
                if field_name in doc:
                    values.append(doc[field_name])
                    
            if values:
                # Check for nullability
                field_info["nullable"] = any(v is None for v in values)
                
                # Get non-null values
                non_null_values = [v for v in values if v is not None]
                
                # Infer type
                if non_null_values:
                    value_types = set(type(v).__name__ for v in non_null_values)
                    
                    if len(value_types) == 1:
                        # Single consistent type
                        field_info["inferred_type"] = next(iter(value_types))
                    else:
                        # Mixed types
                        field_info["inferred_type"] = f"mixed: {', '.join(value_types)}"
                        
                    # Check if array field
                    field_info["array_field"] = any(isinstance(v, list) for v in non_null_values)
                    
                    # Collect unique values (limited to 10 to avoid memory issues)
                    unique_values = set()
                    for v in non_null_values[:20]:  # Only check first 20 values
                        try:
                            # Only add hashable values
                            if isinstance(v, (str, int, float, bool, tuple)):
                                unique_values.add(v)
                        except:
                            pass
                    field_info["unique_values"] = unique_values
                    
                    # Store sample values (limited)
                    field_info["sample_values"] = non_null_values[:5]
                    
                    # Check case sensitivity for string fields
                    if all(isinstance(v, str) for v in non_null_values):
                        string_values = [v for v in non_null_values if v.strip()]  # Non-empty strings
                        if string_values:
                            all_uppercase = all(v == v.upper() for v in string_values)
                            all_lowercase = all(v == v.lower() for v in string_values)
                            
                            if all_uppercase:
                                field_info["case_sensitivity"] = "uppercase"
                            elif all_lowercase:
                                field_info["case_sensitivity"] = "lowercase"
                            else:
                                field_info["case_sensitivity"] = "mixed"
                                
                            # Try to identify common patterns in string values
                            if len(string_values) >= 3:
                                field_info["value_pattern"] = identify_string_pattern(string_values)
            
            collection_analysis[field_name] = field_info
            
        field_analysis[collection] = collection_analysis
        
    return field_analysis

def identify_string_pattern(values: List[str]) -> Optional[str]:
    """
    Try to identify a common pattern in string values.
    
    Args:
        values: List of string values
        
    Returns:
        Pattern description or None
    """
    # Check for email pattern
    email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if all(re.match(email_pattern, v) for v in values):
        return "email"
    
    # Check for URL pattern
    url_pattern = r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(/[-\w%!$&\'()*+,;=:]+)*$'
    if all(re.match(url_pattern, v) for v in values):
        return "url"
    
    # Check for date pattern (simple)
    date_pattern = r'^\d{4}-\d{2}-\d{2}$'
    if all(re.match(date_pattern, v) for v in values):
        return "date (YYYY-MM-DD)"
    
    # Check for phone number pattern (simple)
    phone_pattern = r'^\+?[\d\s-]{7,15}$'
    if all(re.match(phone_pattern, v) for v in values):
        return "phone number"
    
    # Check if all values are single words (no spaces)
    if all(' ' not in v for v in values):
        return "single word"
    
    # Check if values appear to be names (capitalized words)
    name_pattern = r'^[A-Z][a-z]*(\s[A-Z][a-z]*)*$'
    if all(re.match(name_pattern, v) for v in values):
        return "proper name"
    
    return None