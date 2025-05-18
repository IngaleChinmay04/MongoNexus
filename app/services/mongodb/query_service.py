"""MongoDB query service."""
from typing import Dict, List, Any, Optional
import json
from bson import json_util
import bson
from datetime import datetime, date
from bson.objectid import ObjectId

class MongoJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB BSON types."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, bson.Binary):
            return str(obj)
        return super(MongoJSONEncoder, self).default(obj)

def serialize_mongo_doc(doc):
    """
    Serialize MongoDB document to JSON-compatible format.
    Handles BSON types like ObjectId and datetime.
    """
    return json.loads(MongoJSONEncoder().encode(doc))

async def find_documents(
    db,
    collection_name: str,
    filter_query: Dict[str, Any],
    projection: Optional[Dict[str, Any]] = None,
    sort: Optional[List[Dict[str, int]]] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Find documents in a MongoDB collection.
    
    Args:
        db: MongoDB database client
        collection_name: Name of the collection to query
        filter_query: MongoDB filter query document
        projection: Optional MongoDB projection document
        sort: Optional sort criteria
        skip: Optional number of documents to skip
        limit: Optional maximum number of documents to return
        
    Returns:
        List of documents matching the query
    """
    # Get the collection
    collection = db[collection_name]
    
    # Parse ObjectIds in filter_query if they're in string format
    filter_query = await parse_query_object_ids(filter_query)
    
    # Prepare find options
    find_options = {}
    
    if projection:
        find_options["projection"] = projection
        
    # Execute the query
    cursor = collection.find(filter_query, **find_options)
    
    # Apply skip and limit
    if skip > 0:
        cursor = cursor.skip(skip)
    
    if limit > 0:
        cursor = cursor.limit(limit)
    
    # Apply sort if provided
    if sort:
        # Convert sort list of dicts to list of tuples
        sort_list = []
        for sort_item in sort:
            for field, direction in sort_item.items():
                sort_list.append((field, direction))
        
        cursor = cursor.sort(sort_list)
    
    # Convert documents to proper JSON serializable format
    documents = []
    async for doc in cursor:
        serialized_doc = serialize_mongo_doc(doc)
        documents.append(serialized_doc)
    
    return documents

async def count_documents(
    db,
    collection_name: str,
    filter_query: Dict[str, Any]
) -> int:
    """
    Count documents in a MongoDB collection.
    
    Args:
        db: MongoDB database client
        collection_name: Name of the collection to query
        filter_query: MongoDB filter query document
        
    Returns:
        Number of matching documents
    """
    # Get the collection
    collection = db[collection_name]
    
    # Parse ObjectIds in filter_query if they're in string format
    filter_query = await parse_query_object_ids(filter_query)
    
    # Count matching documents
    return await collection.count_documents(filter_query)

async def parse_query_object_ids(query: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse string ObjectIds in a query document into actual ObjectId objects.
    This allows clients to send ObjectIds as strings in their queries.
    
    Args:
        query: MongoDB query document
        
    Returns:
        Query with string ObjectIds converted to ObjectId objects
    """
    if not query:
        return {}
        
    result = {}
    
    for key, value in query.items():
        # Handle ObjectId in _id field
        if key == '_id' and isinstance(value, str) and len(value) == 24:
            try:
                result[key] = ObjectId(value)
                continue
            except:
                # If not a valid ObjectId, use as is
                result[key] = value
                
        # Handle operator expressions like $in, $nin, etc.
        elif isinstance(value, dict) and all(k.startswith('$') for k in value.keys()):
            result[key] = {}
            for op, op_value in value.items():
                # Handle array operators
                if op in ('$in', '$nin') and isinstance(op_value, list):
                    result[key][op] = [
                        ObjectId(v) if isinstance(v, str) and len(v) == 24 else v 
                        for v in op_value
                    ]
                else:
                    result[key][op] = op_value
                    
        # Handle nested documents
        elif isinstance(value, dict):
            result[key] = await parse_query_object_ids(value)
            
        # Regular field
        else:
            result[key] = value
            
    return result