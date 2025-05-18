"""Tests for the MongoDB schema service."""
import pytest
from app.services.mongodb.schema_service import infer_collection_schema

@pytest.mark.asyncio
async def test_infer_collection_schema(test_mongodb_client):
    """Test that schema inference works correctly."""
    # Setup test data
    collection_name = "test_collection"
    test_collection = test_mongodb_client[collection_name]
    
    # Insert test documents with different schemas
    await test_collection.insert_many([
        {"name": "John", "age": 30, "email": "john@example.com"},
        {"name": "Jane", "age": 25, "isActive": True},
        {"name": "Bob", "email": "bob@example.com", "tags": ["user", "admin"]}
    ])
    
    # Call the schema inference function
    schema = await infer_collection_schema(test_mongodb_client, collection_name)
    
    # Verify the result
    assert schema["collection_name"] == collection_name
    assert "name" in schema["fields"]
    assert schema["fields"]["name"] == "str"
    assert "age" in schema["fields"]
    assert schema["fields"]["age"] == "int"
    assert "email" in schema["fields"]
    assert schema["fields"]["email"] == "str"
    assert "isActive" in schema["fields"]
    assert schema["fields"]["isActive"] == "bool"
    assert "tags" in schema["fields"]
    assert schema["fields"]["tags"] == "list"
