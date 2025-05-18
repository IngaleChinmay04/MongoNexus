"""
Knowledge Graph component for the Graph RAG system.
Uses Neo4j to store and query relational information about database schema and entities.
"""
import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Set
import neo4j
from neo4j import GraphDatabase
from app.config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Knowledge Graph component that manages schema information and relationships
    using Neo4j as the graph database backend.
    """
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """
        Initialize the knowledge graph with Neo4j connection parameters.
        
        Args:
            uri: Neo4j URI (defaults to environment variable)
            user: Neo4j username (defaults to environment variable)
            password: Neo4j password (defaults to environment variable)
        """
        self.uri = uri or NEO4J_URI or "bolt://localhost:7687"
        self.user = user or NEO4J_USER or "neo4j"
        self.password = password or NEO4J_PASSWORD or "password"
        self.driver = None
        self.is_connected = False
        
    def connect(self) -> bool:
        """
        Connect to the Neo4j database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connection
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            self.is_connected = True
            logger.info(f"Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.is_connected = False
            return False
            
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            self.is_connected = False
            
    def create_schema_graph(self, db_name: str, schema: Dict[str, Any]) -> bool:
        """
        Create a graph representation of database schema.
        
        Args:
            db_name: Database name
            schema: Database schema (collection -> fields mapping)
            
        Returns:
            True if creation successful, False otherwise
        """
        if not self.is_connected and not self.connect():
            logger.error("Cannot create schema graph: not connected to Neo4j")
            return False
            
        try:
            with self.driver.session() as session:
                # Create database node
                session.run(
                    "MERGE (d:Database {name: $name}) "
                    "SET d.type = 'mongodb'",
                    name=db_name
                )
                
                # Create collection nodes and their relationships to the database
                for collection_name, fields in schema.items():
                    # Create collection node
                    session.run(
                        "MATCH (d:Database {name: $db_name}) "
                        "MERGE (c:Collection {name: $coll_name}) "
                        "MERGE (d)-[:CONTAINS]->(c)",
                        db_name=db_name, coll_name=collection_name
                    )
                    
                    # Create field nodes and their relationships to the collection
                    for field_name, field_type in fields.items():
                        # Create field node with its type
                        session.run(
                            "MATCH (c:Collection {name: $coll_name}) "
                            "MERGE (f:Field {name: $field_name, collection: $coll_name}) "
                            "SET f.type = $field_type "
                            "MERGE (c)-[:HAS_FIELD]->(f)",
                            coll_name=collection_name, field_name=field_name, field_type=str(field_type)
                        )
                
                # Infer and create semantic connections between related fields
                self._create_semantic_connections(session, schema)
                
                logger.info(f"Created schema graph for database {db_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating schema graph: {str(e)}", exc_info=True)
            return False
    
    def _create_semantic_connections(self, session, schema: Dict[str, Any]):
        """
        Create semantic connections between related fields across collections.
        
        Args:
            session: Neo4j session
            schema: Database schema
        """
        # Look for fields with similar names that might represent relationships
        relationship_indicators = [
            ("id", "Id", "_id"),  # ID fields
            ("user", "User"),     # User references
            ("product", "Product"), # Product references
            # Add more common relationship patterns
        ]
        
        for coll1_name, fields1 in schema.items():
            for coll2_name, fields2 in schema.items():
                if coll1_name == coll2_name:
                    continue  # Skip self-relationships
                    
                # Look for potential relationships between collections
                for field1_name in fields1:
                    for indicator_group in relationship_indicators:
                        # Check if this field might be referencing the other collection
                        if any(indicator in field1_name for indicator in indicator_group) and \
                           coll2_name.lower() in field1_name.lower():
                           
                            # Create a REFERENCES relationship
                            session.run(
                                "MATCH (f1:Field {name: $field1, collection: $coll1}) "
                                "MATCH (c2:Collection {name: $coll2}) "
                                "MERGE (f1)-[:REFERENCES]->(c2)",
                                field1=field1_name, coll1=coll1_name, coll2=coll2_name
                            )
                            logger.info(f"Created reference: {coll1_name}.{field1_name} -> {coll2_name}")
    
    def add_entity_examples(self, collection_name: str, examples: List[Dict[str, Any]]):
        """
        Add example entities to the knowledge graph to improve query understanding.
        
        Args:
            collection_name: Collection name
            examples: List of example documents from the collection
        """
        if not self.is_connected and not self.connect():
            logger.error("Cannot add entity examples: not connected to Neo4j")
            return
            
        try:
            with self.driver.session() as session:
                for example in examples[:10]:  # Limit to 10 examples per collection
                    # Create entity node with sample data
                    properties = {k: str(v)[:100] for k, v in example.items() if k != "_id"}
                    properties["id"] = str(example.get("_id", ""))
                    
                    # Create entity node
                    session.run(
                        "MATCH (c:Collection {name: $coll_name}) "
                        "CREATE (e:Entity {id: $id, collection: $coll_name}) "
                        "SET e += $properties "
                        "CREATE (c)-[:HAS_ENTITY]->(e)",
                        coll_name=collection_name, id=properties["id"], properties=properties
                    )
        except Exception as e:
            logger.error(f"Error adding entity examples: {str(e)}", exc_info=True)
    
    def query_schema_info(self, query_terms: List[str]) -> Dict[str, Any]:
        """
        Query the knowledge graph for information related to the given query terms.
        
        Args:
            query_terms: List of terms from the user query
            
        Returns:
            Relevant schema information from the knowledge graph
        """
        if not self.is_connected and not self.connect():
            logger.error("Cannot query schema info: not connected to Neo4j")
            return {}
            
        results = {}
        
        try:
            with self.driver.session() as session:
                # Query for collections matching the terms
                collections = session.run(
                    """
                    MATCH (c:Collection)
                    WHERE any(term IN $query_terms WHERE c.name CONTAINS term)
                    RETURN c.name AS name
                    UNION
                    MATCH (f:Field)-[:REFERENCES]->(c:Collection)
                    WHERE any(term IN $query_terms WHERE f.name CONTAINS term)
                    RETURN c.name AS name
                    """,
                    query_terms=query_terms
                ).values()
                
                if collections:
                    results["collections"] = [c[0] for c in collections]
                
                # Query for fields matching the terms
                fields = session.run(
                    """
                    MATCH (c:Collection)-[:HAS_FIELD]->(f:Field)
                    WHERE any(term IN $query_terms WHERE f.name CONTAINS term)
                    RETURN c.name AS collection, f.name AS field, f.type AS type
                    """,
                    query_terms=query_terms
                ).data()
                
                if fields:
                    results["fields"] = fields
                    
                # Query for semantic connections related to the terms
                connections = session.run(
                    """
                    MATCH (f1:Field)-[r:REFERENCES]->(c:Collection)
                    WHERE any(term IN $query_terms 
                        WHERE f1.name CONTAINS term OR c.name CONTAINS term)
                    RETURN f1.collection AS source_collection, 
                           f1.name AS source_field,
                           c.name AS target_collection
                    """,
                    query_terms=query_terms
                ).data()
                
                if connections:
                    results["connections"] = connections
                
                return results
                
        except Exception as e:
            logger.error(f"Error querying schema info: {str(e)}", exc_info=True)
            return {}
            
    def get_field_suggestions(self, collection_name: str, partial_field_name: str) -> List[str]:
        """
        Get field suggestions based on partial field name.
        
        Args:
            collection_name: Collection name
            partial_field_name: Partial field name to match
            
        Returns:
            List of suggested field names
        """
        if not self.is_connected and not self.connect():
            logger.error("Cannot get field suggestions: not connected to Neo4j")
            return []
            
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Collection {name: $coll_name})-[:HAS_FIELD]->(f:Field)
                    WHERE f.name CONTAINS $partial_name
                    RETURN f.name AS field_name
                    """,
                    coll_name=collection_name, partial_name=partial_field_name
                )
                
                return [record["field_name"] for record in result]
                
        except Exception as e:
            logger.error(f"Error getting field suggestions: {str(e)}", exc_info=True)
            return []
    
    def get_query_suggestions(self, user_query: str) -> Dict[str, Any]:
        """
        Get query suggestions based on the user's natural language query.
        
        Args:
            user_query: User's natural language query
            
        Returns:
            Suggested collections, fields, and conditions
        """
        # Extract terms from query
        terms = [term.lower() for term in user_query.split() if len(term) > 2]
        
        if not self.is_connected and not self.connect():
            logger.error("Cannot get query suggestions: not connected to Neo4j")
            return {}
            
        suggestions = {
            "collections": [],
            "fields": [],
            "conditions": []
        }
        
        try:
            with self.driver.session() as session:
                # Find matching collections
                coll_result = session.run(
                    """
                    MATCH (c:Collection)
                    WHERE any(term IN $terms WHERE toLower(c.name) CONTAINS toLower(term))
                    RETURN c.name AS name, count(*) AS relevance
                    ORDER BY relevance DESC
                    LIMIT 3
                    """,
                    terms=terms
                )
                
                suggestions["collections"] = [record["name"] for record in coll_result]
                
                # Find matching fields
                field_result = session.run(
                    """
                    MATCH (c:Collection)-[:HAS_FIELD]->(f:Field)
                    WHERE any(term IN $terms WHERE toLower(f.name) CONTAINS toLower(term))
                    RETURN c.name AS collection, f.name AS field, 
                           count(*) AS relevance
                    ORDER BY relevance DESC
                    LIMIT 5
                    """,
                    terms=terms
                )
                
                suggestions["fields"] = [
                    {"collection": record["collection"], "field": record["field"]} 
                    for record in field_result
                ]
                
                return suggestions
                
        except Exception as e:
            logger.error(f"Error getting query suggestions: {str(e)}", exc_info=True)
            return suggestions