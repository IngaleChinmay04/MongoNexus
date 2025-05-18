"""
Vector Embedding component for the Graph RAG system.
Uses FAISS for efficient similarity search on embedded queries and schemas.
"""
import logging
import os
import numpy as np
import faiss
import pickle
from typing import Dict, List, Any, Optional, Tuple, Set
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector Store component that handles embedding and retrieval of semantic vectors
    for database schema elements, queries, and examples.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.index = None
            self.metadata = []  # Stores metadata for indexed items
            logger.info(f"Initialized vector store with model {model_name}, dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            self.model = None
            self.dimension = 0
    
    def is_initialized(self) -> bool:
        """Check if the vector store is properly initialized."""
        return self.model is not None and self.dimension > 0
    
    def create_index(self):
        """Create a new FAISS index."""
        if not self.is_initialized():
            logger.error("Cannot create index: vector store not initialized")
            return
            
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []
            logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {str(e)}")
    
    def add_schema_embeddings(self, db_name: str, schema: Dict[str, Any]):
        """
        Add database schema embeddings to the index.
        
        Args:
            db_name: Database name
            schema: Database schema (collection -> fields mapping)
        """
        if not self.is_initialized():
            logger.error("Cannot add schema embeddings: vector store not initialized")
            return
            
        if self.index is None:
            self.create_index()
            
        try:
            # Create descriptions for collections and fields to embed
            texts = []
            metadata_items = []
            
            # Add database
            db_text = f"Database: {db_name}"
            texts.append(db_text)
            metadata_items.append({
                "type": "database",
                "name": db_name
            })
            
            # Add collections and their fields
            for collection_name, fields in schema.items():
                # Embed collection
                coll_text = f"Collection: {collection_name}"
                texts.append(coll_text)
                metadata_items.append({
                    "type": "collection",
                    "name": collection_name,
                    "database": db_name
                })
                
                # Embed fields
                for field_name, field_type in fields.items():
                    field_text = f"Field: {field_name} in collection {collection_name} with type {field_type}"
                    texts.append(field_text)
                    metadata_items.append({
                        "type": "field",
                        "name": field_name,
                        "collection": collection_name,
                        "database": db_name,
                        "field_type": str(field_type)
                    })
                    
                    # Add common queries for this field
                    query_templates = [
                        f"Find {collection_name} with {field_name}",
                        f"Get {collection_name} where {field_name} is",
                        f"Show {collection_name} with {field_name}",
                        f"Count {collection_name} by {field_name}"
                    ]
                    
                    for template in query_templates:
                        texts.append(template)
                        metadata_items.append({
                            "type": "query_template",
                            "text": template,
                            "field": field_name,
                            "collection": collection_name,
                            "database": db_name
                        })
            
            # Create embeddings in batches
            embeddings = self.model.encode(texts, show_progress_bar=False)
            embeddings = np.array(embeddings).astype('float32')
            
            # Add to index
            self.index.add(embeddings)
            self.metadata.extend(metadata_items)
            
            logger.info(f"Added {len(texts)} schema embeddings to the index")
            
        except Exception as e:
            logger.error(f"Failed to add schema embeddings: {str(e)}", exc_info=True)
    
    def add_example_embeddings(self, collection_name: str, examples: List[Dict[str, Any]]):
        """
        Add example document embeddings to the index.
        
        Args:
            collection_name: Collection name
            examples: List of example documents
        """
        if not self.is_initialized():
            logger.error("Cannot add example embeddings: vector store not initialized")
            return
            
        if self.index is None:
            self.create_index()
            
        try:
            # Create descriptions for examples
            texts = []
            metadata_items = []
            
            for example in examples[:10]:  # Limit to 10 examples per collection
                # Create a readable text representation of the document
                example_text = f"Example {collection_name}: "
                for key, value in example.items():
                    if key != "_id":  # Skip _id field
                        example_text += f"{key}={value}, "
                
                texts.append(example_text)
                metadata_items.append({
                    "type": "example",
                    "collection": collection_name,
                    "data": {k: str(v) for k, v in example.items()}
                })
            
            # Create embeddings
            if texts:
                embeddings = self.model.encode(texts, show_progress_bar=False)
                embeddings = np.array(embeddings).astype('float32')
                
                # Add to index
                self.index.add(embeddings)
                self.metadata.extend(metadata_items)
                
                logger.info(f"Added {len(texts)} example embeddings for {collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to add example embeddings: {str(e)}", exc_info=True)
    
    def find_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar items to the query.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of metadata for similar items
        """
        if not self.is_initialized() or self.index is None:
            logger.error("Cannot find similar items: vector store not initialized")
            return []
            
        if self.index.ntotal == 0:
            logger.warning("Cannot find similar items: index is empty")
            return []
            
        try:
            # Embed query
            query_embedding = self.model.encode([query], show_progress_bar=False)
            query_embedding = np.array(query_embedding).astype('float32')
            
            # Search index
            k = min(k, self.index.ntotal)  # Ensure k is not larger than index size
            D, I = self.index.search(query_embedding, k)
            
            # Return metadata for results
            results = []
            for idx in I[0]:
                if 0 <= idx < len(self.metadata):
                    results.append(self.metadata[idx])
                    
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar items: {str(e)}", exc_info=True)
            return []
    
    def save(self, file_path: str):
        """
        Save the vector store to disk.
        
        Args:
            file_path: Path to save the vector store
        """
        if not self.is_initialized() or self.index is None:
            logger.error("Cannot save vector store: not initialized")
            return
            
        try:
            # Save index
            faiss.write_index(self.index, f"{file_path}.index")
            
            # Save metadata
            with open(f"{file_path}.meta", "wb") as f:
                pickle.dump(self.metadata, f)
                
            logger.info(f"Saved vector store to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}", exc_info=True)
    
    def load(self, file_path: str) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            file_path: Path to load the vector store from
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_initialized():
            logger.error("Cannot load vector store: not initialized")
            return False
            
        try:
            # Load index
            self.index = faiss.read_index(f"{file_path}.index")
            
            # Load metadata
            with open(f"{file_path}.meta", "rb") as f:
                self.metadata = pickle.load(f)
                
            logger.info(f"Loaded vector store from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}", exc_info=True)
            return False
    
    def get_query_suggestions(self, query: str) -> Dict[str, Any]:
        """
        Get query suggestions based on the query text.
        
        Args:
            query: Query text
            
        Returns:
            Suggested collections and fields
        """
        suggestions = {
            "collections": [],
            "fields": [],
            "templates": []
        }
        
        results = self.find_similar(query, k=10)
        
        for result in results:
            if result["type"] == "collection":
                if result["name"] not in suggestions["collections"]:
                    suggestions["collections"].append(result["name"])
            elif result["type"] == "field":
                field_info = {
                    "collection": result["collection"],
                    "field": result["name"]
                }
                if field_info not in suggestions["fields"]:
                    suggestions["fields"].append(field_info)
            elif result["type"] == "query_template":
                if result["text"] not in suggestions["templates"]:
                    suggestions["templates"].append(result["text"])
                    
        return suggestions