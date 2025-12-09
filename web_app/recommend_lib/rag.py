import logging
import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
from recommend_lib.abs_api import get_all_items

logger = logging.getLogger(__name__)

_RAG_INSTANCE = None

def init_rag_system(persist_directory: str = "rag_db_v2"):
    """
    Initializes the global RAG system singleton and indexes the library.
    """
    global _RAG_INSTANCE
    if _RAG_INSTANCE is None:
        _RAG_INSTANCE = RAGSystem(persist_directory)
        logger.info("Global RAG System initialized.")
        
        try:
            logger.info("Fetching library items for RAG indexing...")
            items_map, _ = get_all_items()
            _RAG_INSTANCE.index_library(items_map)
        except Exception as e:
            logger.error(f"Failed to index library during initialization: {e}")
            
    else:
        logger.info("Global RAG System already initialized.")

def get_rag_system() -> Optional['RAGSystem']:
    """
    Returns the global RAG system singleton.
    """
    if _RAG_INSTANCE is None:
        # Fallback if not explicitly initialized, though we prefer explicit init
        logger.warning("RAG System accessed before explicit initialization. Initializing now.")
        init_rag_system()
    return _RAG_INSTANCE


class RAGSystem:
    def __init__(self, persist_directory: str = "rag_db_v2"):
        """
        Initializes the RAG system with ChromaDB and Sentence Transformers.
        """
        self.persist_directory = persist_directory
        # Ensure directory exists
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Use a better multilingual model for embeddings
        # intfloat/multilingual-e5-base has better semantic understanding
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="intfloat/multilingual-e5-base"
        )

        # Get or create collection (v2 for new model/content)
        self.collection = self.client.get_or_create_collection(
            name="audiobooks_v2",
            embedding_function=self.embedding_fn
        )
        logger.info(f"RAG System initialized. Database path: {self.persist_directory}")

    def index_library(self, items_map: Dict[str, Dict]):
        """
        Indexes the library items into ChromaDB.
        """
        # IDs to add
        ids = []
        # Documents (text to embed)
        documents = []
        # Metadatas
        metadatas = []

        existing_ids = self.collection.get()["ids"]
        
        count_new = 0

        for item_id, item in items_map.items():
            if item_id in existing_ids:
                continue

            # Build enhanced embedding text with genres and series
            genres = item.get('genres', [])
            genres_str = ', '.join(genres) if genres else ''
            tags = item.get('tags', [])
            tags_str = ', '.join(tags) if tags else ''
            series = item.get('series', '')
            description = item.get('description', '')
            
            # Construct rich embedding text: Title + Author + Genres + Series + Description
            parts = [f"{item['title']} by {item['author']}"]
            if genres_str:
                parts.append(f"Genres: {genres_str}")
            if tags_str:
                parts.append(f"Tags: {tags_str}")
            if series:
                parts.append(f"Series: {series}")
            if description:
                parts.append(description)
            
            text_to_embed = ". ".join(parts)
            
            ids.append(item_id)
            documents.append(text_to_embed)
            metadatas.append({
                "title": item['title'],
                "author": item['author'],
                "genres": ','.join(genres) if genres else '',
                "series": series or '',
                "tags": ','.join(tags) if tags else ''
            })
            count_new += 1

        if ids:
            logger.info(f"Indexing {len(ids)} new items...")
            # Add in batches to avoid hitting limits if any
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                end = min(i + batch_size, len(ids))
                self.collection.add(
                    ids=ids[i:end],
                    documents=documents[i:end],
                    metadatas=metadatas[i:end]
                )
            logger.info("Indexing complete.")
        else:
            logger.info("No new items to index.")

    def retrieve_similar(self, query_text: str, n_results: int = 5) -> List[str]:
        """
        Retrieves similar items based on the query text.
        Returns a list of item IDs.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        # results['ids'] is a list of lists (one per query)
        if results and results['ids']:
            return results['ids'][0]
        return []

    def get_embeddings(self, ids: List[str]) -> List[List[float]]:
        """
        Retrieves embeddings for a list of item IDs.
        """
        if not ids:
            return []
            
        results = self.collection.get(
            ids=ids,
            include=['embeddings']
        )
        
        # Depending on chromadb version, embeddings might be None if not found.
        embeddings = results.get('embeddings')
        if embeddings is not None and len(embeddings) > 0:
            # Ensure it is a list of lists, not numpy array
            if hasattr(embeddings, 'tolist'):
                 return embeddings.tolist()
            return embeddings
        return []

    def retrieve_by_embedding(self, query_embedding: List[float], n_results: int = 50) -> List[str]:
        """
        Retrieves similar items based on a query embedding vector.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if results and results['ids']:
            return results['ids'][0]
        return []
