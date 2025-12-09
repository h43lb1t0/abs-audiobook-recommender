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

        # Use a standardized multilingual model for embeddings
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="audiobooks",
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

            # Construct embedding text: Title + Author + Description
            text_to_embed = f"{item['title']} by {item['author']}. {item.get('description', '')}"
            
            ids.append(item_id)
            documents.append(text_to_embed)
            metadatas.append({
                "title": item['title'],
                "author": item['author']
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
