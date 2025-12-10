import logging
import os
from typing import Dict, List, Optional

import chromadb
import numpy as np
import onnxruntime as ort
from chromadb.utils import embedding_functions
from huggingface_hub import hf_hub_download
from recommend_lib.abs_api import get_all_items
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

_RAG_INSTANCE = None

def init_rag_system(persist_directory: str = "rag_db_v2") -> None:
    """
    Initializes the global RAG system singleton and indexes the library.

    Args:
        persist_directory (str): Directory to persist the RAG system.
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

    Returns:
        Optional['RAGSystem']: The global RAG system singleton.
    """
    if _RAG_INSTANCE is None:
        logger.warning("RAG System accessed before explicit initialization. Initializing now.")
        init_rag_system()
    return _RAG_INSTANCE


class JinaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    Custom embedding function for Jina v3 using SentenceTransformers.
    Supports task-specific embeddings via trust_remote_code=True.
    """

    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3", default_task: str = "retrieval.passage") -> None:
        """
        Custom embedding function for Jina v3 using SentenceTransformers.
        Supports task-specific embeddings via trust_remote_code=True.

        Args:
            model_name (str): Name of the Jina v3 model to use.
            default_task (str): Default task for embeddings.
        """
        cache_folder = os.path.abspath(".cache")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, cache_folder=cache_folder)
        self.default_task = default_task

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        This is called by ChromaDB for indexing (default task: passage)

        Args:
            input (List[str]): List of strings to embed.

        Returns:
            List[List[float]]: List of embeddings for the input strings.
        """
        return self.model.encode(input, task=self.default_task).tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Specific method for queries (task: query)
        
        Args:
            query (str): Query to embed.

        Returns:
            List[float]: Embedding for the query.
        """
        return self.model.encode([query], task="retrieval.query")[0].tolist()


class RAGSystem:
    """
    Initializes the RAG system with ChromaDB and Jina Embeddings v3 ONNX.
    """

    def __init__(self, persist_directory: str = "rag_db_v2", model_repo="alikia2x/jina-embedding-v3-m2v-1024") -> None:
        """
        Initializes the RAG system with ChromaDB and Jina Embeddings v3 ONNX.

        Args:
            persist_directory (str): Directory to persist the RAG system.
            model_repo (str): Jina Embeddings v3 ONNX model repository.
        """
        self.persist_directory = persist_directory
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.embedding_fn = JinaOnnxEmbeddingFunction(
            model_repo=model_repo
        )

        self.collection = self.client.get_or_create_collection(
            name="audiobooks_v3_onnx",
            embedding_function=self.embedding_fn
        )
        logger.info(f"RAG System initialized with ONNX model. Database path: {self.persist_directory}")

    def index_library(self, items_map: Dict[str, Dict]) -> None:
        """
        Indexes the library items into ChromaDB.

        Args:
            items_map (Dict[str, Dict]): Map of item IDs to item data.
        """
        ids = []
        documents = []
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
            
            # Construct rich embedding text
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
            # Add in batches
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

        Args:
            query_text (str): Query text to search for similar items.
            n_results (int): Number of similar items to retrieve.

        Returns:
            List[str]: List of IDs of similar items.
        """

        query_vec = self.embedding_fn.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results
        )
        if results and results['ids']:
            return results['ids'][0]
        return []

    def get_embeddings(self, ids: List[str]) -> List[List[float]]:
        """
        Retrieves embeddings for the given IDs.

        Args:
            ids (List[str]): List of IDs to retrieve embeddings for.

        Returns:
            List[List[float]]: List of embeddings for the given IDs.
        """

        if not ids: return []
        results = self.collection.get(ids=ids, include=['embeddings'])
        embeddings = results.get('embeddings')
        if embeddings is not None and len(embeddings) > 0:
            if hasattr(embeddings, 'tolist'): return embeddings.tolist()
            return embeddings
        return []

    def retrieve_by_embedding(self, query_embedding: List[float], n_results: int = 50) -> List[tuple[str, float]]:
        """
        Retrieves similar items based on the query embedding.

        Args:
            query_embedding (List[float]): Query embedding to search for similar items.
            n_results (int): Number of similar items to retrieve.

        Returns:
            List[tuple[str, float]]: List of tuples containing IDs and distances of similar items.
        """

        if self.collection.count() == 0: return []
        results = self.collection.query(
            query_embeddings=[query_embedding], 
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        if results and results['ids'] and results['distances']: 
            zipped = list(zip(results['ids'][0], results['distances'][0]))
            return zipped
        return []

class JinaOnnxEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    Custom embedding function for Jina v3 (or compatible) via ONNX Runtime.
    """

    def __init__(self, model_repo: str) -> None:
        """
        Custom embedding function for Jina v3 (or compatible) via ONNX Runtime.
        """

        logger.info(f"Initializing Jina ONNX from: {model_repo}")
        
        try:
             logger.info("Attempting to download quantized ONNX model...")
             model_path = hf_hub_download(repo_id=model_repo, filename="onnx/model_INT8.onnx")
        except Exception:
             logger.info("Quantized model not found, falling back to full float32 model...")
             model_path = hf_hub_download(repo_id=model_repo, filename="onnx/model.onnx")
             
        tokenizer_path = hf_hub_download(repo_id=model_repo, filename="tokenizer.json")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=8192) # Jina v3 max length

        self.session = ort.InferenceSession(model_path)
        self.output_name = self.session.get_outputs()[0].name
        
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        This is called by ChromaDB for indexing (default task: passage)

        Args:
            input (List[str]): List of strings to embed.

        Returns:
            List[List[float]]: List of embeddings for the input strings.
        """
        return self._embed(input)

    def embed_query(self, query: str) -> List[float]:
        """
        This is called by ChromaDB for querying (default task: query)

        Args:
            query (str): String to embed.

        Returns:
            List[float]: Embedding for the input string.
        """
        embedding = self._embed([query])
        if embedding:
            return embedding[0]
        return []
        
    def _embed(self, texts: List[str]) -> List[List[float]]:
        """
        This is called by ChromaDB for indexing (default task: passage)

        Args:
            texts (List[str]): List of strings to embed.

        Returns:
            List[List[float]]: List of embeddings for the input strings.
        """

        encoded = self.tokenizer.encode_batch(texts)
        
        input_names = [i.name for i in self.session.get_inputs()]
        
        # Prepare inputs based on model type
        if "offsets" in input_names:
             # Model2Vec / BagOfWords style (concatenated inputs + offsets)
             flat_ids = []
             offsets = []
             current_offset = 0
             
             for e in encoded:
                 offsets.append(current_offset)
                 ids = e.ids
                 # Remove padding (0) if present in ids (Model2Vec usually doesn't need padding in flat array)
                 ids = [i for i in ids if i != 0] 
                 
                 flat_ids.extend(ids)
                 current_offset += len(ids)
                 
             input_ids = np.array(flat_ids, dtype=np.int64)
             offsets = np.array(offsets, dtype=np.int64)
             
             ort_inputs = {
                 "input_ids": input_ids,
                 "offsets": offsets
             }
        else:
             # Standard BERT style
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            if "token_type_ids" in input_names:
                 ort_inputs["token_type_ids"] = np.zeros_like(input_ids)
        

        try:
            outputs = self.session.run(None, ort_inputs)
            output = outputs[0]
        
            if len(output.shape) == 3:
                mask_expanded = np.expand_dims(attention_mask, axis=-1)
                sum_embeddings = np.sum(output * mask_expanded, axis=1)
                sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
                embeddings = sum_embeddings / sum_mask
            else:
                embeddings = output
                
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"ONNX Inference failed for inputs: {list(ort_inputs.keys())}")
            return []
