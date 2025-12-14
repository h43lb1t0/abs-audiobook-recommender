import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import chromadb
import numpy as np
import onnxruntime as ort
from chromadb.utils import embedding_functions
from huggingface_hub import hf_hub_download
from recommend_lib.abs_api import get_all_items
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from defaults import DURATION_BUCKETS
from db import LibraryStats, db
import json
from collections import Counter

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
            items_map = get_all_items()
            _RAG_INSTANCE.index_library(items_map)
        except Exception as e:
            logger.error(f"Failed to index library during initialization: {e}")

    else:
        logger.info("Global RAG System already initialized.")


def get_rag_system() -> Optional["RAGSystem"]:
    """
    Returns the global RAG system singleton.

    Returns:
        Optional['RAGSystem']: The global RAG system singleton.
    """
    if _RAG_INSTANCE is None:
        logger.warning(
            "RAG System accessed before explicit initialization. Initializing now."
        )
        init_rag_system()
    return _RAG_INSTANCE


def get_duration_bucket(duration_seconds: float) -> str:
    """
    Determines the duration bucket for a given duration.

    Args:
        duration_seconds: Duration in seconds.

    Returns:
        str: Bucket name.
    """
    if not duration_seconds:
        return None

    for bucket, limits in DURATION_BUCKETS.items():
        min_val = limits.get("min", 0)
        max_val = limits.get("max", float("inf"))

        if min_val <= duration_seconds < max_val:
            return bucket

    return None


def format_duration(duration_seconds: float) -> str:
    """
    Formats the duration into rounded string representation.

    If < 1h: rounds to 15, 30, 45, 60 minutes.
    If > 1h: minutes part rounds to 0, 15, 30, 45.

    Args:
        duration_seconds: Duration in seconds

    Returns:
        Formatted string (e.g. "30 minutes", "1 hours 15 minutes")
    """
    if not duration_seconds:
        return ""

    minutes = duration_seconds / 60.0

    if minutes < 60:
        if minutes <= 15:
            return "15 minutes"
        elif minutes <= 30:
            return "30 minutes"
        elif minutes <= 45:
            return "45 minutes"
        else:
            return "1 hour"

    else:
        hours = int(minutes // 60)
        rem_minutes = minutes % 60

        rounded_rem = 0
        if rem_minutes < 7.5:
            rounded_rem = 0
        elif rem_minutes < 22.5:
            rounded_rem = 15
        elif rem_minutes < 37.5:
            rounded_rem = 30
        elif rem_minutes < 52.5:
            rounded_rem = 45
        else:
            hours += 1
            rounded_rem = 0

        if rounded_rem == 0:
            return f"{hours} hours"
        else:
            return f"{hours} hours {rounded_rem} minutes"


class JinaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    Custom embedding function for Jina v3 using SentenceTransformers.
    Supports task-specific embeddings via trust_remote_code=True.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        default_task: str = "retrieval.passage",
    ) -> None:
        """
        Custom embedding function for Jina v3 using SentenceTransformers.
        Supports task-specific embeddings via trust_remote_code=True.

        Args:
            model_name (str): Name of the Jina v3 model to use.
            default_task (str): Default task for embeddings.
        """
        cache_folder = Path(".cache").resolve()
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, cache_folder=cache_folder
        )
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

    def __init__(
        self,
        persist_directory: str = "rag_db_v2",
        model_repo="alikia2x/jina-embedding-v3-m2v-1024",
    ) -> None:
        """
        Initializes the RAG system with ChromaDB and Jina Embeddings v3 ONNX.

        Args:
            persist_directory (str): Directory to persist the RAG system.
            model_repo (str): Jina Embeddings v3 ONNX model repository.
        """
        self.persist_directory = Path(persist_directory)
        if not self.persist_directory.exists():
            self.persist_directory.mkdir()

        self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.embedding_fn = JinaOnnxEmbeddingFunction(model_repo=model_repo)

        self.content_collection = self.client.get_or_create_collection(
            name="audiobooks_content_v1", embedding_function=self.embedding_fn
        )

        self.metadata_collection = self.client.get_or_create_collection(
            name="audiobooks_metadata_v1", embedding_function=self.embedding_fn
        )

        logger.info(
            f"RAG System initialized with ONNX model. Database path: {self.persist_directory}"
        )

    def index_library(self, items_map: Dict[str, Dict]) -> int:
        """
        Indexes the library items into ChromaDB.

        Args:
            items_map (Dict[str, Dict]): Map of item IDs to item data.

        Returns:
            int: Number of new items indexed.
        """
        ids = []
        content_documents = []
        metadata_documents = []
        metadatas = []

        # Check existing IDs in content collection (assuming they are synced)
        existing_ids = self.content_collection.get()["ids"]

        count_new = 0

        for item_id, item in items_map.items():
            if item_id in existing_ids:
                continue

            # Build enhanced embedding text with genres and series
            genres = item.get("genres", [])
            genres_str = ", ".join(genres) if genres else ""
            tags = item.get("tags", [])
            tags_str = ", ".join(tags) if tags else ""

            series = item.get("series", "")
            narrator = item.get("narrator", "")
            description = item.get("description", "")

            # --- Content Embedding (About the book) ---
            content_parts = []
            if genres_str:
                content_parts.append(f"Genres: {genres_str}")
            if tags_str:
                content_parts.append(f"Tags: {tags_str}")
            if description:
                content_parts.append(description)

            content_text = ". ".join(content_parts)

            # --- Metadata Embedding (Who/Structure) ---
            metadata_parts = [f"{item['title']} by {item['author']}"]
            if narrator and narrator != "Unknown":
                metadata_parts.append(f"Narrated by {narrator}")
            if series:
                metadata_parts.append(f"Series: {series}")
            if series:
                metadata_parts.append(f"Series: {series}")
            # Duration removed from RAG embedding as per new logic

            metadata_text = ". ".join(metadata_parts)

            ids.append(item_id)
            content_documents.append(content_text)
            metadata_documents.append(metadata_text)

            metadatas.append(
                {
                    "title": item["title"],
                    "author": item["author"],
                    "narrator": narrator or "",
                    "genres": ",".join(genres) if genres else "",
                    "series": series or "",
                    "tags": ",".join(tags) if tags else "",
                }
            )
            count_new += 1

        if ids:
            logger.info(f"Indexing {len(ids)} new items...")
            # Add in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                end = min(i + batch_size, len(ids))

                # Add to content collection
                self.content_collection.add(
                    ids=ids[i:end],
                    documents=content_documents[i:end],
                    metadatas=metadatas[i:end],
                )

                # Add to metadata collection
                self.metadata_collection.add(
                    ids=ids[i:end],
                    documents=metadata_documents[i:end],
                    metadatas=metadatas[i:end],
                )

            logger.info("Indexing complete.")
        else:
            logger.info("No new items to index.")

            logger.info("No new items to index.")

        # --- Calculate and Save Library Stats (Duration Distribution) ---
        try:
            logger.info("Calculating library duration distribution...")
            duration_counts = Counter()
            total_items_with_duration = 0

            for item in items_map.values():
                dur = item.get("duration_seconds")
                bucket = get_duration_bucket(dur)
                if bucket:
                    duration_counts[bucket] += 1
                    total_items_with_duration += 1

            distribution = {}
            if total_items_with_duration > 0:
                for bucket in DURATION_BUCKETS:
                    distribution[bucket] = (
                        duration_counts[bucket] / total_items_with_duration
                    )
            else:
                # Fallback to uniform if empty library?
                for bucket in DURATION_BUCKETS:
                    distribution[bucket] = 0.2

            logger.info(
                f"Library Duration Distribution: { {k: round(v, 2) for k, v in distribution.items()} }"
            )

            from flask import current_app

            if current_app:
                stats_entry = LibraryStats.query.filter_by(
                    key="duration_distribution"
                ).first()
                if not stats_entry:
                    stats_entry = LibraryStats(key="duration_distribution")
                    db.session.add(stats_entry)

                stats_entry.value_json = json.dumps(distribution)
                db.session.commit()
                logger.info("Library stats saved to database.")

        except Exception as e:
            logger.error(f"Failed to calculate/save library stats: {e}")

        return count_new

    def retrieve_similar(
        self, query_text: str, n_results: int = 5, collection_type: str = "content"
    ) -> List[str]:
        """
        Retrieves similar items based on the query text.

        Args:
            query_text (str): Query text to search for similar items.
            n_results (int): Number of similar items to retrieve.
            collection_type (str): 'content' or 'metadata'

        Returns:
            List[str]: List of IDs of similar items.
        """

        collection = (
            self.content_collection
            if collection_type == "content"
            else self.metadata_collection
        )

        query_vec = self.embedding_fn.embed_query(query_text)
        results = collection.query(query_embeddings=[query_vec], n_results=n_results)
        if results and results["ids"]:
            return results["ids"][0]
        return []

    def get_embeddings(self, ids: List[str]) -> Dict[str, List[List[float]]]:
        """
        Retrieves embeddings for the given IDs from BOTH collections.

        Args:
            ids (List[str]): List of IDs to retrieve embeddings for.

        Returns:
            Dict[str, List[List[float]]]: Dictionary with 'content' and 'metadata' keys containing lists of embeddings.
        """

        if not ids:
            return {"content": [], "metadata": []}

        content_results = self.content_collection.get(ids=ids, include=["embeddings"])
        metadata_results = self.metadata_collection.get(ids=ids, include=["embeddings"])

        def extract_embeddings(results):
            embeddings = results.get("embeddings")
            if embeddings is not None and len(embeddings) > 0:
                if hasattr(embeddings, "tolist"):
                    return embeddings.tolist()
                return embeddings
            return []

        return {
            "content": extract_embeddings(content_results),
            "metadata": extract_embeddings(metadata_results),
        }

    def retrieve_by_embedding(
        self,
        query_embedding: List[float],
        n_results: int = 50,
        collection_type: str = "content",
    ) -> List[Tuple[str, float]]:
        """
        Retrieves similar items based on the query embedding.

        Args:
            query_embedding (List[float]): Query embedding to search for similar items.
            n_results (int): Number of similar items to retrieve.
            collection_type (str): 'content' or 'metadata'

        Returns:
            List[tuple[str, float]]: List of tuples containing IDs and distances of similar items.
        """

        collection = (
            self.content_collection
            if collection_type == "content"
            else self.metadata_collection
        )

        if collection.count() == 0:
            return []
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if results and results["ids"] and results["distances"]:
            zipped = list(zip(results["ids"][0], results["distances"][0]))
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
            model_path = hf_hub_download(
                repo_id=model_repo, filename="onnx/model_INT8.onnx"
            )
        except Exception:
            logger.info(
                "Quantized model not found, falling back to full float32 model..."
            )
            model_path = hf_hub_download(repo_id=model_repo, filename="onnx/model.onnx")

        tokenizer_path = hf_hub_download(repo_id=model_repo, filename="tokenizer.json")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=8192)  # Jina v3 max length

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

            ort_inputs = {"input_ids": input_ids, "offsets": offsets}
        else:
            # Standard BERT style
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array(
                [e.attention_mask for e in encoded], dtype=np.int64
            )

            ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in input_names:
                ort_inputs["token_type_ids"] = np.zeros_like(input_ids)

        try:
            outputs = self.session.run(None, ort_inputs)
            output = outputs[0]

            if len(output.shape) == 3:
                mask_expanded = np.expand_dims(attention_mask, axis=-1)
                sum_embeddings = np.sum(output * mask_expanded, axis=1)
                sum_mask = np.clip(
                    np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None
                )
                embeddings = sum_embeddings / sum_mask
            else:
                embeddings = output

            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"ONNX Inference failed for inputs: {list(ort_inputs.keys())}")
            return []
