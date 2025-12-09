import logging
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Set, Optional
import json
import os
from collections import Counter

from recommend_lib.abs_api import get_all_items, get_finished_books
from recommend_lib.llm import generate_book_recommendations
from recommend_lib.rag import get_rag_system

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_language_file(language: str, type: str) -> str:
    """
    Loads the language file

    Args:
        language (str): The language code
        type (str): User or system

    Returns:
        str: The content of the language file
    """
    languages_dir = os.path.join(os.path.dirname(__file__), 'languages')
    language_file = os.path.join(languages_dir, f"{type}/{language}.txt")
    if not os.path.exists(language_file):
        raise ValueError(f"Language file not found: {language_file}")
    
    with open(language_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return content

def rank_candidates(
    unread_books: List[Dict],
    finished_books: List[Dict],
    top_genres: Set[str],
    top_authors: Set[str]
) -> List[Dict]:
    """
    Ranks unread books based on RAG similarity to finished books and user preferences.
    Uses Mean Pooling of finished books to query RAG.
    Annotates books with '_rag_score' and '_match_reasons'.
    """
    
    # Initialize RAG
    rag = get_rag_system()
    
    # 1. Get embeddings for seed books
    seed_ids = [book['id'] for book in finished_books if book.get('id')]
    seed_embeddings = rag.get_embeddings(seed_ids)
    
    candidate_match_counts = Counter()
    # Note: With mean pooling, we lose the direct "this book matched that book" mapping 
    # unless we do a second pass or keep the old logic. 
    # For now, we'll lose the granular "matched because of X" specific book reason in this step,
    # but we can infer it or just use generic "High similarity to your library".
    
    if seed_embeddings and len(seed_embeddings) > 0:
        # 2. Calculate Mean Vector
        # Assuming all embeddings are same length.
        # Simple average.
        vector_length = len(seed_embeddings[0])
        mean_vector = [0.0] * vector_length
        
        for emb in seed_embeddings:
            for i, val in enumerate(emb):
                mean_vector[i] += val
                
        for i in range(vector_length):
            mean_vector[i] /= len(seed_embeddings)
            
        logger.info(f"Calculated mean vector from {len(seed_embeddings)} seed books.")
        
        # 3. Query RAG with mean vector
        # Fetch more results since this is a broad query
        similar_ids = rag.retrieve_by_embedding(mean_vector, n_results=100)
        
        # Create a score map from the order (higher rank = better)
        # similar_ids is ordered by similarity
        for idx, sid in enumerate(similar_ids):
            # Linearly decreasing score based on rank
            # e.g. 1st = 100, 2nd = 99...
            score = max(0, 100 - idx) 
            candidate_match_counts[sid] = score

    ranked_books = []

    for book in unread_books:
        book_id = book['id']
        match_score = candidate_match_counts.get(book_id, 0)
        
        # Calculate preference bonus
        pref_score = 0
        for genre in book.get('genres', []):
            if genre in top_genres:
                pref_score += 10 # Boost genres
        if book.get('author', '') in top_authors:
            pref_score += 15 # Boost authors
            
        total_score = match_score + pref_score
        
        book['_rag_score'] = total_score
        # For mean pooling, we don't have specific seed book matches easily. 
        # But we can indicate it matched the profile.
        if match_score > 0:
            book['_match_reasons'] = ["Matches your reading profile"]
        else:
            book['_match_reasons'] = [] 
        
        ranked_books.append(book)

    # Sort by score descending
    ranked_books.sort(key=lambda x: -x.get('_rag_score', 0))
    
    return ranked_books

def calculate_mean_vector(embeddings: List[List[float]]) -> List[float]:
    """Calculates the mean vector from a list of embeddings."""
    if not embeddings:
        return []
    
    vector_length = len(embeddings[0])
    mean_vector = [0.0] * vector_length
    
    for emb in embeddings:
        for i, val in enumerate(emb):
            mean_vector[i] += val
            
    for i in range(vector_length):
        mean_vector[i] /= len(embeddings)
        
    return mean_vector

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
        
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

def get_collaborative_recommendations(
    current_user_mean_vector: List[float], 
    current_user_id: str, 
    items_map: Dict,
    rag_system
) -> Tuple[List[Dict], str, float]:
    """
    Finds the most similar user and returns their top recommendations.
    Returns: (List of candidate books, Similar User Name, Similarity Score)
    """
    from recommend_lib.abs_api import get_abs_users, get_finished_books
    
    all_users = get_abs_users()
    best_similarity = -1.0
    most_similar_user = None
    most_similar_mean_vector = None
    
    logger.info(f"Checking {len(all_users)} users for collaborative filtering...")
    
    for user in all_users:
        uid = user['id']
        username = user.get('username', 'Unknown')
        
        if uid == current_user_id:
            continue
            
        # Get their finished books
        finished_ids, _, _ = get_finished_books(items_map, user_id=uid)
        
        if not finished_ids:
            continue
            
        # Get embeddings
        # We need to filter IDs that are actually in our items_map/RAG
        valid_ids = [fid for fid in finished_ids if fid in items_map]
        if not valid_ids:
            continue
            
        embeddings = rag_system.get_embeddings(valid_ids)
        if not embeddings:
            continue
            
        user_mean_vector = calculate_mean_vector(embeddings)
        
        sim = cosine_similarity(current_user_mean_vector, user_mean_vector)
        
        # logger.debug(f"User {username} similarity: {sim}")
        
        if sim > best_similarity:
            best_similarity = sim
            most_similar_user = user
            most_similar_mean_vector = user_mean_vector
            
    if most_similar_user and best_similarity > 0.5: # Threshold for "similar enough"
        logger.info(f"Most similar user found: {most_similar_user.get('username')} with score {best_similarity}")
        
        # Get recommendations for this similar user using their mean vector
        similar_ids = rag_system.retrieve_by_embedding(most_similar_mean_vector, n_results=50)
        
        # Convert IDs to book objects
        collab_candidates = []
        for sid in similar_ids:
            if sid in items_map:
                collab_candidates.append(items_map[sid])
                
        return collab_candidates, most_similar_user.get('username'), best_similarity
        
    return [], None, 0.0


def get_recommendations(use_llm: bool = False, user_id: str = None) -> List[Dict[str, str]]:
    """
    Returns the recommendations
    
    Args:
        use_llm (bool): Whether to use the LLM to generate recommendations/reasons. 
                        If False, returns RAG-based recommendations with static reasons.
        user_id (str): The user ID (optional)

    Returns:
        List[Dict[str, str]]: The recommendations
    """

    logger.info(f"Getting recommendations for user_id: {user_id}")
    load_dotenv()

    items_map, series_counts = get_all_items()
    
    # Update RAG index with all items
    rag = get_rag_system()
    rag.index_library(items_map)

    finished_ids, in_progress_ids, finished_keys = get_finished_books(items_map, user_id)
    
    cleaned_finished_keys = set()
    for item_id in finished_ids:
        if item_id in items_map:
            book = items_map[item_id]
            cleaned_finished_keys.add((book['title'], book['author']))

    unread_books_candidates = [] # List of book dicts
    finished_books_list = []
    
    # Group unread books by series to find the next sequence
    series_candidates = {}
    standalone_candidates = []

    for item_id, book in items_map.items():
        if item_id in finished_ids:
            finished_books_list.append(book)
            continue
        
        if item_id in in_progress_ids:
            continue
            
        if (book['title'], book['author']) in cleaned_finished_keys:
            continue
        
        if book['series']:
            if book['series'] not in series_candidates:
                series_candidates[book['series']] = []
            series_candidates[book['series']].append(book)
        else:
            standalone_candidates.append(book)
    
    # Process series to find the next unread book
    for series_name, books in series_candidates.items():
        valid_books = []
        for b in books:
            seq = b.get('series_sequence')
            if seq is not None:
                try:
                    val = float(seq)
                    valid_books.append((val, b))
                except ValueError:
                    pass
        
        if valid_books:
            valid_books.sort(key=lambda x: x[0])
            standalone_candidates.append(valid_books[0][1])
        elif books:
            standalone_candidates.append(books[0])

    unread_books_candidates = standalone_candidates
    
    # --- USER PREFERENCE WEIGHTING ---
    genre_counts = Counter()
    author_counts = Counter()
    
    for book in finished_books_list:
        for genre in book.get('genres', []):
            genre_counts[genre] += 1
        author_counts[book.get('author', '')] += 1
    
    top_genres = set(g for g, _ in genre_counts.most_common(5))
    top_authors = set(a for a, _ in author_counts.most_common(5))
    
    logger.info(f"User preferences - Top genres: {top_genres}, Top authors: {top_authors}")
    
    # --- RANKING ---
    # --- RANKING ---
    ranked_candidates = rank_candidates(unread_books_candidates, finished_books_list, top_genres, top_authors)
    
    # --- COLLABORATIVE FILTERING BONUS ---
    # Need current user's mean vector - recalculate or extract from rank_candidates?
    # rank_candidates calculated it internally but didn't return it. 
    # Let's extract getting the mean vector into a variable if possible, or just recalculate (cheap).
    
    current_seed_ids = [book['id'] for book in finished_books_list if book.get('id')]
    current_embeddings = rag.get_embeddings(current_seed_ids)
    
    if current_embeddings:
        current_mean_vector = calculate_mean_vector(current_embeddings)
        
        collab_recs, similar_user, urgency_score = get_collaborative_recommendations(
            current_mean_vector, 
            user_id if user_id else "me", # Approximate ID logic
            items_map,
            rag
        )
        
        if similar_user and collab_recs:
            collab_ids = set([c['id'] for c in collab_recs])
            logger.info(f"Boosting {len(collab_ids)} books based on similar user {similar_user}")
            
            for book in ranked_candidates:
                if book['id'] in collab_ids:
                    # Apply Boost
                    # Boost logic: If book is highly rated for similar user but low for me?
                    # The prompt said: "sort books that have a low rank for me but a higher rank for the other user higher for me"
                    # We are re-sorting, so simply increasing the score achieves this.
                    # We give a massive boost to ensure they float up.
                    
                    boost_amount = 50 * urgency_score # Significant boost
                    book['_rag_score'] += boost_amount
                    if '_match_reasons' not in book:
                        book['_match_reasons'] = []
                    book['_match_reasons'].append(f"Highly relevant to similar user '{similar_user}'")

            # Re-sort after boosting
            ranked_candidates.sort(key=lambda x: -x.get('_rag_score', 0))

    
    # Filter out candidates with 0 score if we have plenty of candidates? 
    # Or just keep best.
    # We'll take top 50 for consideration in both paths
    top_candidates = ranked_candidates[:50]
    
    if not top_candidates:
        logger.info("No valid candidates found.")
        return []

    # --- NO LLM PATH ---
    if not use_llm:
        logger.info("Generating RAG-only recommendations (use_llm=False)")
        final_recommendations = []
        # Return top 20
        for book in top_candidates[:20]:
            # Generate static reason
            reasons = book.get('_match_reasons', [])
            score = book.get('_rag_score', 0)
            
            if reasons:
                if reasons == ["Matches your reading profile"]:
                   reason_text = f"Recommended based on your reading profile. Score: {score}"
                else:
                   reason_text = f"Recommended based on your history causing a high match score. Similar to: {', '.join(reasons)}"
            else:
                reason_text = f"Recommended based on genre/author preferences. Score: {score}"
                
            final_recommendations.append({
                'id': book['id'],
                'title': book['title'],
                'author': book['author'],
                'reason': reason_text,
                'cover': book['cover']
            })
        return final_recommendations

    # --- LLM PATH ---
    # Prepare prompt with top candidates
    
    unread_str = ""
    prompt_book_map = {} # int id -> book
    
    for idx, book in enumerate(top_candidates):
        prompt_book_map[idx] = book
        
        series_info = ""
        if book['series']:
            seq = book.get('series_sequence')
            if seq is not None:
                series_info = f" (Series: {book['series']} #{seq})"
            else:
                series_info = f" (Series: {book['series']})"
        
        description = book.get('description', '')
        if description:
            description = description.replace('{', '{{').replace('}', '}}')
            if len(description) > 250:
                description = description[:250] + "..."
            
            # Include RAG hint in prompt to help LLM? Maybe not needed if they are already filtered.
            entry = f"ID:{idx} | {book['title']} by {book['author']}{series_info}\nDescription: {description}"
        else:
            entry = f"ID:{idx} | {book['title']} by {book['author']}{series_info}"
        
        unread_str += f"{entry}\n"

    finished_str = ""
    for book in finished_books_list:
        finished_str += f"- {book['title']} by {book['author']}\n"
    
    Language_setting = os.getenv('LANGUAGE', 'de').lower()
    prompt_string = _load_language_file(Language_setting, 'user')
    prompt = prompt_string.format(finished_str=finished_str, unread_str=unread_str)

    logger.info(f"Prompt prepared with {len(top_candidates)} candidates.")
    
    # Save debug prompt
    with open('prompt_debug.txt', 'w', encoding='utf-8') as f:
        f.write(prompt)

    recs = generate_book_recommendations(prompt, language=Language_setting)

    if not recs:
        logger.error("No recommendations generated from LLM")
        return []

    try:
        parsed_recs = json.loads(recs.text)
        recommendations_raw = parsed_recs.get('items', [])
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        return []
    
    final_recommendations = []
        
    for rec in recommendations_raw:
        rec_index = rec.get('id')
        
        if isinstance(rec_index, int) and rec_index in prompt_book_map:
            original_book = prompt_book_map[rec_index]
            
            final_recommendations.append({
                'id': original_book['id'],
                'title': original_book['title'],
                'author': original_book['author'],
                'reason': rec.get('reason'),
                'cover': original_book['cover']
            })
        else:
            logger.warning(f"Invalid index between returned by LLM: {rec_index}")
                
    return final_recommendations
