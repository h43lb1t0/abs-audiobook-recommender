import json
import logging
import os
from collections import Counter
from typing import Dict, List, Set, Tuple

from db import UserLib, User
from dotenv import load_dotenv
from recommend_lib.abs_api import get_abs_users, get_all_items, get_finished_books
from recommend_lib.llm import generate_book_recommendations
from recommend_lib.rag import get_rag_system
from sklearn.cluster import KMeans
from defaults import *


logger = logging.getLogger(__name__)

load_dotenv()


# Rating weights mapping: stars -> weight
RATING_WEIGHTS = {
    5: 2.0,
    4: 1.0,
    3: 0.0,
    2: -0.5,
    1: -1.5
}


def _load_language_file(language: str, type: str) -> str:
    """
    Loads the language file for the LLM

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
    top_authors: Set[str],
    user_id: str = None
) -> List[Dict]:
    """
    Ranks unread books based on RAG similarity to finished books and user preferences.

    Uses a Two-Phase Rating System:
    1. Query phase: Find candidates similar to positively-rated books (4-5★)
    2. Penalty phase: Demote candidates similar to negatively-rated books (1-2★)

    Annotates books with '_rag_score' and '_match_reasons'.

    Args:
        unread_books (List[Dict]): List of unread books
        finished_books (List[Dict]): List of finished books
        top_genres (Set[str]): Set of top genres
        top_authors (Set[str]): Set of top authors
        user_id (str, optional): User ID. Defaults to None.

    Returns:
        List[Dict]: List of ranked books
    """
    
    if  unread_books is None or finished_books is None:
        raise ValueError("unread_books or finished_books is None")

    rag = get_rag_system()
    
    ratings_map = {}
    if user_id:
        ratings_map = get_user_ratings(user_id)
        logger.info(f"Loaded {len(ratings_map)} user ratings")
    

    positive_ids = []  # 4-5 stars (liked)
    negative_ids = []  # 1-2 stars (disliked)
    neutral_ids = []   # 3 stars or unrated
    
    for book in finished_books:
        book_id = book.get('id')
        if not book_id:
            continue
        rating = ratings_map.get(book_id, DEFAULT_RATING)  # Default to neutral
        if rating >= 4:
            positive_ids.append(book_id)
        elif rating <= 2:
            negative_ids.append(book_id)
        else:
            neutral_ids.append(book_id)
    
    logger.info(f"Rating categories: {len(positive_ids)} liked (4-5★), {len(negative_ids)} disliked (1-2★), {len(neutral_ids)} neutral")
    
    candidate_scores = Counter()
    
    query_embeddings = []
    log_prefix = ""
    
    if positive_ids:
        query_embeddings = rag.get_embeddings(positive_ids)
        log_prefix = f"Phase 1: Querying with {len(positive_ids)} positively-rated books"
    else:
        all_ids = [book['id'] for book in finished_books if book.get('id')]
        query_embeddings = rag.get_embeddings(all_ids)
        log_prefix = f"Phase 1 (fallback): Querying with all {len(all_ids)} finished books"

    if query_embeddings:
        cluster_vectors = calculate_cluster_vectors(query_embeddings, max_clusters=MAX_CLUSTERS)
        logger.info(f"{log_prefix} -> Generated {len(cluster_vectors)} clusters")
        
        for i, vector in enumerate(cluster_vectors):
            similar_items = rag.retrieve_by_embedding(vector, n_results=100)
            
            for idx, (sid, dist) in enumerate(similar_items):
                
                safe_dist = max(0.0, dist)
                similarity = 1.0 - (safe_dist / 2.0)
                
                base_score = similarity * 100.0
                
                score_multiplier = 2.0 if positive_ids else 1.0
                
                candidate_scores[sid] += base_score * score_multiplier
                
    
    # === PHASE 2: Penalize similarity to negatively-rated books ===
    if negative_ids:
        negative_embeddings = rag.get_embeddings(negative_ids)
        if negative_embeddings:
            logger.info(f"Phase 2: Penalizing similarity to {len(negative_ids)} negatively-rated books (individual embeddings)")
            
            for i, neg_vector in enumerate(negative_embeddings):
                disliked_similar = rag.retrieve_by_embedding(neg_vector, n_results=50)
                
                for idx, (sid, dist) in enumerate(disliked_similar):
                    safe_dist = max(0.0, dist)
                    similarity = 1.0 - (safe_dist / 2.0)
                    
                    if similarity > 0:
                        penalty = similarity * 100.0 * 1.5
                        candidate_scores[sid] -= penalty

            logger.info("Applied penalties based on negative ratings")

    ranked_books = []

    for book in unread_books:

        book_id = book['id']

        match_score = candidate_scores.get(book_id, 0)
        
        pref_score = 0

        for genre in book.get('genres', []):
            if genre in top_genres:
                pref_score += RECOMMENDATION_BOOST['GENRE'] # Boost genres

        if book.get('author', '') in top_authors:
            pref_score += RECOMMENDATION_BOOST['AUTHOR'] # Boost authors
            
        total_score = match_score + pref_score
        
        book['_rag_score'] = total_score

        reasons = []
        if match_score > 50:
            reasons.append("Similar to books you loved")
        elif match_score > 0:
            reasons.append("Matches your reading profile")
        elif match_score < -50:
            reasons.append("Note: Similar to books you disliked")
        
        book['_match_reasons'] = reasons
        
        ranked_books.append(book)


    ranked_books.sort(key=lambda x: -x.get('_rag_score', 0))
    
    return ranked_books


def calculate_mean_vector(embeddings: List[List[float]]) -> List[float]:
    """
    Calculates the mean vector from a list of embeddings.

    Args:
        embeddings: List of embeddings.

    Returns:
        List of mean vector.
    """

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


def calculate_cluster_vectors(embeddings: List[List[float]], max_clusters: int = 5) -> List[List[float]]:
    """
    Performs K-Means clustering on embeddings to identify distinct taste profiles.
    
    Args:
        embeddings: List of embeddings.
        max_clusters: Maximum number of clusters to form.
    
    Returns:
        List of cluster centers.
    """
    
    if not embeddings:
        return []
    
    if len(embeddings) == 1:
        return embeddings
        
    n_clusters = min(len(embeddings), max_clusters)
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        return kmeans.cluster_centers_.tolist()
    except Exception as e:
        logger.error(f"KMeans failed: {e}. Falling back to mean vector.")
        return [calculate_mean_vector(embeddings)]



def calculate_weighted_user_vector(
    embeddings: List[List[float]], 
    book_ids: List[str],
    ratings_map: Dict[str, int]
) -> List[float]:
    """
    Calculates a weighted user vector based on book ratings.
    
    Formula: V_user = sum(V_book * Weight) / sum(|Weight|)
    
    Args:
        embeddings: List of embedding vectors for each book
        book_ids: List of book IDs corresponding to each embedding
        ratings_map: Dictionary mapping book_id -> rating (1-5)
        
    Returns:
        Weighted user vector
    """
    if not embeddings:
        return []
    
    vector_length = len(embeddings[0])
    weighted_vector = [0.0] * vector_length
    total_abs_weight = 0.0
    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    
    for emb, book_id in zip(embeddings, book_ids):
        # Get rating weight, default to 3 stars (neutral) if unrated
        rating = ratings_map.get(book_id, DEFAULT_RATING)
        weight = RATING_WEIGHTS.get(rating, 0.0)
        
        # Track counts for logging
        if weight > 0:
            positive_count += 1
        elif weight < 0:
            negative_count += 1
        else:
            neutral_count += 1
        
        # Skip neutral weights (3 stars) as they don't contribute
        if weight == 0.0:
            continue
            
        total_abs_weight += abs(weight)
        
        for i, val in enumerate(emb):
            weighted_vector[i] += val * weight
    
    logger.info(f"Rating weights applied: {positive_count} positive (4-5★), {negative_count} negative (1-2★), {neutral_count} neutral (3★/unrated)")
    logger.info(f"Total absolute weight: {total_abs_weight}")
    
    # Normalize by total absolute weight
    if total_abs_weight > 0:
        for i in range(vector_length):
            weighted_vector[i] /= total_abs_weight
    else:
        # Fallback to simple mean if no valid weights
        logger.warning("No valid rating weights found, falling back to simple mean vector")
        return calculate_mean_vector(embeddings)
    
    return weighted_vector


def get_user_ratings(user_id: str) -> Dict[str, int]:
    """
    Fetches user ratings from the database.
    
    Args:
        user_id: The user's ID
        
    Returns:
        Dictionary mapping book_id -> rating (1-5)
    """
    
    try:
        user_ratings = UserLib.query.filter_by(user_id=user_id).all()
        return {r.book_id: r.rating for r in user_ratings if r.rating is not None}
    except Exception as e:
        logger.warning(f"Could not fetch user ratings: {e}")
        return {}


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculates cosine similarity between two vectors.
    
    Args:
        vec_a: First vector
        vec_b: Second vector
    
    Returns:
        Cosine similarity between the two vectors
    """
    
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
        
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)


def calculate_max_cluster_similarity(clusters_a: List[List[float]], clusters_b: List[List[float]]) -> float:
    """
    Calculates the maximum similarity between any pair of clusters from two sets.

    Args:
        clusters_a: List of cluster vectors
        clusters_b: List of cluster vectors

    Returns:
        Maximum similarity between any pair of clusters
    """

    max_sim = 0.0
    for vec_a in clusters_a:
        for vec_b in clusters_b:
            sim = cosine_similarity(vec_a, vec_b)
            if sim > max_sim:
                max_sim = sim
    return max_sim


def get_collaborative_recommendations(
    current_user_clusters: List[List[float]], 
    current_user_id: str, 
    items_map: Dict,
    rag_system
) -> Tuple[List[Dict], str, float]:
    """
    Finds the most similar user and returns their top recommendations.

    Args:
        current_user_clusters: List of cluster vectors for the current user
        current_user_id: ID of the current user
        items_map: Dictionary mapping book_id -> book object
        rag_system: RAG system for generating recommendations

    Returns:
        Tuple of (List of candidate books, Similar User Name, Similarity Score)
    """
    
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

        finished_ids, _, _ = get_finished_books(items_map, user_id=uid)
        
        if not finished_ids:
            continue

        valid_ids = [fid for fid in finished_ids if fid in items_map]

        if not valid_ids:
            continue

        embeddings = rag_system.get_embeddings(valid_ids)

        if not embeddings:
            continue
            
        user_cluster_vectors = calculate_cluster_vectors(embeddings, max_clusters=5)
        
        matching_vectors = []
        max_user_sim = 0.0
        
        for their_vec in user_cluster_vectors:
            # Check if this cluster matches any of my clusters
            best_match_score = 0.0
            for my_vec in current_user_clusters:
                sim = cosine_similarity(my_vec, their_vec)
                if sim > best_match_score:
                    best_match_score = sim
            
            if best_match_score > max_user_sim:
                max_user_sim = best_match_score
            
            # If this cluster is similar enough to one of mine, include it
            if best_match_score > 0.6: # Configurable threshold
                matching_vectors.append(their_vec)
        
        if max_user_sim > best_similarity:
            best_similarity = max_user_sim
            most_similar_user = user
            # ONLY store the clusters that matched
            most_similar_mean_vector = matching_vectors 
            
    if most_similar_user and best_similarity > 0.5 and most_similar_mean_vector: 

        logger.info(f"Most similar user found: {most_similar_user.get('username')} with score {best_similarity}")
        logger.info(f"Using {len(most_similar_mean_vector)} matching clusters for cross-recommendation.")
        
        similar_ids_set = set()
        
        # Iterate over their matching clusters to get books
        for vector in most_similar_mean_vector:
             items = rag_system.retrieve_by_embedding(vector, n_results=20)
             similar_ids_set.update([item[0] for item in items])
             
        similar_ids = list(similar_ids_set)

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

    if user_id is None:
        raise ValueError("User ID is required")
        return []

    user = User.query.get(user_id)

    if not user:
        raise ValueError("User not found")
        return []

    logger.info(f"Getting recommendations for user: {user.username}")

    items_map, series_counts = get_all_items()
    
    rag = get_rag_system()

    finished_ids, in_progress_ids, finished_keys = get_finished_books(items_map, user_id)
    
    cleaned_finished_keys = set()

    for item_id in finished_ids:
        if item_id in items_map:
            book = items_map[item_id]
            cleaned_finished_keys.add((book['title'], book['author']))

    unread_books_candidates = []
    finished_books_list = []

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
    
    top_genres = set(g for g, _ in genre_counts.most_common(MOST_COMMON_GENRES))
    top_authors = set(a for a, _ in author_counts.most_common(MOST_COMMON_AUTHORS))
    
    logger.info(f"User preferences - Top genres: {top_genres}, Top authors: {top_authors}")
    
    # --- RANKING ---

    ranked_candidates = rank_candidates(unread_books_candidates, finished_books_list, top_genres, top_authors, user_id)
    

    # --- COLLABORATIVE FILTERING BONUS ---

    # Use positive-rated books for user profile (consistent with rank_candidates)
    
    ratings_map = get_user_ratings(user_id) if user_id else {}
    
    # Get positive-rated book IDs
    positive_ids = []
    for book in finished_books_list:
        book_id = book.get('id')
        if book_id:
            rating = ratings_map.get(book_id, DEFAULT_RATING)
            if rating >= 4:
                positive_ids.append(book_id)
    
    # Use positive books if available, otherwise fall back to all
    if positive_ids:
        current_embeddings = rag.get_embeddings(positive_ids)
    else:
        current_seed_ids = [book['id'] for book in finished_books_list if book.get('id')]
        current_embeddings = rag.get_embeddings(current_seed_ids)
    
    if current_embeddings:
        # Calculate clusters for me
        current_user_clusters = calculate_cluster_vectors(current_embeddings, max_clusters=5)
        
        collab_recs, similar_user, similarity_score = get_collaborative_recommendations(
            current_user_clusters, 
            user_id if user_id else "me",
            items_map,
            rag
        )
        
        if similar_user and collab_recs:
            collab_ids = set([c['id'] for c in collab_recs])

            logger.info(f"Boosting {len(collab_ids)} books based on similar user {similar_user}")
            
            for book in ranked_candidates:
                if book['id'] in collab_ids:
                    boost_amount = 15 * similarity_score
                    book['_rag_score'] += boost_amount

                    if '_match_reasons' not in book:
                        book['_match_reasons'] = []

                    book['_match_reasons'].append(f"Highly relevant to similar user '{similar_user}'")

            ranked_candidates.sort(key=lambda x: -x.get('_rag_score', 0))

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

