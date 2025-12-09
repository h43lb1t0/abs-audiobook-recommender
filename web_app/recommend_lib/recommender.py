import logging
from dotenv import load_dotenv
from typing import List, Dict
import json
import os

from recommend_lib.abs_api import get_all_items, get_finished_books
from recommend_lib.llm import generate_book_recommendations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_language_file(language: str) -> str:
    """
    Loads the language file

    Args:
        language (str): The language code

    Returns:
        str: The content of the language file
    """
    languages_dir = os.path.join(os.path.dirname(__file__), 'languages')
    language_file = os.path.join(languages_dir, f"{language}.txt")
    if not os.path.exists(language_file):
        raise ValueError(f"Language file not found: {language_file}")
    
    with open(language_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return content

def get_recommendations(use_llm: bool = True, user_id: str = None) -> List[Dict[str, str]]:
    """
    Returns the recommendations

    Returns:
        List[Dict[str, str]]: The recommendations
    """

    logger.info(f"Getting recommendations for user_id: {user_id}")
    load_dotenv()

    items_map, series_counts = get_all_items()
    finished_ids, in_progress_ids, finished_keys = get_finished_books(items_map, user_id)
    
    for item_id in finished_ids:
        if item_id in items_map:
            book = items_map[item_id]
            finished_keys.add((book['title'], book['author']))

    unread_books = [] # List of (index, book)
    finished_books_list = []
    
    current_index = 0
    
    # Group unread books by series to find the next sequence
    series_candidates = {}
    standalone_candidates = []

    for item_id, book in items_map.items():
        if item_id in finished_ids:
            finished_books_list.append(book)
            continue
        
        if item_id in in_progress_ids:
            logger.debug(f"Skipping in-progress book: {book['title']}")
            continue
            
        if (book['title'], book['author']) in finished_keys:
            continue
        
        if book['series']:
            if book['series'] not in series_candidates:
                series_candidates[book['series']] = []
            series_candidates[book['series']].append(book)
        else:
            standalone_candidates.append(book)
    
    # Process series to find the next unread book
    for series_name, books in series_candidates.items():
        # Try to parse sequences
        valid_books = []
        for b in books:
            seq = b.get('series_sequence')
            if seq is not None:
                try:
                    # Handle cases like "1", "1.0", "0.5"
                    val = float(seq)
                    valid_books.append((val, b))
                except ValueError:
                    pass
        
        if valid_books:
            # Sort by sequence
            valid_books.sort(key=lambda x: x[0])
            # Add only the first one (lowest sequence number)
            standalone_candidates.append(valid_books[0][1])
        elif books:
            standalone_candidates.append(books[0])

    # Assign indices
    for book in standalone_candidates:
        book['_index'] = current_index
        unread_books.append(book)
        current_index += 1
    
    # --- RAG INTEGRATION ---
    from recommend_lib.rag import get_rag_system
    
    # Initialize RAG (Persistence ensures we don't re-index everything constantly, but we check for updates)
    rag = get_rag_system()
    rag.index_library(items_map)
    
    rag_candidates_ids = set()
    
    seeds = finished_books_list
    
    logger.info(f"Using {len(seeds)} seed books for RAG retrieval.")
    
    for seed in seeds:
         # Construct query from seed book
        query = f"{seed['title']} by {seed['author']}. {seed.get('description', '')}"
        similar_ids = rag.retrieve_similar(query, n_results=5)
        for sid in similar_ids:
            rag_candidates_ids.add(sid)
                
    rag_unread_books = []
    other_unread_books = []
    
    for book in unread_books:
        if book['id'] in rag_candidates_ids:
            rag_unread_books.append(book)
        else:
            other_unread_books.append(book)
            
    logger.info(f"RAG found {len(rag_unread_books)} relevant unread books.")
    
    # Strategy: Combine them, putting RAG matches first. Limit total context size.
    # We want to give the LLM mostly RAG matches if available.
    
    final_context_books = rag_unread_books + other_unread_books
    # Limit to e.g. 50 books to fit in context? 
    # Or just pass them all sorted.
    # Let's keep existing logic but re-order.
    
    unread_str = ""
    for book in final_context_books:
        # We need to preserve the _index mapping!
        # The _index is the index in unread_books list passed to prompt?
        # No, the prompt uses IDs:0, ID:1 etc.
        # The LLM returns the ID.
        # So we must be careful. 
        # The current implementation uses `unread_books` list index as ID.
        # So if we reorder `final_context_books`, we need to update the prompt text 
        # BUT the verification logic `original_book = unread_books[rec_index]` uses `unread_books`.
        # So we should probably NOT reorder `unread_books` list itself, OR we must re-create it.
        pass

    # Better approach: 
    # Create a NEW list for the prompt `prompt_books`.
    # Update the prompt generation to use `prompt_books`.
    # AND when parsing response, use `prompt_books`.
    
    prompt_books = rag_unread_books
    
    # If we have very few RAG results, fill with others
    if len(prompt_books) < 20: # arbitrary minimum
        needed = 20 - len(prompt_books)
        prompt_books.extend(other_unread_books[:needed])
        
    # Re-assign indices for the PROMPT only
    # We can't easily change the book dicts in place without affecting other things potentially?
    # Actually, we can just build the string and a lookup map.
    
    unread_str = ""
    prompt_book_map = {} # int id -> book
    
    for idx, book in enumerate(prompt_books):
        prompt_book_map[idx] = book
        
        series_info = ""
        if book['series']:
            seq = book.get('series_sequence')
            if seq is not None:
                series_info = f" (Series: {book['series']} #{seq})"
            else:
                series_info = f" (Series: {book['series']})"
        
        # Add description, truncated to ~250 tokens (1000 chars)
        description = book.get('description', '')
        if description:
            # Escape braces for format string safety
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

    prompt_string = _load_language_file(Language_setting)

    prompt = prompt_string.format(finished_str=finished_str, unread_str=unread_str)

    logger.info(f"Prompt: {prompt}")

    with open('prompt_debug.txt', 'w', encoding='utf-8') as f:
        f.write(prompt)

    if not use_llm:
        logger.warning("LLM generation is not enabled (use_llm=False)")
        return []
    
    recs = generate_book_recommendations(prompt)

    if not recs:
        logger.error("No recommendations generated")
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
                'id': original_book['id'], # The REAL ABS ID
                'title': original_book['title'],
                'author': original_book['author'],
                'reason': rec.get('reason'),
                'cover': original_book['cover']
            })
        else:
            logger.warning(f"Invalid index returned by LLM: {rec_index}")
                
    return final_recommendations
