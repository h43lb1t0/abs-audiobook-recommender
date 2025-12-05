import logging
from dotenv import load_dotenv
from typing import List, Dict
import json
import os
from datetime import datetime
from sqlalchemy import desc
from flask_sqlalchemy import SQLAlchemy
from models.db import UserLib, UserLastRecommendation
from recommend_lib.abs_api import get_all_items, get_finished_books
from recommend_lib.gemini import generate_book_recommendations

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

def get_recommendations(use_gemini: bool = True, user_id: str = None, db: SQLAlchemy = None) -> List[Dict[str, str]]:
    """
    Returns the recommendations

    Returns:
        List[Dict[str, str]]: The recommendations
    """

    logger.info(f"Getting recommendations for user_id: {user_id}")
    load_dotenv()

    items_map, series_counts = get_all_items()
    finished_ids, in_progress_ids, finished_keys = get_finished_books(items_map, user_id)
    
    existing_books = db.session.query(UserLib.book_id).filter_by(user_id=user_id).all()
    existing_book_ids = {b[0] for b in existing_books}

    new_books_found = False

    for item_id in finished_ids:
        if item_id in items_map:
            book = items_map[item_id]
            finished_keys.add((book['title'], book['author']))
            if item_id not in existing_book_ids:
                db.session.add(UserLib(user_id=user_id, book_id=item_id, user_name_debug=user_id, book_name_debug=book['title']))
                existing_book_ids.add(item_id)
                new_books_found = True
    db.session.commit()

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
            if item_id not in existing_book_ids:
                db.session.add(UserLib(user_id=user_id, book_id=item_id, user_name_debug=user_id, book_name_debug=book['title']))
                existing_book_ids.add(item_id)
                new_books_found = True
            continue
            
        if (book['title'], book['author']) in finished_keys:
            continue
        
        if book['series']:
            if book['series'] not in series_candidates:
                series_candidates[book['series']] = []
            series_candidates[book['series']].append(book)
        else:
            standalone_candidates.append(book)
    db.session.commit()
    
    if new_books_found:
        logger.info("New finished or in-progress books found, generating recommendations")
    else:
        logger.info("No new finished or in-progress books found, skipping recommendations")
        return []    
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

    # Add finished books to the DB

    
    finished_str = ""
    for book in finished_books_list:
        finished_str += f"- {book['title']} by {book['author']}\n"

            
    unread_str = ""
    for book in unread_books:
        series_info = ""
        if book['series']:
            seq = book.get('series_sequence')
            if seq is not None:
                series_info = f" (Series: {book['series']} #{seq})"
            else:
                series_info = f" (Series: {book['series']})"
        entry = f"ID:{book['_index']} | {book['title']} by {book['author']}{series_info}"
        unread_str += f"{entry}\n"


    
    language_setting = os.getenv('LANGUAGE', 'de').lower()

    prompt_string = _load_language_file(language_setting)

    prompt = prompt_string.format(finished_str=finished_str, unread_str=unread_str)

    logger.debug(f"Prompt: {prompt}")

    if not use_gemini:
        logger.warning("Gemini is not enabled, skipping recommendation generation")
        return []
    
    recs = generate_book_recommendations(prompt)

    if not recs:
        logger.error("No recommendations generated")
        return []

    try:
        parsed_recs = json.loads(recs.text)
        recommendations_raw = parsed_recs.get('items', [])
    except Exception as e:
        logger.error(f"Error parsing Gemini response: {e}")
        return []
    
    final_recommendations = []

    # Delete previous recommendations for this user
    db.session.query(UserLastRecommendation).filter_by(user_id=user_id).delete()
    db.session.commit()
    
    for rec in recommendations_raw:
        rec_index = rec.get('id')
        
        if isinstance(rec_index, int) and 0 <= rec_index < len(unread_books):
            original_book = unread_books[rec_index]
            
            final_recommendations.append({
                'id': original_book['id'], # The REAL ABS ID
                'title': original_book['title'],
                'author': original_book['author'],
                'reason': rec.get('reason'),
                'cover': original_book['cover']
            })
            
            db.session.add(UserLastRecommendation(
                user_id=user_id, 
                book_id=original_book['id'], 
                gemini_reason=rec.get('reason'),
                date=datetime.utcnow()
            ))
        else:
            logger.warning(f"Invalid index returned by Gemini: {rec_index}")
            
    db.session.commit()
                
    return final_recommendations

def get_last_recommendations(user_id: str, db: SQLAlchemy) -> List[Dict[str, str]]:
    """
    Retrieves the last generated recommendations for the user.
    """
    logger.info(f"Fetching last recommendations for user_id: {user_id}")
    
    # query DB for recommendations
    last_recs = db.session.query(UserLastRecommendation).filter_by(user_id=user_id).all()
    
    if not last_recs:
        return []
        
    # Get all items to lookup book details
    items_map, _ = get_all_items()
    
    final_recommendations = []
    
    for rec in last_recs:
        if rec.book_id in items_map:
            book = items_map[rec.book_id]
            final_recommendations.append({
                'id': rec.book_id,
                'title': book['title'],
                'author': book['author'],
                'reason': rec.gemini_reason,
                'cover': book.get('cover'),
                'date': rec.date
            })
        else:
            logger.warning(f"Book ID {rec.book_id} from saved recommendations not found in current library items.")
            
    return final_recommendations
