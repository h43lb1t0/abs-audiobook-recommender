import logging
from dotenv import load_dotenv
from typing import List, Dict
import json
import os
from datetime import datetime
from random import randint
import math
from sqlalchemy import desc
from flask_sqlalchemy import SQLAlchemy
from models.db import User, UserLib, UserLastRecommendation
from recommend_lib.abs_api import get_all_items, get_finished_books
from recommend_lib.gemini import generate_book_recommendations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_max_recommendations_per_user_per_day(db, max_total_recommendations: int = 50) -> int:
    num_users = db.session.query(User).count()
    max_recommendations_per_user = math.floor(max_total_recommendations / num_users)
    
    return max_recommendations_per_user
    


MOCK_BOOKS = {
    'mock-1': {
        'title': 'The Invisible Library',
        'author': 'Genevieve Cogman',
        'cover': None,
        'description': 'Irene is a professional spy for the mysterious Library, which harvests fiction from different realities.'
    },
    'mock-2': {
        'title': 'Project Hail Mary',
        'author': 'Andy Weir',
        'cover': None,
        'description': 'Ryland Grace is the sole survivor on a desperate, last-chance mission—and if he fails, humanity and the earth itself will perish.'
    },
    'mock-3': {
        'title': 'The Midnight Library',
        'author': 'Matt Haig',
        'cover': None,
        'description': 'Between life and death there is a library, and within that library, the shelves go on forever. Every book provides a chance to try another life you could have lived.'
    }
}

def create_mock_recommendations(db: SQLAlchemy, user_id: str) -> List[Dict[str, str]]:
    """
    Mocks the response from the Gemini API using hardcoded mock books.
    """
    logger.info(f"Generating mock recommendations for user_id: {user_id}")
    
    # Delete previous recommendations for this user
    db.session.query(UserLastRecommendation).filter_by(user_id=user_id).delete()
    db.session.commit()

    final_recommendations = []
    
    for book_id, book_data in MOCK_BOOKS.items():
        reason = f"Mock reason: Because you might like '{book_data['title']}' based on your reading history."
        
        # Generate a random seed for this recommendation
        seed = randint(1, 100)
        
        # Create a composite ID to store the seed: "book_id|seed"
        composite_id = f"{book_id}|{seed}"
        
        final_recommendations.append({
            'id': book_id, # Frontend still sees the clean ID
            'title': f"{book_data['title']} {seed}",
            'author': book_data['author'],
            'reason': reason,
            'cover': book_data['cover']
        })
        
        db.session.add(UserLastRecommendation(
            user_id=user_id, 
            book_id=composite_id,  # Save composite ID to DB
            gemini_reason=reason,
            date=datetime.utcnow()
        ))
        
    db.session.commit()

    return final_recommendations

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

    # Rate Limiting Check
    user = db.session.get(User, user_id)
    if user:
        today = datetime.utcnow().date()
        if user.last_recommendation_date is None or user.last_recommendation_date.date() != today:
            current_last_date = user.last_recommendation_date.date() if user.last_recommendation_date else None
            if current_last_date != today:
                logger.info(f"Resetting daily recommendation count for user {user_id}")
                user.daily_recommendation_count = 0
                user.last_recommendation_date = datetime.utcnow()
                pass

        max_recs = calculate_max_recommendations_per_user_per_day(db)
        
        last_date = user.last_recommendation_date
        if last_date is None or last_date.date() != today:
             user.daily_recommendation_count = 0
        
        if user.daily_recommendation_count >= max_recs:
            logger.warning(f"User {user_id} reached daily recommendation limit ({max_recs}). Skipping.")
            return ([], "Daily limit reached")

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
        return ([], "No Update")    
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
        
        # Increment count for mock too, so we can test the limit logic
        if user:
             user.daily_recommendation_count += 1
             user.last_recommendation_date = datetime.utcnow()
        db.session.commit()
        
        return (create_mock_recommendations(db, user_id), "Gemini is not enabled")
    
    recs = generate_book_recommendations(prompt)

    if not recs:
        logger.error("No recommendations generated")
        return ([], "No recommendations generated")

    try:
        parsed_recs = json.loads(recs.text)
        recommendations_raw = parsed_recs.get('items', [])
    except Exception as e:
        logger.error(f"Error parsing Gemini response: {e}")
        return ([], "Error parsing Gemini response")
    
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
            
    # Update user rate limit stats
    if user:
         user.daily_recommendation_count += 1
         user.last_recommendation_date = datetime.utcnow()
         
    db.session.commit()
                
    return (final_recommendations, "Updated")

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
        elif rec.book_id in MOCK_BOOKS:
            # Handle legacy mock IDs (without seed)
            book = MOCK_BOOKS[rec.book_id]
            final_recommendations.append({
                'id': rec.book_id,
                'title': book['title'], # No seed info available for legacy
                'author': book['author'],
                'reason': rec.gemini_reason,
                'cover': book['cover'],
                'date': rec.date
            })
        else:
            # Check for composite mock ID (e.g. "mock-1|42")
            if '|' in rec.book_id:
                parts = rec.book_id.split('|')
                if len(parts) == 2:
                    base_id = parts[0]
                    seed = parts[1]
                    
                    if base_id in MOCK_BOOKS:
                        book = MOCK_BOOKS[base_id]
                        final_recommendations.append({
                            'id': base_id, # Return clean ID to frontend
                            'title': f"{book['title']} {seed}",
                            'author': book['author'],
                            'reason': rec.gemini_reason,
                            'cover': book['cover'],
                            'date': rec.date
                        })
                        continue

            logger.warning(f"Book ID {rec.book_id} from saved recommendations not found in current library items.")
    return final_recommendations
