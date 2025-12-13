from gevent import monkey
monkey.patch_all()
import json
import logging
import os
from datetime import datetime

import requests
from background_tasks import scheduled_indexing, scheduled_user_activity_check
from db import User, UserLib, UserRecommendations, db
from defaults import *
from dotenv import load_dotenv
from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                   request, url_for, send_from_directory)
from flask_apscheduler import APScheduler
from flask_login import (LoginManager, current_user, login_required,
                         login_user, logout_user)
from flask_babel import Babel, gettext as _
from flask_socketio import SocketIO, emit, join_room
from logger_conf import setup_logging
from recommend_lib.abs_api import (get_abs_users, get_all_items,
                                   get_finished_books)
from recommend_lib.rag import get_rag_system, init_rag_system
from recommend_lib.recommender import get_recommendations
from sqlalchemy import inspect, text
from werkzeug.security import check_password_hash, generate_password_hash

setup_logging()
logger = logging.getLogger(__name__)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key") # Needed for sessions
socketio = SocketIO(app)

def get_locale():
    # Check if a custom header or query param is present
    # Priority:
    # 1. 'lang' query parameter (e.g. ?lang=de)
    # 2. 'Accept-Language' header
    
    user_lang = request.args.get('lang')
    if user_lang:
        return user_lang
    
    if current_user.is_authenticated and hasattr(current_user, 'language') and current_user.language:
        return current_user.language
        
    return request.accept_languages.best_match(['en', 'de'])

babel = Babel(app, locale_selector=get_locale)

ABS_URL = os.getenv("ABS_URL")
ABS_TOKEN = os.getenv("ABS_TOKEN")

# Use absolute path for database to avoid instance path confusion
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
db_path = os.path.join(basedir, 'instance', 'site.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
logger.debug(f"Database path: {db_path}")

db.init_app(app)

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, user_id)

def init_db():
    with app.app_context():
        # Ensure instance directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db.create_all()
        
        # Ensure root user exists
        root_user = db.session.get(User, 'root')
        if not root_user:
            logger.info("Creating root user...")
            root_password = os.getenv('ROOT_PASSWORD', 'admin')
            hashed_root_pw = generate_password_hash(root_password)
            new_root = User(
                id='root',
                username='root',
                password=hashed_root_pw
            )
            db.session.add(new_root)
            db.session.commit()
            logger.info("Root user created.")

        sync_abs_users()




def sync_abs_users():
    """Syncs users from ABS to the local database."""
    try:
        abs_users = get_abs_users()
        logger.info(f"Fetched {len(abs_users)} users from ABS.")
        for abs_user in abs_users:
            user = db.session.get(User, abs_user['id'])
            if not user:
                logger.info(f"Creating new user: {abs_user['username']}")
                # Create new user with hashed password (default is username)
                hashed_password = generate_password_hash(abs_user['username'])
                new_user = User(
                    id=abs_user['id'], 
                    username=abs_user['username'], 
                    password=hashed_password,
                    force_password_change=True
                )
                db.session.add(new_user)
            else:
                logger.info(f"Updating existing user: {abs_user['username']}")
                # Update username if changed
                user.username = abs_user['username']
        
        db.session.commit()
        logger.info("Synced users from ABS.")
    except Exception as e:
        logger.error(f"Error syncing users from ABS: {e}")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
        else:
            username = request.form.get('username')
            password = request.form.get('password')

        logger.debug(f"Login attempt for username: {username}")

        user = User.query.filter_by(username=username).first()
        
        if user:
            logger.debug(f"User found: {user.username}, ID: {user.id}")
            if check_password_hash(user.password, password):
                logger.debug("Password match. Logging in.")

                login_user(user)
                if request.is_json:
                    return jsonify({"success": True, "user": {"id": user.id, "username": user.username}})
                return redirect(url_for('index'))
            else:
                logger.debug("Password mismatch.")
                if request.is_json:
                    return jsonify({"error": _("Invalid username or password")}), 401
                flash(_('Invalid username or password'))
        else:
            logger.debug("User not found.")
            if request.is_json:
                return jsonify({"error": _("Invalid username or password")}), 401
            flash(_('Invalid username or password'))
            
    return app.send_static_file('dist/index.html')

@app.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            current_password = data.get('current_password')
            new_password = data.get('new_password')
            confirm_password = data.get('confirm_password')
        else:
            current_password = request.form.get('current_password')
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')

        if not check_password_hash(current_user.password, current_password):
            if request.is_json:
                return jsonify({"error": _("Incorrect current password")}), 400
            flash(_('Incorrect current password'))
        elif check_password_hash(current_user.password, new_password):
             if request.is_json:
                return jsonify({"error": _("New password cannot be the same as the current password")}), 400
             flash(_('New password cannot be the same as the current password'))
        elif new_password != confirm_password:
             if request.is_json:
                return jsonify({"error": _("New passwords do not match")}), 400
             flash(_('New passwords do not match'))
        else:
            current_user.password = generate_password_hash(new_password)
            current_user.force_password_change = False
            db.session.commit()
            if request.is_json:
                return jsonify({"success": True})
            flash(_('Password updated successfully'))
            return redirect(url_for('index'))
            
    return app.send_static_file('dist/index.html')

@app.route('/api/user/language', methods=['POST'])
@login_required
def set_language():
    try:
        data = request.get_json()
        language = data.get('language')
        
        if not language:
            return jsonify({"error": _("Language is required")}), 400
            
        if language not in ['en', 'de']:
             return jsonify({"error": _("Invalid language")}), 400
             
        current_user.language = language
        db.session.commit()
        
        return jsonify({"success": True, "language": language})
    except Exception as e:
        logger.error(f"Error setting language: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/auth/status')
def auth_status():
    if current_user.is_authenticated:
        return jsonify({
            "authenticated": True, 
            "user": {
                "id": current_user.id, 
                "username": current_user.username,
                "language": getattr(current_user, 'language', 'en'),
                "force_password_change": getattr(current_user, 'force_password_change', False)
            }, 
            "abs_url": ABS_URL
        })
    return jsonify({"authenticated": False}), 401

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    """
    Returns the index.html file
    """
    return app.send_static_file('dist/index.html')

@app.route('/history')
@login_required
def listening_history():
    """
    Returns the listening history page
    """
    return app.send_static_file('dist/index.html')

@app.route('/in-progress')
@login_required
def in_progress():
    """
    Returns the in-progress books page
    """
    return app.send_static_file('dist/index.html')

@app.route('/settings')
@login_required
def settings():
    """
    Returns the settings page
    """
    return app.send_static_file('dist/index.html')

@app.route('/api/listening-history')
@login_required
def get_listening_history():
    """
    Returns the user's finished books with their ratings
    """
    try:
        items_map, _ = get_all_items()
        finished_ids, _, _ = get_finished_books(items_map, user_id=current_user.id)
        
        
        # Get user's ratings from database
        user_ratings = UserLib.query.filter_by(user_id=current_user.id).all()
        ratings_map = {r.book_id: r.rating for r in user_ratings}
        
        finished_books = []
        for book_id in finished_ids:
            if book_id in items_map:
                book = items_map[book_id]
                # Parse series_sequence as float for proper sorting (handles "1", "1.5", "2", etc.)
                seq = book.get('series_sequence')
                try:
                    seq_num = float(seq) if seq else float('inf')
                except (ValueError, TypeError):
                    seq_num = float('inf')
                
                finished_books.append({
                    'id': book['id'],
                    'title': book['title'],
                    'author': book['author'],
                    'series': book.get('series'),
                    'series_sequence': book.get('series_sequence'),
                    'series_sequence_num': seq_num,
                    'rating': ratings_map.get(book_id)
                })
        
        # Sort: first by series name (None/standalone at the end), then by sequence within series
        finished_books.sort(key=lambda x: (
            x['series'] is None,  # Books with series come first
            (x['series'] or '').lower(),  # Then alphabetically by series name
            x['series_sequence_num']  # Then by sequence number within series
        ))
        
        # Remove the temporary sorting field before sending
        for book in finished_books:
            del book['series_sequence_num']
        
        return jsonify(finished_books)
    except Exception as e:
        logger.error(f"Error fetching listening history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/in-progress')
@login_required
def get_in_progress():
    """
    Returns the user's in-progress books
    """
    try:
        items_map, _ = get_all_items()
        _, in_progress_ids, _ = get_finished_books(items_map, user_id=current_user.id)
        
        in_progress_books = []
        for book_id, progress in in_progress_ids.items():
            if book_id in items_map:
                book = items_map[book_id]
                # Parse series_sequence as float for proper sorting
                seq = book.get('series_sequence')
                try:
                    seq_num = float(seq) if seq else float('inf')
                except (ValueError, TypeError):
                    seq_num = float('inf')
                
                in_progress_books.append({
                    'id': book['id'],
                    'title': book['title'],
                    'author': book['author'],
                    'series': book.get('series'),
                    'series_sequence': book.get('series_sequence'),
                    'series_sequence_num': seq_num,
                    'cover': book.get('cover'), # Ensure cover is available if needed by frontend directly, usually handled by separate endpoint though
                    'progress': progress,
                    'status': 'reading' # Default to reading
                })
        
        # Fetch local status overrides (e.g. abandoned)
        user_lib_entries = UserLib.query.filter(
            UserLib.user_id == current_user.id,
            UserLib.book_id.in_([b['id'] for b in in_progress_books])
        ).all()
        
        status_map = {e.book_id: e.status for e in user_lib_entries}
        
        for book in in_progress_books:
            if book['id'] in status_map:
                book['status'] = status_map[book['id']]
        
        # Sort: first by series name, then by sequence
        in_progress_books.sort(key=lambda x: (
            x['series'] is None,
            (x['series'] or '').lower(),
            x['series_sequence_num']
        ))
        
        for book in in_progress_books:
            del book['series_sequence_num']
            
        return jsonify(in_progress_books)
    except Exception as e:
        logger.error(f"Error fetching in-progress books: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/rate-book', methods=['POST'])
@login_required
def rate_book():
    """
    Saves or updates a book rating for the current user
    """
    try:
        data = request.get_json()
        book_id = data.get('book_id')
        rating = data.get('rating')
        
        if not book_id:
            return jsonify({"error": _("book_id is required")}), 400
        
        if rating is not None and (rating < 1 or rating > 5):
            return jsonify({"error": _("Rating must be between 1 and 5")}), 400
        
        # Check if rating already exists
        existing = UserLib.query.filter_by(user_id=current_user.id, book_id=book_id).first()

        if existing.status == 'abandoned':
            return jsonify({"error": _("You cannot rate a book that you have abandoned")}), 400
        
        if existing:
            existing.rating = rating
            existing.updated_at = datetime.now().isoformat()
            db.session.commit()
        else:
            return jsonify({"error": _("Book not found in library")}), 404
        return jsonify({"success": True, "book_id": book_id, "rating": rating})
    except Exception as e:
        logger.error(f"Error saving rating: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ratings')
@login_required
def get_ratings():
    """
    Returns all ratings for the current user
    """
    try:
        user_ratings = UserLib.query.filter_by(user_id=current_user.id).all()
        ratings = {r.book_id: r.rating for r in user_ratings if r.rating is not None}
        return jsonify(ratings)
    except Exception as e:
        logger.error(f"Error fetching ratings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/abandon-book', methods=['POST'])
@login_required
def abandon_book():
    """
    Marks a book as abandoned.
    """
    try:
        data = request.get_json()
        book_id = data.get('book_id')
        
        if not book_id:
            return jsonify({"error": _("book_id is required")}), 400
            
        # Check if book exists in library
        existing = UserLib.query.filter_by(user_id=current_user.id, book_id=book_id).first()
        
        if existing:
            if existing.status == 'finished':
                 return jsonify({"error": _("Cannot abandon a finished book")}), 400
                 
            existing.status = 'abandoned'
            existing.updated_at = datetime.now().isoformat()
            db.session.commit()
            return jsonify({"success": True, "book_id": book_id, "status": "abandoned"})
        else:
            # If not in local lib, create it (edge case if not synced yet but in progress)
            # However, typically it should be synced if it's in progress.
            # We'll create it with abandoned status.
            new_entry = UserLib(
                user_id=current_user.id,
                book_id=book_id,
                status='abandoned',
                updated_at=datetime.now().isoformat()
            )
            db.session.add(new_entry)
            db.session.commit()
            return jsonify({"success": True, "book_id": book_id, "status": "abandoned"})
            
    except Exception as e:
        logger.error(f"Error abandoning book: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/reactivate-book', methods=['POST'])
@login_required
def reactivate_book():
    """
    Marks a book as reading (reactivates an abandoned book).
    """
    try:
        data = request.get_json()
        book_id = data.get('book_id')
        
        if not book_id:
            return jsonify({"error": _("book_id is required")}), 400
            
        existing = UserLib.query.filter_by(user_id=current_user.id, book_id=book_id).first()
        
        if existing:
            existing.status = 'reading'
            existing.updated_at = datetime.now().isoformat()
            db.session.commit()
            return jsonify({"success": True, "book_id": book_id, "status": "reading"})
        else:
            return jsonify({"error": _("Book not found in library")}), 404
            
    except Exception as e:
        logger.error(f"Error reactivating book: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend')
@login_required
def recommend():
    """
    Returns the recommendations
    """
    refresh = request.args.get('refresh', 'false').lower() == 'true'
    
    try:
        if not refresh:
            # Try to fetch from DB
            existing_recs = UserRecommendations.query.filter_by(user_id=current_user.id).first()
            if existing_recs:
                logger.info(f"Returning cached recommendations for user {current_user.id}")
                return jsonify({
                    "recommendations": json.loads(existing_recs.recommendations_json),
                    "generated_at": existing_recs.created_at
                })
        
        # Calculate new recommendations
        recs = get_recommendations(user_id=current_user.id)
        
        # Save to DB (overwrite)
        current_time = datetime.now().isoformat()
        
        existing_recs = UserRecommendations.query.filter_by(user_id=current_user.id).first()
        if existing_recs:
            existing_recs.recommendations_json = json.dumps(recs)
            existing_recs.created_at = current_time
        else:
            new_recs = UserRecommendations(
                user_id=current_user.id,
                recommendations_json=json.dumps(recs),
                created_at=current_time
            )
            db.session.add(new_recs)
            
        db.session.commit()
        
        return jsonify({
            "recommendations": recs, 
            "generated_at": current_time
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """
    Handle new websocket connection
    """
    if current_user.is_authenticated:
        logger.info(f"User {current_user.id} connected to websocket. Joining room {current_user.id}")
        join_room(current_user.id)
    else:
        logger.info("Anonymous user connected to websocket")

@socketio.on('get_recommendations')
def handle_get_recommendations(data):
    """
    WebSocket handler for generating recommendations
    """
    if not current_user.is_authenticated:
        emit('error', {'error': _('Authentication required')})
        return

    refresh = data.get('refresh', False)
    
    try:
        if not refresh:
            # Try to fetch from DB
            existing_recs = UserRecommendations.query.filter_by(user_id=current_user.id).first()
            if existing_recs:
                logger.info(f"Returning cached recommendations for user {current_user.id}")
                emit('recommendations_ready', {
                    "recommendations": json.loads(existing_recs.recommendations_json),
                    "generated_at": existing_recs.created_at
                })
                return
        
        # Calculate new recommendations
        recs = get_recommendations(user_id=current_user.id)
        
        # Save to DB (overwrite)
        current_time = datetime.now().isoformat()
        
        existing_recs = UserRecommendations.query.filter_by(user_id=current_user.id).first()
        with app.app_context(): # Ensure DB context
             existing_recs = UserRecommendations.query.filter_by(user_id=current_user.id).first()
             if existing_recs:
                 existing_recs.recommendations_json = json.dumps(recs)
                 existing_recs.created_at = current_time
             else:
                 new_recs = UserRecommendations(
                     user_id=current_user.id,
                     recommendations_json=json.dumps(recs),
                     created_at=current_time
                 )
                 db.session.add(new_recs)
             db.session.commit()
        
        emit('recommendations_ready', {
            "recommendations": recs, 
            "generated_at": current_time
        })
    except Exception as e:
        logger.error(f"Error in websocket recommendation: {e}")
        emit('error', {"error": str(e)})

@app.route('/api/cover/<item_id>')
@login_required
def proxy_cover(item_id):
    """
    Returns the cover image from ABS
    """

    if not ABS_URL or not ABS_TOKEN:
        return _("Server misconfigured"), 500
        
    abs_url = ABS_URL.rstrip('/')
    
    cover_url = f"{abs_url}/api/items/{item_id}/cover"
    headers = {"Authorization": f"Bearer {ABS_TOKEN}"}
    
    resp = requests.get(cover_url, headers=headers, stream=True)
    
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]
               
    return Response(resp.content, resp.status_code, headers)



scheduler.add_job(
    id='scheduled_user_activity_check',
    func=scheduled_user_activity_check,
    trigger='interval',
    minutes=BACKGROUND_TASKS['CREATE_RECOMMENDATIONS_INTERVAL'],
    args=[app, socketio]
)

scheduler.add_job(
    id='scheduled_indexing', 
    func=scheduled_indexing, 
    trigger='interval', 
    hours=BACKGROUND_TASKS['CHECK_NEW_BOOKS_INTERVAL'],
    args=[app]
)

@app.route('/api/admin/force-sync', methods=['POST'])
@login_required
def force_sync():
    """
    Force triggers the library indexing (Root only)
    """
    if current_user.id != 'root':
        return jsonify({"error": _("Unauthorized")}), 403
    
    try:
        logger.info("Force sync triggered by root user.")
        items_map, _ = get_all_items()
        rag = get_rag_system()
        rag.index_library(items_map)
        return jsonify({"status": "success", "message": "Indexing triggered"})
    except Exception as e:
        logger.error(f"Error in force sync: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/force-check')
def debug_force_check():
    scheduled_user_activity_check(app, socketio)
    return "Check triggered"

if __name__ == '__main__':
    init_db()
    with app.app_context():
        init_rag_system()
    socketio.run(app, debug=False)
