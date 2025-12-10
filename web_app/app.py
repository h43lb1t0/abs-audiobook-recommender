from flask import Flask, jsonify, send_from_directory, request, Response, render_template, redirect, url_for, flash
import os
import requests
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_socketio import SocketIO, emit

from recommend_lib.recommender import get_recommendations
from recommend_lib.rag import init_rag_system, get_rag_system
from flask_apscheduler import APScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from recommend_lib.abs_api import get_abs_users, get_all_items, get_finished_books
from db import db, User, UserLib, UserRecommendations

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key") # Needed for sessions
socketio = SocketIO(app)

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
                    password=hashed_password
                )
                db.session.add(new_user)
            else:
                logger.info(f"Updating existing user: {abs_user['username']}")
                # Update username if changed
                user.username = abs_user['username']
                
                # Check if the current password is the plain text username (migration/first run with existing db)
                if user.password == abs_user['username']:
                     logger.debug(f"Migrating password for {user.username} to hash.")
                     user.password = generate_password_hash(abs_user['username'])
                
                # If the password is NOT the username, we assume the user changed it, so we DO NOT overwrite it.
        
        db.session.commit()
        logger.info("Synced users from ABS.")
    except Exception as e:
        logger.error(f"Error syncing users from ABS: {e}")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        logger.debug(f"Login attempt for username: {username}")

        user = User.query.filter_by(username=username).first()
        
        if user:
            logger.debug(f"User found: {user.username}, ID: {user.id}")
            # Check if password matches hash OR if it matches plain text (backward compatibility during migration)
            if check_password_hash(user.password, password) or user.password == password:
                logger.debug("Password match. Logging in.")

                # If it matched plain text, upgrade to hash immediately
                if user.password == password:
                     logger.debug("Upgrading plain text password to hash on login.")
                     user.password = generate_password_hash(password)
                     db.session.commit()

                login_user(user)
                return redirect(url_for('index'))
            else:
                logger.debug("Password mismatch.")
                flash('Invalid username or password')
        else:
            logger.debug("User not found.")
            flash('Invalid username or password')
            
    return render_template('login.html')

@app.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if not check_password_hash(current_user.password, current_password):
            flash('Incorrect current password')
        elif new_password != confirm_password:
            flash('New passwords do not match')
        else:
            current_user.password = generate_password_hash(new_password)
            db.session.commit()
            flash('Password updated successfully')
            return redirect(url_for('index'))
            
    return render_template('change_password.html')

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
    return render_template('index.html')

@app.route('/history')
@login_required
def listening_history():
    """
    Returns the listening history page
    """
    return render_template('listening_history.html')

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
            return jsonify({"error": "book_id is required"}), 400
        
        if rating is not None and (rating < 1 or rating > 5):
            return jsonify({"error": "Rating must be between 1 and 5"}), 400
        
        # Check if rating already exists
        existing = UserLib.query.filter_by(user_id=current_user.id, book_id=book_id).first()
        
        if existing:
            existing.rating = rating
        else:
            new_entry = UserLib(user_id=current_user.id, book_id=book_id, rating=rating)
            db.session.add(new_entry)
        
        db.session.commit()
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

@socketio.on('get_recommendations')
def handle_get_recommendations(data):
    """
    WebSocket handler for generating recommendations
    """
    if not current_user.is_authenticated:
        emit('error', {'error': 'Authentication required'})
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
        return "Server misconfigured", 500
        
    abs_url = ABS_URL.rstrip('/')
    
    cover_url = f"{abs_url}/api/items/{item_id}/cover"
    headers = {"Authorization": f"Bearer {ABS_TOKEN}"}
    
    resp = requests.get(cover_url, headers=headers, stream=True)
    
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]
               
    return Response(resp.content, resp.status_code, headers)

def scheduled_indexing():
    """
    Scheduled task to index the library
    """
    with app.app_context():
        logger.info("Starting scheduled library indexing...")
        try:
             items_map, _ = get_all_items()
             rag = get_rag_system()
             rag.index_library(items_map)
             logger.info("Scheduled library indexing complete.")
        except Exception as e:
             logger.error(f"Error in scheduled indexing: {e}")

# Schedule task to run every 6 hours
scheduler.add_job(id='scheduled_indexing', func=scheduled_indexing, trigger='interval', hours=6)

@app.route('/api/admin/force-sync', methods=['POST'])
@login_required
def force_sync():
    """
    Force triggers the library indexing (Root only)
    """
    if current_user.id != 'root':
        return jsonify({"error": "Unauthorized"}), 403
        
    # Run synchronously for immediate feedback, or trigger background job
    # For now, let's run it essentially synchronously or fire-and-forget but return success
    
    # We can just call the function directly
    # Note: This might timeout if library is huge, but for now it's fine
    try:
        logger.info("Force sync triggered by root user.")
        items_map, _ = get_all_items()
        rag = get_rag_system()
        rag.index_library(items_map)
        return jsonify({"status": "success", "message": "Indexing triggered"})
    except Exception as e:
        logger.error(f"Error in force sync: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_db()
    init_rag_system()
    socketio.run(app, debug=True)
