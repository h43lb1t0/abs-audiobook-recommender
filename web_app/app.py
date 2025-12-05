from flask import Flask, jsonify, send_from_directory, request, Response, render_template, redirect, url_for, flash
import os
import requests
import logging
from dotenv import load_dotenv
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_socketio import SocketIO, emit, join_room

from recommend_lib.recommender import get_recommendations, get_last_recommendations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from recommend_lib.abs_api import get_abs_users
from models.db import db, User

load_dotenv()

def bool_from_env(var):
    if var is None:
        return False
    return var.lower() == 'true'

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key") # Needed for sessions
ABS_URL = os.getenv("ABS_URL")
ABS_FETCH_INTERVAL = int(os.getenv("ABS_FETCH_INTERVAL", 5))
ABS_TOKEN = os.getenv("ABS_TOKEN")
USE_GEMINI = bool_from_env(os.getenv("USE_GEMINI"))

# Use absolute path for database to avoid instance path confusion
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
db_path = os.path.join(basedir, 'instance', 'site.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
logger.debug(f"Database path: {db_path}")

db.init_app(app)
socketio = SocketIO(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, user_id)

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
                    username=abs_user['username'].lower(), 
                    password=hashed_password
                )
                db.session.add(new_user)
            else:
                logger.info(f"Updating existing user: {abs_user['username']}")
                # Update username if changed
                user.username = abs_user['username'].lower()
        db.session.commit()
        logger.info("Synced users from ABS.")
    except Exception as e:
        logger.error(f"Error syncing users from ABS: {e}")

def init_db():
    with app.app_context():
        # Ensure instance directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db.create_all()
        sync_abs_users()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username').lower()
        password = request.form.get('password')

        logger.debug(f"Login attempt for username: {username}")

        user = User.query.filter_by(username=username).first()
        
        if user:
            logger.debug(f"User found: {user.username}, ID: {user.id}")
            if check_password_hash(user.password, password) or user.password == password:
                logger.debug("Password match. Logging in.")

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

@app.route('/api/recommend')
@login_required
def recommend():
    """
    Returns the recommendations
    """
    try:
        recs = get_recommendations(USE_GEMINI, user_id=current_user.id, db=db)
        return jsonify(recs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/last_recommendations')
@login_required
def last_recommendations():
    """
    Returns the last recommendations if they exist
    """
    try:
        recs = get_last_recommendations(user_id=current_user.id, db=db)
        return jsonify(recs)
    except Exception as e:
        logger.error(f"Error fetching last recommendations: {e}")
        return jsonify({"error": str(e)}), 500

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

# WebSocket Events

@socketio.on('connect')
def handle_connect():
    if current_user.is_authenticated:
        logger.info(f"User connected: {current_user.username}")
        join_room(f"user_{current_user.id}")
    else:
        logger.info("Anonymous user connected")

@socketio.on('generate_recommendations')
@login_required
def handle_generate_recommendations():
    try:
        emit('status', {'message': 'Fetching library data...'})
        # Generate recommendations using the existing function
        recs, status = get_recommendations(USE_GEMINI, user_id=current_user.id, db=db)
        emit('recommendations', recs)
        emit('status', {'message': 'Done!'})
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        emit('error', {'message': str(e)})

# Initialize Scheduler
from flask_apscheduler import APScheduler
import time

scheduler = APScheduler()

def scheduled_recommendation_task():
    """
    Background task to generate recommendations for all users.
    Runs every 5 minutes (configured below), but sequentially with a 2-minute pause between users.
    """
    with app.app_context():
        logger.info("Starting scheduled recommendation task...")
        try:
            users = db.session.query(User).all()
            for i, user in enumerate(users):
                logger.info(f"Processing recommendations for user: {user.username} ({user.id})")
                try:
                    recs, status = get_recommendations(USE_GEMINI, user_id=user.id, db=db)
                    logger.info(f"Recommendation status for {user.username}: {status}")
                    
                    if status != "No Update":
                         # Emit to the user's room
                        logger.info(f"Emitting recommendations to user_{user.id}")
                        socketio.emit('recommendations', recs, room=f"user_{user.id}")

                except Exception as e:
                    logger.error(f"Error generating recommendations for user {user.username}: {e}")
                    status = "Error"
                
                if i < len(users) - 1:
                    if status == "No Update":
                        logger.info("No updates found, skipping sleep period.")
                    else:
                        logger.info("Waiting 2 minutes before next user...")
                        time.sleep(120)
        except Exception as e:
            logger.error(f"Error in scheduled task: {e}")
        logger.info("Scheduled recommendation task finished.")

if __name__ == '__main__':
    init_db()
    
    scheduler.add_job(id='scheduled_recommendation_task', func=scheduled_recommendation_task, trigger='interval', minutes=ABS_FETCH_INTERVAL, max_instances=1, coalesce=True)
    scheduler.init_app(app)
    scheduler.start()
    
    socketio.run(app, debug=True, use_reloader=False) # use_reloader=False prevents double execution of scheduler

