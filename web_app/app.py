from flask import Flask, jsonify, send_from_directory, request, Response, render_template, redirect, url_for, flash
import os
import requests
import logging
from dotenv import load_dotenv
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from recommend_lib.recommender import get_recommendations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from recommend_lib.abs_api import get_abs_users
from db import db, User

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key") # Needed for sessions
ABS_URL = os.getenv("ABS_URL")
ABS_TOKEN = os.getenv("ABS_TOKEN")
USE_GEMINI = bool(os.getenv("USE_GEMINI", True))

# Use absolute path for database to avoid instance path confusion
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
db_path = os.path.join(basedir, 'instance', 'site.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
logger.debug(f"Database path: {db_path}")

db.init_app(app)

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

@app.route('/api/recommend')
@login_required
def recommend():
    """
    Returns the recommendations
    """
    try:
        recs = get_recommendations(USE_GEMINI, user_id=current_user.id)
        return jsonify(recs)
    except Exception as e:
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

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
