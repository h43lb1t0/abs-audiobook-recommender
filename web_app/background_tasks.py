import json
import logging
from datetime import datetime, timedelta

from db import BackgroundCheckLog, User, UserLib, UserRecommendations, db
from defaults import BACKGROUND_TASKS
from recommend_lib.abs_api import get_all_items, get_finished_books
from recommend_lib.rag import get_rag_system
from recommend_lib.recommender import get_recommendations

logger = logging.getLogger(__name__)

def scheduled_indexing(app):
    """
    Scheduled task to index the library
    """
    with app.app_context():
        logger.info("Starting scheduled library indexing...")
        try:
             items_map, _ = get_all_items()
             rag = get_rag_system()
             new_count = rag.index_library(items_map)
             
             # Log the check
             current_time = datetime.now().isoformat()
             
             if new_count > 0:
                try:
                    db.session.query(BackgroundCheckLog).delete()
                    
                    new_log = BackgroundCheckLog(
                        checked_new_books_at=current_time
                    )
                    db.session.add(new_log)
                    db.session.commit()
                    logger.info(f"Scheduled library indexing complete. New items: {new_count}. Logged to DB.")
                except Exception as e:
                    logger.error(f"Error logging background check: {e}")
                    db.session.rollback()

        except Exception as e:
             logger.error(f"Error in scheduled indexing: {e}")


def scheduled_user_activity_check(app, socketio_instance):
    """
    Background task to check for new books and user activity to trigger recommendations.
    """
    with app.app_context():
        logger.debug("Starting background check task...")
        try:
            # 1. Check for new books being added
            last_check_log = BackgroundCheckLog.query.order_by(BackgroundCheckLog.id.desc()).first()
            
            new_books_trigger = False
            
            if last_check_log:
                # If checked_new_books_at is older than interval AND created_recommendations is False
                if last_check_log.checked_new_books_at:
                    checked_at = datetime.fromisoformat(last_check_log.checked_new_books_at)                
                    time_threshold = datetime.now() - timedelta(hours=BACKGROUND_TASKS['CHECK_NEW_BOOKS_INTERVAL'])
                    
                    if checked_at < time_threshold and not last_check_log.created_recommendations:
                        logger.info("Found pending new books check. Triggering global recommendation update.")
                        new_books_trigger = True
                        
            
            users = User.query.all()
            
            check_time = datetime.now().isoformat()
            
            for user in users:
                user_id = user.id
                if user_id == 'root':
                    continue
                should_generate = False
                
                # Global trigger (New books added)
                if new_books_trigger:
                    should_generate = True
                
                if not should_generate:
                    last_rec = UserRecommendations.query.filter_by(user_id=user_id).first()
                    last_rec_time = datetime.min
                    
                    if last_rec and last_rec.created_at:
                        try:
                            last_rec_time = datetime.fromisoformat(last_rec.created_at)
                        except ValueError:
                            pass

                    items_map, _ = get_all_items()
                    get_finished_books(items_map, user_id=user_id)
            
                    
                    recent_changes = UserLib.query.filter(
                        UserLib.user_id == user_id,
                        UserLib.updated_at > last_rec_time.isoformat()
                    ).first()
                    
                    if recent_changes:
                        logger.info(f"User {user.username} has recent activity since {last_rec_time}. triggering recommendations.")
                        should_generate = True

                if should_generate:
                    logger.info(f"BGT: Generating background recommendations for {user.username} (ID: {user_id})")
                    try:
                        # Wrap in request context for Flask-Babel (gettext/get_locale)
                        # We pass the user's language as a query param to satisfy the locale selector
                        lang = user.language if user.language else 'en'
                        with app.test_request_context(f"/?lang={lang}"):
                            recs = get_recommendations(user_id=user_id)
                            
                            existing_recs = UserRecommendations.query.filter_by(user_id=user_id).first()
                            
                            if existing_recs:
                                existing_recs.recommendations_json = json.dumps(recs)
                                existing_recs.created_at = check_time
                            else:
                                new_recs = UserRecommendations(
                                    user_id=user_id,
                                    recommendations_json=json.dumps(recs),
                                    created_at=check_time
                                )
                                db.session.add(new_recs)
                            
                            db.session.commit()
                            logger.info(f"BGT: Recommendations updated for {user.username}. Emitting to room {user_id}")
                            
                            # Broadcast websocket event to notify user of new recommendations
                            try:
                                logger.info(f"BGT: Attempting emit to {user_id} with socketio {socketio_instance}")
                                socketio_instance.emit('recommendations_ready', {
                                    'recommendations': recs,
                                    'generated_at': check_time
                                }, room=user_id)
                                logger.info(f"BGT: Successfully emitted recommendations_ready event for {user.username} to room {user_id}")
                            except Exception as ws_error:
                                logger.error(f"BGT: Failed to broadcast websocket event: {ws_error}")
                        
                    except Exception as e:
                        logger.error(f"BGT: Error generating recommendations for {user.username}: {e}")
                else:
                    logger.info(f"BGT: No recommendations needed for {user.username}. changes={bool(recent_changes) if 'recent_changes' in locals() else 'N/A'}, global={new_books_trigger}")
            
            # Update the log if we processed the global trigger
            if new_books_trigger and last_check_log:
                last_check_log.created_recommendations = True
                db.session.commit()
                
        except Exception as e:
            logger.error(f"BGT: Error in background check task: {e}")
