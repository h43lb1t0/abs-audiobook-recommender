from gevent import monkey

monkey.patch_all()

import sys
from pathlib import Path

# Ensure the current directory is in sys.path so that imports work as expected
sys.path.append(str(Path(__file__).resolve().parent))

from app import app, init_db, init_rag_system  # noqa: E402

# Initialize the database and RAG system on startup
init_db()
with app.app_context():
    init_rag_system()

# Expose the application instance for Gunicorn
application = app
