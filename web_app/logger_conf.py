# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

LOG_FILE = "app.log"

def setup_logging():
    formatter = logging.Formatter(
        u'%(asctime)-s %(levelname)s [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    # File handler (DEBUG)
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console/Terminal handler (INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    
    if root.handlers:
        for handler in root.handlers:
            handler.close()
        root.handlers = []
        
    root.addHandler(file_handler)
    root.addHandler(console_handler)
