import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class Config:
    # Flask app settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key_change_in_production_1234')
    
    # SQLAlchemy configuration
    # Create the database directory automatically
    DB_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'database')
    os.makedirs(DB_DIR, exist_ok=True)
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', f"sqlite:///{os.path.join(DB_DIR, 'pcos_app.db')}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Flask-Mail configuration
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_USERNAME')

    # REST API key controls
    API_KEYS = {
        'key1': os.getenv('API_KEY_1', 'default_rest_api_key_pcos_app_xyz_789')
    }

    # Google Gemini configuration
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
