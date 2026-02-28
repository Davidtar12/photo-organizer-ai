from __future__ import annotations

from flask import Flask, send_from_directory
from flask_cors import CORS
import os

from config import Config, ensure_directories
from database import engine
from models import Base
from routes import persons, system, suggestions, pets, pet_search


def create_app() -> Flask:
    ensure_directories()
    Base.metadata.create_all(bind=engine)

    app = Flask(__name__)
    app.config.update(
        FACE_BROWSER_DB=str(Config.DB_PATH),
        FACE_BROWSER_ORGANIZED=str(Config.ORGANIZED_DIR),
    )

    CORS(app, supports_credentials=False)

    app.register_blueprint(system.bp)
    app.register_blueprint(persons.bp)
    app.register_blueprint(suggestions.bp)
    app.register_blueprint(pets.bp)
    app.register_blueprint(pet_search.bp)
    
    print(app.url_map)  # Debug: print all registered routes
    
    # Serve web UI
    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'web')
    
    @app.route('/')
    def index():
        return send_from_directory(web_dir, 'index.html')
    
    @app.route('/pets.html')
    def pets_page():
        return send_from_directory(web_dir, 'pets.html')

    @app.route('/max.html')
    def max_page():
        return send_from_directory(web_dir, 'max.html')

    @app.route('/search.html')
    def search_page():
        return send_from_directory(web_dir, 'search.html')

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5052, debug=False, use_reloader=False)
