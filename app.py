import logging

from flask import Flask, request, jsonify, render_template

from config import Config
from nlp.pipeline import load_model, analyze_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

def create_app() -> Flask:
    app = Flask(__name__)

    load_model(Config.MODEL_PATH)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return jsonify({"status": "healthy"})

    @app.route("/analyze", methods=["POST"])
    def analyze():
        json_data = request.get_json()
        if not json_data or "text" not in json_data:
            return jsonify({"error": "Missing 'text' field in request body."}), 400

        raw_text = json_data["text"].strip()
        if not raw_text:
            return jsonify({"error": "The 'text' field cannot be empty."}), 400

        try:
            result = analyze_text(raw_text)
            return jsonify(result)
        except Exception as e:
            logger.exception("Analysis failed")
            return jsonify({"error": "Analysis failed. Please try again."}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
