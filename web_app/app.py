from flask import Flask, jsonify, send_from_directory, request, Response
import os
import requests
from dotenv import load_dotenv
import os

from recommend_lib.recommender import get_recommendations


load_dotenv()

app = Flask(__name__)
ABS_URL = os.getenv("ABS_URL")
ABS_TOKEN = os.getenv("ABS_TOKEN")

@app.route('/')
def index():
    """
    Returns the index.html file
    """
    return send_from_directory('templates', 'index.html')

@app.route('/api/recommend')
def recommend():
    """
    Returns the recommendations
    """
    try:
        recs = get_recommendations(False)
        return jsonify(recs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cover/<item_id>')
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
    app.run(debug=True)
