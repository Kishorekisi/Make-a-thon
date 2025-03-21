from flask import Flask, render_template, jsonify, send_from_directory
import sqlite3
import os
from flask_socketio import SocketIO

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app)

# Database connection
def get_db_connection():
    conn = sqlite3.connect('seal_detection.db')
    conn.row_factory = sqlite3.Row
    return conn

# Main route to serve the dashboard
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve CSS and other static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# API endpoint to get current vote counts
@app.route('/api/votes', methods=['GET'])
def get_votes():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get all candidates and their vote counts
    cursor.execute("SELECT sno, name, count FROM detections ORDER BY count DESC")
    rows = cursor.fetchall()
    
    # Calculate total votes
    total_votes = sum(row['count'] for row in rows)
    
    # Format data for the dashboard
    candidates = []
    for row in rows:
        percent = round((row['count'] / total_votes * 100), 1) if total_votes > 0 else 0
        candidates.append({
            'sno': row['sno'],
            'name': row['name'] or f"Candidate {row['sno']}",
            'votes': row['count'],
            'percent': percent
        })
    
    conn.close()
    
    return jsonify({
        'totalVotes': total_votes,
        'candidates': candidates,
        'lastUpdated': datetime.datetime.now().strftime("%B %d, %Y - %H:%M")
    })

# WebSocket event for real-time updates
@socketio.on('update_votes')
def handle_update(data):
    socketio.emit('votes_updated', data)

if __name__ == '__main__':
    import datetime
    
    # Make sure directory structure exists
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Copy the HTML to templates folder
    #with open('index.html', 'r') as src_file, open('templates/index.html', 'w') as dest_file:
        #dest_file.write(src_file.read())
    
    # Copy the CSS to static folder
    # open('styles.css', 'r') as src_file, open('static/styles.css', 'w') as dest_file:
        #dest_file.write(src_file.read())
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)