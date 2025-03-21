import sqlite3
import json
import os
import datetime
from flask import Flask, render_template, jsonify, send_from_directory
from flask import Flask, request, jsonify


app = Flask(__name__)

# Database helper functions
def get_db_connection():
    """Connect to the election database"""
    conn = sqlite3.connect('election.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_vote_counts():
    """Get the current vote counts for all candidates"""
    conn = get_db_connection()
    votes = conn.execute('SELECT candidate_id, candidate_name, vote_count FROM votes ORDER BY vote_count DESC').fetchall()
    conn.close()
    
    # Convert to list of dictionaries
    result = []
    for row in votes:
        result.append({
            'candidate_id': row['candidate_id'],
            'candidate_name': row['candidate_name'],
            'vote_count': row['vote_count']
        })
    return result

def get_detection_logs(limit=10):
    """Get the most recent detection logs"""
    conn = get_db_connection()
    logs = conn.execute('''
        SELECT id, timestamp, candidate_id, candidate_name, confidence 
        FROM detection_log 
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,)).fetchall()
    conn.close()
    
    # Convert to list of dictionaries
    result = []
    for row in logs:
        result.append({
            'id': row['id'],
            'timestamp': row['timestamp'],
            'candidate_id': row['candidate_id'],
            'candidate_name': row['candidate_name'],
            'confidence': row['confidence']
        })
    return result

def get_election_summary():
    """Get summary statistics about the election"""
    conn = get_db_connection()
    
    # Get total votes
    total_votes = conn.execute('SELECT SUM(vote_count) as total FROM votes').fetchone()['total']
    
    # Get total candidates
    total_candidates = conn.execute('SELECT COUNT(*) as total FROM votes').fetchone()['total']
    
    # Get winner info
    winner = conn.execute('''
        SELECT candidate_name, vote_count 
        FROM votes 
        ORDER BY vote_count DESC 
        LIMIT 1
    ''').fetchone()
    
    conn.close()
    
    return {
        'total_votes': total_votes if total_votes else 0,
        'total_candidates': total_candidates,
        'winner_name': winner['candidate_name'] if winner and winner['vote_count'] > 0 else 'No winner yet',
        'winner_votes': winner['vote_count'] if winner else 0
    }

# Flask routes
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/votes')
def api_votes():
    return jsonify(get_vote_counts())

@app.route('/api/logs')
def api_logs():
    limit = request.args.get('limit', default=10, type=int)
    return jsonify(get_detection_logs(limit))

@app.route('/api/summary')
def api_summary():
    return jsonify(get_election_summary())

@app.route('/api/export')
def export_csv():
    """Export the current results to CSV"""
    from flask import send_file
    import csv
    
    # Generate a filename with current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"election_results_{timestamp}.csv"
    
    # Get data
    votes = get_vote_counts()
    logs = get_detection_logs(limit=100)  # Get up to 100 log entries
    summary = get_election_summary()
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header and summary
        writer.writerow(['ELECTION RESULTS SUMMARY'])
        writer.writerow(['Total Votes', summary['total_votes']])
        writer.writerow(['Winner', summary['winner_name']])
        writer.writerow(['Winner Votes', summary['winner_votes']])
        writer.writerow([])
        
        # Write vote counts
        writer.writerow(['VOTE COUNTS'])
        writer.writerow(['Candidate ID', 'Candidate Name', 'Vote Count'])
        for vote in votes:
            writer.writerow([vote['candidate_id'], vote['candidate_name'], vote['vote_count']])
        writer.writerow([])
        
        # Write detection logs
        writer.writerow(['DETECTION LOG'])
        writer.writerow(['ID', 'Timestamp', 'Candidate ID', 'Candidate Name', 'Confidence'])
        for log in logs:
            writer.writerow([log['id'], log['timestamp'], log['candidate_id'], log['candidate_name'], log['confidence']])
    
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Save the dashboard HTML to the templates directory
    with open('templates/dashboard.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Election Results Dashboard</title>
    <!-- Add your CSS here or link to external stylesheet -->
    <style>
        /* Copy the CSS from the HTML artifact */
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Election Results Dashboard</h1>
            <p>Live vote counting and election statistics</p>
        </div>
        
        <div id="dashboard-content">
            <!-- This will be populated by JavaScript -->
            <p>Loading dashboard data...</p>
        </div>
        
        <div class="actions">
            <button class="btn btn-secondary" id="export-btn">Export Results</button>
            <button class="btn btn-primary" id="refresh-btn">Refresh Data</button>
        </div>
        
        <p class="refresh-time" id="refresh-time">Last updated: Loading...</p>
    </div>

    <script>
        // Function to update the dashboard
        function updateDashboard() {
            $.when(
                $.getJSON('/api/votes'),
                $.getJSON('/api/logs'),
                $.getJSON('/api/summary')
            ).done(function(votesData, logsData, summaryData) {
                const votes = votesData[0];
                const logs = logsData[0];
                const summary = summaryData[0];
                
                // Build HTML content
                let html = buildDashboardHTML(votes, logs, summary);
                
                // Update the content
                $('#dashboard-content').html(html);
                
                // Update refresh time
                const now = new Date();
                $('#refresh-time').text('Last updated: ' + now.toLocaleString());
            });
        }
        
        function buildDashboardHTML(votes, logs, summary) {
            // Calculate percentages
            const totalVotes = summary.total_votes;
            votes.forEach(candidate => {
                candidate.percentage = totalVotes > 0 ? 
                    Math.round((candidate.vote_count / totalVotes) * 100) : 0;
            });
            
            // Build the HTML
            let html = `
                <div class="dashboard-grid">
                    <div class="card">
                        <div class="card-header">
                            <h2>Election Summary</h2>
                            <span class="badge">Live</span>
                        </div>
                        <div class="stats-grid">
                            <div class="stat-card">
                                <p class="stat-value">${summary.total_votes}</p>
                                <p class="stat-label">Total Votes</p>
                            </div>
                            <div class="stat-card">
                                <p class="stat-value">${summary.total_candidates}</p>
                                <p class="stat-label">Candidates</p>
                            </div>
                            <div class="stat-card">
                                <p class="stat-value">${summary.winner_votes}</p>
                                <p class="stat-label">Winner Votes</p>
                            </div>
                        </div>
                        <div class="results">
                            <h3>Current Results</h3>
                            
                            ${votes.map((candidate, index) => `
                                <div class="candidate-bar">
                                    <div class="candidate-info">
                                        <span class="candidate-name">
                                            ${candidate.candidate_name}
                                            ${index === 0 && candidate.vote_count > 0 ? 
                                                '<span class="winner-badge">WINNER</span>' : ''}
                                        </span>
                                        <span class="candidate-votes">
                                            ${candidate.vote_count} votes (${candidate.percentage}%)
                                        </span>
                                    </div>
                                    <div class="progress-bar">
                                        <div class="progress ${index === 0 ? 'winner' : index === 1 ? 'runner-up' : 'other'}" 
                                             style="width: ${candidate.percentage}%"></div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <h2>Vote Detection Log</h2>
                            <span class="badge">Recent Activity</span>
                        </div>
                        <table>
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Candidate</th>
                                    <th>Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${logs.length > 0 ? logs.map(log => `
                                    <tr>
                                        <td>${new Date(log.timestamp).toLocaleTimeString()}</td>
                                        <td>${log.candidate_name}</td>
                                        <td>
                                            <span class="badge-confidence ${
                                                log.confidence > 0.8 ? 'high' : 
                                                log.confidence > 0.6 ? 'medium' : 'low'
                                            }">
                                                ${log.confidence.toFixed(2)}
                                            </span>
                                        </td>
                                    </tr>
                                `).join('') : `
                                    <tr>
                                        <td colspan="3" style="text-align: center;">No detection logs yet</td>
                                    </tr>
                                `}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h2>Vote Distribution</h2>
                    </div>
                    <div id="chart" class="chart-container">
                        <!-- Dynamic SVG chart based on vote data -->
                        ${generateBarChart(votes, totalVotes)}
                    </div>
                </div>
            `;
            
            return html;
        }
        
        function generateBarChart(votes, totalVotes) {
            // Calculate bar heights and positions
            const barWidth = 80;
            const spacing = 150;
            const maxHeight = 200;
            const baseline = 250;
            
            let bars = '';
            let xLabels = '';
            let values = '';
            
            votes.forEach((candidate, index) => {
                const x = 150 + index * spacing;
                const percentage = totalVotes > 0 ? (candidate.vote_count / totalVotes) : 0;
                const height = Math.max(percentage * maxHeight, 1); // Ensure at least 1px height
                const y = baseline - height;
                
                // Add bar
                bars += `<rect x="${x}" y="${y}" width="${barWidth}" height="${height}" fill="#4361ee" opacity="0.8"/>`;
                
                // Add x-axis label
                xLabels += `<text x="${x + barWidth/2}" y="${baseline + 20}" text-anchor="middle" font-size="12">${candidate.candidate_name}</text>`;
                
                // Add value label if there are votes
                if (candidate.vote_count > 0) {
                    values += `<text x="${x + barWidth/2}" y="${y - 5}" text-anchor="middle" font-size="12" fill="black" font-weight="bold">${candidate.vote_count}</text>`;
                }
            });
            
            return `
                <svg width="100%" height="100%" viewBox="0 0 800 300">
                    <!-- Background grid -->
                    <line x1="100" y1="50" x2="100" y2="250" stroke="#e9ecef" stroke-width="1"/>
                    <line x1="100" y1="250" x2="750" y2="250" stroke="#333" stroke-width="2"/>
                    <line x1="100" y1="50" x2="750" y2="50" stroke="#e9ecef" stroke-width="1"/>
                    <line x1="100" y1="150" x2="750" y2="150" stroke="#e9ecef" stroke-width="1"/>
                    
                    <!-- Y-axis labels -->
                    <text x="90" y="250" text-anchor="end" font-size="12">0</text>
                    <text x="90" y="150" text-anchor="end" font-size="12">50%</text>
                    <text x="90" y="50" text-anchor="end" font-size="12">100%</text>
                    
                    <!-- Bars -->
                    ${bars}
                    
                    <!-- X-axis labels -->
                    ${xLabels}
                    
                    <!-- Values -->
                    ${values}
                </svg>
            `;
        }
        
        // Initial load
        $(document).ready(function() {
            updateDashboard();
            
            // Set up refresh button
            $('#refresh-btn').click(function() {
                updateDashboard();
            });
            
            // Set up export button
            $('#export-btn').click(function() {
                window.location.href = '/api/export';
            });
            
            // Auto-refresh every 30 seconds
            setInterval(updateDashboard, 30000);
        });
    </script>
</body>
</html>
        ''')
    
    print("Dashboard template created in templates/dashboard.html")
    app.run(debug=True, host='0.0.0.0', port=5000)