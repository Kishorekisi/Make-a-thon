import cv2
import os
import sqlite3
import datetime
import requests
from google.generativeai import GenerativeModel
import google.generativeai as genai
import socketio

# Initialize Socket.IO client
sio = socketio.Client()
try:
    sio.connect('http://localhost:5000')
    socket_connected = True
    print("Connected to WebSocket server")
except Exception as e:
    socket_connected = False
    print(f"Could not connect to WebSocket server: {e}")

# Replace with your actual Gemini API key
my_key = ""  # You need to fill in your API key here
genai.configure(api_key=my_key)

# Set up SQLite database
def setup_database():
    conn = sqlite3.connect('seal_detection.db')
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sno TEXT NOT NULL,
        name TEXT,
        detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        count INTEGER DEFAULT 1
    )
    ''')
    
    conn.commit()
    return conn, cursor

# Initialize database
conn, cursor = setup_database()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load counters from database
def load_counters():
    sno_counter = {}
    cursor.execute("SELECT sno, count FROM detections")
    rows = cursor.fetchall()
    for row in rows:
        sno_counter[row[0]] = row[1]
    return sno_counter

# Function to send updates to the web dashboard
def notify_dashboard_update():
    # Get current vote data
    try:
        # Notify via WebSocket if connected
        if socket_connected:
            sio.emit('update_votes', {'timestamp': datetime.datetime.now().isoformat()})
        
        # Also notify via direct API call as fallback
        requests.get('http://localhost:5000/api/votes')
    except Exception as e:
        print(f"Error sending update to dashboard: {e}")

# Initial counter load
sno_counter = load_counters()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Election Vote Scanner', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Capture frame when 'c' is pressed
        captured_frame = frame.copy()
        cv2.imwrite("captured_image.jpg", captured_frame)  # Save the frame as JPG
        
        # Send to Gemini API
        try:
            # Initialize the model
            model = GenerativeModel('gemini-2.0-flash')
            
            # Read image
            with open("captured_image.jpg", "rb") as image_file:
                image_data = image_file.read()
            
            # Generate content
            context = "You are a intelligent object detection model. You are trained to detect the seal in the image."
            prompt = "Extract only the respective row text S.No and Name where the seal is detected."
            output_format = "The output format should be 'Name : <the name of the person, S.No : <the serial number of the person>'"
            response = model.generate_content(
                [
                    context+prompt+output_format,
                    {"mime_type": "image/jpeg", "data": image_data}
                ]
            )
            
            # Get the response text
            full_response = response.text.strip()
            print(f"Full Gemini Response: {full_response}")
            
            # Try to extract S.No and name from the response
            import re
            sno_match = re.search(r'S[\.\s]?No[\s:]+(\d+)', full_response, re.IGNORECASE)
            name_match = re.search(r'name:?\s*([^\n]+)', full_response, re.IGNORECASE)
            print(f"S.No Match: {sno_match}, Name Match: {name_match}")
            
            if sno_match:
                sno = sno_match.group(1)
                name = name_match.group(1).strip() if name_match else "Unknown"
                
                # Update counter for this S.No
                if sno in sno_counter:
                    sno_counter[sno] += 1
                    # Update database
                    cursor.execute("UPDATE detections SET count = ?, detection_time = CURRENT_TIMESTAMP WHERE sno = ?", 
                                  (sno_counter[sno], sno))
                else:
                    sno_counter[sno] = 1
                    # Insert into database
                    cursor.execute("INSERT INTO detections (sno, name, count) VALUES (?, ?, ?)", 
                                  (sno, name, 1))
                
                conn.commit()
                print(f"S.No {sno} ({name}) detected (Count: {sno_counter[sno]})")
                
                # Notify dashboard of the update
                notify_dashboard_update()
            else:
                print(f"Could not extract S.No from response")
        
        except Exception as e:  # Handle any errors
            print(f"An error occurred: {e}")
    
    elif key == ord('r'):  # Reset counters when 'r' is pressed
        sno_counter = {}
        cursor.execute("DELETE FROM detections")
        conn.commit()
        print("All counters reset and database cleared")
        notify_dashboard_update()
    
    elif key == ord('p'):  # Print all counters when 'p' is pressed
        print("Current S.No counters:")
        cursor.execute("SELECT sno, name, count, detection_time FROM detections ORDER BY CAST(sno AS INTEGER)")
        rows = cursor.fetchall()
        for row in rows:
            print(f"S.No {row[0]} ({row[1]}): {row[2]} detections, last detected at {row[3]}")
    
    elif key == ord('q'):  # Exit when 'q' is pressed
        break

# Close connections before exiting
if socket_connected:
    sio.disconnect()
conn.close()
cap.release()
cv2.destroyAllWindows()