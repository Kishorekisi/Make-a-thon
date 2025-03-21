import cv2
import numpy as np
import re
import sqlite3
import time
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Database setup functions
def setup_database(db_name="election.db"):
    """
    Set up the database connection and create necessary tables
    
    Args:
        db_name (str): Name of the SQLite database file
        
    Returns:
        tuple: (connection, cursor) - Database connection and cursor objects
    """
    # Connect to SQLite database (will be created if it doesn't exist)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Create votes table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS votes (
        candidate_id INTEGER PRIMARY KEY,
        candidate_name TEXT,
        vote_count INTEGER DEFAULT 0
    )
    """)
    
    # Create a log table to track all detections with timestamps
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detection_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        candidate_id INTEGER,
        candidate_name TEXT,
        row_text TEXT,
        confidence REAL,
        FOREIGN KEY (candidate_id) REFERENCES votes (candidate_id)
    )
    """)
    
    # Create settings table to store system configuration
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        setting_name TEXT PRIMARY KEY,
        setting_value TEXT
    )
    """)
    
    conn.commit()
    return conn, cursor

def initialize_candidates(cursor, conn, candidates):
    """
    Initialize the candidates in the database
    
    Args:
        cursor: Database cursor
        conn: Database connection
        candidates (dict): Dictionary of candidate IDs and names
    """
    # Reset vote counts to zero
    cursor.execute("UPDATE votes SET vote_count = 0")
    
    # Insert or update candidates in the database
    for candidate_id, name in candidates.items():
        cursor.execute("""
        INSERT OR REPLACE INTO votes (candidate_id, candidate_name, vote_count) 
        VALUES (?, ?, 0)
        """, (candidate_id, name))
    
    conn.commit()

def log_vote_to_database(cursor, conn, candidate_id, candidate_name, row_text, confidence):
    """
    Log a vote detection to the database
    
    Args:
        cursor: Database cursor
        conn: Database connection
        candidate_id (int): ID of the detected candidate
        candidate_name (str): Name of the detected candidate
        row_text (str): OCR text extracted from the row
        confidence (float): Detection confidence score
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Log the detection
    cursor.execute("""
    INSERT INTO detection_log (timestamp, candidate_id, candidate_name, row_text, confidence)
    VALUES (?, ?, ?, ?, ?)
    """, (timestamp, candidate_id, candidate_name, row_text, confidence))
    
    # Update vote count
    cursor.execute("""
    UPDATE votes SET vote_count = vote_count + 1 
    WHERE candidate_id = ?
    """, (candidate_id,))
    
    conn.commit()

def get_vote_counts(cursor):
    """
    Get current vote counts from database
    
    Args:
        cursor: Database cursor
        
    Returns:
        list: List of (candidate_name, vote_count) tuples
    """
    cursor.execute("SELECT candidate_name, vote_count FROM votes ORDER BY vote_count DESC")
    return cursor.fetchall()

def export_results_to_csv(cursor, filename="election_results.csv"):
    """
    Export election results to a CSV file
    
    Args:
        cursor: Database cursor
        filename (str): Name of the CSV file to create
        
    Returns:
        str: Path to the created CSV file
    """
    import csv
    
    # Get vote counts
    cursor.execute("SELECT candidate_id, candidate_name, vote_count FROM votes ORDER BY vote_count DESC")
    results = cursor.fetchall()
    
    # Get detection log
    cursor.execute("SELECT * FROM detection_log ORDER BY timestamp")
    log_entries = cursor.fetchall()
    
    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header and results
        writer.writerow(['ELECTION RESULTS'])
        writer.writerow(['Candidate ID', 'Candidate Name', 'Vote Count'])
        for row in results:
            writer.writerow(row)
            
        # Add empty row as separator
        writer.writerow([])
        
        # Write log entries
        writer.writerow(['DETECTION LOG'])
        writer.writerow(['ID', 'Timestamp', 'Candidate ID', 'Candidate Name', 'Row Text', 'Confidence'])
        for entry in log_entries:
            writer.writerow(entry)
    
    return os.path.abspath(filename)

# Main application code
def run_ballot_counter():
    # Connect to database
    conn, cursor = setup_database()
    
    # Define candidates
    candidates = {1: "Chandran.S", 2: "Gopal.G.K", 3: "Sumathi.R", 4: "Venkat.K"}
    
    # Initialize database with candidates
    initialize_candidates(cursor, conn, candidates)
    
    try:
        # Load YOLO model
        model = YOLO('trained_model.pt')
        print("YOLO model loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
        
    try:
        # Initialize PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)
        print("PaddleOCR initialized successfully!")
    except Exception as e:
        print(f"Error initializing PaddleOCR: {e}")
        return
    
    # Variables for processing
    last_detected_serial = None
    last_detection_time = 0
    cooldown_period = 5  # 5 seconds between processing ballots
    processing_mode = True  # Start with processing enabled
    debug_mode = False      # Debug visualization
    
    print("\n=== Ballot Counting System ===")
    print("\nInstructions:")
    print("- Press 'p' to toggle processing mode")
    print("- Press 's' to show current vote counts")
    print("- Press 'r' to reset all vote counts")
    print("- Press 'd' to toggle debug mode")
    print("- Press 'e' to export results to CSV")
    print("- Press 'q' to quit\n")
    
    # Show initial vote counts
    votes = get_vote_counts(cursor)
    print("\n📊 Initial Vote Counts:")
    print("=" * 30)
    for candidate, count in votes:
        print(f"{candidate:10}: {count:5} votes")
    print("=" * 30)
    
    detection_history = []  # Store recent detections for verification
    
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
                
            if processing_mode:
                # Perform object detection
                results = model.predict(source=frame, conf=0.5, imgsz=640, show=False)
                
                # Process the detection results
                for result in results:
                    boxes = result.boxes  # Boxes object for bbox outputs
                    
                    # Sort boxes by confidence (higher first)
                    sorted_indices = boxes.conf.argsort(descending=True)
                    
                    for idx in sorted_indices:
                        # Extract bounding box coordinates and confidence
                        x1, y1, x2, y2 = map(int, boxes.xyxy[idx].tolist())
                        confidence = float(boxes.conf[idx])
                        
                        # Draw a rectangle around the detected seal
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Seal ({confidence:.2f})", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Extract text with improved method
                        row_text, serial_num = extract_row_text_with_columns(frame, (x1, y1, x2, y2), ocr, debug_mode)
                        
                        # Draw a box around the row area
                        seal_y = (y1 + y2) // 2
                        row_height = 60
                        row_start_y = max(0, seal_y - row_height//2)
                        row_end_y = min(frame.shape[0], seal_y + row_height//2)
                        cv2.rectangle(display_frame, (0, row_start_y), (frame.shape[1], row_end_y), (255, 0, 0), 2)
                        
                        # Display extracted text
                        if row_text:
                            text_y = row_start_y - 5
                            cv2.putText(display_frame, f"Text: {row_text[:30]}...", (10, text_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # If a serial number is detected and cooldown period has passed
                        current_time = time.time()
                        if (serial_num is not None and 
                            (current_time - last_detection_time) > cooldown_period):
                            
                            # Add to history for verification
                            detection_history.append(serial_num)
                            if len(detection_history) > 3:
                                detection_history.pop(0)
                            
                            # Only register vote if same serial number detected consistently
                            # or if it's a new detection
                            if serial_num != last_detected_serial or (
                              len(detection_history) >= 2 and all(x == serial_num for x in detection_history)):
                                
                                # Log vote to database
                                log_vote_to_database(cursor, conn, serial_num, candidates[serial_num], row_text, confidence)
                                
                                # Log to console
                                timestamp = time.strftime("%H:%M:%S")
                                print(f"[{timestamp}] ✅ Vote recorded for {candidates[serial_num]} (ID: {serial_num})")
                                print(f"  - Row text: \"{row_text}\"")
                                print(f"  - Confidence: {confidence:.2f}")
                                
                                # Update last detection info
                                last_detected_serial = serial_num
                                last_detection_time = current_time
                                
                                # Update votes
                                votes = get_vote_counts(cursor)
                                print("\n📊 Current Vote Counts:")
                                print("=" * 30)
                                for candidate, count in votes:
                                    print(f"{candidate:10}: {count:5} votes")
                                print("=" * 30)
                        
                        # Display serial number if found
                        if serial_num is not None:
                            # Display with color based on confidence
                            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                            cv2.putText(display_frame, f"Candidate: {candidates[serial_num]} (ID: {serial_num})", 
                                      (x2 + 10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw vote counts on display frame
            display_frame = draw_vote_counts(display_frame, votes)
            
            # Display processing mode status
            status = f"Processing: {'ON' if processing_mode else 'OFF'}"
            
            # Add cooldown timer if in processing mode
            if processing_mode:
                time_since_last = time.time() - last_detection_time
                if time_since_last < cooldown_period:
                    remaining = cooldown_period - time_since_last
                    status += f" | Cooldown: {remaining:.1f}s"
                if debug_mode:
                    status += " | DEBUG ON"
            
            cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Ballot Counter', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                processing_mode = not processing_mode
                print(f"Processing mode {'enabled' if processing_mode else 'disabled'}")
            elif key == ord('s'):
                votes = get_vote_counts(cursor)
                print("\n📊 Current Vote Counts:")
                print("=" * 30)
                for candidate, count in votes:
                    print(f"{candidate:10}: {count:5} votes")
                print("=" * 30)
                
                # Determine the winner
                cursor.execute("SELECT candidate_name, MAX(vote_count) FROM votes")
                winner = cursor.fetchone()
                if winner[1] > 0:
                    print(f"\n🏆 Current Leader: {winner[0]} with {winner[1]} votes!")
            elif key == ord('r'):
                # Reset votes
                cursor.execute("UPDATE votes SET vote_count = 0")
                conn.commit()
                print("\n🔄 All vote counts have been reset to zero!")
                
                votes = get_vote_counts(cursor)
                print("\n📊 Current Vote Counts:")
                print("=" * 30)
                for candidate, count in votes:
                    print(f"{candidate:10}: {count:5} votes")
                print("=" * 30)
                
                # Reset detection variables
                last_detected_serial = None
                last_detection_time = 0
                detection_history = []
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
            elif key == ord('e'):
                # Export results to CSV
                csv_path = export_results_to_csv(cursor)
                print(f"\n📄 Results exported to: {csv_path}")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user...")
    finally:
        # Show final results
        cursor.execute("SELECT candidate_name, vote_count FROM votes ORDER BY vote_count DESC")
        results = cursor.fetchall()
        
        print("\n")
        print("=" * 40)
        print("       FINAL ELECTION RESULTS       ")
        print("=" * 40)
        
        # Find the winner
        winner_name = results[0][0]
        winner_votes = results[0][1]
        
        for candidate, votes in results:
            is_winner = candidate == winner_name
            if is_winner:
                print(f"🏆 {candidate:10}: {votes:5} votes  <-- WINNER")
            else:
                print(f"   {candidate:10}: {votes:5} votes")
        
        print("=" * 40)
        print(f"\nWinner: {winner_name} with {winner_votes} votes")
        print("\nElection completed successfully!")
        print("=" * 40)
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Close database connection
        conn.close()

# This function is copied from your original code
def extract_row_text_with_columns(frame, seal_bbox, ocr_reader, debug=False):
    """
    Extracts text from the row containing the detected seal, with improved column detection.
    
    Args:
        frame (numpy.ndarray): The input frame from the camera.
        seal_bbox (tuple): The bounding box of the detected seal (x1, y1, x2, y2).
        ocr_reader (PaddleOCR): The PaddleOCR reader object.
        debug (bool): Whether to show debugging information.
        
    Returns:
        tuple: (row_text, candidate_id) - The extracted text and detected candidate ID.
    """
    x1, y1, x2, y2 = seal_bbox
    seal_y = (y1 + y2) // 2  # Center Y-coordinate of seal
    
    # Calculate row bounds - expand vertically to capture entire row
    h, w = frame.shape[:2]
    row_height = 60  # Default row height - adjust based on your table
    
    # Calculate row start and end
    row_start_y = max(0, seal_y - row_height//2)
    row_end_y = min(h, seal_y + row_height//2)
    
    # Extract the entire row
    row_image = frame[row_start_y:row_end_y, :]
    
    if debug:
        cv2.imshow("Row Image", row_image)
    
    # First, focus on the area to the left of the seal to find serial number
    # Typically serial numbers are in the first column
    serial_column_width = x1  # Area from left edge to seal
    serial_area = frame[row_start_y:row_end_y, 0:serial_column_width]
    
    # Apply preprocessing to improve OCR accuracy for serial numbers
    gray_serial = cv2.cvtColor(serial_area, cv2.COLOR_BGR2GRAY)
    _, binary_serial = cv2.threshold(gray_serial, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    processed_serial = cv2.dilate(binary_serial, kernel, iterations=1)
    
    if debug:
        cv2.imshow("Processed Serial Area", processed_serial)
    
    # Save preprocessed image temporarily to file (PaddleOCR requires file path or PIL/numpy image)
    temp_serial_path = "temp_serial.jpg"
    cv2.imwrite(temp_serial_path, processed_serial)
    
    # Perform OCR specifically on the serial number area with higher precision settings
    serial_results = ocr_reader.ocr(temp_serial_path, cls=True)
    
    # Extract serial number text
    serial_text = ""
    if serial_results and serial_results[0]:  # Check if results and first page exist
        for line in serial_results[0]:
            # Each line is in format [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
            if line[1][1] > 0.4:  # Set minimum confidence threshold
                serial_text += line[1][0] + " "
    
    # Also perform OCR on the full row for context and candidate name
    # Save row image temporarily
    temp_row_path = "temp_row.jpg"
    cv2.imwrite(temp_row_path, row_image)
    
    results = ocr_reader.ocr(temp_row_path, cls=True)
    row_text = ""
    if results and results[0]:  # Check if results and first page exist
        for line in results[0]:
            row_text += line[1][0] + " "
    
    # Parse for candidate ID
    candidate_id = extract_serial_number(serial_text)
    
    # If not found in serial area, try with full row text
    if candidate_id is None:
        candidate_id = extract_serial_number(row_text)
    
    # If still not found, try with stricter number pattern matching
    if candidate_id is None:
        numbers = re.findall(r'\b[1-4]\b', row_text)  # Look for standalone digits 1-4
        if numbers:
            for num in numbers:
                try:
                    candidate_id = int(num)
                    if 1 <= candidate_id <= 4:  # Assuming candidate IDs are 1-4
                        break
                except ValueError:
                    continue
    
    return row_text.strip(), candidate_id

# This function is copied from your original code
def extract_serial_number(text):
    """
    Extract candidate serial number from the OCR text with improved pattern matching.
    
    Args:
        text (str): Text extracted from OCR
        
    Returns:
        int: Serial number found in the text, or None if not found
    """
    # Look for patterns like "1." "2." "No. 3" "Serial 4" etc.
    serial_patterns = [
        r'\b([1-4])[.\s]',  # Match 1. 2. 3. 4. with optional space
        r'[Nn][Oo][.\s]*([1-4])\b',  # Match No. 1, No.2, etc.
        r'[Ss][Ee][Rr][Ii][Aa][Ll][.\s]*([1-4])\b',  # Match Serial 1, Serial2
        r'\b([1-4])\b'  # Match standalone 1,2,3,4
    ]
    
    for pattern in serial_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                candidate_id = int(match)
                if 1 <= candidate_id <= 4:  # Assuming candidate IDs are 1-4
                    return candidate_id
            except ValueError:
                continue
    
    # If specific patterns fail, try any digit as a last resort
    numbers = re.findall(r'\d+', text)
    for num in numbers:
        try:
            candidate_id = int(num)
            if 1 <= candidate_id <= 4:  # Assuming candidate IDs are 1-4
                return candidate_id
        except ValueError:
            continue
            
    return None

# This function is copied from your original code
def draw_vote_counts(frame, votes):
    """
    Draw vote counts on the frame with improved styling.
    
    Args:
        frame (numpy.ndarray): The frame to draw on
        votes (list): List of (candidate_name, vote_count) tuples
        
    Returns:
        numpy.ndarray: Frame with vote counts
    """
    h, w = frame.shape[:2]
    
    # Create an overlay for vote counts with semi-transparent background
    overlay = np.zeros((h, 250, 3), dtype=np.uint8)
    cv2.rectangle(overlay, (0, 0), (250, h), (20, 20, 20), -1)
    
    y_offset = 40
    
    # Draw title with highlight
    cv2.rectangle(overlay, (0, 0), (250, 35), (0, 70, 120), -1)
    cv2.putText(overlay, "VOTE COUNTS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw each candidate's vote count with better formatting
    for i, (candidate, count) in enumerate(votes):
        # Alternate row colors for better readability
        if i % 2 == 0:
            cv2.rectangle(overlay, (0, y_offset-25), (250, y_offset+5), (30, 30, 40), -1)
        
        # Draw candidate name
        cv2.putText(overlay, f"{candidate}:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw vote count with bold text
        cv2.putText(overlay, f"{count}", (180, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        
        y_offset += 35
    
    # Combine with the main image
    try:
        combined = np.hstack((frame, overlay))
        return combined
    except:
        # If shapes don't match, resize overlay
        overlay_resized = cv2.resize(overlay, (250, h))
        combined = np.hstack((frame, overlay_resized))
        return combined

# Main entry point
if __name__ == "__main__":
    run_ballot_counter()