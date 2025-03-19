import cv2
import os
from google.generativeai import GenerativeModel
import google.generativeai as genai

# Replace with your actual Gemini API key
my_key = ""  # You need to fill in your API key here
genai.configure(api_key=my_key)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Counter dictionary to track S.No occurrences
sno_counter = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Webcam Capture', frame)
    
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
            response = model.generate_content(
                [
                    "Extract the respective row text S.No and name where the seal is detected. Return only the S.No value as a number.",
                    {"mime_type": "image/jpeg", "data": image_data}
                ]
            )
            
            # Get the S.No value from the response
            sno_text = response.text.strip()
            
            # Try to extract a number from the response
            import re
            sno_match = re.search(r'\d+', sno_text)
            
            if sno_match:
                sno = sno_match.group()
                
                # Update counter for this S.No
                if sno in sno_counter:
                    sno_counter[sno] += 1
                else:
                    sno_counter[sno] = 1
                
                print(f"S.No {sno} detected (Count: {sno_counter[sno]})")
                print(f"Full Gemini Response: {response.text}")
            else:
                print(f"Could not extract S.No from response: {response.text}")
        
        except Exception as e:  # Handle any errors
            print(f"An error occurred: {e}")
    
    elif key == ord('r'):  # Reset counters when 'r' is pressed
        sno_counter = {}
        print("All counters reset")
    
    elif key == ord('p'):  # Print all counters when 'p' is pressed
        print("Current S.No counters:")
        for sno, count in sorted(sno_counter.items(), key=lambda x: int(x[0])):
            print(f"S.No {sno}: {count}")
    
    elif key == ord('q'):  # Exit when 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()