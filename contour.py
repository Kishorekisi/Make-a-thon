import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('polled_pic.png', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)


# Variables to track the rightmost vertical line
rightmost_x = -1
rightmost_line = None

#variables to track the second rightmost vertical line
second_rightmost_x = -1
second_rightmost_line = None

# Define a minimum distance threshold (in pixels)
MIN_DISTANCE_THRESHOLD = 5

# Detect lines using Probabilistic Hough Transform
lines = cv2.HoughLinesP(
    edges,              # Binary edge-detected image
    rho=1,             # Distance resolution in pixels
    theta=np.pi/180,   # Angle resolution in radians
    threshold=50,      # Minimum number of votes to detect a line
    minLineLength=50,  # Minimum length of a line
    maxLineGap=10      # Maximum gap between line segments to treat them as a single line
)
# copy the original image 
output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

#draw the detected lines 
if lines is not None:
    for line in lines:
        print(line)
        x1,y1,x2,y2 = line[0]

        #calculate the length 
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        #calculate the angle 
        angle = np.arctan2(y2-y1, x2-x1)*180/np.pi

        if (angle < 10 or abs(angle -90) < 10) and length > 500:

            if x1 > rightmost_x:
                # Shift the current rightmost to second rightmost if it's far enough
                if rightmost_line is not None and abs(x1 - rightmost_x) >= MIN_DISTANCE_THRESHOLD:
                    second_rightmost_x = rightmost_x
                    second_rightmost_line = rightmost_line

                # Update the new rightmost
                rightmost_x = x1
                rightmost_line = (x1, y1, x2, y2)
            elif x1 > second_rightmost_x and abs(x1 - rightmost_x) >= MIN_DISTANCE_THRESHOLD:
                # Update the second rightmost if it's far enough from the rightmost
                second_rightmost_x = x1
                second_rightmost_line = (x1, y1, x2, y2)
            cv2.line(output, (x1,y1), (x2,y2), (0,255,0), 2)

print(f"Second right most line :",{second_rightmost_line})
print(f"Right most line :",{rightmost_line})    

if rightmost_line is not None:
    x1, y1, x2, y2 = rightmost_line
    cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue color
if second_rightmost_line is not None:
    x1,y1,x2,y2 = second_rightmost_line
    cv2.line(output, (x1,y1), (x2,y2), (255,0,0), 2)  # cyan color


cv2.imshow('Lines', output)
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the edges to a file
# Optionally, save the output to a file
cv2.imwrite('contour_output.jpg', output)