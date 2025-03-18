
#this code doesn't work for our requirements
#it is just a reference to understand how template matching works
#It uses fixed matching template and image to find the template in the image

import cv2
import numpy as np

# Load the images
image1 = cv2.imread('polled_pic.png', cv2.IMREAD_GRAYSCALE)  # Larger image
template = cv2.imread('seal_symbol1.png', cv2.IMREAD_GRAYSCALE)  # Smaller image (template)

# Get the dimensions of the template
h, w = template.shape
print(h,w)  
# Perform template matching
result = cv2.matchTemplate(image1, template, cv2.TM_CCOEFF_NORMED)
print(result)
# Set a threshold to determine if the template is found
threshold = 0.8 # Adjust based on your use case
loc = np.where(result >= threshold)

# Check if the template is found
if len(loc[0]) > 0:
    print("Image 2 is inside Image 1")
    # Draw a rectangle around the matched region
    for pt in zip(*loc[::-1]):  # Switch columns and rows
        cv2.rectangle(image1, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    # Display the result
    cv2.imshow('Match', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image 2 is NOT inside Image 1")