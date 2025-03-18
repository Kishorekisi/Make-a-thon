import cv2 
import numpy as np
def brute_force_matcher(image1, image2):


    detector = cv2.SIFT_create()

    keypoints1 = detector.detect(image1,None)
    keypoints2 = detector.detect(image2,None)

    # Compute descriptors for the keypoints
    keypoints1, descriptors1 = detector.compute(image1, keypoints1)
    keypoints2, descriptors2 = detector.compute(image2, keypoints2)

    print(descriptors1.dtype)
    print(descriptors2.dtype)
    # Create a Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)


    # Draw matches
    output = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    print(good_matches)  
    
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # Display the result
    cv2.imshow('Matches', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Flann_based_matcher(image1, image2):
    detector = cv2.SIFT_create()

    keypoints1 = detector.detect(image1,None)
    keypoints2 = detector.detect(image2,None)

    # Compute descriptors for the keypoints
    keypoints1, descriptors1 = detector.compute(image1, keypoints1)
    keypoints2, descriptors2 = detector.compute(image2, keypoints2)

    # Create a FLANN Matcher
    flann = cv2.FlannBasedMatcher()

    # Match descriptors
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    print(good_matches)
    # Draw matches
    output = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    cv2.imshow('Matches', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image1 = cv2.imread('polled_pic.png')
    image2 = cv2.imread('seal_symbol2.png')
    brute_force_matcher(image1, image2)
    Flann_based_matcher(image1, image2)