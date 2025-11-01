import cv2
import numpy as np
from scipy import ndimage

def compute_optical_flow(prev_frame, next_frame):
    """Compute optical flow between two frames"""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 
                                       pyr_scale=0.5, levels=3, winsize=15, 
                                       iterations=3, poly_n=5, poly_sigma=1.2, 
                                       flags=0)
    return flow

def compute_flow_magnitude(flow):
    """Compute magnitude of optical flow"""
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return np.mean(magnitude)

def extract_features(frame):
    """Extract feature descriptors from frame"""
    # Use ORB feature detector
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(frame, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    """Match features between two frames"""
    if desc1 is None or desc2 is None:
        return 0
    
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    
    # Return number of good matches
    return len(matches)