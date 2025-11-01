import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def evaluate_reconstruction(original_video, reconstructed_video):
    """Evaluate similarity between original and reconstructed video"""
    
    # Read both videos
    cap_orig = cv2.VideoCapture(original_video)
    cap_recon = cv2.VideoCapture(reconstructed_video)
    
    similarities = []
    frame_count = 0
    
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_recon, frame_recon = cap_recon.read()
        
        if not ret_orig or not ret_recon:
            break
        
        # Resize frames to same size for comparison
        frame_orig = cv2.resize(frame_orig, (320, 240))
        frame_recon = cv2.resize(frame_recon, (320, 240))
        
        # Convert to grayscale
        gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        gray_recon = cv2.cvtColor(frame_recon, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM
        score, _ = ssim(gray_orig, gray_recon, full=True)
        similarities.append(score)
        
        frame_count += 1
    
    cap_orig.release()
    cap_recon.release()
    
    avg_similarity = np.mean(similarities) * 100
    print(f"Average frame similarity: {avg_similarity:.2f}%")
    print(f"Frames compared: {frame_count}")
    
    return avg_similarity, similarities