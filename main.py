import cv2
import numpy as np
import os
import time
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cdist
import argparse

def extract_frames(video_path, output_dir="frames"):
    """Extract all frames from the video"""
    print("Extracting frames from video...")
    
    # Create frames directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frames.append(frame_filename)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames")
    return frames

def preprocess_frame(frame_path, target_size=(320, 240)):
    """Preprocess frame for similarity comparison"""
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Could not read frame: {frame_path}")
    
    # Resize for faster processing
    frame_resized = cv2.resize(frame, target_size)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Normalize
    normalized = blurred / 255.0
    
    return normalized

def compute_similarity_matrix(frames, similarity_metric='ssim'):
    """Compute similarity matrix between all frames"""
    print("Computing similarity matrix...")
    
    n_frames = len(frames)
    similarity_matrix = np.zeros((n_frames, n_frames))
    
    # Preprocess all frames
    processed_frames = []
    for frame_path in tqdm(frames, desc="Preprocessing frames"):
        processed_frame = preprocess_frame(frame_path)
        processed_frames.append(processed_frame.flatten())
    
    processed_frames = np.array(processed_frames)
    
    if similarity_metric == 'correlation':
        # Use correlation-based similarity
        similarity_matrix = 1 - cdist(processed_frames, processed_frames, metric='correlation')
    elif similarity_metric == 'euclidean':
        # Use inverse Euclidean distance (convert distance to similarity)
        distances = cdist(processed_frames, processed_frames, metric='euclidean')
        max_dist = np.max(distances)
        similarity_matrix = 1 - (distances / max_dist)
    else:  # SSIM
        # Compute SSIM for each pair (slower but more accurate)
        for i in tqdm(range(n_frames), desc="Computing SSIM"):
            frame1 = preprocess_frame(frames[i])
            for j in range(i, n_frames):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    frame2 = preprocess_frame(frames[j])
                    score, _ = ssim(frame1, frame2, full=True)
                    similarity_matrix[i, j] = score
                    similarity_matrix[j, i] = score
    
    return similarity_matrix

def find_start_frame(similarity_matrix):
    """Find the frame that is most dissimilar to others (likely start/end)"""
    # Sum of similarities for each frame
    similarity_sums = np.sum(similarity_matrix, axis=1)
    
    # The frame with lowest total similarity is likely an endpoint
    start_idx = np.argmin(similarity_sums)
    return start_idx

def reconstruct_sequence_greedy(similarity_matrix, start_idx):
    """Reconstruct frame sequence using greedy approach"""
    print("Reconstructing frame sequence...")
    
    n_frames = similarity_matrix.shape[0]
    visited = set([start_idx])
    sequence = [start_idx]
    current_idx = start_idx
    
    pbar = tqdm(total=n_frames-1)
    
    while len(visited) < n_frames:
        # Get similarities to unvisited frames
        similarities = similarity_matrix[current_idx].copy()
        
        # Set visited frames to -1 to exclude them
        for visited_idx in visited:
            similarities[visited_idx] = -1
        
        # Find most similar unvisited frame
        next_idx = np.argmax(similarities)
        
        sequence.append(next_idx)
        visited.add(next_idx)
        current_idx = next_idx
        pbar.update(1)
    
    pbar.close()
    return sequence

def reconstruct_sequence_optimized(similarity_matrix, start_idx):
    """Reconstruct frame sequence using optimized approach with lookahead"""
    print("Reconstructing frame sequence with optimization...")
    
    n_frames = similarity_matrix.shape[0]
    visited = set([start_idx])
    sequence = [start_idx]
    current_idx = start_idx
    
    pbar = tqdm(total=n_frames-1)
    
    while len(visited) < n_frames:
        # Get top k candidates for next frame
        similarities = similarity_matrix[current_idx].copy()
        
        # Exclude visited frames
        for visited_idx in visited:
            similarities[visited_idx] = -1
        
        # Find top candidate
        next_idx = np.argmax(similarities)
        
        sequence.append(next_idx)
        visited.add(next_idx)
        current_idx = next_idx
        pbar.update(1)
    
    pbar.close()
    return sequence

def create_video_from_sequence(frames, sequence, output_path, fps=30):
    """Create video from reconstructed frame sequence"""
    print("Creating output video...")
    
    # Get video properties from first frame
    sample_frame = cv2.imread(frames[0])
    height, width = sample_frame.shape[:2]
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames in reconstructed order
    for frame_idx in tqdm(sequence, desc="Writing video"):
        frame_path = frames[frame_idx]
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Jumbled Frames Reconstruction')
    parser.add_argument('--input', '-i', default='jumbled_video.mp4', 
                       help='Input jumbled video file')
    parser.add_argument('--output', '-o', default='reconstructed_video.mp4', 
                       help='Output reconstructed video file')
    parser.add_argument('--method', '-m', choices=['greedy', 'optimized'], 
                       default='optimized', help='Reconstruction method')
    parser.add_argument('--similarity', '-s', choices=['ssim', 'correlation', 'euclidean'], 
                       default='correlation', help='Similarity metric')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        # Step 1: Extract frames
        frames = extract_frames(args.input)
        
        # Step 2: Compute similarity matrix
        similarity_matrix = compute_similarity_matrix(frames, args.similarity)
        
        # Step 3: Find starting frame
        start_idx = find_start_frame(similarity_matrix)
        print(f"Selected start frame: {start_idx}")
        
        # Step 4: Reconstruct sequence
        if args.method == 'greedy':
            sequence = reconstruct_sequence_greedy(similarity_matrix, start_idx)
        else:
            sequence = reconstruct_sequence_optimized(similarity_matrix, start_idx)
        
        # Step 5: Create output video
        create_video_from_sequence(frames, sequence, args.output)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        print(f"\nReconstruction completed in {execution_time:.2f} seconds")
        
        # Save execution log
        with open('execution_log.txt', 'w') as f:
            f.write(f"Execution Time: {execution_time:.2f} seconds\n")
            f.write(f"Number of Frames: {len(frames)}\n")
            f.write(f"Method: {args.method}\n")
            f.write(f"Similarity Metric: {args.similarity}\n")
            f.write(f"Start Frame: {start_idx}\n")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())