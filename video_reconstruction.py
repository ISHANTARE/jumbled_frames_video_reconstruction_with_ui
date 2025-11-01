import cv2
import numpy as np
import os
import time
import argparse
import shutil


class VideoReconstructor:
    def __init__(self):
        self.frames = []
        self.sequence = []

    #defining a function to extract the frames from the video
    def extract_frames(self, video_path, output_dir="frames"):
        print("Extracting frames from video...")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frames.append(frame_filename)
            frame_count += 1

        cap.release()
        print(f"✓ Extracted {frame_count} frames")
        return frames

    # Preprocessing the frams to find similarity
    def preprocess_frame(self, frame_path, target_size=(120, 90)):
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                return None
            frame_resized = cv2.resize(frame, target_size)
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            normalized = blurred / 255.0

            return normalized.flatten()
        except Exception as e:
            print(f"Error processing frame {frame_path}: {e}")
            return None

    # calculating th frame difference
    def compute_frame_differences(self, frames):
        print("Computing frame differences...")
        n_frames = len(frames)
        differences = np.zeros((n_frames, n_frames))
        print("Preprocessing frames...")
        processed_frames = []

        for i in range(n_frames):
            processed = self.preprocess_frame(frames[i])
            if processed is not None:
                processed_frames.append(processed)
            else:
                processed_frames.append(np.zeros(120 * 90))

        print("Calculating frame similarities...")
        for i in range(n_frames):
            if i % 1 == 0:
                print(f"  Processed {i}/{n_frames} frames")
            for j in range(n_frames):
                if i != j:
                    diff = np.sqrt(np.sum((processed_frames[i] - processed_frames[j]) ** 2))
                    differences[i, j] = diff

        return differences

    # finding the frames with the most difference to get start and end of video
    def find_start_frame(self, differences):
        difference_sums = np.sum(differences, axis=1)
        start_idx = np.argmax(difference_sums)
        return start_idx

    # correctly arranging frames using nearest neighbor method
    def reconstruct_sequence(self, differences, start_idx):
        print("Reconstructing frame sequence...")

        n_frames = differences.shape[0]
        visited = set([start_idx])
        sequence = [start_idx]
        current_idx = start_idx

        for step in range(n_frames - 1):
            if step % 50 == 0:  # Show progress
                print(f"  Reconstructed {step}/{n_frames - 1} frames")

            current_differences = differences[current_idx].copy()
            for visited_idx in visited:
                current_differences[visited_idx] = float('inf')

            next_idx = np.argmin(current_differences)

            sequence.append(next_idx)
            visited.add(next_idx)
            current_idx = next_idx

        return sequence

    # making the video from the corrected sequence
    def create_video_from_sequence(self, frames, sequence, output_path, fps=30):
        print("Creating output video...")
        sample_frame = cv2.imread(frames[0])
        height, width = sample_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i, frame_idx in enumerate(sequence):
            if i % 50 == 0:  # Show progress
                print(f"  Written {i}/{len(sequence)} frames")
            frame_path = frames[frame_idx]
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)

        out.release()
        print(f"✓ Video saved: {output_path}")

    # ADD THIS METHOD FOR THE WEB INTERFACE
    def reconstruct_video(self, input_path, output_path="reconstructed_video.mp4"):
        """Main reconstruction function for web interface"""
        start_time = time.time()

        try:
            if not os.path.exists(input_path):
                return False, "Input file not found!"

            # Step 1: Extract frames
            frames = self.extract_frames(input_path)

            if len(frames) == 0:
                return False, "No frames extracted from video"

            print(f"Processing {len(frames)} frames...")

            # Step 2: Compute frame differences
            differences = self.compute_frame_differences(frames)

            # Step 3: Find starting frame
            start_idx = self.find_start_frame(differences)
            print(f"Selected start frame: {start_idx}")

            # Step 4: Reconstruct sequence
            sequence = self.reconstruct_sequence(differences, start_idx)

            # Step 5: Create output video
            self.create_video_from_sequence(frames, sequence, output_path)

            execution_time = time.time() - start_time

            # Clean up temporary frames directory
            if os.path.exists("frames"):
                shutil.rmtree("frames")

            return True, {
                "execution_time": execution_time,
                "frame_count": len(frames),
                "output_path": output_path
            }

        except Exception as e:
            # Clean up on error
            if os.path.exists("frames"):
                shutil.rmtree("frames")
            return False, f"Error during reconstruction: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description='Jumbled Frames Reconstruction')
    parser.add_argument('--input', '-i', default='jumbled_video.mp4',
                        help='Input jumbled video file')
    parser.add_argument('--output', '-o', default='reconstructed_video.mp4',
                        help='Output reconstructed video file')

    args = parser.parse_args()

    print("=== Jumbled Frames Reconstruction ===")
    print("Minimal version - Only OpenCV and NumPy")
    start_time = time.time()

    try:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found!")
            print("Please download the jumbled video and save it as 'jumbled_video.mp4'")
            return 1

        # Create reconstructor instance and use its methods
        reconstructor = VideoReconstructor()
        frames = reconstructor.extract_frames(args.input)

        if len(frames) == 0:
            print("Error: No frames extracted from video")
            return 1

        print(f"Processing {len(frames)} frames...")

        differences = reconstructor.compute_frame_differences(frames)

        start_idx = reconstructor.find_start_frame(differences)
        print(f"Selected start frame: {start_idx}")

        sequence = reconstructor.reconstruct_sequence(differences, start_idx)

        reconstructor.create_video_from_sequence(frames, sequence, args.output)

        execution_time = time.time() - start_time

        print(f"\nReconstruction Complete!")
        print(f"Execution Time: {execution_time:.2f} seconds ({execution_time / 60:.2f} minutes)")
        print(f"Output: {args.output}")
        print(f"Frames Processed: {len(frames)}")

        # Saving execution log
        with open('execution_log.txt', 'w') as f:
            f.write("=== Jumbled Frames Reconstruction Log ===\n")
            f.write(f"Execution Time: {execution_time:.2f} seconds\n")
            f.write(f"Number of Frames: {len(frames)}\n")
            f.write(f"Start Frame: {start_idx}\n")
            f.write(f"Algorithm: Frame Difference + Nearest Neighbor\n")
            f.write(f"Input: {args.input}\n")
            f.write(f"Output: {args.output}\n")

        print("Execution log saved to: execution_log.txt")

    except Exception as e:
        print(f"Error during reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())