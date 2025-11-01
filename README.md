# Jumbled Frames Reconstruction

## Overview
This project reconstructs jumbled video frames into their correct sequential order using frame similarity analysis and optimization algorithms.

## Algorithm Explanation

### Approach
The solution uses a multi-step process:

1. **Frame Extraction**: Extract all frames from the jumbled video
2. **Similarity Analysis**: Compute pairwise similarity between all frames using:
   - Structural Similarity Index (SSIM)
   - Correlation coefficients
   - Euclidean distance metrics
3. **Sequence Reconstruction**: Use greedy optimization to find the most probable frame sequence
4. **Video Generation**: Create output video from reconstructed sequence

### Key Techniques
- **Similarity Metrics**: Multiple metrics for robust comparison
- **Greedy Optimization**: Efficient sequence reconstruction
- **Frame Preprocessing**: Normalization and resizing for performance

### Design Considerations
- **Accuracy**: Uses SSIM for high-quality similarity measurement
- **Performance**: Optimized with matrix operations and efficient algorithms
- **Robustness**: Multiple similarity metrics for different video types

## Installation

```bash
pip install -r requirements.txt