# CV 1000-Puzzle Piece Project: Puzzle Reconstruction by Images

**Author:** Armando Abelho  
**Institution:** University of the Witwatersrand  

## Project Overview

This repository contains my end-to-end computer vision pipeline designed to reconstruct a 1000-piece puzzle from unlabelled images. The project tackles the complex task of segmenting puzzle pieces from visual images, identifying the four corners of each piece, and matching the contours and images of the pieces to reconstruct the original puzzle via adjacency graph construction. 

The documentation details the early successes of the project, particularly in feature extraction, alongside the struggles encountered during the final reconstruction.

Note that the full set of images have not been included as they are not mine but my 500 generated masks are all included which had very high accuracy.
---

## System Architecture & Pipeline

I designed the reconstruction pipeline across four distinct stages:

### 1. Segmentation
* **Model:** I utilized a U-Net architecture with an EfficientNet-B4 encoder.
* **Approach:** The model is trained to isolate individual puzzle pieces from the background by predicting binary masks.
* **Refinement:** During the inference stage, I incorporated Test Time Augmentation (TTA). By averaging predictions across multiple augmented versions (flips and rotations) of the input image, the final model produced consistent binary masks with smoother boundaries.

### 2. Corner Detection
* **Model:** I employed a specialized U-Net with an EfficientNet-B0 encoder, treating corner detection as a dense regression problem.
* **Approach:** The network predicts a 256 x 256 heatmap probability field where bright hotspots indicate corner locations.
* **Post-Processing:** * Centroid prediction to find the approximate corner location using heatmap peaks.
  * Harris Refinement to prioritize sharp geometric corners over rounded tab tips.
  * Contour snapping, a custom algorithm that snaps the predicted point to the nearest pixel on the actual mask contour to ensure geometric validity.

### 3. Graph Construction (Adjacency)
* **Feature Extraction:** I resampled each side contour to exactly 50 equidistant points, applied pose normalization to achieve translation and rotation invariance, and extracted the average pixel color converted to the CIELAB color space.
* **Matching:** Pair-wise adjacency was determined via a weighted scoring function combining the Euclidean distance between normalized shape vectors and the Delta-E color difference.
* **Locking Mechanism:** To prevent conflicting connections, I computed all $N^{2}$ potential match scores and implemented a strict one-to-one global locking mechanism.

### 4. Final Assembly
* **Transformation:** I treated the assembly as a sequence of transformations, solving for the optimal Affine Transformation matrix using Partial Affine Estimation. This restricted the degrees of freedom to rotation and translation, which is physically accurate for rigid puzzle pieces:

$$
\begin{bmatrix} x^{\prime} \\ y^{\prime} \end{bmatrix} = \begin{bmatrix} \cos \theta & -\sin \theta & t_x \\ \sin \theta & \cos \theta & t_y \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}
$$

* **Stitching:** I utilized a Breadth-First Search (BFS) combined with a max-blend compositing technique. This pixel-wise maximum operation merges new pieces onto the canvas without overwriting or introducing black artifacts from mask bounding boxes.

---

## Results & Metrics

* **Segmentation:** The EfficientNet-B4 model demonstrated exceptional performance, achieving a Training IoU of 0.9863 and a Validation IoU of 0.9787. The minimal divergence between these scores indicates that the model generalized well without significant overfitting.
* **Corner Detection:** The deep learning approach successfully learned the semantic appearance of puzzle corners, making it robust to shape variations and ignoring most tab tips that trap geometric algorithms in local minima.
* **Assembly:** While the mathematical framework for stitching was sound, the system proved volatile to upstream errors in feature extraction. The strict global locking mechanism locked onto incorrect neighbors early, resulting in a sparse graph. The final assembly script generated small, disjointed islands of 2-3 pieces, highlighting how small alignment errors accumulate over pieces without a global optimization framework.

---

## Future Work

To bridge the gap between local feature successes and global assembly, future iterations of this project will implement:
1. **Larger Training Set:** Feeding 5000+ labelled puzzle pieces into the model to improve accuracy across complex piece geometries.
2. **Global Relaxation:** Formulating the assembly as a global optimization problem (e.g., minimizing total energy in a Mass-Spring system) to distribute error across the entire grid and correct geometric drift.
3. **Soft Matching:** Replacing the binary locking logic with a probabilistic graph where edges have confidence weights, allowing the assembly algorithm to backtrack and break weak links if a better global configuration is found.

---

## Usage & Reproducibility

To ensure this project can be reliably reproduced, please follow the setup and execution instructions below. 

### Prerequisites

* Python 3.8 or higher
* Git

It is highly recommended to run this project within an isolated Python virtual environment (such as `venv` or `conda`) to prevent dependency conflicts.

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/1000-puzzle-reconstruction.git](https://github.com/yourusername/1000-puzzle-reconstruction.git)
   cd 1000-puzzle-reconstruction

2. **Virtual Environment:**        
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. **Dependencies:**  
pip install -r requirements.txt