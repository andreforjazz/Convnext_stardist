# Integration of nuclear morphology and 3D imaging for profiling cellular neighborhoods

This project implements a deep learning-based approach for nucleus segmentation in Hematoxylin and Eosin (H&E) stained histological images. The workflow includes model training, whole slide image (WSI) segmentation, and feature extraction capabilities.

## Installation

### Prerequisites

1. Python 3.9
2. CUDA-capable GPU (recommended for training and inference)
3. OpenSlide library for handling whole slide images

### Setup

1. Create a new conda environment:
```bash
conda create -n CODA_HE_nucleus_segmentation python=3.9
conda activate CODA_HE_nucleus_segmentation
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- TensorFlow < 2.11
- StarDist (for nucleus detection)
- OpenSlide Python
- OpenCV
- NumPy
- Pandas
- Jupyter

## Project Structure

- `workflow/`: Contains the main implementation files
  - `1_train_test_new_model.ipynb`: Notebook for training and testing the nucleus segmentation model
  - `2_Segment_WSIs.ipynb`: Notebook for segmenting whole slide images
  - `training_functions.py`: Functions for model training and evaluation
  - `extract_features_functions.py`: Functions for extracting features from segmented nuclei
  - `functions.py`: General utility functions
- `Analysis/`: Contains analysis scripts and notebooks
- `individual_codes/`: Contains additional implementation files

## Usage

1. **Model Training**
   - Use `1_train_test_new_model.ipynb` to train and evaluate the nucleus segmentation model
   - The notebook includes data preprocessing, model architecture, training loop, and evaluation metrics

2. **Whole Slide Image Segmentation**
   - Use `2_Segment_WSIs.ipynb` to apply the trained model to segment nuclei in whole slide images
   - Supports various WSI formats through OpenSlide

3. **Feature Extraction**
   - Use the functions in `extract_features_functions.py` to compute various morphological and intensity features from the segmented nuclei

## Requirements

- GPU with CUDA support (recommended)
- Sufficient RAM for processing large whole slide images
- Storage space for training data and model checkpoints

## License

MIT License

Copyright (c) 2024 André Forjaz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact
André Forjaz, aperei13@jh.edu
