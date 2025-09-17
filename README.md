# SAGE (Soot Aggregate Geometry Extraction)
SAGE is a machine learning model for primary particle segmentation in HRTEM/TEM images of soot aggregates.  Utilizing the capabilites of Mask R-CNN, SAGE aims to eliminate the bottleneck of manually segmenting TEM images to obtain morphological information about soot aggregates, allowing for more streamlined quantitative analysis and generalization of soot structures. By training models with a combination of synethetically generated TEM images followed by fine-tuning with a smaller amount of manually segmented TEM images, SAGE is able to generalize to a larger spectrum of samples.


**__Coming Soon ...__**

**Note:**
This is a placeholder README. This will be updated with first release of the code soon.

## Features
- **SAGE Training Pipeline**: Train new models, or further fine-tune SAGE models using the pipeline found in `SAGE_train.ipynb`


- **SAGE Pretrained Models**: Analyze images and compare results using 5 different pretrained models. These models are downloaded in either notebook from 
    - SAGE<sub>0</sub>: Model trained using sythetically generated TEM images of soot
    - SAGE<sub>1</sub>: Fine-tuned version of SAGE<sub>0</sub>, implementing additional training on manual segmentations 
    - SAGE<sub>2</sub>: Further fine-tuned model, trained on a second set of manual segmentations for better generalization.
    - COCO<sub>1</sub>: Model trained using same images/segmentations as SAGE<sub>1</sub>, but initialized on COCO weights rather than SAGE<sub>0</sub> weights
    - COCO<sub>2</sub>: Model trained using same images/segmentations as SAGE<sub>1</sub>, but initialized with COCO<sub>1</sub> model rather than SAGE<sub>1</sub>
- **Model Visualizations**: Visualize and save predictions made by SAGE models on your dataset (`SAGE_ANALYZE.ipynb`)
- **Model Comparison**: Compare various model's performance metrics (`SAGE_ANALYZE.ipynb`)
- **Geometry Extraction**: Extract morphological information (dp, Rg, dF, etc) from TEM images (`SAGE_ANALYZE.ipynb`)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/comp-comb/SAGE.git
```
### 2. Install [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (Recommended) or [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### 3. Change to repository folder

```bash
cd SAGE
```

### 4. Create a virtual environment from requirements.yml
**Micromamba**
```bash
micromamba env create -f requirements.yml -y
```
**Conda**
```bash
conda env create -f requirements.yml
```

### 4. Activate virtual environment
**Micromamba**
```bash
micromamba activate SAGE-env
```
**Conda**
```bash
conda activate SAGE-env
```

### 5. Open Jupyter notebook to access files and notebooks:
Once your virtual environment is activated, launch Jupyter Notebook to view and run the notebooks included in this repository. 

**Launch Jupyter Notebook**
```bash
jupyter notebook
```
Copy and paste one of the links into your browser, showing the folder contents of the repository. Usage on remote machines may require some port forwarding. 

## Usage

### SAGE_train
This notebook demonstrates the process for training a model to detect primary particles in TEM images of soot.

### SAGE_ANALYZE
This notebook demonstrates how to perform analysis of images using trained models, as well as methods to compare performance between different models.


## Hardware information
The models are trained and test used Nvidia Tesla K-80 GPUs.

** Later TensorFlow versions (TF 2.11+) losses support for Tesla K-80 GPUs. For newer GPUs, a compatible version of CUDA, CUDANN, and TensorFlow is recommended, and should work with minimal adjustments to existing the codebase. **


## Back Matter
### Citing this repository

Please cite the corresponding paper for any work related to using SAGE models for primary particle segmentation or building upon this workflow:
> [Day, T.P., Mukut, K.M., Klacik, L., O'Donnel, R., Wasilewski, J., & Roy, S.P. (2025). SAGE: A machine learning model for primary particle segmentation in TEM images of soot aggregates. Proceedings of the Combustion Institute, 41, 105821.][sage]

### References

[Abdulla, W. (2017). Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow. Github Repository.][matterportmaskrcnn]

[Sipkens, T. A., Rogak, S. N. (2021). Technical note: Using k-means to identify soot aggregates in transmission electron microscopy images. Journal of Aerosol Science, 105699.][atems]

[Frei, Max and Kruis, Frank Einar. (2019). Image-Based Size Analysis of Agglomerated and Partially Sintered Particles via Convolutional Nerual Networks. Poweder Technology, 360, 324-336.][paddle]

[sage]: https://doi.org/10.1016/j.proci.2025.105821
[matterportmaskrcnn]: https://github.com/matterport/Mask_RCNN
[atems]: https://doi.org/10.1016/j.jaerosci.2020.105699
[paddle]: https://doi.org/10.1016/j.powtec.2019.10.020