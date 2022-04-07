# COVID-MTL
Multitask learning for automated COVID 19 diagnosis and severity assessment using CT scans

## Citation
Guoqing Bao<sup>\*</sup>, Huai Chen, Tongliang Liu, Guanzhong Gong, Yong Yin, Lisheng Wang, Xiuying Wang<sup>\*</sup>, "COVID-MTL: Multitask Learning with Shift3D and Random-weighted Loss for COVID-19 Diagnosis and Severity Assessment", Pattern Recognition, 2021, 108499, ISSN 0031-3203, https://doi.org/10.1016/j.patcog.2021.108499.

<sup>*</sup> Corresponding authors.

## Prerequisites
The following python libraries are required:

pytorch, cv2, radiomics, pydicom, sklearn, skimage, matplotlib, pandas, scipy, tensorflow and keras

## The dataset
We provide the segmented CT lung volumes (about **7GB**, *.dtm and *.dcm; losslessly compressed) and corresponding extracted CT lung features (9 MB; radiofeatures.csv) of the COVID-19 cohort (1329 cases) under the **"data"** folder. The preprocessed dataset were obtained from the raw data (http://ictcf.biocuckoo.cn/HUST-19.php; about **300GB - 400GB**) using proposed unsupervised segmentation method. All of the preprocessed data has corresponding clinical features and outcomes (Patients.xlsx).

## The framework
The framework is composed of unsupervised lung segmentation, data preprocessing pipeline, feature extractioin, multitask learning model (3D CNN with Shift3D and Random-weighted loss), see more details in our paper: [https://www.sciencedirect.com/science/article/pii/S0031320321006750](https://www.sciencedirect.com/science/article/pii/S0031320321006750)

## The Code

### 1. Unsupervised lung segmentation from COVID-19 CT scans (LungSegmentation.py)
A classical method that widely used in Kaggle competition for lung segmentation has been improved. Given the Kaggle method was based on thresholding, there are some under-segmentation especially in COVID-19 infected areas, the contours of the initial segmentation results were refined in this algorithm.

### 2. Preprocessing (CropSave.py)
The segmented results were aligned under same bounding box and saved as compressed file (see folder for our compression and de-compression algorithm)

### 3. Feature Extraction (FeatureExtractor.py, HOGFeatureExtractor.py)
The features were extracted from segmented lung volumes. There are two types of feature extractors, i.e., radiomics (FeatureExtractor.py), HOG (HOGFeatureExtractor.py), supports both DICOM and NII formats. Note: the extracted features were used for enhancement of the multitask learning model (which composed of two network branches, i.e., 3D CNN branch, and feed-forward branch)

### 4. COVID-MTL Model (COVID-MTL.py)
The proposed model in the paper, which composed of two network branches, i.e., 3D CNN branch, and feed-forward branch. The Shift3D module was proposed to speed up the convergence of 3D CNN branch and improve the accuracy performance, and the Random-weighted loss was proposed to enhance the joint learning performance (learning three tasks, including diagnosis and severity assessment, simutaneously). 

### 5. Shift3D (Shift3D.py)
The real-time 3D augmentation method that used in COVID-MTL (ShiftNet3D is the SqueezeNet3D with Shift3D).

### 6. Competing Models (COVID-SingleTask.py, COVID-ML.py)
Single task models (COVID-SingleTask.py), 3D CNN models, i.e., ResNet3D, SqueezeNet3D and DeCovNet (a method published previously for COVID-19 detection), and two machine learning models (COVID-ML.py) were used as comparisons.

### 7. Helper Functions (utils.py)
Performance measurements and results visualization.


