# Major_Project

# INTRODUCTION <br>
Diabetic Retinopathy (DR) is a progressive microvascular disorder that affects the
retina, leading to visual impairment and blindness among working-age adults in the
Western population. Itis the most common microvascular complication associated with
diabetes mellitus. Prolonged exposure to high blood glucose levels and metabolic
abnormalities contributes to the development and progression of DR. To prevent the
vision-threatening damage caused by DR, strict glycaemic control, early detection, and
appropriate management are essential. This study examines thepathogenesis, diagnosis,
and management of DR, emphasizing the collaborative efforts of an interprofessional
team in evaluating and treating patients with this condition. By understanding the
underlying mechanisms, implementing timely interventions, and coordinating
comprehensive care,healthcare professionals can effectively address the challenges
posed by DR and improve patient outcomes.
<be>
#  Objectives
1. Implementing CLAHE for Image Enhancement
2. Implementation of DCGAN on Enhanced Images
3. Using GAN-FID to Compare GAN Images
4. Application of DENSENET-201 For Classification on CLAHE Images
5. Comparison between Proposed Model (DENSENET-201) and Existing Model (ALEXNET)

#  Methodology <br>
- Image Enhancement
- Data Augmentation
- Classification using Deep Learning Models
-   - DenseNet-201
    - AlexNet
      
# PROPOSED SYSTEM <br>
![Screenshot (206)](https://github.com/Mini-project4ppl/Major_Project/assets/132262814/dd8151fb-3046-4659-893e-6d767476b320)
<br>
The system architecture provides a detailed description of the steps involved in identifying Diabetic Retinopathy (DR) using the mentioned approach. <br>
## Retinal Dataset:
Start by obtaining a retinal image dataset, such as those found on platforms like
Kaggle. This dataset includes retinal images along with labels indicating the
presence and severity of diabetic retinopathy. Additionally, real-time data was
gathered from LVPEI Hospital and AIMER.
## Data Preprocessing using CLAHE Algorithm:
Data preprocessing is a critical step to prepare the retinal images for analysis.
Utilize the CLAHE (Contrast Limited Adaptive Histogram Equalization)
algorithm to enhance contrast and image details. This process standardizes imag
quality and improves the model's ability to detect features associated with diabetic
retinopathy.[2]
## Data Augmentation using GAN:
Enhance the diversity and quality of the training dataset through data
augmentation using a GenerativeAdversarial Network (GAN). The use of GANs
involves generating synthetic data samples that closely resemble the original
dataset. This diversification process aims to create additional data points to
improve the model's ability to generalize.
The augmentation process specifically focuses on improving the Fréchet
Inception Distance (FID), a metric that quantifies the similarity between real
and generated data by considering the distribution of feature vectors extracted
from InceptionNet. Reducing the FID indicates a successful data augmentation
process, as it means that the generated data closely matchesthe real data in terms
of statistical characteristics.
## Feature Extraction with DENSENET-201 and ALEXNET:
o Utilize Pre-trained Models: Employ pre-trained DENSENET-201 and
ALEXNET models to extract hierarchical features from retinal images.
o Hierarchical Feature Representation: Leverage the hierarchical feature
representations learned by DENSENET-201 and ALEXNET to capture
complex patterns in the retinal images related to diabetic retinopathy.[14]
## Classification:
o Model Training: Train a classifier, such as a fully connected neural network,
on top of the features extracted by DENSENET-201 and ALEXNET.
o Output Prediction: Use the trained classifier to predict the severity levels of
diabetic retinopathy for each input image, classifying them into categories
such as normal, mild, moderate, severe, or proliferative diabetic retinopathy
based on the predicted output probabilities The success of this architectural approach relies on the quality and size of the retinal
dataset, theeffectiveness of data preprocessing steps, the diversity introduced through
data augmentation. Regular monitoring and evaluation of the model's performance are
essential to ensure it can effectively identify diabetic retinopathy in retinal images <br>

# IMPLEMENTATION OF MODULES
## Dataset Description
1. Kaggle Datasets:
The project leverages two Kaggle datasets, each containing over 35,000 images
sourced from the Kaggle platform. These datasets collectively provide a diverse
range of diabetic retinopathy images for model training and evaluation. The images
are meticulously categorized into five distinct classes:
- No DR (Diabetic Retinopathy)
- Mild DR
- Moderate DR
- Severe DR
- PDR (Proliferative Diabetic Retinopathy)
2. Real-Time Datasets:
Two real-time datasets were obtained from LV Prasad Eye Hospital and
AIMER (Artificial Intelligence in Medical Epidemiological Research) to
supplement the Kaggle datasets. These datasets, combined with the Kaggle data,
offer a comprehensive collection of diabetic retinopathy images for model training
and validation.
- LV Prasad Eye Hospital Dataset:
 - This dataset comprises approximately 400 images, classified into four classes:
 - Mild DR
 - Moderate DR
 - Severe DR
 - PDR (Proliferative Diabetic Retinopathy)
- AIMER Dataset:
 With a collection of around 2500 images, the AIMER dataset enriches the model
training process with additional real-world samples. The images in this dataset are
categorized into four classes: Mild, Moderate, Severe, and PDR.
**Combining Datasets:**
The combination of Kaggle datasets, each with over 35,000 images, along with
the real-time datasets from LV Prasad Eye Hospital and AIMER, forms a robust
dataset for training and evaluating the diabetic retinopathy classification model.
This diverse dataset facilitates the development of a model capable of accurately
categorizing the severity of diabetic retinopathy across various scenarios.
## Description of Technology used
1. Programming Language:
− Python: A high-level, versatile programming language widely used for various
purposes including web development, data analysis, machine learning, and
more.
2. Development Environments/Tools:
- Jupyter: An open-source web application that allows you to create and share
documents containing live code, equations, visualizations, and narrative text.
- Colab: Google Colaboratory, a cloud-based Jupyter notebook environment that
enables free access to GPU and TPU resources for executing code.
- Streamlit: A Python library that allows you to create interactive web apps for
machine learning and data science projects with simple Python scripts.
- cv2 (OpenCV): Open-Source Computer Vision Library, a library of
programming functions mainly aimed at real-time computer vision.
- os: Python module providing a portable way of using operating systemdependent functionality.
- opendatasets: A Python library for downloading and working with datasets from
online sources.
- shutil: Python module providing a higher-level interface for file operations.
3. Data Manipulation and Analysis:
- Pandas: A powerful Python library for data manipulation and analysis,
particularly useful for working with structured data.
- numpy: A fundamental package for scientific computing with Python, providing
support for multi-dimensional arrays and matrices.
4. Image Processing and Computer Vision:
- cv2 (OpenCV): Used for image processing and computer vision tasks.
- clahe: Contrast Limited Adaptive Histogram Equalization, a method for
improving the contrast of an image.

5. Deep Learning:
- torch: PyTorch, an open-source deep learning platform that provides a seamless
path from research prototyping to production deployment.
- torchvision: A package consisting of popular datasets, model architectures, and
common image transformations for computer vision tasks in PyTorch.
- dcgan: Deep Convolutional Generative Adversarial Networks, a type of
generative model for generating realistic images.

6. Machine Learning Frameworks:
- TensorFlow: An open-source machine learning framework developed by
Google for building and training machine learning models.
- Keras: A high-level neural networks API, written in Python and capable of
running on top of TensorFlow, Theano, or Microsoft Cognitive Toolkit (CNTK).

7. Optimization Algorithms:
- Adam Optimization: An optimization algorithm used for training deep learning
models, particularly effective for models with large datasets and parameters.

8. Pre-trained Models:
- Densenet-201: A convolutional neural network architecture that has achieved
state-of-the-art performance on various image classification tasks.
- AlexNet: A deep convolutional neural network architecture that was one of the
first models to demonstrate significant improvement over traditional methods
on the ImageNet dataset.
9. Data Visualization:
- matplotlib: A comprehensive library for creating static, animated, and
interactive visualizations in Python.<br>
## Module Description <br>
- **CLAHE Algorithm**
CLAHE, which stands for Contrast Limited Adaptive Histogram Equalization,
is a technique used to enhance the contrast and improve the overall quality of images.
It's particularly useful in medical image processing, including the analysis of retinal
images for conditions like Diabetic Retinopathy. Here's how the CLAHE module
works: <br>
- **Prep-rocessing Step**: CLAHE is usually applied as a preprocessing step before
feeding the images into machine learning models, such as convolutional neural
networks (CNNs), for automated analysis and diagnosis.<br>
- **Histogram Equalization:** The CLAHE technique is based on the concept of
histogram equalization. The histogram of an image represents the distribution of
pixel intensity values. In traditional histogram equalization, the goal is to spread
out the intensity values in the image so that the full range is utilized, enhancing
contrast.<br>
- **Adaptive Approach:** What sets CLAHE apart is its adaptability. Unlike
standard histogramequalization, CLAHE divides the image into smaller blocks
or tiles. It computes a separate histogram and equalization for each of these tiles.
This adaptive approach allows CLAHE toenhance local contrast. <br>
- **Contrast Limiting:** To prevent over-amplification of noise and outliers in small
regions, CLAHE includes a contrast limiting parameter. This parameter limits
the amplification of the local contrast in each tile. It ensures that the contrast
enhancement remains controlled and doesn't lead to artifacts or excessive noise.
Histogram Calculation: For each tile, CLAHE calculates a histogram of pixel
intensity values. The histogram reflects the distribution of intensity values
within that local region. <br>
- **Equalization**: After the histograms are calculated, CLAHE performs histogram
equalizationon each tile separately. This equalization process redistributes the
pixel intensity values in away that enhances local contrast
