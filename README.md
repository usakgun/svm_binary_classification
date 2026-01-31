# Bank Note Authentication System

This project implements a Support Vector Machine (SVM) classifier to detect genuine and forged bank notes based on features extracted from images.

## Project Overview
Authentication of currency is a critical task in banking and commerce. This system automates the process by analyzing distinct features of bank note images using machine learning. The model is trained to classify notes into two categories:
* **Class 0:** Genuine / Authentic
* **Class 1:** Forged / Fake

## Dataset & Features
The model utilizes the "Bill Authentication" dataset. Data was extracted from images that were taken from genuine and forged banknote-like specimens. Wavelet Transform tools were used to extract the following features:
1.  **Variance** of Wavelet Transformed image
2.  **Skewness** of Wavelet Transformed image
3.  **Curtosis** of Wavelet Transformed image
4.  **Entropy** of image

## Methodology
The project follows a standard supervised learning pipeline:
* **Data Preprocessing:** Loading data and separating features (attributes) from labels.
* **Model Selection:** Support Vector Machine (SVM) with a Linear Kernel is used due to its effectiveness in high-dimensional binary classification.
* **Training:** The dataset is split into training (80%) and testing (20%) sets.
* **Evaluation:** The model's performance is measured using specific classification metrics.

## Performance Metrics
The script evaluates the trained model using:
* **Confusion Matrix:** To visualize true positives, true negatives, false positives, and false negatives.
* **Classification Report:** Provides Precision, Recall, and F1-Score for each class.
* **Accuracy Score:** The overall percentage of correctly classified bank notes.

## Technologies
* Python
* Pandas (Data Management)
* Scikit-Learn (SVM Algorithm & Metrics)
* NumPy
* Matplotlib
