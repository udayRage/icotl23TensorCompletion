# rasterImputation

## Introduction 
Raster Imputation: Enhancing Data Completeness and Quality by prediction of missing pixels in raster images using Tensor-Based Imputation Techniques 

Imputation is a powerful technique used to enhance the completeness and quality of raster data by predicting and filling missing or corrupted pixel values. Raster data, often represented as gridded matrices of pixels, are prevalent in various fields such as remote sensing, geographic information systems (GIS), and image analysis. However, real-world raster datasets frequently contain gaps or areas with corrupted or unobserved values due to sensor limitations, data acquisition issues, or other factors. Raster imputation provides a solution to address these data gaps and improve the usability of such datasets.

One of the key benefits of raster imputation is its ability to preserve spatial and contextual integrity during the data enhancement process. Algorithms consider not only the pixel values themselves but also their spatial relationships, allowing for accurate reconstruction of missing values while respecting the overall structure of the dataset. This is especially valuable in applications like land cover classification, environmental monitoring, and terrain analysis, where preserving spatial coherence is crucial.

## Algorithms 
- Simple Low-Rank Tensor Completion (siLRTC)
- High-accuracy Low-Rank Tensor Completion (HaLRTC)
- Canonical Polyadic-Alternating Least Squares (CP-ALS)
- Coupled Matrix and Tensor Factorization Optimization (CMTF OPT)
- Coupled Matrix and Tensor factorization Optimization For Satellite Images (CMTF4SI)

## Files
- algos.py : Implementation of all the above mentioned algorithms in python. 
- helpers.py : Mathematical Helper functions required for the algorithm.
- imputation.py : Necessary functions required to run imputation algorithms from algos.py
- UIhandler.py : Handles GUI if user wants to give manual input to create missing pixels.
- experiments.py : Imputation of raster images.
- ThesisTutorial.py : Tutorial to explain how to run imputation algorithms.
- experimentsTutorial.py : Tutorial to explain how experiments were conducted.
- QGIS Plugin - Folder containg imputation zip file which can be downloaded and installed in QGIS to perform imputation on raster images in the application. (You can also find usage tutorial in the folder)


### This work is done by EDULA Raashika at the University of Aizu under the supervision of Prof.RAGE Uday Kiran for the fulfillment of the requirements for the Master's Thesis. 


