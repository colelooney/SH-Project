# SH-Project  
**Search for CP violation in llqq final states using deep learning**

---

## ✅ Project Overview  
This repository contains a machine-learning workflow developed to study **CP-violating effects** in the semi-leptonic final state (ℓℓqq) of heavy boson decays.  
The goal is to train a **Deep Neural Network (DNN)** to discriminate between events with **positive vs. negative effective luminosity weights (`Lumi_weight`)** and thereby probe **CP asymmetries**.

Key components include:  
- Preprocessing of large HDF5 datasets  
- Standardization and structured data splitting (train/dev/test)  
- A PyTorch-based DNN architecture with BatchNorm, Dropout, and weighted loss support  
- Performance evaluation via ROC, discriminant distributions, and Lumi-weighted histograms  
- Diagnostics for training behaviour and data-splitting strategies  

---
