# SH-Project
**Probing CP Violation in ℓℓqq Final States with Deep Learning**

---

## ✅ Project Overview
This repository contains a machine learning project focused on the search for **Charge-Parity (CP) violation** in the semi-leptonic final state (ℓℓqq) of heavy boson decays using deep learning techniques. The primary goal is to train a **Deep Neural Network (DNN)** to distinguish between events with positive and negative effective luminosity weights, which serves as a powerful method for probing CP asymmetries.

The project handles the entire pipeline, from data preprocessing of large HDF5 datasets to model training, evaluation, and visualization of the results.

**Key Features:**
- **Data Handling:** Preprocessing and standardization of large HDF5 datasets.
- **Robust Splitting:** Structured splitting of data into training, development, and testing sets.
- **Deep Neural Network:** A PyTorch-based DNN architecture with BatchNorm, Dropout, and weighted loss support.
- **Performance Evaluation:** Comprehensive model evaluation using ROC curves, discriminant distributions, and luminosity-weighted histograms.
- **Graph Neural Network (GNN):** Includes a GNN implementation for exploring graph-based learning approaches.

---

## 📂 Repository Structure
- DNN/ # Contains the Deep Neural Network implementation
- GNN/ # Contains the Graph Neural Network implementation
- data/ # Data files for the project
- plots/ # Output directory for plots and visualizations
- .gitignore # Git ignore file
- README.md # Project README file
- requirements.txt # Python dependencies

---

## 🛠️ Technology Stack
- Python 3.x
- PyTorch
- Jupyter Notebook
- Pandas & NumPy
- Matplotlib
- scikit-learn

---

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- pip

### Installation
1.  **Clone the repository:**
    ```sh
    git clone https://github.com/colelooney/SH-Project.git
    cd SH-Project
    ```

2.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

### Usage
Navigate to the `DNN` or `GNN` directory and run the python scripts to train models and visualize results.
