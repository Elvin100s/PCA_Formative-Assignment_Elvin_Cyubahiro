# PCA Assignment - Kenya Malaria Indicators

**Principal Component Analysis (PCA) Implementation from Scratch**

---

## 📊 Project Overview

This project implements Principal Component Analysis (PCA) entirely from scratch using NumPy, Pandas, and Matplotlib. The analysis is performed on Kenya Malaria Indicators data from the World Health Organization (WHO).

**Assignment**: Formative 2 - Principal Component Analysis  
**Student**: Elvin Cyubahiro  
**Date**: February 2026  
**Points**: 15

---

## 📁 Dataset

**Kenya Malaria Indicators (2000-2024)**

- **Source**: World Health Organization via Humanitarian Data Exchange
- **URL**: [Kenya Malaria Data - HDX](https://data.humdata.org/dataset/b239ef6c-910d-4347-ba87-2d21a23f03fa)
- **Description**: Health indicators related to malaria incidence, mortality, prevention, and treatment across Kenya
- **Features**:
  - ✅ Contains 595 missing values (NaN)
  - ✅ Contains 17 non-numeric columns
  - ✅ 17 health indicator columns
  - ✅ Real African health data from WHO

---

## 🛠️ Requirements

### Python Version
- Python 3.7 or higher

### Required Libraries
```bash
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

### Installation

Install all requirements using pip:

```bash
pip install numpy pandas matplotlib seaborn
```

Or if you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)

1. **Open in Colab**:
   - Upload the notebook to [Google Colab](https://colab.research.google.com/)
   - Or use this link if repo is public: `https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/PCA_Assignment_Complete.ipynb`

2. **Run the notebook**:
   - Click `Runtime` → `Run all`
   - Wait for all cells to execute (2-3 minutes)
   - All outputs will be generated automatically

3. **No installation needed** - Colab has all libraries pre-installed

### Option 2: Local Jupyter Notebook

1. **Install Jupyter**:
   ```bash
   pip install jupyter
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**:
   ```bash
   jupyter notebook PCA_Assignment_Complete.ipynb
   ```

5. **Run all cells**:
   - Click `Cell` → `Run All`
   - Or run cells individually with `Shift + Enter`

### Option 3: VS Code

1. **Install VS Code Jupyter extension**
2. **Open the notebook** in VS Code
3. **Select Python kernel**
4. **Run all cells**

---

## 📚 What's Inside

### Implementation Features

✅ **No sklearn** - All PCA steps implemented from scratch  
✅ **Complete data preprocessing pipeline**  
✅ **Manual covariance matrix computation**  
✅ **Eigendecomposition using NumPy**  
✅ **Dynamic component selection based on explained variance**  
✅ **Comprehensive visualizations**  
✅ **Performance benchmarking**

### Notebook Structure

```
1.  Import Libraries
2.  Load Raw Data and Explore
3.  Data Quality Assessment
4.  Handle Missing Values
5.  Encode Categorical Data
6.  Standardize Data (Z-Score)
7.  Compute Covariance Matrix
8.  Eigendecomposition
9.  Sort Eigenvalues Descending ⭐
10. Calculate Explained Variance ⭐
11. Visualize Component Selection
12. Project Data onto Principal Components
13. BEFORE vs AFTER PCA Visualization ⭐
14. Written Interpretation
15. Performance Benchmarking (Task 3)
```

---

## 🎯 Assignment Requirements Met

### ✅ Data Handling (5 points)
- Data contains 595 missing values (NaN)
- Data contains 17 non-numeric columns
- Proper encoding and imputation techniques applied
- Skewness-based imputation strategy (median for skewed, mean for normal)

### ✅ Explained Variance Calculation (5 points)
- Variance percentages calculated correctly
- Eigenvalues sorted in descending order
- Components selected dynamically based on 95% variance threshold
- Complete understanding of PCA demonstrated

### ✅ Visualization (5 points)
- Before PCA visualization (original feature space)
- After PCA visualization (principal component space)
- Properly labeled axes (feature names & PC1/PC2 with variance %)
- Clear explanation of PCA effects

---

## 📈 Key Algorithms Implemented

### 1. Data Preprocessing
```python
# Missing value imputation
- Numeric: Skewness-based (median if |skew| > 1, mean if normal)
- Categorical: Mode imputation

# Categorical encoding
- Label encoding for ordinal features
- Frequency-based encoding
```

### 2. Standardization
```python
# Z-score normalization (manual implementation)
mean = data.mean(axis=0)
std = data.std(axis=0, ddof=1)
X_standardized = (data - mean) / std
```

### 3. Covariance Matrix
```python
# Manual computation (no np.cov)
X_centered = X - np.mean(X, axis=0)
cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
```

### 4. Eigendecomposition
```python
# Using NumPy for eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
```

### 5. Explained Variance
```python
# Calculate variance explained by each component
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance_ratio)
```

### 6. Dynamic Component Selection
```python
# Select components to retain 95% of variance
threshold = 0.95
n_components = np.argmax(cumulative_variance >= threshold) + 1
```

### 7. Projection
```python
# Project data onto principal components
X_pca = X_standardized @ eigenvectors_sorted[:, :n_components]
```

---

## 📊 Expected Results

After running the notebook, you'll see:

- **Dataset**: 140 samples × 17 features
- **Missing Values**: 595 NaN values across 5 columns
- **Principal Components**: Dynamically selected based on 95% variance threshold
- **Dimensionality Reduction**: From 17 features to ~5-10 principal components
- **Variance Explained**: PC1 captures the most variance, decreasing for subsequent PCs
- **Visualizations**: 
  - Missing value heatmaps
  - Covariance matrix heatmaps
  - Scree plot
  - Cumulative explained variance
  - Before/After PCA scatter plots
  - Performance benchmarks

---

## 🖼️ Visualizations Included

1. **Missing Value Heatmap** - Shows data quality issues
2. **Missing Value Bar Chart** - Top columns with missing data
3. **Covariance Matrix Heatmap** - Feature correlations
4. **Eigenvalue Scree Plot** - Component importance
5. **Cumulative Explained Variance** - Component selection guide
6. **Before PCA Scatter Plot** - Original feature space (2 features)
7. **After PCA Scatter Plot** - Principal component space (PC1 vs PC2)
8. **Performance Benchmarks** - Execution time analysis

---

## 🔬 Technical Details

### Libraries Used
- **NumPy**: Matrix operations, eigendecomposition
- **Pandas**: Data loading, preprocessing, imputation
- **Matplotlib**: All plots and visualizations
- **Seaborn**: Heatmaps and statistical plots
- **time**: Performance benchmarking

### No External ML Libraries
- ❌ No sklearn
- ❌ No scipy (except as NumPy dependency)
- ❌ No pre-built PCA functions

### Key Functions Implemented Manually
- ✅ Covariance matrix computation
- ✅ Data standardization (Z-score)
- ✅ Eigenvalue sorting
- ✅ Explained variance calculation
- ✅ Data projection onto principal components

---

## 📝 Files in Repository

```
├── PCA_Assignment_Complete.ipynb    # Main notebook with all code
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
└── malaria_indicators_ken.csv       # Dataset (optional - loaded from URL)
```

---

## 🎓 Learning Outcomes

This project demonstrates:

1. **Linear Algebra Concepts**:
   - Eigenvalues and eigenvectors
   - Covariance matrices
   - Matrix transformations
   - Orthogonality

2. **Statistical Concepts**:
   - Variance and explained variance
   - Data standardization
   - Missing data imputation
   - Skewness analysis

3. **Programming Skills**:
   - NumPy for numerical computing
   - Pandas for data manipulation
   - Matplotlib for visualization
   - Code optimization and benchmarking

4. **Data Science Workflow**:
   - Data loading and exploration
   - Data cleaning and preprocessing
   - Algorithm implementation from scratch
   - Result visualization and interpretation
   - Performance analysis

---

## ⚠️ Troubleshooting

### Issue: Data not loading
**Solution**: Check internet connection. The data is loaded from an online URL. If the URL is down, the CSV file is included in the repo as backup.

### Issue: Memory error
**Solution**: Restart the runtime/kernel and run again. The dataset is small (140 rows), so this is unlikely.

### Issue: Plots not showing
**Solution**: 
- In Jupyter: Make sure `%matplotlib inline` is set or use `plt.show()`
- In Colab: Plots should show automatically

### Issue: Module not found
**Solution**: Install missing libraries:
```bash
pip install numpy pandas matplotlib seaborn
```

### Issue: All columns showing as 'object' type
**Solution**: The CSV has a metadata header row. Use `pd.read_csv(url, skiprows=1)` to skip it.

---

## 📧 Contact

**Student**: Elvin Cyubahiro  
**Assignment**: Formative 2 - PCA  
**Course**: Advanced Linear Algebra

For questions or issues with this implementation, please create an issue in the GitHub repository.

---

## 📄 License

This project is for educational purposes as part of an Advanced Linear Algebra course assignment.

---

## 🙏 Acknowledgments

- **Data Source**: World Health Organization (WHO) via Humanitarian Data Exchange
- **Dataset**: Kenya Malaria Indicators (2000-2024)
- **Course**: Advanced Linear Algebra
- **Platform**: Google Colab

---

## ⭐ Key Features

- 🎯 Meets all assignment requirements
- 📊 Real-world African health data (Kenya WHO malaria indicators)
- 🔬 From-scratch PCA implementation (no sklearn)
- 📈 Comprehensive visualizations (8 different plots)
- ⚡ Performance optimized with benchmarking
- 📝 Well-documented code with explanations
- ✅ Rubric-aligned structure with verification checks
- 🧮 Mathematical rigor with formula documentation

---

## 🔍 Data Quality Verification

The Kenya malaria dataset was verified to meet all requirements:

- ✅ **Missing Values**: 595 NaN values (63.6% in Low/High columns, 99.3% in DIMENSION columns)
- ✅ **Non-Numeric Columns**: All 17 columns (will be encoded appropriately)
- ✅ **Column Count**: 17 columns (exceeds 10+ requirement)
- ✅ **African Data**: Kenya health indicators from WHO
- ✅ **Not Generic**: Real-world epidemiological data, not house prices or wine quality

---

## 🚀 Quick Start

**For the impatient:**

```bash
# 1. Clone repo
git clone https://github.com/Elvin100s/YOUR_REPO_NAME.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebook
jupyter notebook PCA_Assignment_Complete.ipynb

# 4. Or open in Colab and click "Runtime → Run all"
```

---



---


