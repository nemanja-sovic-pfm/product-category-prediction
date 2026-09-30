# Product Category Prediction

Machine Learning project for automatic product category prediction based on product titles from an e-commerce platform.

## Project Overview

Online marketplaces receive thousands of new products every day. Manually assigning categories is time consuming and prone to errors.

The goal of this project is to build a machine learning model that automatically predicts the appropriate category for a product based on its title.

Example:

Input:
Samsung Galaxy A52 128GB

Output:
Mobile Phones

---

## Dataset

The dataset contains more than 30,000 products with the following information:

- Product ID
- Product Title
- Merchant ID
- Category Label
- Product Code
- Number of Views
- Merchant Rating
- Listing Date

Target variable:

- Category Label

---

## Project Structure

```text
product-category-prediction/
|-- data/
|   `-- processed/
|       |-- products_clean.csv
|       |-- train.csv
|       `-- test.csv
|-- models/
|   |-- product-category-linear-svc.joblib
|   `-- product-category-linear-svc-metrics.json
|-- src/
|   |-- train-model.py
|   `-- test_model.py
|-- IMLP6_TASK_03-products.csv
|-- JypyterWorkbook.ipynb
`-- README.md
```

The `.joblib` model is generated locally and excluded from Git. The metrics JSON
contains the evaluation result and can be committed.

---

## Environment Setup

Python 3.11 or a compatible newer Python version is recommended.

Create and activate a virtual environment if desired, then install the required
packages:

```powershell
python -m pip install numpy pandas scikit-learn joblib
```

Run all commands from the project root directory.

---

## Build and Save the Model

The training script uses the already prepared datasets:

- `data/processed/train.csv`
- `data/processed/test.csv`

If these files are missing, run the data preparation notebook first.

Build the model with:

```powershell
python .\src\train-model.py
```

The script performs the following steps:

1. Loads and validates the prepared training and test datasets.
2. Builds the preprocessing pipeline using word TF-IDF, numeric scaling, and
   One-Hot encoding.
3. Trains and evaluates the selected `LinearSVC` classifier on the fixed test
   set.
4. Prints the classification report, accuracy, and macro F1 score.
5. Retrains a fresh pipeline on all prepared data.
6. Saves the complete preprocessing and classification pipeline for later use.

Generated files:

```text
models/product-category-linear-svc.joblib
models/product-category-linear-svc-metrics.json
```

The current evaluation result is approximately:

```text
Accuracy: 0.9616
Macro F1: 0.9642
```

---

## Test the Saved Model

The model must be built before starting the interactive test script.

Run:

```powershell
python .\src\test_model.py
```

The script asks for:

- Product title
- Number of views, which may be left empty
- Merchant rating from 1 to 5, which may be left empty

Merchant ID and listing date are treated as unknown and are not requested.
The saved pipeline performs all TF-IDF, scaling, and encoding transformations
before predicting the product category.

Example:

```text
Model je uspesno ucitan!
Unesite 'exit' u bilo kom trenutku za prekid.

Naslov proizvoda: Samsung Galaxy A52 128GB
Broj pregleda (Enter ako nije poznat): 860
Ocena trgovca 1-5 (Enter ako nije poznata): 4.5

Predvidjena kategorija: Mobile Phones
--------------------------------------------------
```

Enter `exit` at any prompt to stop the script.
