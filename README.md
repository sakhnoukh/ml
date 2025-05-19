# ML Marketplace: Computer Price Prediction

## Project Overview

This project aims to predict computer prices based on their hardware specifications. It features a machine learning model trained on computer data and a Streamlit web application that allows users to input hardware details and receive an estimated price range. The application focuses on providing a user-friendly interface with features like accordion-style input grouping and realistic price adjustments for specific brands (e.g., Apple) and high-end components.

## Key Features

*   **Streamlit Web Application (`src/app.py`)**:
    *   Interactive UI for inputting computer specifications (CPU, RAM, SSD, Screen Size, Graphics Card, etc.).
    *   Accordion-style sections for organized input: "Processor & Memory," "Storage, Display & Graphics," and "Other Features."
    *   Enriched dropdown for graphics cards showing detailed names.
    *   Dynamic price prediction based on user inputs.
    *   Special pricing considerations for Apple products and high-performance configurations.
*   **Machine Learning Model**:
    *   Predicts price based on a hybrid approach, combining a rule-based system with model predictions.
    *   Handles various hardware components and their impact on pricing.
*   **Market Segmentation and Analysis**:
    *   Identifies which market segment the user's configuration belongs to.
    *   Provides detailed insights about different market segments and their characteristics.
    *   Visualizes segment distribution and market share.
*   **Price Breakdown**:
    *   Detailed visualization of how different components contribute to the final price.
    *   Percentage-based breakdown of component costs.
    *   Cost optimization recommendations based on user configuration.
*   **Similarity Search**:
    *   Finds and presents similar computer configurations based on the user's inputs.
    *   Explains why each alternative is similar to the user's configuration.
    *   Provides visual comparison between user configuration and alternatives.
    *   Collects user feedback on the relevance and helpfulness of alternatives.
*   **Data Processing & Feature Engineering**:
    *   Scripts for cleaning raw data (`src/preprocessing.py`, `src/data_cleaner.py`).
    *   Feature engineering to prepare data for modeling (`src/feature_engineering.py`, `src/feature_builder.py`).

## Directory Structure

/MLF_22_2/
├── archive/         # Older models and scripts
├── data/            # Raw, processed, and intermediate datasets
├── models/          # Trained models, scalers, and preprocessing pipelines
├── notebooks/       # Jupyter notebooks for EDA and experimentation (e.g., eda_simplified.py)
├── outputs/         # Generated plots, summaries, and model results
├── src/             # Source code
│   ├── app.py       # Main Streamlit application
│   ├── model.py     # Model training and prediction logic
│   ├── preprocessing.py # Data cleaning and preparation
│   ├── feature_engineering.py # Feature creation
│   └── ...          # Other utility scripts
├── venv/            # Python virtual environment (if created locally)
└── README.md        # This file

## Setup and Usage

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/sakhnoukh/ML-.git
    cd ML-
    ```

2.  **Install Git LFS (if you haven't already):**
    Ensure Git LFS is installed to handle large model files.
    ```bash
    git lfs install
    git lfs pull
    ```
    (Instructions: [https://git-lfs.github.com](https://git-lfs.github.com))

3.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install dependencies:**
    A `requirements.txt` file would typically be here. For now, ensure you have the main libraries:
    ```bash
    pip install streamlit pandas numpy scikit-learn joblib
    ```
    *(If a `requirements.txt` is added later, use `pip install -r requirements.txt`)*

5.  **Run the Streamlit Application:**
    ```bash
    streamlit run src/app.py
    ```
    This will open the application in your web browser.

## Technologies Used

*   Python
*   Streamlit (for the web UI)
*   Pandas (for data manipulation)
*   NumPy (for numerical operations)
*   Scikit-learn (for machine learning tasks)
*   Joblib (for saving/loading models)
*   Git & Git LFS (for version control and large file storage)

---

This README provides a good starting point. We can refine it further if needed.
