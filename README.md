# ML Marketplace

A Python-based, end-to-end machine learning web application for analyzing computer systems, predicting prices, and finding similar offers using Streamlit.

## Overview

ML Marketplace is a comprehensive web application that demonstrates the entire machine learning lifecycle from data preprocessing to deploying a user-friendly interface. The application focuses on computer systems data, allowing users to explore the dataset, predict prices based on specifications, and find similar computer systems using machine learning techniques.

## Features

- **Data Preprocessing**: Automated cleaning of numeric columns with units, handling missing values, and merging duplicate features.
- **Descriptive Analytics**: Interactive visualizations, summary statistics, and K-means clustering of computer systems.
- **Predictive Modeling**: Price prediction using machine learning models with feature importance explanations.
- **Similarity Search**: Find similar computer systems based on specifications using k-NN or cosine similarity.
- **User Feedback Collection**: Integrated feedback mechanism to gather user input on the application functionality.

## Project Structure

```
ml_marketplace/
  ├─ data/              # Raw & processed CSV files
  ├─ notebooks/         # Exploratory analysis notebooks
  ├─ src/
  │   ├─ preprocessing.py  # Data cleaning and preprocessing
  │   ├─ features.py       # Feature engineering and selection
  │   ├─ models.py         # Predictive model training and evaluation
  │   ├─ similarity.py     # Similarity search functionality
  │   └─ app.py            # Streamlit application entry point
  ├─ models/            # Trained model artifacts
  ├─ feedback/          # Saved user feedback
  ├─ README.md          # Project documentation
  └─ requirements.txt   # Project dependencies
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ml_marketplace.git
   cd ml_marketplace
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Prepare the data:
   - Place your `db_computers_2025_raw.csv` file in the `data/` directory

## Running the Application

1. Train the models (if not already trained):
   ```
   python src/train_models.py
   ```

2. Launch the Streamlit application:
   ```
   streamlit run src/app.py
   ```

3. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

## Training Pipeline

The training pipeline consists of several components:

1. **Data Preprocessing** (`preprocessing.py`):
   - Loads raw data
   - Cleans numeric columns by stripping text and normalizing units
   - Merges duplicate features
   - Handles missing values using configurable strategies

2. **Feature Engineering** (`features.py`):
   - Identifies core features for machine learning
   - Implements multi-label categorical encoding
   - Scales numeric features
   - Creates a reusable feature pipeline

3. **Model Training** (`models.py`):
   - Trains multiple regression models (Random Forest, Gradient Boosting, MLP)
   - Evaluates models with cross-validation
   - Selects the best model based on RMSE
   - Computes SHAP values for feature interpretation

4. **Similarity Search** (`similarity.py`):
   - Implements k-NN or cosine similarity on feature vectors
   - Provides feature-wise distance calculation
   - Creates visualizations comparing similar items

## Web Application

The Streamlit application (`app.py`) provides an intuitive interface with four main pages:

1. **Home**: Overview of the dataset and application features
2. **Descriptive Analytics**: Data exploration, visualizations, and clustering
3. **Predict Price**: Interactive form for predicting computer prices based on specifications
4. **Similar Offers**: Find and compare similar computer systems

## Extending the Project

- **Add New Models**: Implement additional machine learning models in `models.py`
- **Enhance Features**: Add more feature engineering techniques in `features.py`
- **Improve Visualizations**: Create new data visualizations in the Streamlit application
- **Deploy to Cloud**: Deploy the application to a cloud platform like Heroku or AWS

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
