# ERCOT Electricity Load Forecasting with GRU and CNN+KMeans

## Project Overview
This project aims to forecast electricity load for the Electric Reliability Council of Texas (ERCOT) using deep learning models, specifically a **Gated Recurrent Unit (GRU)** model and a **CNN+KMeans clustering** approach. By leveraging historical electricity consumption data combined with weather data from multiple stations, the goal is to enhance the accuracy of load forecasting.

## Features
### Data Preprocessing
- **Data Merging:** Combines ERCOT's hourly electricity consumption data with weather data from three weather stations.
- **Missing Data Handling:** Fills missing values using forward and backward filling methods to maintain time-series integrity.
- **Feature Engineering:** Extracts time-based features such as Hour, Day, Month, and Weekday to improve the time-series forecasting model.
- **Normalization:** Normalizes data using **Min-Max Scaling** to improve model performance and convergence speed.

### Modeling Approaches
#### GRU Model
- **Input:** Utilizes the past 24 hours of load data to predict the next time step.
- **Architecture:** A two-layer **GRU** network with **dropout** regularization to prevent overfitting.
- **Optimizer:** Uses **Adam optimizer** for adaptive learning rates.
- **Loss Function:** **Mean Squared Error (MSE)** is used for training the model.

#### CNN+KMeans Model
- **CNN for Feature Extraction:** Convolutional Neural Networks (CNN) extract spatial features from the time-series data, capturing local patterns.
- **KMeans Clustering:** KMeans clustering refines predictions by identifying patterns and grouping similar data points, improving forecast accuracy.

### Performance Evaluation
- **Metrics:** Model performance is evaluated using:
  - **MAPE** (Mean Absolute Percentage Error)
  - **RMSE** (Root Mean Squared Error)
  - **NMAE** (Normalized Mean Absolute Error)
  - **NRMSE** (Normalized Root Mean Squared Error)
  - **R² Score** (Coefficient of Determination)
- **Model Comparison:** Evaluates the performance of both the GRU model and the CNN+KMeans model for different seasons (e.g., Summer, Winter).
- **Visualization:** Compares predicted values against actual electricity load data through various plots.

### Visualizations and Insights
- **Data Distribution:** Histograms and bar charts to visualize the distribution of features.
- **Time-Series Comparison:** Plots showing actual vs. predicted load values to evaluate model performance over time.
- **Model Performance:** Graphs that compare the performance metrics of GRU vs. CNN+KMeans models.

## Dataset
- **ERCOT Load Data:** Hourly electricity consumption records from ERCOT.
- **Weather Data:** Meteorological data including temperature, humidity, and other weather factors collected from three weather stations in the ERCOT region.

## Results Summary
### Model Performance (Seasonal Comparison)

| Model          | Season | MAPE  | RMSE   | NMAE   | NRMSE  |
|----------------|--------|-------|--------|--------|--------|
| CNN            | Summer | 3.95  | 0.250  | 0.026  | 0.046  |
| CNN            | Winter | 12.55 | 0.261  | 0.028  | 0.047  |
| CNN+KMeans     | Summer | 3.05  | 0.219  | 0.024  | 0.040  |
| CNN+KMeans     | Winter | 7.41  | 0.239  | 0.025  | 0.042  |

## Installation Instructions
To set up and run the project, follow the steps below:

### 1. Install dependencies
Install the required Python packages using `pip`:
```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn
