# ERCOT Electricity Load Forecasting with GRU and CNN+KMeans

## Overview
This project focuses on electricity load forecasting using deep learning models, specifically GRU (Gated Recurrent Unit) and CNN+KMeans clustering. The objective is to improve prediction accuracy using historical ERCOT load data combined with weather data from multiple stations.

## Features
- **Data Preprocessing:**
  - Merging ERCOT load data with weather data from three weather stations.
  - Handling missing values using forward and backward filling.
  - Feature extraction (Hour, Day, Month, Weekday) for time series analysis.
  - Normalization using Min-Max Scaling.

- **Modeling Approaches:**
  - **GRU Model:**
    - Uses past 24 hours of data to predict the next time step.
    - Two-layer GRU with dropout to prevent overfitting.
    - Adam optimizer and Mean Squared Error (MSE) loss function.
  - **CNN+KMeans:**
    - CNN extracts spatial features from the time-series data.
    - KMeans clustering refines the predictions by segmenting data patterns.
    
- **Performance Evaluation:**
  - Metrics used: MAPE, RMSE, NMAE, NRMSE, R² Score.
  - Comparison of GRU and CNN+KMeans models.
  - Visualization of predictions vs actual load data.
  
- **Visualizations and Insights:**
  - Histograms of features to analyze data distribution.
  - Time series plots for actual vs predicted values.
  - Performance comparison of models.
  
## Dataset
- **ERCOT Load Data:** Contains hourly electricity consumption records.
- **Weather Data:** Temperature, humidity, and other meteorological factors from three stations.

## Results
| Model | Season | MAPE | RMSE | NMAE | NRMSE |
|--------|--------|--------|--------|--------|--------|
| CNN | Summer | 3.95 | 0.250 | 0.026 | 0.046 |
| CNN | Winter | 12.55 | 0.261 | 0.028 | 0.047 |
| CNN+KMeans | Summer | 3.05 | 0.219 | 0.024 | 0.040 |
| CNN+KMeans | Winter | 7.41 | 0.239 | 0.025 | 0.042 |

## Installation
```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib seaborn
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ercot-load-forecasting.git
cd ercot-load-forecasting
```
2. Run the preprocessing script:
```bash
python preprocess.py
```
3. Train the model:
```bash
python train_model.py
```
4. Visualize results:
```bash
python visualize.py
```

## Future Enhancements
- Integrate additional meteorological factors.
- Hyperparameter tuning for improved accuracy.
- Deploy the model as a web API for real-time predictions.

## Contributors
- **Muhammad Mohsin** - Lead Developer

## License
This project is licensed under the MIT License.

