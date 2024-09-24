# Time Series Forecasting Using Neural Networks and Monte Carlo Simulations

## Overview
This project focuses on time series forecasting for fund net values using various machine learning models. The data is scraped in real-time from financial websites, and models including Simple Neural Networks, Bayesian Neural Networks (BNN), and Monte Carlo (MC) simulations are used for prediction and optimization. The goal is to explore different modeling approaches and assess their performance in forecasting future net values.

## Data Source
The data used in this project is scraped from the [Tian Tian Fund website](http://fund.eastmoney.com/). It consists of daily net values of selected funds over the past two years.

## Project Structure
- **`FundNetValueData.py`**: Script for scraping and saving fund net values data.
- **`LSTM_Predicted_30days.py`**: LSTM-based model for predicting future fund net values.
- **`MonteCarlo_Simulation.py`**: Monte Carlo simulation to generate predictions with confidence intervals.
- **`Predicted_RMSE_BNN.py`**: Optimization using Bayesian Neural Networks (BNN).
- **`PredictionErrors_basedwithErrorofTraining.py`**: Analysis of prediction errors based on training set performance.

## Dependencies
To run the scripts, you need to install the following Python packages:

```bash
pip install pandas numpy tensorflow matplotlib requests beautifulsoup4
```

## How to Run

### 1. Data Scraping
Run the `FundNetValueData.py` script to scrape the latest fund net values and save them as CSV files:

```bash
python FundNetValueData.py
```

### 2. Simple Neural Network Prediction
Run the `MonteCarlo_Simulation.py` script to perform time series forecasting using a simple neural network and Monte Carlo simulation:

```bash
python MonteCarlo_Simulation.py
```

### 3. Bayesian Neural Network (BNN) Optimization
Run the `Predicted_RMSE_BNN.py` script to train and evaluate a Bayesian Neural Network for the fund net value data:

```bash
python Predicted_RMSE_BNN.py
```

### 4. Error Analysis
Run the `PredictionErrors_basedwithErrorofTraining.py` script to analyze prediction errors based on training set performance:

```bash
python PredictionErrors_basedwithErrorofTraining.py
```

## Detailed Workflow

### 1. Data Preprocessing
- The data is read from CSV files and preprocessed by converting dates, sorting, and forward filling missing values.
- A sliding window technique is used to create training and testing datasets, allowing the models to learn from sequences of historical data.

### 2. Modeling
- **Simple Neural Network**: A basic neural network with two hidden layers is built to predict the fund net values.
- **Monte Carlo Simulation**: Used to calculate confidence intervals for predictions, providing a range of possible outcomes.
- **Bayesian Neural Network**: Incorporates uncertainty in predictions, yielding more robust predictions compared to traditional models.

### 3. Evaluation
- **Root Mean Square Error (RMSE)** is used to evaluate the model performance on both training and testing datasets.
- The **standard deviation of prediction errors** is calculated to measure model uncertainty.

### 4. Visualization
- Actual vs. predicted net values are plotted to visually compare model performance.
- Future predictions with confidence intervals are plotted using Monte Carlo simulations.
- Distribution of predictions for specific future dates is shown to assess prediction spread and uncertainty.

## Results and Analysis
- The models are evaluated based on their RMSE and prediction accuracy.
- The Monte Carlo simulation provides a 95% confidence interval for future predictions, giving insights into potential future values.
- Bayesian Neural Network models improve prediction robustness by considering uncertainty in training data.

## Future Work
- Implement advanced models like Transformers for time series prediction.
- Extend the data sources to include multiple financial indicators and news sentiment analysis.
- Automate the model selection and hyperparameter tuning process.

## Contact
For any questions or suggestions, feel free to reach out via GitHub issues or email.

