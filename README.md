# VNIndex Closing Price Prediction

## Goal

The goal of this project is to make stock predictions, specifically predicting the closing price of the VNIndex (Vietnam's main stock market index) using machine learning regression models.

## Why These Models

I read several research papers by other scientists who attempted to predict stock market prices. Ridge Regression, Decision Trees, and Random Forests kept coming up as models used in that research, so I decided to try them myself. To understand them properly, I implemented them from scratch with NumPy instead of only calling library functions.

## Dataset

- Source: [Cafef.vn](https://cafef.vn)
- The original `.xlsx` file was exported from Cafef and converted to `vnindex.csv` using Excel.
- Period: 1 January 2025 to 1 August 2025.
- Columns used: `Open`, `High`, `Low`, `Volume` (features) and `Close` (target).

## Models

### Ridge Regression
A linear regression model with an L2 penalty. It fits a straight-line relationship between the features and the target, while the penalty (`lambda`) shrinks large weights to reduce overfitting. It is simple, fast, and a good baseline.

### Decision Tree
A model that repeatedly splits the data with yes/no questions on a feature (e.g. "is `High` <= some value?"). Each split is chosen to minimize the squared error in the resulting groups. At the end of each branch (a leaf), the prediction is the average target value of the samples that landed there. It can capture non-linear patterns but overfits easily on its own.

### Random Forest
An ensemble of many decision trees. Each tree is trained on a bootstrap sample (random sampling with replacement) of the training data, and the final prediction is the average of all trees. This reduces the variance of a single tree.

## What I Did in the Code

1. **Data preparation**
   - Loaded `vnindex.csv`, renamed columns, dropped the unused column, parsed dates, converted numeric columns (removing thousands separators), and sorted by date.
   - Split the data chronologically (no shuffling):
     - Train: before 2025-05-01
     - Validation: 2025-05-01 to before 2025-06-26
     - Test: from 2025-06-26 onward
   - Saved the splits as `train.csv`, `val.csv`, and `test.csv`.
   - Removed rows where `Open`, `High`, or `Low` was >= 10 (outlier filtering), then inspected each split with `describe()`.

2. **Features and target**
   - Features: `Open`, `High`, `Low`, `Volume`
   - Target: `Close`

3. **Ridge Regression (from scratch)**
   - Standardized the features using the training set's mean and standard deviation, and applied the same values to validation and test data.
   - Added a bias term and initialized the weights to zero.
   - Trained with gradient descent (learning rate 0.01, 250 epochs, `lambda` = 1.0, no regularization on the bias).
   - Tracked training and validation MSE every epoch and plotted the loss curves.
   - Evaluated on the test set with R² and MSE.

4. **Decision Tree (from scratch)**
   - Built a recursive regression tree: for every feature and every unique value as a threshold, it picks the split with the lowest weighted MSE.
   - Stopping conditions: maximum depth, minimum samples to split, or all target values being identical.
   - Leaves predict the mean of the target values.

5. **Random Forest (from scratch)**
   - Built on the decision tree above with bootstrap sampling and averaging of predictions.
   - Settings: up to 150 trees, max depth 15, min samples split 2.
   - Added trees one at a time and recorded training and validation MSE after each tree.
   - Applied early stopping based on validation loss (patience 100) and kept the number of trees with the lowest validation MSE.
   - Plotted training vs. validation loss per tree and evaluated on the test set with R² and MSE.

6. **Scikit-learn Random Forest (comparison)**
   - Repeated the same tree-by-tree growth and early-stopping procedure with scikit-learn's `RandomForestRegressor` (`warm_start=True`) to compare against my own implementation.
   - Retrained the final model with the best number of trees and evaluated it on the test set.

## Evaluation Metrics

- R² score
- Mean Squared Error (MSE)

## Limitations

- The features (`Open`, `High`, `Low`, `Volume`) come from the same day as the `Close` target, so the model estimates the closing price from same-day information rather than forecasting a future price. High scores should be read with this in mind.
- The dataset is small (about 7 months).
- Results are not financial advice.

## Possible Future Work

- Predict the next day's (or next week's) closing price using lagged features and moving averages.
- Compare against a naive baseline ("tomorrow = today").
- Add more metrics (RMSE, MAE, directional accuracy) and a predicted-vs-actual plot.
- Use time-series cross-validation.
