import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------
# 1. Load Dataset
# --------------------------------

df = pd.read_csv("train.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())


# --------------------------------
# 2. Check Missing Values
# --------------------------------

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))


# --------------------------------
# 3. Select Features
# --------------------------------

features = [
    "OverallQual",
    "GrLivArea",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "FullBath",
    "GarageCars",
    "TotalBsmtSF",
    "LotArea",
    "YearBuilt",
    "Neighborhood"
]

X = df[features]
y = df["SalePrice"]


# --------------------------------
# 4. Define Feature Types
# --------------------------------

numeric_features = [
    "OverallQual",
    "GrLivArea",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "FullBath",
    "GarageCars",
    "TotalBsmtSF",
    "LotArea",
    "YearBuilt"
]

categorical_features = [
    "Neighborhood"
]


# --------------------------------
# 5. Preprocessing
# --------------------------------

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# --------------------------------
# 6. Train-Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------
# 7. Create Linear Regression Model
# --------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# --------------------------------
# 8. Train Model
# --------------------------------

model.fit(X_train, y_train)

print("\nModel training completed.")


# --------------------------------
# 9. Predictions
# --------------------------------

y_pred = model.predict(X_test)


# --------------------------------
# 10. Model Evaluation
# --------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# --------------------------------
# 11. Actual vs Predicted
# --------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.show()


# --------------------------------
# 12. House Price Distribution
# --------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["SalePrice"], bins=30)

plt.xlabel("House Price")
plt.ylabel("Number of Houses")
plt.title("Distribution of House Prices")

plt.show()


# --------------------------------
# 13. Correlation Heatmap
# --------------------------------

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()