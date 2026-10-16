import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
data = pd.read_csv("house_prices.csv")
print("Original dataset shape:", data.shape)


# 2. Convert price values into rupees
def convert_to_rupees(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()
    value = value.replace(",", "").replace("₹", "").replace("rs.", "").replace("rs", "")

    try:
        if "cr" in value:
            return float(value.replace("cr", "").strip()) * 10000000
        elif "lac" in value:
            return float(value.replace("lacs", "").replace("lac", "").strip()) * 100000
        elif "lakh" in value:
            return float(value.replace("lakhs", "").replace("lakh", "").strip()) * 100000
        elif "k" in value:
            return float(value.replace("k", "").strip()) * 1000
        else:
            return float(value)
    except (ValueError, TypeError):
        return np.nan


data["Total Price"] = data["Amount(in rupees)"].apply(convert_to_rupees)


# 3. Clean numeric columns
data["Carpet Area"] = pd.to_numeric(
    data["Carpet Area"].astype(str).str.replace(",", "", regex=False)
    .str.extract(r"(\d+\.?\d*)")[0],
    errors="coerce"
)

data["Bathroom"] = pd.to_numeric(
    data["Bathroom"].astype(str).str.extract(r"(\d+\.?\d*)")[0],
    errors="coerce"
)


# 4. Clean categorical columns
data["location"] = data["location"].fillna("Unknown").astype(str).str.strip().str.lower()
data["Furnishing"] = data["Furnishing"].fillna("Unknown").astype(str).str.strip()
data["Transaction"] = data["Transaction"].fillna("Unknown").astype(str).str.strip()


# 5. Select features and target
features = [
    "Carpet Area",
    "Bathroom",
    "location",
    "Furnishing",
    "Transaction"
]

target = "Total Price"

data = data[features + [target]].copy()


# 6. Remove missing and unrealistic values
data = data.dropna(subset=["Carpet Area", "Bathroom", target])

data = data[
    (data["Carpet Area"] >= 100) &
    (data["Carpet Area"] <= 20000) &
    (data["Bathroom"] >= 1) &
    (data["Bathroom"] <= 20) &
    (data[target] >= 100000) &
    (data[target] <= 1000000000)
].copy()

print("Rows after cleaning:", len(data))


# 7. Separate features and target
X = data[features].copy()
y = data[target].copy()


# 8. Grouped train/test split
# Keep identical feature combinations together to reduce overlap.
groups = X.astype(str).agg("|".join, axis=1)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()
y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

print("\n--- Grouped split ---")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training feature groups:", groups.iloc[train_idx].nunique())
print("Testing feature groups:", groups.iloc[test_idx].nunique())


# 9. Verify there is no overlap between feature groups
train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])
overlap = train_groups.intersection(test_groups)

print("\n--- Train/test overlap check ---")
print("Overlapping groups:", len(overlap))

seen_in_train = groups.iloc[test_idx].isin(train_groups)
print("Test rows with feature combinations seen in training:", seen_in_train.sum())
print(
    "Percentage with previously seen feature combinations:",
    round(seen_in_train.mean() * 100, 2), "%"
)


# 10. Prepare features for modeling
# Fit dummy-column definitions on training data only.
X_train_encoded = pd.get_dummies(
    X_train,
    columns=["location", "Furnishing", "Transaction"],
    dtype=int
)

X_test_encoded = pd.get_dummies(
    X_test,
    columns=["location", "Furnishing", "Transaction"],
    dtype=int
)

X_test_encoded = X_test_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value=0
)


# 11. Median baseline model
baseline = DummyRegressor(strategy="median")
baseline.fit(X_train_encoded, y_train)

baseline_predictions = baseline.predict(X_test_encoded)
baseline_mae = mean_absolute_error(y_test, baseline_predictions)

print("\n--- Baseline ---")
print(f"Median baseline MAE: ₹{baseline_mae:,.0f}")


# 12. Train Random Forest models
# Basic features: Carpet Area, Bathroom, location
basic_columns = [
    col for col in X_train_encoded.columns
    if col in ["Carpet Area", "Bathroom"] or col.startswith("location_")
]

rf_basic = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_basic.fit(X_train_encoded[basic_columns], y_train)
basic_predictions = rf_basic.predict(X_test_encoded[basic_columns])


# Enhanced features: include Furnishing and Transaction
rf_enhanced = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_enhanced.fit(X_train_encoded, y_train)
enhanced_predictions = rf_enhanced.predict(X_test_encoded)


# 13. Evaluation function
def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return {
        "Model": name,
        "MAE (₹)": round(mae),
        "RMSE (₹)": round(rmse),
        "R²": round(r2, 4)
    }


# 14. Compare basic and enhanced features
comparison = pd.DataFrame([
    evaluate_model("Basic features", y_test, basic_predictions),
    evaluate_model("With extra features", y_test, enhanced_predictions)
])

print("\n--- Feature comparison ---")
print(comparison.to_string(index=False))


# 15. Train and compare Decision Tree
decision_tree = DecisionTreeRegressor(
    random_state=42,
    min_samples_leaf=5
)

decision_tree.fit(X_train_encoded, y_train)
tree_predictions = decision_tree.predict(X_test_encoded)

model_comparison = pd.DataFrame([
    evaluate_model("Decision Tree", y_test, tree_predictions),
    evaluate_model("Random Forest", y_test, enhanced_predictions)
])

print("\n--- Decision Tree vs Random Forest ---")
print(model_comparison.to_string(index=False))

best_model_name = model_comparison.sort_values("MAE (₹)").iloc[0]["Model"]
print("Best model by MAE:", best_model_name)


# 16. Prediction errors by price range
results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": enhanced_predictions
})

results["Absolute Error"] = (
    results["Actual Price"] - results["Predicted Price"]
).abs()

results["Price Range"] = pd.cut(
    results["Actual Price"],
    bins=[0, 2500000, 5000000, 10000000, float("inf")],
    labels=[
        "Under ₹25 lakh",
        "₹25–50 lakh",
        "₹50 lakh–₹1 crore",
        "Above ₹1 crore"
    ]
)

error_analysis = results.groupby(
    "Price Range", observed=False
).agg(
    Listings=("Absolute Error", "count"),
    MAE=("Absolute Error", "mean")
).reset_index()

print("\n--- Prediction Errors by Price Range ---")
error_analysis["MAE"] = error_analysis["MAE"].apply(
    lambda value: f"₹{value:,.0f}"
)
print(error_analysis.to_string(index=False))


# 17. Feature importance
importance = pd.DataFrame({
    "Feature": X_train_encoded.columns,
    "Importance": rf_enhanced.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n--- Top 10 Important Features ---")
print(importance.head(10).to_string(index=False))


# 18. Display sample predictions
sample_predictions = pd.DataFrame({
    "Actual Price (₹)": y_test.iloc[:10].values,
    "Predicted Price (₹)": enhanced_predictions[:10]
})

sample_predictions["Actual Price (₹)"] = sample_predictions[
    "Actual Price (₹)"
].round(0).astype(int)

sample_predictions["Predicted Price (₹)"] = sample_predictions[
    "Predicted Price (₹)"
].round(0).astype(int)

print("\n--- Sample Predictions ---")
print(sample_predictions.to_string(index=False))


# 19. Actual vs predicted price visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, enhanced_predictions, alpha=0.4)

min_price = min(y_test.min(), enhanced_predictions.min())
max_price = max(y_test.max(), enhanced_predictions.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.xlabel("Actual House Price (₹)")
plt.ylabel("Predicted House Price (₹)")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()
