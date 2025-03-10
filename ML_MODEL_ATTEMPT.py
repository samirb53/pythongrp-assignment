import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Step 1: Load Preprocessed Data
logging.info("📂 Loading preprocessed training and testing datasets...")
train_df = pd.read_csv("cleaned_stock_data_train.csv", low_memory=False)
test_df = pd.read_csv("cleaned_stock_data_test.csv", low_memory=False)
logging.info(f"✅ Train dataset shape: {train_df.shape}, Test dataset shape: {test_df.shape}")

# ✅ Step 2: Standardize column names
logging.info("🔤 Standardizing column names...")
train_df.columns = train_df.columns.str.strip().str.lower().str.replace(' ', '_')
test_df.columns = test_df.columns.str.strip().str.lower().str.replace(' ', '_')

# ✅ Step 3: Encode Categorical Variables
logging.info("🛠 Encoding categorical columns...")
categorical_cols = train_df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    
    # Fit only on train data
    train_df[col] = train_df[col].astype(str)
    le.fit(train_df[col])
    
    # Transform train data
    train_df[col] = le.transform(train_df[col])
    
    # ✅ Fix: Use a dictionary mapping to safely transform test data
    encoding_dict = {label: index for index, label in enumerate(le.classes_)}
    test_df[col] = test_df[col].astype(str).map(encoding_dict).fillna(-1).astype(int)
    
    label_encoders[col] = le

# Convert categorical columns to integers
test_df[categorical_cols] = test_df[categorical_cols].astype(int)
train_df[categorical_cols] = train_df[categorical_cols].astype(int)
logging.info("✅ Categorical encoding completed!")

# ✅ Step 4: Identify Adjusted Close Column
logging.info("📌 Identifying adjusted close column...")
adj_close_col = next((col for col in train_df.columns if 'adj' in col.lower() and 'close' in col.lower()), None)
if adj_close_col is None:
    raise KeyError("❌ No column found for adjusted close price!")

logging.info(f"✅ Identified adjusted close column: {adj_close_col}")

# ✅ Step 5: Create Target Variable (Price Up/Down)
logging.info("📊 Creating classification target variable (Price Movement)...")
train_df['price_movement'] = (train_df[adj_close_col].shift(-1) > train_df[adj_close_col]).astype(int)
test_df['price_movement'] = (test_df[adj_close_col].shift(-1) > test_df[adj_close_col]).astype(int)
logging.info("✅ Target variable created!")

# ✅ Step 6: Define Features and Target
target = 'price_movement'

# ✅ Fix: Ensure PCA columns are unique before adding
pca_features = list(set([col for col in train_df.columns if "pca_" in col]))

# ✅ Fix: Extract target variable **before** feature selection
y_train = train_df[target]
y_test = test_df[target]

# ✅ Fix: Features must exclude target variables
features = [col for col in train_df.columns if col not in ['ticker', 'date', 'close', adj_close_col, 'price_movement']]
features.extend(pca_features)

# Remove duplicates while keeping order
features = list(dict.fromkeys(features))

logging.info(f"🔎 Feature selection completed! Using {len(features)} features.")

# ✅ Step 7: Ensure All Features Exist Before Converting
logging.info("🔄 Validating feature list...")

# Find missing columns
missing_features = [col for col in features if col not in train_df.columns]
if missing_features:
    logging.error(f"❌ Missing expected features in dataset: {missing_features}")
    raise KeyError(f"Missing expected features: {missing_features}")

logging.info("✅ All features are present in the dataset.")

# ✅ Step 8: Convert All Features to Numeric Format
logging.info("🔄 Converting all features to numeric format...")
X_train = train_df[features].apply(pd.to_numeric, errors='coerce').astype(np.float32)
X_test = test_df[features].apply(pd.to_numeric, errors='coerce').astype(np.float32)

# ✅ Step 9: Train XGBoost Model
logging.info("🚀 Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200, 
    max_depth=6, 
    learning_rate=0.05, 
    objective='binary:logistic', 
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# ✅ Step 10: Evaluate Model Performance
logging.info("📊 Evaluating model performance...")
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"📈 XGBoost Accuracy: {accuracy:.4f}")

# ✅ Step 11: Save Model
logging.info("💾 Saving trained model...")
joblib.dump((xgb_model, features), "best_trading_model.pkl")
logging.info("✅ XGBoost model saved as best_trading_model.pkl")

# ✅ Step 12: Define Prediction Function
def predict_signals(data):
    """Function to predict trading signals from the trained XGBoost model."""
    if data.empty:
        logging.warning("⚠️ No data provided for prediction.")
        return pd.DataFrame({"Error": ["No data available for prediction."]})
    
    # Load model at prediction time
    model, feature_list = joblib.load("best_trading_model.pkl")

    # ✅ Ensure the data only contains required columns
    missing_features = [col for col in feature_list if col not in data.columns]
    if missing_features:
        logging.error(f"❌ Missing expected features in provided data: {missing_features}")
        return pd.DataFrame({"Error": [f"Missing features: {missing_features}"]})

    data = data[feature_list]
    data = data.apply(pd.to_numeric, errors='coerce').astype(np.float32)

    predictions = model.predict(data)
    
    data["Predicted Signal"] = predictions
    return data
