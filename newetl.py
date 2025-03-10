import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_data(selected_ticker=None):
    """
    Extract, Transform, and Load (ETL) pipeline for financial data.
    Processes historical stock prices, company info, and optional financials.
    Optimized for machine learning.

    :param selected_ticker: If provided, processes only this company's data first.
    :return: Processed dataset
    """
    try:
        logging.info("📂 Loading datasets...")

        # ✅ Step 1: Extract Data - Load CSVs
        shareprices_df = pd.read_csv("us-shareprices-daily.csv", sep=";")
        companies_df = pd.read_csv("us-companies.csv", sep=";")
        income_df = pd.read_csv("us-income-quarterly.csv", sep=";")  # Optional

        # ✅ Step 2: Standardize column names
        shareprices_df.columns = shareprices_df.columns.str.lower()
        companies_df.columns = companies_df.columns.str.lower()
        income_df.columns = income_df.columns.str.lower()

        logging.info("✅ Data loaded successfully!")

        # ✅ Step 3: Ensure required columns exist
        required_columns = {"ticker", "date", "close", "adj. close", "volume"}
        missing_columns = required_columns - set(shareprices_df.columns)
        if missing_columns:
            raise KeyError(f"❌ Missing columns in share prices dataset: {missing_columns}")

        # ✅ Step 4: Convert Date Column to Datetime
        shareprices_df["date"] = pd.to_datetime(shareprices_df["date"], errors="coerce")

        # ✅ Step 5: Convert numeric columns to proper format
        numeric_cols = ["open", "high", "low", "close", "adj. close", "volume"]
        for col in numeric_cols:
            if col in shareprices_df.columns:
                shareprices_df[col] = pd.to_numeric(shareprices_df[col], errors="coerce")

        # ✅ Step 6: Handle Missing Values
        shareprices_df.dropna(subset=["ticker", "close"], inplace=True)
        shareprices_df["dividend"] = shareprices_df["dividend"].fillna(0)
        shareprices_df["shares outstanding"] = shareprices_df["shares outstanding"].fillna(
            shareprices_df["shares outstanding"].median()
        )

        companies_df["industryid"] = companies_df["industryid"].fillna("Unknown Industry")
        companies_df["isin"] = companies_df["isin"].fillna("Unknown")

        # Drop unnecessary columns
        companies_df.drop(columns=["business summary", "number employees", "cik"], inplace=True, errors="ignore")

        logging.info("✅ Missing values handled successfully!")

        # ✅ Step 7: Process One Company First (If Required)
        if selected_ticker:
            shareprices_df = shareprices_df[shareprices_df["ticker"] == selected_ticker]
            logging.info(f"🔍 Processing only selected company: {selected_ticker}")

        # ✅ Step 8: Find Top 10 Most Traded Stocks (Expand After Testing One)
        top_tickers = (
            shareprices_df.groupby("ticker")["volume"]
            .sum()
            .nlargest(10)  # Select top 10 tickers by total traded volume
            .index
        )

        logging.info(f"📊 Top 10 Most Traded Companies: {list(top_tickers)}")

        # ✅ Step 9: Filter dataset to keep only those 10 tickers
        shareprices_df = shareprices_df[shareprices_df["ticker"].isin(top_tickers)]

        # ✅ Step 10: Merge Data with Company Info --- issue review
        merged_df = shareprices_df.merge(companies_df, on="ticker", how="left")
        logging.info(f"✅ Data merged successfully! New dataset shape: {merged_df.shape}")

        # ✅ Step 11: Feature Engineering
        logging.info("⚙️ Applying feature engineering...")

        # Ensure 'adj. close' has no missing values before applying rolling calculations
        merged_df["adj. close"] = merged_df["adj. close"].ffill()
        merged_df["adj. close"] = merged_df["adj. close"].bfill()

        # Apply moving averages and volatility
        merged_df["ma_5"] = merged_df["adj. close"].rolling(window=5, min_periods=1).mean()
        merged_df["ma_20"] = merged_df["adj. close"].rolling(window=20, min_periods=1).mean()
        merged_df["volatility_10"] = merged_df["adj. close"].rolling(window=10, min_periods=1).std()

        merged_df[["ma_5", "ma_20", "volatility_10"]] = merged_df[["ma_5", "ma_20", "volatility_10"]].bfill()

        logging.info("✅ Feature engineering completed!")

        # ✅ Step 12: Add Target Variable (Price Movement)
        logging.info("📊 Creating classification target variable (Price Movement)...")
        merged_df["price_movement"] = (merged_df["adj. close"].shift(-1) > merged_df["adj. close"]).astype(int)

        logging.info("✅ Target variable created!")

        # ✅ Step 13: Train-Test Split
        logging.info("📊 Splitting dataset into training and testing sets...")
        train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42, shuffle=False)
        logging.info(f"✅ Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        # ✅ Step 14: Apply PCA (Optional)
        numeric_features = train_df.select_dtypes(include=["float32", "float64"]).columns.tolist()

        if len(numeric_features) >= 5:
            logging.info("🧪 Applying PCA for feature reduction...")
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_df[numeric_features].fillna(0))
            test_scaled = scaler.transform(test_df[numeric_features].fillna(0))

            pca = PCA(n_components=5)
            train_pca = pca.fit_transform(train_scaled)
            test_pca = pca.transform(test_scaled)

            pca_columns = [f"pca_{i+1}" for i in range(5)]
            train_pca_df = pd.DataFrame(train_pca, columns=pca_columns, index=train_df.index)
            test_pca_df = pd.DataFrame(test_pca, columns=pca_columns, index=test_df.index)

            train_df = pd.concat([train_df, train_pca_df], axis=1)
            test_df = pd.concat([test_df, test_pca_df], axis=1)

            logging.info("✅ PCA applied successfully!")

        # ✅ Step 15: Save Cleaned Data
        logging.info("💾 Saving cleaned datasets...")
        train_df.to_csv("cleaned_stock_data_train.csv", index=False)
        test_df.to_csv("cleaned_stock_data_test.csv", index=False)
        logging.info("✅ Cleaned train and test datasets saved successfully!")

        return merged_df  # Return processed data for further use

    except Exception as e:
        logging.error(f"❌ Error in processing data: {e}")
        return None

# Run ETL Process
if __name__ == "__main__":
    process_data()
