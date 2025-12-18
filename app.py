from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "NoCode ML Backend Running ✅",
        "endpoints": [
            "/upload (POST)",
            "/preprocess (POST)",
            "/split (POST)",
            "/train (POST)"
        ]
    })


# Global dataset and split
dataset = None
X_train = X_test = y_train = y_test = None

@app.route("/upload", methods=["POST"])
def upload():
    global dataset
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        filename = file.filename.lower()
        if filename.endswith(".csv"):
            df = pd.read_csv(file)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400

        dataset = df  # assign to global
        return jsonify({
            "message": "Dataset uploaded successfully ✅",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/preprocess", methods=["POST"])
def preprocess():
    global dataset
    try:
        if dataset is None:
            return jsonify({"error": "Upload dataset first"}), 400

        data = request.get_json(force=True) or {}
        method = data.get("method", "standard")

        # Remove empty columns
        df = dataset.dropna(axis=1, how="all")

        if df.shape[1] < 2:
            return jsonify({"error": "Dataset must have at least 2 columns (features + target)"}), 400

        X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df.iloc[:, -1]

        # Encode target if categorical
        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
        else:
            y = y.values

        # Scaling
        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        dataset = pd.DataFrame(X_scaled, columns=X.columns)
        dataset["label"] = y

        return jsonify({
            "message": "Preprocessing completed ✅",
            "features_used": X.shape[1],
            "rows": X.shape[0]
        })
    except Exception as e:
        print("❌ PREPROCESS ERROR:", e)
        return jsonify({"error": str(e)}), 400
    
@app.route("/split", methods=["POST", "OPTIONS"])
def split():
    global X_train, X_test, y_train, y_test, dataset
    try:
        if dataset is None:
            return jsonify({"error": "Preprocess dataset first"}), 400

        data = request.get_json(silent=True) or {}
        ratio = float(data.get("ratio", 0.8))  # dynamic ratio from frontend

        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        if X.shape[0] < 2:
            return jsonify({"error": "Not enough rows to split"}), 400

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - ratio, random_state=42
        )

        return jsonify({
            "message": "Train-test split completed ✅",
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        })
    except Exception as e:
        print("❌ SPLIT ERROR:", e)
        return jsonify({"error": "Split failed"}), 400

@app.route("/train", methods=["POST", "OPTIONS"])
def train():
    global X_train, X_test, y_train, y_test
    try:
        if X_train is None or y_train is None:
            return jsonify({"error": "Complete all previous steps first"}), 400

        data = request.get_json(silent=True) or {}
        model_type = data.get("model", "logistic")  # dynamic from frontend

        model = LogisticRegression(max_iter=500) if model_type == "logistic" else DecisionTreeClassifier()
        model.fit(X_train, y_train)
        accuracy = round(model.score(X_test, y_test) * 100, 2)

        return jsonify({
            "message": "Training completed ✅",
            "model": model_type,
            "accuracy": accuracy
        })
    except Exception as e:
        print("❌ TRAIN ERROR:", e)
        error_msg = str(e)
        if "samples of at least 2 classes" in error_msg:
            user_error = "Your dataset's target variable has only one unique value. Classification requires at least 2 different classes (e.g., 0 and 1). Please check your data."
        else:
            user_error = "Training failed due to an unexpected error. Please check your data format."
        return jsonify({"error": user_error}), 400



@app.route("/reset", methods=["POST"])
def reset():
    global dataset, X_train, X_test, y_train, y_test
    dataset = None
    X_train = X_test = y_train = y_test = None
    return jsonify({"message": "Analysis reset successfully"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)