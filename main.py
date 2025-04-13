from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

# Autoriser accès frontend plus tard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Répertoire de stockage
UPLOAD_DIR = "../data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Détection automatique des colonnes
def detect_column(columns, keywords):
    for col in columns:
        for keyword in keywords:
            if keyword.lower() in col.lower():
                return col
    return None

# ✅ Endpoint : Upload de fichier
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        df = pd.read_csv(file_location)
        columns = df.columns.tolist()

        cycle_col = detect_column(columns, ["cycle", "index"])
        capacity_col = detect_column(columns, ["capacity", "cap", "q_discharge"])

        if not cycle_col or not capacity_col:
            return {"error": "Impossible de détecter les colonnes 'cycle' et 'capacity'."}

        return {
            "filename": file.filename,
            "columns": columns,
            "cycle_column": cycle_col,
            "capacity_column": capacity_col,
            "preview": df[[cycle_col, capacity_col]].head(5).to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}

# ✅ Endpoint : Prédiction automatique
@app.post("/predict")
def predict_capacity(file_name: str, cycle_column: str, capacity_column: str, model_type: str = "linear"):
    try:
        file_path = os.path.join(UPLOAD_DIR, file_name)
        df = pd.read_csv(file_path)

        df = df[[cycle_column, capacity_column]].dropna()
        df[capacity_column] = df[capacity_column].rolling(window=5, center=True).mean()
        df = df.dropna()

        X = np.array(df[cycle_column]).reshape(-1, 1)
        y = np.array(df[capacity_column]).reshape(-1, 1)

        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "forest":
            model = RandomForestRegressor(n_estimators=100, random_state=0)
        else:
            return {"error": "Modèle non supporté. Choisir 'linear' ou 'forest'."}

        model.fit(X, y.ravel())
        future_cycles = np.arange(X.min(), X.max() + 50).reshape(-1, 1)
        y_pred = model.predict(future_cycles)

        preview = {
            "cycle": future_cycles.flatten().tolist(),
            "predicted_capacity": y_pred.flatten().tolist()
        }

        return {
            "model_used": model_type,
            "original_file": file_name,
            "cycle_column": cycle_column,
            "capacity_column": capacity_column,
            "predicted": preview
        }

    except Exception as e:
        return {"error": str(e)}

PLOT_DIR = "../data/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

@app.get("/plot")
def generate_plot(file_name: str, cycle_column: str, capacity_column: str, model_type: str = "linear"):
    try:
        file_path = os.path.join(UPLOAD_DIR, file_name)
        df = pd.read_csv(file_path)

        df = df[[cycle_column, capacity_column]].dropna()
        df[capacity_column] = df[capacity_column].rolling(window=5, center=True).mean()
        df = df.dropna()

        X = np.array(df[cycle_column]).reshape(-1, 1)
        y = np.array(df[capacity_column]).reshape(-1, 1)

        # Modèle ML
        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "forest":
            model = RandomForestRegressor(n_estimators=100, random_state=0)
        else:
            return {"error": "Modèle non supporté."}

        model.fit(X, y.ravel())
        future_cycles = np.arange(X.min(), X.max() + 50).reshape(-1, 1)
        y_pred = model.predict(future_cycles)

        # Tracer
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X.flatten(), y=y.flatten(), mode='lines+markers', name="Réel"))
        fig.add_trace(go.Scatter(x=future_cycles.flatten(), y=y_pred.flatten(), mode='lines', name="Prédiction", line=dict(dash='dot')))
        fig.update_layout(title="Prédiction de la capacité", xaxis_title="Cycle", yaxis_title="Capacité (Ah)")

        plot_path = os.path.join(PLOT_DIR, "graph.html")
        fig.write_html(plot_path)

        return FileResponse(plot_path, media_type='text/html')

    except Exception as e:
        return {"error": str(e)}

