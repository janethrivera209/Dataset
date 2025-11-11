import io
import base64
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI

import matplotlib.pyplot as plt
import pandas as pd
from django.shortcuts import render
from sklearn.model_selection import train_test_split
import arff
from pathlib import Path

# Ruta del dataset
DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "KDDTrain+.arff"

# --- Cargar dataset ---
def load_kdd_dataset():
    with open(DATA_PATH, 'r') as file:
        dataset = arff.load(file)
        attributes = [attr[0] for attr in dataset["attributes"]]
        df = pd.DataFrame(dataset["data"], columns=attributes)
    return df

# --- Generar gráficas ---
def generate_plots(df):
    if isinstance(df['protocol_type'].iloc[0], bytes):
        df['protocol_type'] = df['protocol_type'].apply(lambda x: x.decode() if isinstance(x, bytes) else x)

    train, test = train_test_split(df, test_size=0.4, random_state=42, stratify=df['protocol_type'])
    val, test = train_test_split(test, test_size=0.5, random_state=42, stratify=test['protocol_type'])

    figs = []
    for data, title in [
        (df, "Dataset completo"),
        (train, "Entrenamiento"),
        (val, "Validación"),
        (test, "Prueba")
    ]:
        fig, ax = plt.subplots()
        data['protocol_type'].value_counts().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel('Protocol Type')
        ax.set_ylabel('Frecuencia')
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        graphic = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        figs.append(graphic)
        plt.close(fig)

    return figs

# --- Vista principal ---
def index(request):
    df = load_kdd_dataset()
    charts = generate_plots(df)
    return render(request, "index.html", {"charts": charts})
