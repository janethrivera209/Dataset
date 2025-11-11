import io
import os
import base64
import matplotlib
matplotlib.use('Agg')  # Para que no intente usar la pantalla

import matplotlib.pyplot as plt
import pandas as pd
from django.shortcuts import render
from sklearn.model_selection import train_test_split
import arff
from django.conf import settings

# --- Cargar dataset ---
def load_kdd_dataset():
    data_path = os.path.join(settings.BASE_DIR, "datasets", "KDDTrain+.arff")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo {data_path}")
    
    with open(data_path, 'r') as file:
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

    sizes = {
        "total": len(df),
        "train": len(train),
        "val": len(val),
        "test": len(test)
    }

    protocol_counts = df['protocol_type'].value_counts().to_dict()

    # Crear gráficos
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
        plt.close(fig)
        figs.append(graphic)

    return figs, sizes, protocol_counts

# --- Vista principal ---
def index(request):
    try:
        df = load_kdd_dataset()
        charts, sizes, protocol_counts = generate_plots(df)
        return render(request, "index.html", {
            "charts": charts,
            "sizes": sizes,
            "protocol_counts": protocol_counts
        })
    except Exception as e:
        return render(request, "index.html", {
            "error": str(e)
        })
