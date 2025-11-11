# views.py
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Evita abrir ventanas gráficas en servidores
import matplotlib.pyplot as plt
import pandas as pd
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from django.conf import settings
import arff

# --- Cargar dataset ---
def load_kdd_dataset():
    try:
        # datasets está al mismo nivel que dataset_project/
        data_path = settings.BASE_DIR / "datasets" / "KDDTrain+.arff"
        with open(data_path, 'r') as file:
            dataset = arff.load(file)  # liac-arff usa .load()
            attributes = [attr[0] for attr in dataset["attributes"]]
            df = pd.DataFrame(dataset["data"], columns=attributes)
        return df, None
    except Exception as e:
        return None, str(e)

# --- Generar gráficas ---
def generate_plots(df):
    # Asegurarse que las columnas categóricas sean strings
    if isinstance(df['protocol_type'].iloc[0], bytes):
        df['protocol_type'] = df['protocol_type'].apply(lambda x: x.decode() if isinstance(x, bytes) else x)

    # Dividir en train / val / test
    train, test = train_test_split(df, test_size=0.4, random_state=42, stratify=df['protocol_type'])
    val, test = train_test_split(test, test_size=0.5, random_state=42, stratify=test['protocol_type'])

    # Crear gráficas
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

        # Convertir gráfico a base64
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
    df, error = load_kdd_dataset()
    if error:
        return render(request, "index.html", {"error": error, "charts": []})

    charts = generate_plots(df)
    return render(request, "index.html", {"charts": charts, "error": None})
