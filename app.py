from flask import Flask, request, render_template
from waitress import serve
import pickle
import numpy as np

# Last inn den lagrede modellen
model = pickle.load(open('best_model.pkl', 'rb'))

# Initialiser Flask-applikasjonen
app = Flask(__name__)

# Rute for hjemmesiden som laster inn HTML-skjemaet
@app.route('/')
def home():
    return render_template('index.html')

# Rute for å håndtere prediksjon
@app.route('/predict', methods=['POST'])
def predict():
    # Hent data fra skjemaet
    features = [
        float(request.form['fysiologisk_score']),
        float(request.form['apache_fysiologisk_score']),
        float(request.form['overlevelsesestimat_2mnd']),
        float(request.form['lege_overlevelsesestimat_6mnd']),
        float(request.form['sykdom_underkategori_ARF_MOSF']),
        float(request.form['alder']),
        float(request.form['blodtrykk']),
        float(request.form['lungefunksjon']),
        float(request.form['serumalbumin']),
        float(request.form['kreatinin']),
        float(request.form['blod_ph']),
        float(request.form['gjennomsnitt_overlevelsesestimat_estimat_1']),
        float(request.form['gjennomsnitt_overlevelsesestimat_estimat_2']),
        float(request.form['helsetilstand']),
    ]
    
    # Generer prediksjon
    prediction = model.predict([features])[0]

    # Send prediksjonen tilbake til index.html
    return render_template(
        'index.html',
        prediction_text=f'Predikert verdi: {prediction:.2f}'
    )

# Kjør serveren lokalt
if __name__ == '__main__':
    print("http://localhost:8080")
    serve(app, host='0.0.0.0', port=8080)
