import numpy as np
import pandas as pd
import pickle

from flask import Flask, request, jsonify, render_template
from waitress import serve

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
        float(request.form['feature1']),
        float(request.form['feature2']),
        # Legg til flere input-felter hvis modellen din bruker flere funksjoner
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
    serve(app, host='0.0.0.0', port=8080)
