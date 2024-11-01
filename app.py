from flask import Flask, request, render_template
from waitress import serve
import pickle
import numpy as np
import pandas as pd

# Load the saved model and imputer
with open('model_with_imputer.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

model = loaded_data['model']
imputer = loaded_data['imputer']

# Initialize the Flask application
app = Flask(__name__)

# Route for the homepage that loads the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from the form and store it in a DataFrame
    data = {
        'fysiologisk_score': float(request.form['fysiologisk_score']),
        'apache_fysiologisk_score': float(request.form['apache_fysiologisk_score']),
        'overlevelsesestimat_2mnd': float(request.form['overlevelsesestimat_2mnd']),
        'lege_overlevelsesestimat_6mnd': float(request.form['lege_overlevelsesestimat_6mnd']),
        'alder': float(request.form['alder']),
        'blodtrykk': float(request.form['blodtrykk']),
        'lungefunksjon': float(request.form['lungefunksjon']),
        'serumalbumin': float(request.form['serumalbumin']),
        'blod_ph': float(request.form['blod_ph']),
        'sykdom_underkategori_ARF/MOSF w/Sepsis': float(request.form['sykdom_underkategori_ARF/MOSF w/Sepsis']),
        'tidlig_overlevelsesestimat_gjennomsnitt': float(request.form['tidlig_overlevelsesestimat_gjennomsnitt']),
        'sen_overlevelsesestimat_gjennomsnitt': float(request.form['sen_overlevelsesestimat_gjennomsnitt']),
        'helsetilstand': float(request.form['helsetilstand']),
    }
    
    # Convert data to DataFrame
    input_df = pd.DataFrame([data])

    # Impute missing values
    input_df_imputed = imputer.transform(input_df)

    # Generate prediction
    prediction = model.predict(input_df_imputed)[0]

    # Send the prediction back to index.html
    return render_template(
        'index.html',
        prediction_text=f'Predikert verdi: {prediction:.2f}'
    )

# Run the server locally
if __name__ == '__main__':
    print("http://localhost:8080")
    serve(app, host='0.0.0.0', port=8080)
