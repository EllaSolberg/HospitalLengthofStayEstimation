from flask import Flask, request, render_template
from waitress import serve
import pickle
import numpy as np
import pandas as pd

with open('model_with_imputer.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

model = loaded_data['model']
imputer = loaded_data['imputer']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
    'antall_komorbiditeter': float(1.840850),
    'koma_score': float(11.579310),
    'fysiologisk_score': float(24.967973),
    'apache_fysiologisk_score': float(request.form['apache_fysiologisk_score']),
    'overlevelsesestimat_2mnd': float(request.form['overlevelsesestimat_2mnd']),
    'overlevelsesestimat_6mnd': float(0.535006),
    'diabetes': float(0.213793),
    'demens': float(0.037931),
    'lege_overlevelsesestimat_2mnd': float(0.631950),
    'lege_overlevelsesestimat_6mnd': float(request.form['lege_overlevelsesestimat_6mnd']),
    'alder': float(request.form['alder']),
    'utdanning': float(11.756316),
    'inntekt': float(0.353856),
    'blodtrykk': float(request.form['blodtrykk']),
    'hvite_blodlegemer': float(11.553840),
    'hjertefrekvens': float(96.412931),
    'respirasjonsfrekvens': float(22.766379),
    'kroppstemperatur': float(37.070798),
    'lungefunksjon': float(request.form['lungefunksjon']),
    'serumalbumin': float(request.form['serumalbumin']),
    'kreatinin': float(1.721019),
    'natrium': float(137.439655),
    'blod_ph': float(request.form['blod_ph']),
    'glukose': float(135.306539),
    'blodurea_nitrogen': float(13.360405),
    'urinmengde': float(2389.054310),
    'sykdom_underkategori_ARF/MOSF w/Sepsis': float(request.form['sykdom_underkategori_ARF/MOSF w/Sepsis']),
    'sykdom_underkategori_CHF': float(0.156034),
    'sykdom_underkategori_COPD': float(0.112931	),
    'sykdom_underkategori_Cirrhosis': float(0.053448),
    'sykdom_underkategori_Colon Cancer': float(0.062069),
    'sykdom_underkategori_Coma': float(0.062069),
    'sykdom_underkategori_Lung Cancer': float(0.094828),
    'sykdom_underkategori_MOSF w/Malig': float(0.087069),
    'kreft_metastatic': float(0.206034),
    'kreft_no': float(0.656034),
    'kreft_yes': float(0.137931),
    'kjønn_female': float(0.439655),
    'kjønn_male': float(0.560345),
    'etnisitet_asian': float(0.002586),
    'etnisitet_black': float(0.137069),
    'etnisitet_hispanic': float(0.035345),
    'etnisitet_other': float(0.013793),
    'etnisitet_white': float(0.807759),
    'etnisitet_nan': float(0.003448),
    'tidlig_overlevelsesestimat_gjennomsnitt': float(request.form['tidlig_overlevelsesestimat_gjennomsnitt']),
    'sen_overlevelsesestimat_gjennomsnitt': float(request.form['sen_overlevelsesestimat_gjennomsnitt']),
    'helsetilstand': float(request.form['helsetilstand'])
}

    
    input_df = pd.DataFrame([data])

    input_df_imputed = imputer.set_output(transform = "pandas").transform(input_df)

    prediction = model.predict(input_df_imputed)[0]

    return render_template(
        'index.html',
        prediction_text=f'Predikert verdi: {prediction:.2f}'
    )

if __name__ == '__main__':
    print("http://localhost:8080")
    serve(app, host='0.0.0.0', port=8080)

