from flask_sqlalchemy import SQLAlchemy
import pdfkit
from sqlalchemy.exc import IntegrityError
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
from flask_mail import Mail, Message
import random
import string
from werkzeug.utils import secure_filename
import os
import plotly.express as px
import numpy as np
# from tensorflow.keras.preprocessing import image
# import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np,pandas as pd
import os
import csv
from dotenv import load_dotenv
# import pdfkit
from reportlab.pdfgen import canvas
from io import BytesIO
from flask.helpers import send_file
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

    
# ============================================================ model ============================================================ 


data = pd.read_csv(os.path.join("static","Data", "Training.csv"))
df = pd.DataFrame(data)
cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

dt = DecisionTreeClassifier()
clf_dt=dt.fit(x_train,y_train)

indices = [i for i in range(132)]
symptoms = df.columns.values[:-1]

dictionary = dict(zip(symptoms,indices))

def predict(symptom):
    user_input_symptoms = symptom
    user_input_label = [0 for i in range(132)]
    for i in user_input_symptoms:
        idx = dictionary[i]
        user_input_label[idx] = 1

    user_input_label = np.array(user_input_label)
    user_input_label = user_input_label.reshape((-1, 1)).transpose()

    # Get probabilities for all diseases
    probabilities = dt.predict_proba(user_input_label)[0]
    # Get indices of top 4 predictions sorted by probability
    top_4_indices = np.argsort(probabilities)[-4:][::-1]
    
    # Get the maximum probability for normalization
    max_prob = np.max(probabilities)
    
    predictions = []
    for idx in top_4_indices:
        disease = dt.classes_[idx]
        # Calculate relative confidence score
        confidence = (probabilities[idx] / max_prob) * 100
        predictions.append({
            'disease': disease,
            'confidence': round(float(confidence), 2)  # Round to 2 decimal places
        })
    
    return predictions

with open('static/Data/Testing.csv', newline='') as f:
        reader = csv.reader(f)
        symptoms = next(reader)
        symptoms = symptoms[:len(symptoms)-1]

# ============================================================ scans ============================================================ 
    
@app.route('/disease_predict', methods=['POST'])
def disease_predict():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        if not symptoms or len(symptoms) != 5:
            return jsonify({'error': 'Please provide exactly 5 symptoms'}), 400
            
        predictions = predict(symptoms)
        
        return jsonify({
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_symptoms')
def get_symptoms():
    try:
        # Get symptoms from the model's dictionary
        symptoms_list = list(dictionary.keys())
        return jsonify({'symptoms': symptoms_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)