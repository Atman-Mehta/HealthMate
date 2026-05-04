# HealthMate — AI-Powered Healthcare Platform

## Overview
Built an integrated AI healthcare platform addressing 
the lack of accessible diagnostic tools in India. 
Combines disease prediction, lung cancer detection 
from medical imaging, and intelligent patient assistance 
in one unified system.

## Results
- 87% accuracy on disease prediction
- 82% sensitivity on lung cancer detection
- 14,000+ medical records processed
- IBM Watson chatbot handling 200+ patient query intents
- Prioritized Recall/Sensitivity over accuracy — 
  missing a sick patient is far more dangerous 
  than a false alarm

## Platform Components

### 1. Disease Prediction
- RandomForestClassifier trained on structured 
  symptom data
- Multi-class classification across common diseases
- Feature importance analysis to identify key symptoms

### 2. Lung Cancer Detection
- TensorFlow CNN trained on CT scan images
- Binary classification — malignant vs benign
- Optimized for sensitivity — catching positive 
  cases is critical in cancer diagnosis
- Image preprocessing and augmentation pipeline

### 3. Patient Assistant Chatbot
- IBM Watson powered conversational agent
- Handles 200+ patient query intents
- Symptom checking, appointment guidance, 
  medication reminders

## Why Sensitivity Over Accuracy?
In medical diagnosis, a false negative — missing 
a sick patient — is far more dangerous than a 
false positive. HealthMate specifically optimizes 
for Recall/Sensitivity to minimize missed diagnoses, 
even at the cost of some precision.

## Tech Stack
Python, TensorFlow, Keras, Scikit-learn, 
IBM Watson, Pandas, NumPy, Matplotlib
