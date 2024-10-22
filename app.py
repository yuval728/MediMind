import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load the model
artifacts_dir = 'artifacts/'
data_dir = 'dataset/'

model = joblib.load(artifacts_dir + 'DecisionTreeClassifier.pkl')
st.session_state.model = model

label_encoder = joblib.load(artifacts_dir + 'label_encoder.pkl')
st.session_state.label_encoder = label_encoder

with open(artifacts_dir + 'model_inputs.txt', 'r') as file:
    model_input = file.read().strip().split('\n')
st.session_state.model_input = model_input

# Load the data
sym_des = pd.read_csv(f"{data_dir}symtoms_df.csv", index_col = 0)
precautions = pd.read_csv(f"{data_dir}precautions_df.csv", index_col = 0)
workout = pd.read_csv(f"{data_dir}workout_df.csv", index_col = 0)
description = pd.read_csv(f"{data_dir}description.csv")
medications = pd.read_csv(f"{data_dir}medications.csv")
diets = pd.read_csv(f"{data_dir}diets.csv")


def get_description(disease):
    return description[description['Disease'] == disease]['Description'].values

def get_precautions(disease):
    return precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values

def get_workout(disease):
    return workout[workout['disease'] == disease]['workout'].values

def get_medications(disease):
    return medications[medications['Disease'] == disease]['Medication'].values

def get_diets(disease):
    return diets[diets['Disease'] == disease]['Diet'].values


def predict_disease(symptoms, model,  le):
    prediction = model.predict(symptoms)
    disease = le.inverse_transform(prediction)
    return disease[0]
    
def get_recommendations(disease):
    return {
        'description': get_description(disease)[0],
        'precautions': get_precautions(disease)[0],
        'workout': get_workout(disease)[0],
        'medications': get_medications(disease)[0],
        'diets': get_diets(disease)[0]
    }
    
def get_symptoms(symptoms, model_inputs):
    # symptoms = symptoms.split(',')
    symptoms = [1 if item in symptoms else 0 for item in model_inputs]

    return pd.DataFrame([symptoms], columns = model_inputs)
    


def main():
    st.title("Medicine Recommendation System")
    
    symtoms = st.multiselect("Select the symtoms",st.session_state.model_input)
    symptoms = get_symptoms(symtoms, st.session_state.model_input)
    
    if st.button("Predict"):
        disease = predict_disease(symptoms, st.session_state.model, st.session_state.label_encoder)
        recommendations = get_recommendations(disease)
        st.subheader(f"Prediction is: {disease}")
        st.subheader("Recommendations")
        st.write(f"Description: {recommendations['description']}")
        st.write(f"Precautions: {recommendations['precautions']}")
        st.write(f"Workout: {recommendations['workout']}")
        st.write(f"Medications: {recommendations['medications']}")
        st.write(f"Diets: {recommendations['diets']}")
        
if __name__ == '__main__':
    main()
    
    