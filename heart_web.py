#importar librerias
import streamlit as st
import pickle
import pandas as pd

#Extrar los archivos pickle
with open('lin_reg.pkl', 'rb') as li:
    lin_reg = pickle.load(li)

with open('log_reg.pkl', 'rb') as lo:
    log_reg = pickle.load(lo)

with open('svc_m.pkl', 'rb') as sv:
    svc_m = pickle.load(sv)


#funcion para clasificar las plantas 
def classify(num):
    if num == 0:
        return 'HEART DISEASE = NO'
    elif num == 1:
        return 'HEART DISEASE = YES'
    else:
        return 'NO ONE'

def main():
    #titulo
    st.title('Modelamiento de Heart Disease')
    #titulo de sidebar
    st.sidebar.header('User Input Parameters')

    #funcion para poner los parametros en el sidebar
    
    def user_input_parameters():
        age = st.sidebar.slider('Age', 18, 100, 50)
        gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
        cholesterol = st.sidebar.slider('Cholesterol', 100, 400, 200)
        blood_pressure = st.sidebar.slider('Blood Pressure', 80, 200, 120)
        heart_rate = st.sidebar.slider('Heart Rate', 40, 200, 70)
        smoking = st.sidebar.selectbox('Smoking', ['Never', 'Former', 'Current'])
        alcohol = st.sidebar.selectbox('Alcohol Intake', ['None', 'Moderate', 'Heavy'])
        exercise_hours = st.sidebar.slider('Exercise Hours per Week', 0, 20, 3)
        family_history = st.sidebar.selectbox('Family History of Heart Disease', ['Yes', 'No'])
        diabetes = st.sidebar.selectbox('Diabetes', ['Yes', 'No'])
        obesity = st.sidebar.selectbox('Obesity', ['Yes', 'No'])
        stress_level = st.sidebar.slider('Stress Level (1–10)', 1, 10, 5)
        blood_sugar = st.sidebar.slider('Blood Sugar', 50, 300, 100)
        exercise_angina = st.sidebar.selectbox('Exercise Induced Angina', ['Yes', 'No'])
        chest_pain_type = st.sidebar.selectbox('Chest Pain Type', [
            'Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'
        ])

        data = {
            'Age': age,
            'Gender': gender,
            'Cholesterol': cholesterol,
            'Blood Pressure': blood_pressure,
            'Heart Rate': heart_rate,
            'Smoking': smoking,
            'Alcohol Intake': alcohol,
            'Exercise Hours': exercise_hours,
            'Family History': family_history,
            'Diabetes': diabetes,
            'Obesity': obesity,
            'Stress Level': stress_level,
            'Blood Sugar': blood_sugar,
            'Exercise Induced Angina': exercise_angina,
            'Chest Pain Type': chest_pain_type,
        }

        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()

    #escoger el modelo preferido
    #option = ['Linear Regression', 'Logistic Regression', 'SVM']
    option = ['SVM']

    model = st.sidebar.selectbox('El modelo a usar es: ', option)

    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'Linear Regression':
            st.success(classify(lin_reg.predict(df)))
        elif model == 'Logistic Regression':
            st.success(classify(log_reg.predict(df)))
        else:
            st.success(classify(svc_m.predict(df)))


if __name__ == '__main__':
    main()
    
