from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler #se plantea usar para conversión
from sklearn.linear_model import LogisticRegression
import pandas as pd

#cargar los datos en un dataset
#iris = datasets.load_iris()
df = pd.read_csv("data/heart_disease_dataset.csv")

# Definir variables categóricas nominales y ordinales
#nominales = ["Gender", "Smoking", "Alcohol Intake", "Family History", "Obesity", "Exercise Induced Angina", "Chest Pain Type"]
nominales = ["Gender", "Family History", "Obesity", "Exercise Induced Angina", "Diabetes", "Alcohol Intake","Smoking"]
#ordinales = ["Diabetes"]
ordinales = ["Chest Pain Type"]

#establecer el orden a las variables ordinales
ordinal_encoder = OrdinalEncoder(categories=[
    #["Never", "Former", "Current"],             # Smoking
    #["None", "Moderate", "Heavy"],              # Alcohol Intake
    ["Non-anginal Pain", "Asymptomatic", "Atypical Angina", "Typical Angina"]  # Chest Pain Type
])

# Transformadores
preprocesador = ColumnTransformer([
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), nominales), #170425
    ("ordinal", ordinal_encoder, ordinales), #170425
    ("scaler", StandardScaler(), ["Age", "Cholesterol", "Blood Pressure", "Heart Rate", "Exercise Hours", "Stress Level", "Blood Sugar"])
])

# Crear el pipeline general
pipeline = Pipeline(steps=[
    ('preprocessor', preprocesador),
    ('classifier', LogisticRegression())])  # Se definirá el modelo posteriormente

# Aplicar transformación
#df["Alcohol Intake"] = df["Alcohol Intake"].fillna("None") #190425
X = df.drop(columns=["Heart Disease"])  # Variables ind
y = df["Heart Disease"]  # Variable objetivo


#separar los datos en entrenamiento y prueba
#x_train, x_test, y_train, y_test = train_test_split(X, y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#entrenar modelos y SVC optimizado
pipeline.set_params(classifier=LinearRegression())
lin_regr = pipeline.fit(x_train, y_train)
 
pipeline.set_params(classifier=LogisticRegression())
log_regr = pipeline.fit(x_train, y_train)

pipeline.set_params(
    classifier=SVC(kernel='rbf', gamma='scale', C=4.6415888336127775)
)
svc_mo = pipeline.fit(x_train, y_train)

with open('lin_reg.pkl', 'wb') as li:
    pickle.dump(lin_regr, li)

with open('log_reg.pkl', 'wb') as lo:
    pickle.dump(log_regr, lo)

with open('svc_m.pkl', 'wb') as sv:
    pickle.dump(svc_mo, sv)
