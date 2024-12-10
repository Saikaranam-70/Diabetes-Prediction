import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


diabetes_dataset = pd.read_csv('diabetes.csv')
diabetes_dataset.shape

diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scalar = StandardScaler()
scalar.fit(X)
standardized_data = scalar.transform(X)
# print(standardized_data)

X= standardized_data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# print(X_train.shape, X_test.shape, X.shape)


classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)


X_train_predicton = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predicton, Y_train)

# print("Accuracy Score :", training_data_accuracy)

print("\nEnter the details for prediction:")
try:
    pregnancies = float(input("Number of Pregnancies : "))
    glucose = float(input("Glucose Level (e.g., 120): "))
    blood_pressure = float(input("Blood Pressure (e.g., 80): "))
    skin_thickness = float(input("Skin Thickness (e.g., 25): "))
    insulin = float(input("Insulin Level (e.g., 120 or 0 if unknown): "))
    weight = float(input("Weight in kg (e.g., 70): "))
    height = float(input("Height in meters (e.g., 1.75): "))
    if height <= 0:  
        raise ValueError("Height must be greater than zero.")
    

    bmi = weight / (height ** 2)
    print(f"Calculated BMI: {bmi:.2f}")
    
    diabetes_pedigree_function = float(input("Diabetes Pedigree Function (e.g., 0.587): "))
    age = float(input("Age (e.g., 45): "))


    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    input_data_as_numpy_array = np.array(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)


    std_data = scalar.transform(input_data_reshaped)


    prediction = classifier.predict(std_data)
    
    if prediction[0] == 0:
        print("\nThe person is NOT diabetic.")
    else:
        print("\nThe person iS diabetic.")

except ValueError as e:
    print(f"Invalid input! {e}")