import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # or any other model
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# Load your preprocessed medical dataset
df = pd.read_csv(r"C:\Users\bhara\Desktop\HEART_DISEASE_PREDICTOR\medical_data_4000.csv")
# print(df.head())
# print(df.isnull().sum())
# print(df.shape)
df.drop('education', axis=1, inplace=True)
# print(df['TenYearCHD'].value_counts())
# Separate majority and minority classes
df_majority = df[df.TenYearCHD == 0]
df_minority = df[df.TenYearCHD == 1]    
# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                    replace=True,     # sample with replacement
                                    n_samples=2000,    # to match majority class
                                    random_state=42) # reproducible results
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
# Display new class counts
# print(df_upsampled['TenYearCHD'].value_counts())
# Separate features (X) and target label (y)
X = df_upsampled.drop('TenYearCHD', axis=1)    
y = df_upsampled['TenYearCHD']  # target variable
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
# Predict and evaluate
y_pred = rf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# # Save the trained model for later deployment
import pickle
pickle.dump(rf, open(r"C:\Users\bhara\Desktop\HEART_DISEASE_PREDICTOR\models\disease_model.pkl", 'wb'))
pickle.dump(scaler, open(r"C:\Users\bhara\Desktop\HEART_DISEASE_PREDICTOR\models\scaler.pkl", 'wb'))
# You can also try other models
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print(f"{name} Accuracy: {accuracy_score(y_test, y_pred)}")
    # print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))
    # print(f"{name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# append result to dataframe
results = []
for name, clf in classifiers.items():
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))
results_df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
# print(results_df)

# print(rf.predict(X_test[:5]))  # Example prediction on first 5 test samples
X_test = X_test[10].reshape(-1, 1)  # Reshape a single test sample
# print(rf.predict(X_test.T))  # Predict on the reshaped sample
#  TEST 1
# print("predicted class :" , rf.predict(X_test.reshape(1, -1))[0])
# print("actual class :" , y_test.iloc[10])


# load random classifier model
with open(r"C:\Users\bhara\Desktop\HEART_DISEASE_PREDICTOR\models\disease_model.pkl", 'rb') as f:
    loaded_model = pickle.load(f)

# load scaler model
with open(r"C:\Users\bhara\Desktop\HEART_DISEASE_PREDICTOR\models\scaler.pkl", 'rb') as f:
    loaded_scaler = pickle.load(f)


# Predict using the loaded model
def predict_disease(gender, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose):
    gender_encoded = 1 if gender.lower() == "male" else 0
    currentSmoker_encoded = 1 if currentSmoker.lower() == "yes" else 0
    BPMeds_encoded = 1 if BPMeds.lower() == "yes" else 0
    prevalentStroke_encoded = 1 if prevalentStroke.lower() == "yes" else 0
    prevalentHyp_encoded = 1 if prevalentHyp.lower() == "yes" else 0
    diabetes_encoded = 1 if diabetes.lower() == "yes" else 0
    feature = np.array([[gender_encoded, age, currentSmoker_encoded, cigsPerDay, BPMeds_encoded, prevalentStroke_encoded,
                         prevalentHyp_encoded, diabetes_encoded, totChol, sysBP, diaBP, BMI, heartRate, glucose]])
    feature_scaled = loaded_scaler.transform(feature)

    predicted_class = loaded_model.predict(feature_scaled)

    message = "High risk of heart disease" if predicted_class[0] == 1 else "Low risk of heart disease"
    return message, predicted_class[0]
    

# Example usage
result = predict_disease("male", 55, "no", 20, "no", "no", "no", "no", 240, 140, 90, 28.5, 80, 100)
message, prediction = result
print(f"Prediction: {message} (class: {prediction})")
