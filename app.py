from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

#Load model
model = pickle.load(open('rf_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


def predict(model, scaler, male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes,  totChol, sysBP, diaBP, BMI, heartRate, glucose):
    # Encode categorical variables
    male_encoded = 1 if male and male.lower() == "male" else 0
    currentSmoker_encoded = 1 if currentSmoker and currentSmoker.lower() == "yes" else 0
    BPMeds_encoded = 1 if BPMeds and BPMeds.lower() == "yes" else 0
    prevalentStroke_encoded = 1 if prevalentStroke and prevalentStroke.lower() == "yes" else 0
    prevalentHyp_encoded = 1 if prevalentHyp and prevalentHyp.lower() == "yes" else 0
    diabetes_encoded = 1 if diabetes and diabetes.lower() == "yes" else 0


    # Prepare features array
    features = np.array([[male_encoded, age, currentSmoker_encoded, cigsPerDay, BPMeds_encoded, prevalentStroke_encoded,  prevalentHyp_encoded, diabetes_encoded, totChol, sysBP, diaBP, BMI, heartRate, glucose]])

    # Scale the features
    scaled_features = scaler.transform(features)

    # Predict using the model
    result = model.predict(scaled_features)

    return result[0]

@app.route('/')
def index():
    return render_template("index.html")   # render a template

@app.route('/pred', methods=['POST','GET'])
def pred():
    if request.method == 'POST':
        male = request.form.get('male')
        age = int(request.form['age'])
        currentSmoker = request.form['currentSmoker']
        cigsPerDay = float(request.form['cigsPerDay'])
        BPMeds = request.form['BPMeds']
        prevalentStroke = request.form['prevalentStroke']
        prevalentHyp = request.form['prevalentHyp']
        diabetes = request.form['diabetes']
        totChol = float(request.form['totChol'])
        sysBP = float(request.form['sysBP'])
        diaBP = float(request.form['diaBP'])
        BMI = float(request.form['BMI'])
        heartRate = float(request.form['heartRate'])
        glucose = float(request.form['glucose'])

        prediction = predict( model, scaler ,male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose)
        prediction_text = "The Patient has Heart Disease" if prediction == 1 else "The Patient has No Heart Disease"
        
        return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)