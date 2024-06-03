from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
filename = 'Breast_Cancer_Data.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    clump_thickness = float(request.form['clump_thickness'])
    uniformity_of_cell_size = float(request.form['uniformity_of_cell_size'])
    uniformity_of_cell_shape = float(request.form['uniformity_of_cell_shape'])
    marginal_adhesion = float(request.form['marginal_adhesion'])
    single_epithelial_cell_size = float(request.form['single_epithelial_cell_size'])
    bare_nuclei = float(request.form['bare_nuclei'])
    bland_chromatin = float(request.form['bland_chromatin'])
    normal_nucleoli = float(request.form['normal_nucleoli'])
    mitoses = float(request.form['mitoses'])

    # Create the feature array for prediction
    features = np.array([[clump_thickness, uniformity_of_cell_size, uniformity_of_cell_shape,
                          marginal_adhesion, single_epithelial_cell_size, bare_nuclei,
                          bland_chromatin, normal_nucleoli, mitoses]])

    # Make prediction
    prediction = model.predict(features)

    return render_template('index.html', prediction=str(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
