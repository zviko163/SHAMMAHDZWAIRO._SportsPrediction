from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the scaler object used during training
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the trained model (Example: DecisionTreeRegressor)
model = pickle.load(open('DecisionTreeRegressor.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getprediction', methods=['POST'])
def getprediction():
    try:
        # Extract input data from form
        input_data = [float(x) for x in request.form.values()]
        
        # Convert input data to numpy array
        final_input = np.array(input_data).reshape(1, -1)

        # Scale the input data using the loaded scaler
        scaled_input = scaler.transform(final_input)

        # Debugging output
        print("Input Data:", input_data)
        print("Scaled Input:", scaled_input)

        # Make prediction using the loaded model
        prediction = model.predict(scaled_input)

        # Additional debugging output
        print("Prediction:", prediction)

        # Render prediction result on index.html
        return render_template('index.html', output='Predicted Overall Rating: {}'.format(prediction[0]))

    except Exception as e:
        return render_template('index.html', output='Error: {}'.format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)
