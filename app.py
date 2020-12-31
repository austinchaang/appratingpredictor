from flask import Flask, request, jsonify
import joblib
import pandas as pd

# 1. create an instance of the Flask class
app = Flask(__name__)

# 2. define a prediction function
def return_prediction(model, input_json):

    features = ['Category', 'Type', 'Content Rating', 'Genres']
    input_data = pd.DataFrame(input_json)
    prediction = model.predict(input_data)[0]

    return prediction

# 3. load our abalone age predictor model
model = joblib.load('app_predictor.joblib')

# 4. set up our home page
@app.route("/")
def index():
    return """
    <h1>Welcome to our App Prediction Service</h1>
    To use this service, make a JSON post request to the /predict url with the following fields:
    <ul>
    <li>Category</li>
    <li>Type</li>
    <li>Content Rating</li>
    <li>Genre</li>
    </ul>
    """

# 5. define a new route which will accept POST requests and return our model predictions
@app.route('/predict', methods=['POST'])
def app_prediction():

    content = request.json
    results = return_prediction(model, content)
    return jsonify(results)

# 6. allows us to run flask using $ python app.py
if __name__ == '__main__':
    app.run()
