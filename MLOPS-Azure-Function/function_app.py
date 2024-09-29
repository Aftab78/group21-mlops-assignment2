import azure.functions as func
import logging
import joblib
import json
import numpy as np

# Load the model at the start so that it doesn't need to be loaded on every request
model = joblib.load('best_random_forest_model.pkl')

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="predict")
def predict(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Parse the request body
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            "Invalid request body. Please pass valid JSON data.", 
            status_code=400
        )
    
    # Ensure that the necessary input data is present
    if 'data' not in req_body:
        return func.HttpResponse(
            "Please provide the input data in the request body as a 'data' field.", 
            status_code=400
        )

    try:
        # Assuming the input is a list of lists (for multiple predictions)
        input_data = np.array(req_body['data'])

        # Perform the prediction using the loaded model
        predictions = model.predict(input_data)

        # Convert the predictions into a JSON response
        result = {
            "predictions": predictions.tolist()
        }

        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return func.HttpResponse(
            "An error occurred while performing prediction. Please check your input data and try again.",
            status_code=500
        )
