from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model
with open('sleep_disorder_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler used for normalization
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Create a FastAPI app
app = FastAPI()


@app.add_route("/")
def foobar():
    return {
        "foo":"bar"
    }

# Define the input data model using Pydantic
class InputData(BaseModel):
    quality_of_sleep: float
    sleep_duration: float
    stress_level: float
    physical_activity_level: float

# Define the endpoint for making predictions
@app.get("/predict")
def predict(data: InputData):
    # Convert input data to a Pandas DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Normalize input data using the scaler
    input_data_scaled = scaler.transform(input_data)

    # Make predictions with the model
    prediction = model.predict(input_data_scaled)

    # Return the prediction
    return {"prediction": prediction[0]}

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)