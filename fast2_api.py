import numpy as np
from fastapi import FastAPI, Form
import pickle
app = FastAPI()

model = pickle.load(open('sleep_disorder_model.pkl', 'rb'))

@app.post("/predict")
def predict(quality_of_sleep: int = Form(...), sleep_duration: float = Form(...), stress_level: int = Form(...), physical_activity_level: int = Form(...)):
    '''
    For returning prediction result as JSON
    '''
    final_features = np.array([[quality_of_sleep, sleep_duration, stress_level, physical_activity_level]])
    prediction = model.predict(final_features)

    if prediction[0] is not None and prediction[0] != 'None':
        output = round(float(prediction[0]), 2)
        return {"prediction_text": f'Detected $ {output}'}
    else:
        return {"error_message": 'Error: Unable to make a prediction for the given input.'}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)