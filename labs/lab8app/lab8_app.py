from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

class HappinessFeatures(BaseModel):
    Year: float
    GDP_per_Capita: float
    Social_Support: float
    Healthy_Life_Expectancy: float
    Freedom: float
    Generosity: float
    Corruption_Perception: float
    Unemployment_Rate: float
    Education_Index: float
    Population: float
    Urbanization_Rate: float
    Life_Satisfaction: float
    Public_Trust: float
    Mental_Health_Index: float
    Income_Inequality: float
    Public_Health_Expenditure: float
    Climate_Index: float
    Work_Life_Balance: float
    Internet_Access: float
    Crime_Rate: float
    Political_Stability: float
    Employment_Rate: float
    Country: str

model = None
encoder = None
country_list = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoder, country_list
    print("Loading Model...")
    mlflow.set_tracking_uri('http://localhost:5001')
    model_uri = "models:/happiness_lr_model/1"
    model = mlflow.pyfunc.load_model(model_uri)

    country_list = ["USA", "UK", "Canada", "China", "Brazil", "France", "Germany", 
                       "India", "Australia", "South Africa"]
        
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoder.fit(np.array(country_list).reshape(-1, 1))
    print("Finished Loading")
    yield

app = FastAPI(
    title="ML Model Prediction API",
    description="API for making predictions using a model loaded from MLFlow.",
    version="0.1",
    lifespan=lifespan
)

@app.get('/')
def root():
    return {"message": "ML Prediction API. Use POST /predict for predictions."}

@app.post('/predict')
def predict(features: HappinessFeatures):

    feature_dict = features.dict()

    country = feature_dict.pop('Country')
    
    country_encoded = encoder.transform(np.array([country]).reshape(-1, 1))
    country_columns = encoder.get_feature_names_out(['Country'])
    
    feature_df = pd.DataFrame([feature_dict])
    country_df = pd.DataFrame(country_encoded, columns=country_columns)
    
    combined_df = pd.concat([feature_df, country_df], axis=1)

    prediction = model.predict(combined_df)

    if hasattr(prediction, 'tolist'):
        prediction = prediction.tolist()

    return {"prediction": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
