from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from xgboost import Booster, DMatrix
import numpy as np

# Initialize FastAPI application
app = FastAPI()

# Load trained XGBoost model
model = Booster()
model.load_model("model/model.json")


# Define input data structure
class SalesInput(BaseModel):
    sales: list[float] = Field(min_length=6, max_length=6)


# Display the HTML page
@app.get("/")
def home():
    return FileResponse("frontend/index.html")


# Predict next month's car sales
@app.post("/predict")
def predict(data: SalesInput):
    try:
        # Input order: oldest month to most recent month
        # Convert to Lag_1, Lag_2, ..., Lag_6
        features = np.array(
            data.sales[::-1], dtype=float
        ).reshape(1, -1)

        # Match training feature names
        dmatrix = DMatrix(
            features,
            feature_names=[
                f"Lag_{i}" for i in range(1, 7)
            ]
        )

        prediction = model.predict(dmatrix)

        return {
            "predicted_sales": round(
                float(prediction[0]), 2
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
