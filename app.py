from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from us_visa_approval_prediction.constants import APP_HOST, APP_PORT
from us_visa_approval_prediction.pipeline.prediction_pipeline import USvisaData, USvisaClassifier
from us_visa_approval_prediction.pipeline.training_pipeline import TrainingPipeline

import os
import requests
from dotenv import load_dotenv
from fastapi import HTTPException, status, Query
from fastapi.responses import FileResponse

import json
import pickle
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from us_visa_approval_prediction.cloud_storage.gdrive_storage import GoogleDriveStorageService
from us_visa_approval_prediction.entity.estimator import USvisaModel

# --- 1️⃣ Load .env ---
load_dotenv()
TRAIN_PASS = os.getenv("TRAIN_PASS")
TOKEN_URL = os.getenv("TOKEN_URL")
TOKEN_DIR = "gdrive_setup"
TOKEN_PATH = os.path.join(TOKEN_DIR, "token.pickle")

# Create directory if it doesn't exist
os.makedirs(TOKEN_DIR, exist_ok=True)

# Download token.pickle only if not exists
if not os.path.exists(TOKEN_PATH):
    try:
        response = requests.get(TOKEN_URL)
        response.raise_for_status()
        with open(TOKEN_PATH, "wb") as f:
            f.write(response.content)
        print(f"token.pickle downloaded to {TOKEN_PATH}")
    except Exception as e:
        print(f"Failed to download token.pickle: {e}")
        raise
else:
    print(f"token.pickle already exists at {TOKEN_PATH}, skipping download.")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.continent: Optional[str] = None
        self.education_of_employee: Optional[str] = None
        self.has_job_experience: Optional[str] = None
        self.requires_job_training: Optional[str] = None
        self.no_of_employees: Optional[str] = None
        self.company_age: Optional[str] = None
        self.region_of_employment: Optional[str] = None
        self.prevailing_wage: Optional[str] = None
        self.unit_of_wage: Optional[str] = None
        self.full_time_position: Optional[str] = None


    async def get_usvisa_data(self):
        form = await self.request.form()
        self.continent = form.get("continent")
        self.education_of_employee = form.get("education_of_employee")
        self.has_job_experience = form.get("has_job_experience")
        self.requires_job_training = form.get("requires_job_training")
        self.no_of_employees = form.get("no_of_employees")
        self.company_age = form.get("company_age")
        self.region_of_employment = form.get("region_of_employment")
        self.prevailing_wage = form.get("prevailing_wage")
        self.unit_of_wage = form.get("unit_of_wage")
        self.full_time_position = form.get("full_time_position")


@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "index.html",{"request": request, "context": "Rendering"})

@app.get("/drift-report")
async def get_drift_report():
    report_path = "us_visa_approval_prediction/notebooks/visa_data_drift_report.html"
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Drift report not found")
    return FileResponse(report_path, media_type="text/html")


@app.get("/train")
async def trainRouteClient(password: str = Query(..., description="Training password")):
    if password != TRAIN_PASS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password for training"
        )

    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_usvisa_data()

        # Convert numeric fields to proper types
        try:
            no_of_employees = int(form.no_of_employees) if form.no_of_employees else None
            company_age = int(form.company_age) if form.company_age else None
            prevailing_wage = float(form.prevailing_wage) if form.prevailing_wage else None
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f"Invalid numeric input: {ve}")

        usvisa_data = USvisaData(
                                continent= form.continent,
                                education_of_employee = form.education_of_employee,
                                has_job_experience = form.has_job_experience,
                                requires_job_training = form.requires_job_training,
                                no_of_employees= no_of_employees,
                                company_age= company_age,
                                region_of_employment = form.region_of_employment,
                                prevailing_wage= prevailing_wage,
                                unit_of_wage= form.unit_of_wage,
                                full_time_position= form.full_time_position,
                                )

        usvisa_df = usvisa_data.get_usvisa_input_data_frame()

        model_predictor = USvisaClassifier()

        value = model_predictor.predict(dataframe=usvisa_df)[0]

        status = None
        if value == 1:
            status = "Visa-approved"
        else:
            status = "Visa Not-Approved"

        return HTMLResponse(f'<h1 class="display-4">Visa Prediction Status: {status}</h1>')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Model reports
BDT = ZoneInfo("Asia/Dhaka")
def bdt_now_iso() -> str:
    """Return current time in Bangladesh as ISO string."""
    return datetime.now(BDT).isoformat()


@app.get("/model-report")
async def get_model_report():
    """
    Comprehensive model report with metrics, features, model info.
    """
    try:
        gdrive = GoogleDriveStorageService(folder_name="Visa Approval ML Project")
        report = {
            "timestamp": bdt_now_iso(),
            "model_info": {},
            "metrics": {},
            "features": {},
            "model_health": {},
            "training_info": {},
        }

        # 1️⃣ Model health
        try:
            if gdrive.gdrive_file_exists("model.pkl"):
                model = gdrive.load_model("model.pkl")
                report["model_health"] = {
                    "model_exists": True,
                    "model_loadable": True,
                    "status": "healthy",
                }

                if hasattr(model, "trained_model_object"):
                    mobj = model.trained_model_object
                    report["model_info"]["model_type"] = type(mobj).__name__
                    report["model_info"]["model_class"] = str(type(mobj))
                    if hasattr(mobj, "get_params"):
                        try:
                            report["model_info"]["parameters"] = mobj.get_params()
                        except Exception:
                            report["model_info"]["parameters"] = "Unable to extract"

                if hasattr(model, "preprocessing_object"):
                    report["model_info"]["preprocessing_type"] = type(
                        model.preprocessing_object
                    ).__name__
            else:
                report["model_health"] = {
                    "model_exists": False,
                    "model_loadable": False,
                    "status": "unhealthy",
                    "message": "model.pkl not found",
                }
        except Exception as e:
            report["model_health"] = {"status": "unhealthy", "error": str(e)}

        # 2️⃣ Metrics
        try:
            if gdrive.gdrive_file_exists("metrics.json"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                tmp.close()
                try:
                    gdrive.download_file("metrics.json", tmp.name)
                    with open(tmp.name, "r", encoding="utf-8") as f:
                        report["metrics"] = json.load(f)
                finally:
                    os.unlink(tmp.name)
            else:
                report["metrics"] = {"error": "metrics.json not found"}
        except Exception as e:
            report["metrics"] = {"error": str(e)}

        # 3️⃣ Features
        try:
            if gdrive.gdrive_file_exists("feature_names.pkl"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
                tmp.close()
                try:
                    gdrive.download_file("feature_names.pkl", tmp.name)
                    with open(tmp.name, "rb") as f:
                        names = pickle.load(f)
                    report["features"] = {
                        "feature_names": list(names)
                        if hasattr(names, "__iter__")
                        else [str(names)],
                        "total_features": len(names)
                        if hasattr(names, "__len__")
                        else 0,
                    }
                finally:
                    os.unlink(tmp.name)
            else:
                report["features"] = {"error": "feature_names.pkl not found"}
        except Exception as e:
            report["features"] = {"error": str(e)}

        # 4️⃣ Training info
        metrics = report.get("metrics", {})
        if isinstance(metrics, dict):
            if "best_parameters" in metrics:
                report["training_info"]["best_parameters"] = metrics["best_parameters"]
            if "best_cv_score" in metrics:
                report["training_info"]["cross_validation_score"] = metrics[
                    "best_cv_score"
                ]

        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e}")


@app.get("/model-metrics")
async def get_model_metrics():
    """Return only metrics (timestamp in BDT)."""
    try:
        gdrive = GoogleDriveStorageService(folder_name="Visa Approval ML Project")
        if not gdrive.gdrive_file_exists("metrics.json"):
            raise HTTPException(status_code=404, detail="metrics.json not found")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()
        try:
            gdrive.download_file("metrics.json", tmp.name)
            with open(tmp.name, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        finally:
            os.unlink(tmp.name)

        return {"timestamp": bdt_now_iso(), "metrics": metrics}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading metrics: {e}")


@app.get("/model-features")
async def get_model_features():
    """Return feature list and count."""
    try:
        gdrive = GoogleDriveStorageService(folder_name="Visa Approval ML Project")
        if not gdrive.gdrive_file_exists("feature_names.pkl"):
            raise HTTPException(status_code=404, detail="feature_names.pkl not found")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        tmp.close()
        try:
            gdrive.download_file("feature_names.pkl", tmp.name)
            with open(tmp.name, "rb") as f:
                names = pickle.load(f)
        finally:
            os.unlink(tmp.name)

        return {
            "timestamp": bdt_now_iso(),
            "feature_names": list(names)
            if hasattr(names, "__iter__")
            else [str(names)],
            "total_features": len(names) if hasattr(names, "__len__") else 1,
            "feature_types": {"categorical": [], "numerical": [], "encoded": []},
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading features: {e}")


@app.get("/model-info")
async def get_model_info():
    """Return model architecture and preprocessing info."""
    try:
        gdrive = GoogleDriveStorageService(folder_name="Visa Approval ML Project")
        if not gdrive.gdrive_file_exists("model.pkl"):
            raise HTTPException(status_code=404, detail="model.pkl not found")

        model = gdrive.load_model("model.pkl")
        info = {
            "timestamp": bdt_now_iso(),
            "model_architecture": {},
            "preprocessing": {},
            "configuration": {},
        }

        if hasattr(model, "trained_model_object"):
            mobj = model.trained_model_object
            info["model_architecture"]["type"] = type(mobj).__name__
            info["model_architecture"]["full_class"] = str(type(mobj))
            if hasattr(mobj, "get_params"):
                try:
                    info["configuration"]["parameters"] = mobj.get_params()
                except Exception:
                    info["configuration"]["parameters"] = "Unable to extract"
            if hasattr(mobj, "n_estimators"):
                info["model_architecture"]["n_estimators"] = mobj.n_estimators
            if hasattr(mobj, "max_depth"):
                info["model_architecture"]["max_depth"] = mobj.max_depth

        if hasattr(model, "preprocessing_object"):
            pobj = model.preprocessing_object
            info["preprocessing"]["type"] = type(pobj).__name__
            if hasattr(pobj, "steps"):
                info["preprocessing"]["pipeline_steps"] = [
                    {"name": n, "transformer": type(t).__name__}
                    for n, t in pobj.steps
                ]

        return info

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading model info: {e}")


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)