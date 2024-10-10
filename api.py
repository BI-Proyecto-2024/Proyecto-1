import io
import joblib
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Cargamos el modelo preentrenado
pipe = joblib.load('')