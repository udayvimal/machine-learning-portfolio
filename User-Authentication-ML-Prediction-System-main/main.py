from fastapi import FastAPI, Depends, HTTPException, Form, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pathlib import Path
import pickle
import numpy as np
from pydantic import BaseModel
from typing import List

# ✅ Import Database & Models
from backend.database import SessionLocal, engine
from backend.models import User

# ✅ Define Base Directory (Fixed Path Setup)
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend"
if not STATIC_DIR.exists():
    raise RuntimeError(f"Directory '{STATIC_DIR}' does not exist. Check your project structure.")

# ✅ Create FastAPI App
app = FastAPI()

# ✅ Serve Static Files (Frontend)
app.mount("/frontend", StaticFiles(directory=STATIC_DIR), name="frontend")

# 🔐 Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 🔑 JWT Token Configuration
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ✅ Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ Password Hashing & Verification
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# ✅ JWT Token Generation
def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ✅ Decode JWT & Get Current User
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

# ✅ Serve HTML Pages
def serve_html(file_name: str):
    file_path = STATIC_DIR / file_name
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read())
    return HTMLResponse(content=f"<h1>{file_name} Not Found</h1>", status_code=404)

@app.get("/", response_class=HTMLResponse)
def serve_home(): return serve_html("index.html")

@app.get("/signup", response_class=HTMLResponse)
def serve_signup(): return serve_html("signup.html")

@app.get("/login", response_class=HTMLResponse)
def serve_login(): return serve_html("login.html")

@app.get("/predictions", response_class=HTMLResponse)
def serve_predictions(): return serve_html("predictions.html")

# ✅ Signup Route
@app.post("/signup/")
def signup(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        return RedirectResponse(url="/signup?error=Email+already+registered", status_code=303)
    
    hashed_password = hash_password(password)
    db_user = User(name=name, email=email, password=hashed_password)

    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    except IntegrityError:
        db.rollback()
        return RedirectResponse(url="/signup?error=Error+saving+user", status_code=303)
    
    return RedirectResponse(url="/login?success=Account+created,+please+login", status_code=303)

# ✅ Login Route (Returns JWT Token)
@app.post("/login")
def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password):  
        raise HTTPException(status_code=400, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": user.email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

# ✅ Prediction API (Requires Authentication)
class PredictionInput(BaseModel):
    features: List[float]

import pickle

from backend.crud import save_prediction  # ✅ Import save_prediction function

@app.post("/predict/")
def predict(
    input_data: PredictionInput,
    current_user: User = Depends(get_current_user),  # Get user from token
    db: Session = Depends(get_db)  # ✅ Add database session
):
    model_path = BASE_DIR / "C:/Users/2k22c/myenv/venv/PROJECTSTRUCTURE/backend/decision_tree_model.pkl"  # ✅ Ensure correct model path

    try:
        # ✅ Load model using pickle
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # ✅ Convert input features to numpy array & reshape for prediction
        features_array = np.array(input_data.features).reshape(1, -1)

        # ✅ Make prediction
        prediction = model.predict(features_array)

        # ✅ Convert prediction to a string format (if necessary)
        prediction_result = str(prediction[0])

        # ✅ Prepare feature dictionary (Ensure correct indexing)
        feature_dict = {
            "feature_1": input_data.features[0],
            "feature_2": input_data.features[1],
            "feature_3": input_data.features[2],
            "feature_4": input_data.features[3],
            "feature_5": input_data.features[4],
            "feature_6": input_data.features[5]
        }

        # ✅ Save prediction to database
        save_prediction(db, current_user.id, feature_dict, prediction_result)

        return {"prediction": prediction.tolist()}

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found")
    except IndexError:
        raise HTTPException(status_code=400, detail="Incorrect number of features provided")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")