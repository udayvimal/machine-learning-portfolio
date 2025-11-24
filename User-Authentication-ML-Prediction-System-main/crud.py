from sqlalchemy.orm import Session
from backend.models import User

def create_user(db: Session, name: str, email: str, password: str):
    new_user = User(name=name, email=email, password=password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

def get_users(db: Session, skip: int = 0, limit: int = 10):
    """
    Retrieve a paginated list of users.
    
    :param db: Database session
    :param skip: Number of records to skip (for pagination)
    :param limit: Maximum number of records to return
    :return: List of users
    """
    return db.query(User).order_by(User.id).offset(skip).limit(limit).all()
from backend.models import Prediction

def save_prediction(db: Session, user_id: int, features: dict, prediction: str):
    db_prediction = Prediction(
        user_id=user_id,
        feature_1=features["feature_1"],
        feature_2=features["feature_2"],
        feature_3=features["feature_3"],
        feature_4=features["feature_4"],
        feature_5=features["feature_5"],
        feature_6=features["feature_6"],
        prediction_result=prediction
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction
