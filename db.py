import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

import shutil
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IS_VERCEL = bool(os.environ.get("VERCEL"))

# Support Cloud Databases (PostgreSQL / Supabase / Neon) or SQLite
database_url = os.environ.get("DATABASE_URL")

if database_url:
    # Standardize postgres:// to postgresql:// for SQLAlchemy
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(database_url, echo=False)
    DB_PATH = None
else:
    if IS_VERCEL:
        # In Vercel serverless functions, the root filesystem is read-only.
        # Use writable temp directory and seed with existing database if available.
        temp_dir = tempfile.gettempdir()
        DB_PATH = os.path.join(temp_dir, "database.db")
        seed_db = os.path.join(BASE_DIR, "database.db")
        if not os.path.exists(DB_PATH) and os.path.exists(seed_db):
            try:
                shutil.copyfile(seed_db, DB_PATH)
            except Exception as e:
                print(f"Notice: Could not copy seed database to temp: {e}")
    else:
        DB_PATH = os.path.join(BASE_DIR, "database.db")

    # In SQLite on Windows, forward slashes avoid URI escaping issues
    sqlite_path = DB_PATH.replace("\\", "/") if DB_PATH else ""
    engine = create_engine(f"sqlite:///{sqlite_path}", echo=False)

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(UserMixin, Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(120), unique=True, index=True)
    password_hash = Column(String(255))
    name = Column(String(100))
    resend_api_key = Column(String(255), nullable=True)
    notification_email = Column(String(120), nullable=True)
    
    predictions = relationship("Prediction", back_populates="user")
    chats = relationship("ChatMessage", back_populates="user")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    plant_name = Column(String(100))
    disease_name = Column(String(100))
    confidence = Column(Float)
    image_filename = Column(String(255), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="predictions")

class ChatMessage(Base):
    __tablename__ = "chats"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    session_id = Column(String(100), index=True)
    role = Column(String(20)) # 'user' or 'ai'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="chats")

# Create tables
Base.metadata.create_all(bind=engine)

# Migrate: add new columns to existing tables if missing
def _migrate():
    if not DB_PATH or not os.path.exists(DB_PATH):
        return
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(users)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if "resend_api_key" not in existing_cols:
            cursor.execute("ALTER TABLE users ADD COLUMN resend_api_key VARCHAR(255)")
        if "notification_email" not in existing_cols:
            cursor.execute("ALTER TABLE users ADD COLUMN notification_email VARCHAR(120)")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Notice: Migration skipped or completed: {e}")

_migrate()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_user(email, password, name):
    db = SessionLocal()
    user = User(email=email, name=name)
    user.set_password(password)
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    return user

def get_user_by_email(email):
    db = SessionLocal()
    user = db.query(User).filter(User.email == email).first()
    db.close()
    return user

def get_user_by_id(user_id):
    db = SessionLocal()
    user = db.query(User).filter(User.id == int(user_id)).first()
    db.close()
    return user

def update_user_profile(user_id, name=None, notification_email=None, resend_api_key=None):
    db = SessionLocal()
    user = db.query(User).filter(User.id == int(user_id)).first()
    if user:
        if name is not None:
            user.name = name
        if notification_email is not None:
            user.notification_email = notification_email
        if resend_api_key is not None:
            user.resend_api_key = resend_api_key
        db.commit()
        db.refresh(user)
    db.close()
    return user

def save_prediction(plant, disease, confidence, image_filename=None, user_id=None):
    db = SessionLocal()
    pred = Prediction(
        user_id=user_id,
        plant_name=plant,
        disease_name=disease,
        confidence=confidence,
        image_filename=image_filename
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    db.close()
    return pred.id

def get_recent_predictions(limit=20, user_id=None):
    db = SessionLocal()
    query = db.query(Prediction)
    if user_id:
        query = query.filter(Prediction.user_id == user_id)
    preds = query.order_by(Prediction.timestamp.desc()).limit(limit).all()
    result = [{
        "id": p.id,
        "plant": p.plant_name,
        "disease": p.disease_name,
        "confidence": p.confidence,
        "timestamp": p.timestamp.isoformat()
    } for p in preds]
    db.close()
    return result

def save_chat_message(session_id, role, content, user_id=None):
    db = SessionLocal()
    msg = ChatMessage(session_id=session_id, role=role, content=content, user_id=user_id)
    db.add(msg)
    db.commit()
    db.close()

def get_chat_history(session_id, limit=50, user_id=None):
    db = SessionLocal()
    query = db.query(ChatMessage).filter(ChatMessage.session_id == session_id)
    if user_id:
        query = query.filter(ChatMessage.user_id == user_id)
    msgs = query.order_by(ChatMessage.timestamp.asc()).limit(limit).all()
    result = [{"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()} for m in msgs]
    db.close()
    return result
