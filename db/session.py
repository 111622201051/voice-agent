# db/session.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_PATH = os.path.join(DATA_DIR, "agent.db").replace('\\', '/')
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# check_same_thread=False is required for SQLite in multi-threaded apps
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()