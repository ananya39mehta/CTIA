from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLite file in project root
DATABASE_URL = "sqlite:///./ctia.db"

# Use check_same_thread=False for SQLite + multithread/async servers
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Session factory
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Base class for models
Base = declarative_base()

def init_db():
    # Import models here to ensure they are registered on Base
    # (models import must be placed after Base is defined)
    from app import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
