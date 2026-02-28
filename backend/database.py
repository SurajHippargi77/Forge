from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import event
import os

import os

# Get the absolute path to the database file
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "slingshot.db")
SQLALCHEMY_DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH.replace(chr(92), '/')}"

# Create async engine
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=True  # Set to False in production
)

# Enable SQLite foreign key enforcement
@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()

# Dependency to get async database session
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Function to create all tables
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Function to drop all tables (for testing)
async def drop_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
