from app.api.server import create_app
from app.db_main import router as db_router
from app.db import engine, init_db
from app.models import Base

init_db()
Base.metadata.create_all(bind=engine)

app = create_app()
app.include_router(db_router)