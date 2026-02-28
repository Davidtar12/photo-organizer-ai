from __future__ import annotations

from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from config import Config, ensure_directories

ensure_directories()

engine = create_engine(f"sqlite:///{Config.DB_PATH}", future=True, echo=False)
_session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
Session = scoped_session(_session_factory)


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""

    session = Session()
    try:
        yield session
        session.commit()
    except Exception:  # pragma: no cover - rollback for safety
        session.rollback()
        raise
    finally:
        session.close()
