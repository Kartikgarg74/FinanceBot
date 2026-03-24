"""Simple database migration - create all tables."""

from .models import init_db


def run_migrations(db_path: str = "data/kartikai.db") -> None:
    """Run database migrations (create tables if they don't exist)."""
    init_db(db_path)
    print(f"Database initialized at {db_path}")


if __name__ == "__main__":
    run_migrations()
