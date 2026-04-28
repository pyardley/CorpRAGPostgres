.venv\Scripts\python.exe - <<'PY'
from app.utils import get_db, load_all_credentials
from models.user import User
with get_db() as db:
    for u in db.query(User).all():
        creds = load_all_credentials(db, u.id, "sql")
        if creds:
            print(u.email, "→")
            for k, v in creds.items():
                print(f"   {k} = {v}")
PY