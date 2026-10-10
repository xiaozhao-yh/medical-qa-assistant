import os
import json
import hashlib

USERS_FILE = "users.json"

def hash_password(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()

def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def verify_user(username: str, password: str) -> bool:
    users = load_users()
    if username not in users:
        return False
    return users[username] == hash_password(password)

def add_user(username: str, password: str):
    users = load_users()
    users[username] = hash_password(password)
    save_users(users)
