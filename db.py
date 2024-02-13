from app import app,db
from app import User
from flask import Flask

# Access the database and query the User table
with app.app_context():
    users = User.query.all()

    for user in users:
        print(f"ID: {user.id}, Name: {user.name}, Email: {user.email}")
