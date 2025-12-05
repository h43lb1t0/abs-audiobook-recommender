from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    username: Mapped[str] = mapped_column(String(255), unique=True)
    password: Mapped[str] = mapped_column(String(255))
    daily_recommendation_count: Mapped[int] = mapped_column(Integer, default=0)
    last_recommendation_date: Mapped[DateTime] = mapped_column(DateTime, nullable=True)

class UserLib(db.Model):
    __tablename__ = "user_lib"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), ForeignKey("users.id"))
    book_id: Mapped[str] = mapped_column(String(255))
    user_name_debug: Mapped[str] = mapped_column(String(255))
    book_name_debug: Mapped[str] = mapped_column(String(255))

class UserLastRecommendation(db.Model):
    __tablename__ = "user_last_recommendation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[DateTime] = mapped_column(DateTime)
    user_id: Mapped[str] = mapped_column(String(255), ForeignKey("users.id"))
    book_id: Mapped[str] = mapped_column(String(255))
    gemini_reason: Mapped[str] = mapped_column(String(1024))
