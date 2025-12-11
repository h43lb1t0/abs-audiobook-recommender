from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    username: Mapped[str] = mapped_column(String(255), unique=True)
    password: Mapped[str] = mapped_column(String(255))

class UserLib(db.Model):
    __tablename__ = "user_lib"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), ForeignKey("users.id"))
    book_id: Mapped[str] = mapped_column(String(255))
    rating: Mapped[int] = mapped_column(Integer, nullable=True)  # 1-5 stars, nullable for unrated books




class UserRecommendations(db.Model):
    __tablename__ = "user_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), ForeignKey("users.id"))
    recommendations_json: Mapped[str] = mapped_column(db.Text)
    created_at: Mapped[str] = mapped_column(String(255)) # ISO8601 string

class BackgroundCheckLog(db.Model):
    __tablename__ = "background_check_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    checked_at: Mapped[str] = mapped_column(String(255)) # ISO8601 string
    new_books_found: Mapped[bool] = mapped_column(db.Boolean)

