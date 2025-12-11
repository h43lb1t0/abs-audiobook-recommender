from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Integer, String, ForeignKey, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """
    User model for authentication and authorization.

    The id is the user's id from the ABS API.
    """
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    username: Mapped[str] = mapped_column(String(255), unique=True)
    password: Mapped[str] = mapped_column(String(255))

class UserLib(db.Model):
    """
    User library model.

    This contains all the books that a user has either finished or is currently reading and their status.
    For finished books, it also contains the rating.
    """
    __tablename__ = "user_lib"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), ForeignKey("users.id"))
    book_id: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(255)) # finished, reading
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1-5 stars, nullable for unrated books

    __table_args__ = (
        CheckConstraint("(rating >= 1 AND rating <= 5) OR (rating IS NULL)", name="valid_rating"),
        CheckConstraint("status IN ('finished', 'reading')", name="valid_status"),
        CheckConstraint(
            "(status = 'finished') OR (rating IS NULL)",
            name="valid_finished_rating"
        )
    )


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

