from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Transcript(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    video_id = db.Column(db.String(11), unique=True, nullable=False)
    transcript_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    summaries = db.relationship(
        "Summary", backref="transcript", lazy=True, cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Transcript {self.video_id}>"


class Summary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transcript_id = db.Column(
        db.Integer, db.ForeignKey("transcript.id"), nullable=False
    )
    summary_type = db.Column(
        db.String(50), nullable=False
    )  # e.g., 'concise', 'detailed', 'key_points'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def __repr__(self):
        return f"<Summary {self.summary_type} for Transcript {self.transcript_id}>"
