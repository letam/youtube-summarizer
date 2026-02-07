from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Transcript(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    source_type = db.Column(db.String(20), nullable=False, default="youtube")
    source_id = db.Column(db.String(255), unique=True, nullable=False)
    transcript_text = db.Column(db.Text, nullable=False)
    generated_title = db.Column(db.String(200), nullable=True)
    original_filename = db.Column(db.String(255), nullable=True)
    file_path = db.Column(db.String(500), nullable=True)
    source_duration = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    summaries = db.relationship(
        "Summary", backref="transcript", lazy=True, cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Transcript {self.source_type}:{self.source_id}>"


class Summary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transcript_id = db.Column(
        db.Integer, db.ForeignKey("transcript.id"), nullable=False
    )
    summary_type = db.Column(
        db.String(50), nullable=False
    )  # e.g., 'concise', 'detailed', 'key_points'
    content = db.Column(db.Text, nullable=False)
    generation_duration = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def __repr__(self):
        return f"<Summary {self.summary_type} for Transcript {self.transcript_id}>"
