import os
import re

from dotenv import load_dotenv
from flask import Flask, render_template_string, request
from openai import OpenAI
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from models import Transcript, db

app = Flask(__name__)

# Load environment variables
load_dotenv()

# ==== Configuration ====
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o"  # or "o3-mini"

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

# Initialize database
with app.app_context():
    db.create_all()

# ==== Helper Functions ====


def extract_video_id(url):
    # Match regular YouTube URLs (v=VIDEO_ID)
    match = re.search(r"(?:v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def fetch_transcript(video_id):
    try:
        # Check if transcript exists in database
        existing_transcript = Transcript.query.filter_by(video_id=video_id).first()
        if existing_transcript:
            return existing_transcript.transcript_text

        # If not in database, fetch from YouTube
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([t["text"] for t in transcript])

        # Save to database
        new_transcript = Transcript(video_id=video_id, transcript_text=full_text)
        db.session.add(new_transcript)
        db.session.commit()

        return full_text
    except (TranscriptsDisabled, NoTranscriptFound):
        return None


def summarize_transcript(text):
    response = client.responses.create(
        model=MODEL,
        instructions="Summarize this YouTube transcript clearly and concisely.",
        input=text,
        temperature=0.5,
        max_output_tokens=500,
    )
    return response.output_text


# ==== Routes ====


@app.route("/", methods=["GET", "POST"])
def index():
    summary = error = ""
    if request.method == "POST":
        url = request.form["url"]
        video_id = extract_video_id(url)
        if not video_id:
            error = "Invalid YouTube URL."
        else:
            transcript = fetch_transcript(video_id)
            if not transcript:
                error = "Transcript not available for this video."
            else:
                summary = summarize_transcript(transcript)
                # Update summary in database
                transcript_record = Transcript.query.filter_by(
                    video_id=video_id
                ).first()
                if transcript_record:
                    transcript_record.summary = summary
                    db.session.commit()
    # Get all processed videos
    processed_videos = Transcript.query.order_by(Transcript.created_at.desc()).all()
    return render_template_string(TEMPLATE, summary=summary, error=error, processed_videos=processed_videos)


# ==== Template ====

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>YouTube Transcript Summarizer</title>
    <style>
        body { font-family: sans-serif; max-width: 700px; margin: 2rem auto; padding: 1rem; }
        textarea { width: 100%; height: 200px; margin-top: 1rem; }
        .error { color: red; }
        .video-list { margin-top: 2rem; }
        .video-item {
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .video-item h3 { margin-top: 0; }
        .video-item .timestamp {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 0.5rem;
        }
        .video-link {
            color: #0066cc;
            text-decoration: none;
        }
        .video-link:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>YouTube Transcript Summarizer</h1>
    <form method="post">
        <input name="url" type="text" placeholder="Enter YouTube URL" style="width:100%; padding: 0.5rem;" required>
        <button type="submit" style="margin-top:1rem; padding:0.5rem 1rem;">Summarize</button>
    </form>
    {% if error %}
        <p class="error">{{ error }}</p>
    {% endif %}
    {% if summary %}
        <h2>Summary:</h2>
        <textarea readonly>{{ summary }}</textarea>
    {% endif %}

    <div class="video-list">
        <h2>Processed Videos</h2>
        {% if processed_videos %}
            {% for video in processed_videos %}
                <div class="video-item">
                    <h3>
                        <a href="https://youtube.com/watch?v={{ video.video_id }}" class="video-link" target="_blank">
                            Video ID: {{ video.video_id }}
                        </a>
                    </h3>
                    <div class="timestamp">
                        Processed: {{ video.created_at.strftime('%Y-%m-%d %H:%M:%S') }}
                    </div>
                    {% if video.summary %}
                        <div class="summary">
                            <strong>Summary:</strong>
                            <p>{{ video.summary }}</p>
                        </div>
                    {% endif %}
                </div>
            {% endfor %}
        {% else %}
            <p>No videos have been processed yet.</p>
        {% endif %}
    </div>
</body>
</html>
"""

# ==== Run Server ====

if __name__ == "__main__":
    app.run(debug=True)
