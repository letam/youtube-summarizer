from flask import Flask, request, render_template_string
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import openai
import re
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

# ==== Configuration ====
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"  # or "o3-mini"

# ==== Helper Functions ====

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([t["text"] for t in transcript])
        return full_text
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

def summarize_transcript(text):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Summarize this YouTube transcript clearly and concisely."},
            {"role": "user", "content": text}
        ],
        temperature=0.5,
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]

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
    return render_template_string(TEMPLATE, summary=summary, error=error)

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
</body>
</html>
"""

# ==== Run Server ====

if __name__ == "__main__":
    app.run(debug=True)

