import os
import re
from typing import List

from dotenv import load_dotenv
from flask import Flask, render_template_string, request
from openai import OpenAI
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from models import Summary, Transcript, db

app = Flask(__name__)

# Load environment variables
load_dotenv()

# ==== Configuration ====
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODELS = {
    "latest": "gpt-5.2",
    "last": "gpt-5.1",
    "mini": "gpt-5-mini",
    "nano": "gpt-5-nano",
}
MODEL = MODELS["latest"]
MAX_TOKENS_PER_CHUNK = 4000  # Conservative estimate to stay within rate limits
CHUNK_OVERLAP = 200  # Number of tokens to overlap between chunks

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


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.
    This is a rough estimation based on OpenAI's tokenizer behavior:
    - ~4 chars per token for English text
    - Special characters and numbers count differently
    """
    # Count words and special characters
    words = len(text.split())
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    numbers = len(re.findall(r'\d+', text))

    # Weight different elements
    return (words * 1.3) + (special_chars * 0.5) + (numbers * 0.5)


def find_split_point(text: str, max_tokens: int) -> int:
    """Find the best point to split text while respecting sentence boundaries."""
    # First try to split on paragraph breaks
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        current_length = 0
        for i, para in enumerate(paragraphs):
            current_length += estimate_tokens(para)
            if current_length > max_tokens:
                return text.find('\n\n', text.find(paragraphs[i-1]))

    # If no good paragraph break, try sentence breaks
    sentences = re.split(r'(?<=[.!?])\s+', text)
    current_length = 0
    for i, sentence in enumerate(sentences):
        current_length += estimate_tokens(sentence)
        if current_length > max_tokens:
            # Find the position of the last complete sentence
            return text.find(sentence, text.find(sentences[i-1]))

    return len(text)


def chunk_transcript(text: str, max_tokens: int = MAX_TOKENS_PER_CHUNK) -> List[str]:
    """Split transcript into smaller chunks based on token count while maintaining context.

    Args:
        text: The transcript text to split
        max_tokens: Maximum tokens per chunk

    Returns:
        List of text chunks
    """
    chunks = []
    remaining_text = text

    while remaining_text:
        # Calculate the split point
        split_point = find_split_point(remaining_text, max_tokens)

        # Extract the chunk
        chunk = remaining_text[:split_point].strip()
        if chunk:
            chunks.append(chunk)

        # Move the remaining text forward, including some overlap
        if split_point < len(remaining_text):
            # Find the last sentence in the current chunk
            last_sentence = re.search(r'[^.!?]+[.!?]', chunk[::-1])
            if last_sentence:
                overlap_start = split_point - len(last_sentence.group(0))
                remaining_text = remaining_text[overlap_start:]
            else:
                remaining_text = remaining_text[split_point:]
        else:
            break

    return chunks


SUMMARY_INSTRUCTIONS = {
    'concise': "Summarize this portion of a YouTube transcript in a concise manner, focusing on the main points.",
    'detailed': "Provide a detailed summary of this portion of a YouTube transcript, including important details and context.",
    'key_points': "Extract the key points and main takeaways from this portion of a YouTube transcript."
}

# Token config by summary type
TOKEN_CONFIG = {
    'concise': {'chunk_pct': 0.08, 'final_pct': 0.12, 'chunk_min': 300, 'chunk_max': 1000, 'final_min': 500, 'final_max': 2000},
    'detailed': {'chunk_pct': 0.30, 'final_pct': 0.40, 'chunk_min': 2000, 'chunk_max': 4000, 'final_min': 4000, 'final_max': 16000},
    'key_points': {'chunk_pct': 0.10, 'final_pct': 0.15, 'chunk_min': 500, 'chunk_max': 1500, 'final_min': 800, 'final_max': 3000},
}


def calculate_max_tokens(text: str, summary_type: str, is_final: bool = False) -> int:
    """Calculate max output tokens based on transcript length and summary type."""
    input_tokens = estimate_tokens(text)
    config = TOKEN_CONFIG[summary_type]
    context = 'final' if is_final else 'chunk'

    percentage = config[f'{context}_pct']
    min_bound = config[f'{context}_min']
    max_bound = config[f'{context}_max']

    return max(min_bound, min(int(input_tokens * percentage), max_bound))


def generate_summary(text: str, summary_type: str) -> str:
    """Generate a single type of summary for the given text."""
    if summary_type not in SUMMARY_INSTRUCTIONS:
        raise ValueError(f"Invalid summary type: {summary_type}")

    chunks = chunk_transcript(text)
    instruction = SUMMARY_INSTRUCTIONS[summary_type]

    # Single chunk - use final config since this is the only output
    if len(chunks) == 1:
        max_tokens = calculate_max_tokens(text, summary_type, is_final=True)
        response = client.responses.create(
            model=MODEL,
            instructions=instruction,
            input=chunks[0],
            temperature=0.5,
            max_output_tokens=max_tokens,
        )
        if app.debug:
            actual = response.usage.output_tokens
            app.logger.debug(f"[{summary_type}] final: {actual}/{max_tokens} tokens used")
        return response.output_text

    # Multiple chunks - summarize each, then combine
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        max_tokens = calculate_max_tokens(chunk, summary_type)
        response = client.responses.create(
            model=MODEL,
            instructions=instruction,
            input=chunk,
            temperature=0.5,
            max_output_tokens=max_tokens,
        )
        chunk_summaries.append(response.output_text)
        if app.debug:
            actual = response.usage.output_tokens
            app.logger.debug(f"[{summary_type}] chunk {i+1}/{len(chunks)}: {actual}/{max_tokens} tokens used")

    combined_summary = " ".join(chunk_summaries)
    max_tokens = calculate_max_tokens(text, summary_type, is_final=True)
    response = client.responses.create(
        model=MODEL,
        instructions=f"Create a coherent final {summary_type} summary from these partial summaries.",
        input=combined_summary,
        temperature=0.5,
        max_output_tokens=max_tokens,
    )
    if app.debug:
        actual = response.usage.output_tokens
        app.logger.debug(f"[{summary_type}] final: {actual}/{max_tokens} tokens used")
    return response.output_text


def summarize_transcript(text):
    """Generate all summary types for the given text."""
    return {
        summary_type: generate_summary(text, summary_type)
        for summary_type in SUMMARY_INSTRUCTIONS
    }


# ==== Routes ====


@app.route("/", methods=["GET", "POST"])
def index():
    summaries = {}
    error = ""
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
                summaries = summarize_transcript(transcript)
                # Update summaries in database
                transcript_record = Transcript.query.filter_by(video_id=video_id).first()
                if transcript_record:
                    # Delete existing summaries
                    Summary.query.filter_by(transcript_id=transcript_record.id).delete()
                    # Add new summaries
                    for summary_type, content in summaries.items():
                        new_summary = Summary(
                            transcript_id=transcript_record.id,
                            summary_type=summary_type,
                            content=content
                        )
                        db.session.add(new_summary)
                    db.session.commit()

    # Get all processed videos
    processed_videos = Transcript.query.order_by(Transcript.created_at.desc()).all()
    return render_template_string(TEMPLATE, summaries=summaries, error=error, processed_videos=processed_videos)


@app.route("/summarize/<video_id>/<summary_type>", methods=["POST"])
def summarize(video_id, summary_type):
    error = ""
    transcript_record = Transcript.query.filter_by(video_id=video_id).first()

    if not transcript_record:
        error = "Video not found."
    elif summary_type not in SUMMARY_INSTRUCTIONS:
        error = "Invalid summary type."
    else:
        new_content = generate_summary(transcript_record.transcript_text, summary_type)
        # Update or create the summary
        existing_summary = Summary.query.filter_by(
            transcript_id=transcript_record.id,
            summary_type=summary_type
        ).first()
        if existing_summary:
            existing_summary.content = new_content
        else:
            new_summary = Summary(
                transcript_id=transcript_record.id,
                summary_type=summary_type,
                content=new_content
            )
            db.session.add(new_summary)
        db.session.commit()

    processed_videos = Transcript.query.order_by(Transcript.created_at.desc()).all()
    return render_template_string(TEMPLATE, summaries={}, error=error, processed_videos=processed_videos)


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
        .summary-section {
            margin-top: 1rem;
            padding: 1rem;
            background: #f5f5f5;
            border-radius: 4px;
        }
        .summary-type {
            font-weight: bold;
            color: #333;
            margin-bottom: 0.5rem;
        }
        .summary-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        .resummarize-btn {
            padding: 0.25rem 0.5rem;
            font-size: 0.8em;
            background: #0066cc;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .resummarize-btn:hover {
            background: #0052a3;
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
    {% if summaries %}
        <h2>Summaries:</h2>
        {% for summary_type, content in summaries.items() %}
            <div class="summary-section">
                <div class="summary-type">{{ summary_type|title }} Summary:</div>
                <textarea readonly>{{ content }}</textarea>
            </div>
        {% endfor %}
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
                    {% if video.summaries %}
                        {% for summary in video.summaries %}
                            <div class="summary-section">
                                <div class="summary-header">
                                    <div class="summary-type">{{ summary.summary_type|title }} Summary:</div>
                                    <form method="post" action="/summarize/{{ video.video_id }}/{{ summary.summary_type }}" style="margin: 0;">
                                        <button type="submit" class="resummarize-btn">Resummarize</button>
                                    </form>
                                </div>
                                <p>{{ summary.content | replace('\n', '<br>') | safe }}</p>
                            </div>
                        {% endfor %}
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
