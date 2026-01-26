import os
import re
from typing import List

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request
from openai import OpenAI
from sqlalchemy.orm import defer
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
    special_chars = len(re.findall(r"[^a-zA-Z0-9\s]", text))
    numbers = len(re.findall(r"\d+", text))

    # Weight different elements
    return (words * 1.3) + (special_chars * 0.5) + (numbers * 0.5)


def find_split_point(text: str, max_tokens: int) -> int:
    """Find the best point to split text while respecting sentence boundaries."""
    # First try to split on paragraph breaks
    paragraphs = text.split("\n\n")
    if len(paragraphs) > 1:
        current_length = 0
        for i, para in enumerate(paragraphs):
            current_length += estimate_tokens(para)
            if current_length > max_tokens:
                return text.find("\n\n", text.find(paragraphs[i - 1]))

    # If no good paragraph break, try sentence breaks
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current_length = 0
    for i, sentence in enumerate(sentences):
        current_length += estimate_tokens(sentence)
        if current_length > max_tokens:
            # Find the position of the last complete sentence
            return text.find(sentence, text.find(sentences[i - 1]))

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
            last_sentence = re.search(r"[^.!?]+[.!?]", chunk[::-1])
            if last_sentence:
                overlap_start = split_point - len(last_sentence.group(0))
                remaining_text = remaining_text[overlap_start:]
            else:
                remaining_text = remaining_text[split_point:]
        else:
            break

    return chunks


SUMMARY_INSTRUCTIONS = {
    "concise": "Summarize this portion of a YouTube transcript in a concise manner, focusing on the main points.",
    "detailed": "Provide a detailed summary of this portion of a YouTube transcript, including important details and context.",
    "key_points": "Extract the key points and main takeaways from this portion of a YouTube transcript.",
}

# Token config by summary type
TOKEN_CONFIG = {
    "concise": {
        "chunk_pct": 0.08,
        "final_pct": 0.12,
        "chunk_min": 300,
        "chunk_max": 1000,
        "final_min": 500,
        "final_max": 2000,
    },
    "detailed": {
        "chunk_pct": 0.30,
        "final_pct": 0.40,
        "chunk_min": 2000,
        "chunk_max": 4000,
        "final_min": 4000,
        "final_max": 16000,
    },
    "key_points": {
        "chunk_pct": 0.10,
        "final_pct": 0.15,
        "chunk_min": 500,
        "chunk_max": 1500,
        "final_min": 800,
        "final_max": 3000,
    },
}


def calculate_max_tokens(text: str, summary_type: str, is_final: bool = False) -> int:
    """Calculate max output tokens based on transcript length and summary type."""
    input_tokens = estimate_tokens(text)
    config = TOKEN_CONFIG[summary_type]
    context = "final" if is_final else "chunk"

    percentage = config[f"{context}_pct"]
    min_bound = config[f"{context}_min"]
    max_bound = config[f"{context}_max"]

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
            app.logger.debug(
                f"[{summary_type}] final: {actual}/{max_tokens} tokens used"
            )
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
            app.logger.debug(
                f"[{summary_type}] chunk {i+1}/{len(chunks)}: {actual}/{max_tokens} tokens used"
            )

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
                transcript_record = Transcript.query.filter_by(
                    video_id=video_id
                ).first()
                if transcript_record:
                    # Delete existing summaries
                    Summary.query.filter_by(transcript_id=transcript_record.id).delete()
                    # Add new summaries
                    for summary_type, content in summaries.items():
                        new_summary = Summary(
                            transcript_id=transcript_record.id,
                            summary_type=summary_type,
                            content=content,
                        )
                        db.session.add(new_summary)
                    db.session.commit()

    # Get all processed videos (defer transcript_text for eco-friendly loading)
    processed_videos = (
        Transcript.query.options(defer(Transcript.transcript_text))
        .order_by(Transcript.created_at.desc())
        .all()
    )
    return render_template_string(
        TEMPLATE, summaries=summaries, error=error, processed_videos=processed_videos
    )


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
            transcript_id=transcript_record.id, summary_type=summary_type
        ).first()
        if existing_summary:
            existing_summary.content = new_content
        else:
            new_summary = Summary(
                transcript_id=transcript_record.id,
                summary_type=summary_type,
                content=new_content,
            )
            db.session.add(new_summary)
        db.session.commit()

    # Defer transcript_text for eco-friendly loading
    processed_videos = (
        Transcript.query.options(defer(Transcript.transcript_text))
        .order_by(Transcript.created_at.desc())
        .all()
    )
    return render_template_string(
        TEMPLATE, summaries={}, error=error, processed_videos=processed_videos
    )


@app.route("/api/video/<video_id>/transcript")
def get_transcript(video_id):
    """API endpoint to fetch transcript on demand."""
    transcript_record = Transcript.query.filter_by(video_id=video_id).first()
    if not transcript_record:
        return jsonify({"error": "Video not found"}), 404
    return jsonify({"transcript": transcript_record.transcript_text})


@app.route("/api/video/<video_id>/summaries")
def get_summaries(video_id):
    """API endpoint to fetch summaries on demand."""
    transcript_record = Transcript.query.filter_by(video_id=video_id).first()
    if not transcript_record:
        return jsonify({"error": "Video not found"}), 404
    summaries = [
        {"type": s.summary_type, "content": s.content}
        for s in transcript_record.summaries
    ]
    return jsonify({"summaries": summaries, "video_id": video_id})


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
        .resummarize-btn, .copy-btn {
            padding: 0.25rem 0.5rem;
            font-size: 0.8em;
            background: #0066cc;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .resummarize-btn:hover, .copy-btn:hover {
            background: #0052a3;
        }
        .copy-btn {
            background: #28a745;
        }
        .copy-btn:hover {
            background: #218838;
        }
        .btn-group {
            display: flex;
            gap: 0.5rem;
        }
        .transcript-toggle {
            padding: 0.4rem 0.8rem;
            font-size: 0.9em;
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin-bottom: 0.5rem;
        }
        .transcript-toggle:hover {
            background: #5a6268;
        }
        .transcript-section {
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            padding: 1rem;
            background: #e9ecef;
            border-radius: 4px;
            display: none;
        }
        .transcript-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        .transcript-label {
            font-weight: bold;
            color: #333;
        }
        .transcript-content {
            white-space: pre-wrap;
            font-size: 0.9em;
            max-height: 300px;
            overflow-y: auto;
            background: white;
            padding: 0.75rem;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
    </style>
    <script>
        function copyToClipboard(btn) {
            const section = btn.closest('.summary-section, .transcript-section');
            const content = section.querySelector('.summary-content, .transcript-content');
            const text = content.innerText || content.textContent;
            navigator.clipboard.writeText(text).then(() => {
                const original = btn.textContent;
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = original, 1500);
            });
        }

        async function toggleTranscript(btn, videoId) {
            const videoItem = btn.closest('.video-item');
            const section = videoItem.querySelector('.transcript-section');
            const content = section.querySelector('.transcript-content');
            const isHidden = section.style.display === 'none' || !section.style.display;

            if (isHidden) {
                if (!content.dataset.loaded) {
                    content.textContent = 'Loading...';
                    section.style.display = 'block';
                    btn.textContent = 'Hide Transcript';
                    try {
                        const res = await fetch(`/api/video/${videoId}/transcript`);
                        const data = await res.json();
                        content.textContent = data.transcript || data.error;
                        content.dataset.loaded = 'true';
                    } catch (e) {
                        content.textContent = 'Failed to load transcript.';
                    }
                } else {
                    section.style.display = 'block';
                    btn.textContent = 'Hide Transcript';
                }
            } else {
                section.style.display = 'none';
                btn.textContent = 'Show Transcript';
            }
        }

        async function toggleSummaries(btn, videoId) {
            const videoItem = btn.closest('.video-item');
            const container = videoItem.querySelector('.summaries-container');
            const isHidden = container.style.display === 'none' || !container.style.display;

            if (isHidden) {
                if (!container.dataset.loaded) {
                    container.innerHTML = '<p>Loading summaries...</p>';
                    container.style.display = 'block';
                    btn.textContent = 'Hide Summaries';
                    try {
                        const res = await fetch(`/api/video/${videoId}/summaries`);
                        const data = await res.json();
                        if (data.summaries && data.summaries.length > 0) {
                            container.innerHTML = data.summaries.map(s => `
                                <div class="summary-section">
                                    <div class="summary-header">
                                        <div class="summary-type">${s.type.charAt(0).toUpperCase() + s.type.slice(1)} Summary:</div>
                                        <div class="btn-group">
                                            <button type="button" class="copy-btn" onclick="copyToClipboard(this)">Copy</button>
                                            <form method="post" action="/summarize/${videoId}/${s.type}" style="margin: 0;">
                                                <button type="submit" class="resummarize-btn">Resummarize</button>
                                            </form>
                                        </div>
                                    </div>
                                    <p class="summary-content">${s.content.replace(/\\n/g, '<br>')}</p>
                                </div>
                            `).join('');
                        } else {
                            container.innerHTML = '<p>No summaries available.</p>';
                        }
                        container.dataset.loaded = 'true';
                    } catch (e) {
                        container.innerHTML = '<p>Failed to load summaries.</p>';
                    }
                } else {
                    container.style.display = 'block';
                    btn.textContent = 'Hide Summaries';
                }
            } else {
                container.style.display = 'none';
                btn.textContent = 'Show Summaries';
            }
        }
    </script>
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
                <div class="summary-header">
                    <div class="summary-type">{{ summary_type|title }} Summary:</div>
                    <button type="button" class="copy-btn" onclick="copyToClipboard(this)">Copy</button>
                </div>
                <textarea readonly class="summary-content">{{ content }}</textarea>
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
                    <div class="btn-group" style="margin-bottom: 0.5rem;">
                        <button type="button" class="transcript-toggle" onclick="toggleTranscript(this, '{{ video.video_id }}')">Show Transcript</button>
                        <button type="button" class="transcript-toggle" onclick="toggleSummaries(this, '{{ video.video_id }}')">Show Summaries</button>
                    </div>
                    <div class="transcript-section">
                        <div class="transcript-header">
                            <span class="transcript-label">Transcript:</span>
                            <button type="button" class="copy-btn" onclick="copyToClipboard(this)">Copy</button>
                        </div>
                        <div class="transcript-content"></div>
                    </div>
                    <div class="summaries-container" style="display: none;"></div>
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
