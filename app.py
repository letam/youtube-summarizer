import hashlib
import os
import re
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request
from openai import OpenAI
from sqlalchemy.orm import defer
from werkzeug.utils import secure_filename
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

# Audio configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance", "uploads")
MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25MB (Whisper API limit)
ALLOWED_AUDIO_EXTENSIONS = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"}

# Database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
db.init_app(app)

# Initialize database and upload folder
with app.app_context():
    db.create_all()
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== Helper Functions ====


def extract_video_id(url):
    # Match regular YouTube URLs (v=VIDEO_ID)
    match = re.search(r"(?:v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def fetch_transcript(video_id):
    try:
        # Check if transcript exists in database
        existing_transcript = Transcript.query.filter_by(
            source_type="youtube", source_id=video_id
        ).first()
        if existing_transcript:
            return existing_transcript.transcript_text

        # If not in database, fetch from YouTube
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([t["text"] for t in transcript])

        # Save to database
        new_transcript = Transcript(
            source_type="youtube",
            source_id=video_id,
            transcript_text=full_text,
        )
        db.session.add(new_transcript)
        db.session.commit()

        return full_text
    except (TranscriptsDisabled, NoTranscriptFound):
        return None


# ==== Audio Helper Functions ====


def allowed_audio_file(filename):
    """Check if file extension is allowed for audio uploads."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


def generate_audio_source_id(filename):
    """Generate unique source_id for audio files."""
    unique_string = f"{filename}_{datetime.utcnow().isoformat()}"
    hash_value = hashlib.sha256(unique_string.encode()).hexdigest()[:16]
    return f"audio_{hash_value}"


def save_audio_file(uploaded_file, source_id):
    """Save uploaded audio file to uploads folder.

    Returns:
        Path to saved file
    """
    filename = secure_filename(uploaded_file.filename)
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else "mp3"
    save_filename = f"{source_id}.{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, save_filename)
    uploaded_file.save(file_path)
    return file_path


def transcribe_audio(file_path):
    """Transcribe audio file using OpenAI Whisper API.

    Returns:
        dict with 'text' and optional 'duration' keys
    """
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
        )
    return {
        "text": response.text,
        "duration": getattr(response, "duration", None),
    }


def process_audio_upload(uploaded_file):
    """Process uploaded audio file: validate, save, transcribe, and store.

    Args:
        uploaded_file: Flask FileStorage object

    Returns:
        Transcript record

    Raises:
        ValueError: Invalid file type or size
        RuntimeError: Transcription failed
    """
    if not uploaded_file or not uploaded_file.filename:
        raise ValueError("No file provided")

    if not allowed_audio_file(uploaded_file.filename):
        raise ValueError(f"Invalid file type. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}")

    # Generate unique ID
    source_id = generate_audio_source_id(uploaded_file.filename)
    original_filename = secure_filename(uploaded_file.filename)

    # Save file
    file_path = save_audio_file(uploaded_file, source_id)

    try:
        # Transcribe
        result = transcribe_audio(file_path)

        # Save to database
        transcript = Transcript(
            source_type="audio",
            source_id=source_id,
            transcript_text=result["text"],
            original_filename=original_filename,
            file_path=file_path,
            audio_duration_seconds=int(result["duration"]) if result.get("duration") else None,
        )
        db.session.add(transcript)
        db.session.commit()

        return transcript

    except Exception as e:
        # Clean up file on failure
        if os.path.exists(file_path):
            os.unlink(file_path)
        raise RuntimeError(f"Transcription failed: {e}") from e


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
                    source_type="youtube", source_id=video_id
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

    # Get all processed YouTube videos (defer transcript_text for eco-friendly loading)
    processed_videos = (
        Transcript.query.filter_by(source_type="youtube")
        .options(defer(Transcript.transcript_text))
        .order_by(Transcript.created_at.desc())
        .all()
    )
    return render_template_string(
        TEMPLATE, summaries=summaries, error=error, processed_videos=processed_videos
    )


@app.route("/summarize/<video_id>/<summary_type>", methods=["POST"])
def summarize(video_id, summary_type):
    error = ""
    transcript_record = Transcript.query.filter_by(
        source_type="youtube", source_id=video_id
    ).first()

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
        Transcript.query.filter_by(source_type="youtube")
        .options(defer(Transcript.transcript_text))
        .order_by(Transcript.created_at.desc())
        .all()
    )
    return render_template_string(
        TEMPLATE, summaries={}, error=error, processed_videos=processed_videos
    )


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    """API endpoint to summarize a video and return JSON."""
    data = request.get_json()
    url = data.get("url", "")
    video_id = extract_video_id(url)

    if not video_id:
        return jsonify({"error": "Invalid YouTube URL."}), 400

    transcript = fetch_transcript(video_id)
    if not transcript:
        return jsonify({"error": "Transcript not available for this video."}), 404

    summaries = summarize_transcript(transcript)

    # Update summaries in database
    transcript_record = Transcript.query.filter_by(
        source_type="youtube", source_id=video_id
    ).first()
    if transcript_record:
        Summary.query.filter_by(transcript_id=transcript_record.id).delete()
        for summary_type, content in summaries.items():
            new_summary = Summary(
                transcript_id=transcript_record.id,
                summary_type=summary_type,
                content=content,
            )
            db.session.add(new_summary)
        db.session.commit()

    return jsonify(
        {
            "video_id": video_id,
            "timestamp": transcript_record.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "summaries": [{"type": k, "content": v} for k, v in summaries.items()],
        }
    )


@app.route("/api/video/<video_id>/resummarize/<summary_type>", methods=["POST"])
def api_resummarize(video_id, summary_type):
    """API endpoint to regenerate a single summary type."""
    transcript_record = Transcript.query.filter_by(
        source_type="youtube", source_id=video_id
    ).first()

    if not transcript_record:
        return jsonify({"error": "Video not found."}), 404
    if summary_type not in SUMMARY_INSTRUCTIONS:
        return jsonify({"error": "Invalid summary type."}), 400

    new_content = generate_summary(transcript_record.transcript_text, summary_type)

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

    return jsonify({"type": summary_type, "content": new_content})


@app.route("/api/video/<video_id>/transcript")
def get_transcript(video_id):
    """API endpoint to fetch transcript on demand."""
    transcript_record = Transcript.query.filter_by(
        source_type="youtube", source_id=video_id
    ).first()
    if not transcript_record:
        return jsonify({"error": "Video not found"}), 404
    return jsonify({"transcript": transcript_record.transcript_text})


@app.route("/api/video/<video_id>/summaries")
def get_summaries(video_id):
    """API endpoint to fetch summaries on demand."""
    transcript_record = Transcript.query.filter_by(
        source_type="youtube", source_id=video_id
    ).first()
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
        .status-message {
            padding: 0.75rem;
            margin-top: 1rem;
            border-radius: 4px;
            display: none;
        }
        .status-message.loading {
            display: block;
            background: #e3f2fd;
            color: #1565c0;
        }
        .status-message.error {
            display: block;
            background: #ffebee;
            color: #c62828;
        }
        .status-message.success {
            display: block;
            background: #e8f5e9;
            color: #2e7d32;
        }
        #submit-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .new-result {
            margin-top: 1rem;
            padding: 1rem;
            border: 2px solid #28a745;
            border-radius: 4px;
            background: #f8fff8;
        }
        .new-result h3 { margin-top: 0; }
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

        function createVideoItemHTML(videoId, timestamp, summaries) {
            const summariesHTML = summaries.map(s => `
                <div class="summary-section">
                    <div class="summary-header">
                        <div class="summary-type">${s.type.charAt(0).toUpperCase() + s.type.slice(1)} Summary:</div>
                        <div class="btn-group">
                            <button type="button" class="copy-btn" onclick="copyToClipboard(this)">Copy</button>
                            <button type="button" class="resummarize-btn" onclick="resummarize(this, '${videoId}', '${s.type}')">Resummarize</button>
                        </div>
                    </div>
                    <p class="summary-content">${s.content.replace(/\\n/g, '<br>')}</p>
                </div>
            `).join('');

            return `
                <div class="video-item" data-video-id="${videoId}">
                    <h3>
                        <a href="https://youtube.com/watch?v=${videoId}" class="video-link" target="_blank">
                            Video ID: ${videoId}
                        </a>
                    </h3>
                    <div class="timestamp">Processed: ${timestamp}</div>
                    <div class="btn-group" style="margin-bottom: 0.5rem;">
                        <button type="button" class="transcript-toggle" onclick="toggleTranscript(this, '${videoId}')">Show Transcript</button>
                        <button type="button" class="transcript-toggle" onclick="toggleSummaries(this, '${videoId}')">Show Summaries</button>
                    </div>
                    <div class="transcript-section">
                        <div class="transcript-header">
                            <span class="transcript-label">Transcript:</span>
                            <button type="button" class="copy-btn" onclick="copyToClipboard(this)">Copy</button>
                        </div>
                        <div class="transcript-content"></div>
                    </div>
                    <div class="summaries-container" style="display: none;" data-loaded="true">${summariesHTML}</div>
                </div>
            `;
        }

        async function handleSubmit(e) {
            e.preventDefault();
            const form = e.target;
            const urlInput = form.querySelector('input[name="url"]');
            const submitBtn = form.querySelector('#submit-btn');
            const statusDiv = document.getElementById('status-message');
            const resultDiv = document.getElementById('new-result');
            const videoList = document.getElementById('video-list-container');

            const url = urlInput.value.trim();
            if (!url) return;

            // Update UI for loading state
            submitBtn.disabled = true;
            submitBtn.textContent = 'Summarizing...';
            statusDiv.className = 'status-message loading';
            statusDiv.textContent = 'Fetching transcript and generating summaries...';
            resultDiv.innerHTML = '';
            resultDiv.style.display = 'none';

            try {
                const res = await fetch('/api/summarize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url })
                });
                const data = await res.json();

                if (!res.ok) {
                    throw new Error(data.error || 'Failed to summarize video');
                }

                // Success - update UI
                statusDiv.className = 'status-message success';
                statusDiv.textContent = 'Summary complete!';

                // Remove existing entry for this video if present
                const existing = videoList.querySelector(`[data-video-id="${data.video_id}"]`);
                if (existing) existing.remove();

                // Add new video item at the top
                const newItemHTML = createVideoItemHTML(data.video_id, data.timestamp, data.summaries);
                videoList.insertAdjacentHTML('afterbegin', newItemHTML);

                // Remove "no videos" message if present
                const noVideos = videoList.querySelector('p');
                if (noVideos && noVideos.textContent.includes('No videos')) {
                    noVideos.remove();
                }

                // Clear input
                urlInput.value = '';

                // Hide status after delay
                setTimeout(() => { statusDiv.style.display = 'none'; }, 3000);

            } catch (err) {
                statusDiv.className = 'status-message error';
                statusDiv.textContent = err.message;
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Summarize';
            }
        }

        async function resummarize(btn, videoId, summaryType) {
            const section = btn.closest('.summary-section');
            const content = section.querySelector('.summary-content');
            const originalText = content.innerHTML;

            btn.disabled = true;
            btn.textContent = 'Working...';
            content.innerHTML = '<em>Regenerating summary...</em>';

            try {
                const res = await fetch(`/api/video/${videoId}/resummarize/${summaryType}`, {
                    method: 'POST'
                });
                const data = await res.json();

                if (!res.ok) {
                    throw new Error(data.error || 'Failed to resummarize');
                }

                content.innerHTML = data.content.replace(/\\n/g, '<br>');
            } catch (err) {
                content.innerHTML = originalText;
                alert('Failed to resummarize: ' + err.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Resummarize';
            }
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
    <form onsubmit="handleSubmit(event)">
        <input name="url" type="text" placeholder="Enter YouTube URL" style="width:100%; padding: 0.5rem;" required>
        <button type="submit" id="submit-btn" style="margin-top:1rem; padding:0.5rem 1rem;">Summarize</button>
    </form>
    <div id="status-message" class="status-message"></div>
    <div id="new-result"></div>

    <div class="video-list">
        <h2>Processed Videos</h2>
        <div id="video-list-container">
        {% if processed_videos %}
            {% for video in processed_videos %}
                <div class="video-item" data-video-id="{{ video.source_id }}">
                    <h3>
                        <a href="https://youtube.com/watch?v={{ video.source_id }}" class="video-link" target="_blank">
                            Video ID: {{ video.source_id }}
                        </a>
                    </h3>
                    <div class="timestamp">
                        Processed: {{ video.created_at.strftime('%Y-%m-%d %H:%M:%S') }}
                    </div>
                    <div class="btn-group" style="margin-bottom: 0.5rem;">
                        <button type="button" class="transcript-toggle" onclick="toggleTranscript(this, '{{ video.source_id }}')">Show Transcript</button>
                        <button type="button" class="transcript-toggle" onclick="toggleSummaries(this, '{{ video.source_id }}')">Show Summaries</button>
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
    </div>
</body>
</html>
"""

# ==== Run Server ====

if __name__ == "__main__":
    app.run(debug=True)
