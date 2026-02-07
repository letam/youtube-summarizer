#!/usr/bin/env python3
"""
YouTube Transcript Summarizer - CLI Version

A command-line tool to summarize YouTube video transcripts and audio files using OpenAI's GPT models.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

# Import functions from the main app
from app import (
    ALLOWED_AUDIO_EXTENSIONS,
    allowed_audio_file,
    chunk_transcript,
    estimate_tokens,
    extract_video_id,
    fetch_transcript,
    generate_audio_source_id,
    save_audio_file,
    summarize_transcript,
    transcribe_audio,
    UPLOAD_FOLDER,
)
from models import Transcript, db

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class DatabaseManager:
    """Manages database operations for CLI usage."""

    def __init__(self):
        self.app = None
        self._setup_app()

    def _setup_app(self):
        """Create and configure Flask app for database operations."""
        from flask import Flask

        self.app = Flask(__name__)
        self.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
        self.app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

        # Initialize database with the app
        db.init_app(self.app)

        # Create tables
        with self.app.app_context():
            db.create_all()

    def get_app(self):
        """Get the Flask app instance."""
        return self.app

    def execute_in_context(self, func, *args, **kwargs):
        """Execute a function within the Flask application context."""
        with self.app.app_context():
            return func(*args, **kwargs)


# Global database manager instance
db_manager = DatabaseManager()


def setup_database():
    """Legacy function for backward compatibility - returns the app instance."""
    return db_manager.get_app()


def print_colored(text: str, color: str = "white") -> None:
    """Print colored text to terminal."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }

    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def format_summary_output(summaries: dict, video_id: str) -> None:
    """Format and display summaries in a nice terminal layout."""
    print_colored(f"\n{'='*60}", "cyan")
    print_colored(f"YouTube Video: https://youtube.com/watch?v={video_id}", "bold")
    print_colored(f"{'='*60}\n", "cyan")

    summary_types = {
        "concise": ("📝", "cyan"),
        "detailed": ("📋", "blue"),
        "key_points": ("🔑", "green"),
    }

    for summary_type, (icon, color) in summary_types.items():
        if summary_type in summaries:
            print_colored(f"{icon} {summary_type.upper()} SUMMARY", color)
            print_colored("-" * 40, "white")

            # Wrap text for better readability
            content = summaries[summary_type]
            lines = content.split("\n")
            for line in lines:
                if line.strip():
                    print(f"  {line.strip()}")

            print()  # Empty line between summaries


def list_processed_videos(limit: int = 10, source_type: str = None) -> None:
    """List previously processed videos/audio."""

    def _list_items():
        query = Transcript.query
        if source_type:
            query = query.filter_by(source_type=source_type)
        items = query.order_by(Transcript.created_at.desc()).limit(limit).all()

        if not items:
            print_colored("No items have been processed yet.", "yellow")
            return

        print_colored(f"\n{'='*50}", "cyan")
        title = "RECENTLY PROCESSED"
        if source_type == "youtube":
            title += " VIDEOS"
        elif source_type == "audio":
            title += " AUDIO"
        else:
            title += " ITEMS"
        print_colored(title, "bold")
        print_colored(f"{'='*50}\n", "cyan")

        for i, item in enumerate(items, 1):
            if item.source_type == "youtube":
                print_colored(f"{i}. [YouTube] Video ID: {item.source_id}", "white")
                print_colored(
                    f"   URL: https://youtube.com/watch?v={item.source_id}", "blue"
                )
            else:
                print_colored(f"{i}. [Audio] {item.original_filename}", "white")
                print_colored(f"   Source ID: {item.source_id}", "blue")
                if item.source_duration:
                    mins, secs = divmod(item.source_duration, 60)
                    print_colored(f"   Duration: {mins}m {secs}s", "blue")

            print_colored(
                f"   Processed: {item.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                "yellow",
            )

            # Show summary types available
            summary_types = [s.summary_type for s in item.summaries]
            if summary_types:
                print_colored(f"   Summaries: {', '.join(summary_types)}", "green")

            print()

    db_manager.execute_in_context(_list_items)


def format_audio_output(transcript_record, summaries: dict = None) -> None:
    """Format and display audio transcription results."""
    print_colored(f"\n{'='*60}", "cyan")
    print_colored(f"Audio File: {transcript_record.original_filename}", "bold")
    print_colored(f"Source ID: {transcript_record.source_id}", "blue")
    if transcript_record.source_duration:
        mins, secs = divmod(transcript_record.source_duration, 60)
        print_colored(f"Duration: {mins}m {secs}s", "blue")
    print_colored(f"{'='*60}\n", "cyan")

    if summaries:
        summary_types = {
            "concise": ("📝", "cyan"),
            "detailed": ("📋", "blue"),
            "key_points": ("🔑", "green"),
        }

        for summary_type, (icon, color) in summary_types.items():
            if summary_type in summaries:
                print_colored(f"{icon} {summary_type.upper()} SUMMARY", color)
                print_colored("-" * 40, "white")

                content = summaries[summary_type]
                lines = content.split("\n")
                for line in lines:
                    if line.strip():
                        print(f"  {line.strip()}")

                print()


def process_audio(file_path: str, summarize: bool = False, verbose: bool = False) -> None:
    """Process an audio file and optionally generate summaries."""
    print_colored("🎵 Audio Transcription", "bold")
    print_colored(f"Processing file: {file_path}", "blue")

    # Validate file exists
    if not os.path.exists(file_path):
        print_colored(f"❌ Error: File not found: {file_path}", "red")
        sys.exit(1)

    # Validate file extension
    filename = os.path.basename(file_path)
    if not allowed_audio_file(filename):
        print_colored(f"❌ Error: Invalid file type.", "red")
        print_colored(f"Allowed types: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}", "yellow")
        sys.exit(1)

    # Check file size
    file_size = os.path.getsize(file_path)
    max_size = 25 * 1024 * 1024
    if file_size > max_size:
        print_colored(f"❌ Error: File too large ({file_size // (1024*1024)}MB).", "red")
        print_colored(f"Maximum size: {max_size // (1024*1024)}MB", "yellow")
        sys.exit(1)

    if verbose:
        print_colored(f"✅ File validated: {filename} ({file_size // 1024}KB)", "green")
        print_colored("🔍 Transcribing audio...", "yellow")

    def _process_audio():
        # Generate source ID
        source_id = generate_audio_source_id(filename)

        # Ensure upload folder exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Copy file to uploads folder
        ext = filename.rsplit(".", 1)[1].lower() if "." in filename else "mp3"
        save_path = os.path.join(UPLOAD_FOLDER, f"{source_id}.{ext}")

        import shutil
        shutil.copy2(file_path, save_path)

        try:
            # Transcribe
            result = transcribe_audio(save_path)

            if verbose:
                print_colored(f"✅ Transcription complete: {len(result['text'])} characters", "green")

            # Save to database
            transcript = Transcript(
                source_type="audio",
                source_id=source_id,
                transcript_text=result["text"],
                original_filename=filename,
                file_path=save_path,
                source_duration=int(result["duration"]) if result.get("duration") else None,
            )
            db.session.add(transcript)
            db.session.commit()

            return transcript, result["text"]

        except Exception as e:
            # Clean up on failure
            if os.path.exists(save_path):
                os.unlink(save_path)
            raise

    try:
        print_colored("🤖 Transcribing with OpenAI Whisper...", "magenta")
        transcript_record, transcript_text = db_manager.execute_in_context(_process_audio)

        summaries = None
        if summarize:
            if verbose:
                tokens = estimate_tokens(transcript_text)
                print_colored(f"📦 Transcript: ~{tokens} tokens", "blue")
                chunks = chunk_transcript(transcript_text)
                print_colored(f"📦 Split into {len(chunks)} chunks for summarization", "blue")

            print_colored("🤖 Generating summaries with OpenAI...", "magenta")

            def _summarize():
                return summarize_transcript(transcript_text)

            summaries = db_manager.execute_in_context(_summarize)

            # Save summaries to database
            def _save_summaries():
                from models import Summary
                for summary_type, content in summaries.items():
                    new_summary = Summary(
                        transcript_id=transcript_record.id,
                        summary_type=summary_type,
                        content=content,
                    )
                    db.session.add(new_summary)
                db.session.commit()

            db_manager.execute_in_context(_save_summaries)

        format_audio_output(transcript_record, summaries)
        print_colored("✨ Processing complete! Data saved to database.", "green")
        print_colored(f"   Source ID: {transcript_record.source_id}", "blue")

    except Exception as e:
        print_colored(f"❌ Error processing audio: {e!s}", "red")
        if verbose:
            import traceback
            print_colored(traceback.format_exc(), "red")
        sys.exit(1)


def process_video(url: str, verbose: bool = False) -> None:
    """Process a YouTube video URL and generate summaries."""
    print_colored("🎬 YouTube Transcript Summarizer", "bold")
    print_colored(f"Processing URL: {url}", "blue")

    # Extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        print_colored("❌ Error: Invalid YouTube URL format.", "red")
        print_colored("Make sure the URL contains a valid YouTube video ID.", "yellow")
        sys.exit(1)

    if verbose:
        print_colored(f"✅ Video ID extracted: {video_id}", "green")
        print_colored("🔍 Fetching transcript...", "yellow")

    def _fetch_transcript():
        return fetch_transcript(video_id)

    transcript = db_manager.execute_in_context(_fetch_transcript)

    if not transcript:
        print_colored("❌ Error: Could not fetch transcript for this video.", "red")
        print_colored("This might be because:", "yellow")
        print_colored(
            "  - The video doesn't have captions/transcripts available", "white"
        )
        print_colored("  - The video has disabled transcripts", "white")
        print_colored("  - The video is private or unavailable", "white")
        sys.exit(1)

    def _process_transcript():
        if verbose:
            transcript_tokens = estimate_tokens(transcript)
            print_colored(
                f"✅ Transcript fetched: {len(transcript)} characters, ~{transcript_tokens} tokens",
                "green",
            )

            chunks = chunk_transcript(transcript)
            print_colored(f"📦 Split into {len(chunks)} chunks for processing", "blue")

        return summarize_transcript(transcript)

    # Generate summaries
    print_colored("🤖 Generating summaries with OpenAI...", "magenta")

    try:
        summaries = db_manager.execute_in_context(_process_transcript)

        if verbose:
            print_colored("✅ Summaries generated successfully", "green")

        # Display results
        format_summary_output(summaries, video_id)

        print_colored("✨ Summary complete! Video data saved to database.", "green")

    except Exception as e:
        print_colored(f"❌ Error generating summaries: {e!s}", "red")
        if verbose:
            import traceback

            print_colored(traceback.format_exc(), "red")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="YouTube Transcript Summarizer - CLI Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s --audio /path/to/recording.mp3
  %(prog)s --audio /path/to/recording.mp3 --summarize
  %(prog)s --list
  %(prog)s --list --limit 5
  %(prog)s --help
        """,
    )

    parser.add_argument("url", nargs="?", help="YouTube video URL to summarize")

    parser.add_argument(
        "-a", "--audio",
        type=str,
        metavar="FILE",
        help="Path to audio file to transcribe",
    )

    parser.add_argument(
        "-s", "--summarize",
        action="store_true",
        help="Generate summaries (for audio files)",
    )

    parser.add_argument(
        "-l", "--list", action="store_true", help="List previously processed items"
    )

    parser.add_argument(
        "--list-type",
        type=str,
        choices=["all", "youtube", "audio"],
        default="all",
        help="Filter list by type (default: all)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of items to list (default: 10)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed processing information",
    )

    args = parser.parse_args()

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print_colored("Error: OPENAI_API_KEY environment variable not set.", "red")
        print_colored(
            "Please set your OpenAI API key in a .env file or environment.", "yellow"
        )
        sys.exit(1)

    # Handle list command
    if args.list:
        source_type = None if args.list_type == "all" else args.list_type
        list_processed_videos(args.limit, source_type)
        return

    # Handle audio file input
    if args.audio:
        process_audio(args.audio, args.summarize, args.verbose)
        return

    # Handle URL input
    if not args.url:
        parser.print_help()
        print_colored(
            "\nError: Please provide a YouTube URL, --audio file, or use --list.",
            "red",
        )
        sys.exit(1)

    process_video(args.url, args.verbose)


if __name__ == "__main__":
    main()
