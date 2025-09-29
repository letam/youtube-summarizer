#!/usr/bin/env python3
"""
YouTube Transcript Summarizer - CLI Version

A command-line tool to summarize YouTube video transcripts using OpenAI's GPT models.
"""

import argparse
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled

# Import functions from the main app
from app import (
    extract_video_id,
    fetch_transcript,
    summarize_transcript,
    estimate_tokens,
    chunk_transcript,
    find_split_point,
    MAX_TOKENS_PER_CHUNK,
    CHUNK_OVERLAP,
    MODEL
)
from models import db, Transcript, Summary

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
        "reset": "\033[0m"
    }

    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def format_summary_output(summaries: dict, video_id: str) -> None:
    """Format and display summaries in a nice terminal layout."""
    print_colored(f"\n{'='*60}", "cyan")
    print_colored(f"YouTube Video: https://youtube.com/watch?v={video_id}", "bold")
    print_colored(f"{'='*60}\n", "cyan")

    summary_types = {
        'concise': ("📝", "cyan"),
        'detailed': ("📋", "blue"),
        'key_points': ("🔑", "green")
    }

    for summary_type, (icon, color) in summary_types.items():
        if summary_type in summaries:
            print_colored(f"{icon} {summary_type.upper()} SUMMARY", color)
            print_colored("-" * 40, "white")

            # Wrap text for better readability
            content = summaries[summary_type]
            lines = content.split('\n')
            for line in lines:
                if line.strip():
                    print(f"  {line.strip()}")

            print()  # Empty line between summaries


def list_processed_videos(limit: int = 10) -> None:
    """List previously processed videos."""
    def _list_videos():
        videos = Transcript.query.order_by(Transcript.created_at.desc()).limit(limit).all()

        if not videos:
            print_colored("No videos have been processed yet.", "yellow")
            return

        print_colored(f"\n{'='*50}", "cyan")
        print_colored("RECENTLY PROCESSED VIDEOS", "bold")
        print_colored(f"{'='*50}\n", "cyan")

        for i, video in enumerate(videos, 1):
            print_colored(f"{i}. Video ID: {video.video_id}", "white")
            print_colored(f"   Processed: {video.created_at.strftime('%Y-%m-%d %H:%M:%S')}", "yellow")
            print_colored(f"   URL: https://youtube.com/watch?v={video.video_id}", "blue")

            # Show summary types available
            summary_types = [s.summary_type for s in video.summaries]
            if summary_types:
                print_colored(f"   Summaries: {', '.join(summary_types)}", "green")

            print()

    db_manager.execute_in_context(_list_videos)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="YouTube Transcript Summarizer - CLI Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s --list
  %(prog)s --list --limit 5
  %(prog)s --help
        """
    )

    parser.add_argument(
        "url",
        nargs="?",
        help="YouTube video URL to summarize"
    )

    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List previously processed videos"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of videos to list (default: 10)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed processing information"
    )

    args = parser.parse_args()

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print_colored("Error: OPENAI_API_KEY environment variable not set.", "red")
        print_colored("Please set your OpenAI API key in a .env file or environment.", "yellow")
        sys.exit(1)

    # Handle list command
    if args.list:
        list_processed_videos(args.limit)
        return

    # Handle URL input
    if not args.url:
        parser.print_help()
        print_colored("\nError: Please provide a YouTube URL or use --list to see processed videos.", "red")
        sys.exit(1)

    # Process the URL
    print_colored("🎬 YouTube Transcript Summarizer", "bold")
    print_colored(f"Processing URL: {args.url}", "blue")

    # Extract video ID
    video_id = extract_video_id(args.url)
    if not video_id:
        print_colored("❌ Error: Invalid YouTube URL format.", "red")
        print_colored("Make sure the URL contains a valid YouTube video ID.", "yellow")
        sys.exit(1)

    if args.verbose:
        print_colored(f"✅ Video ID extracted: {video_id}", "green")
        print_colored("🔍 Fetching transcript...", "yellow")

    def _fetch_transcript():
        return fetch_transcript(video_id)

    transcript = db_manager.execute_in_context(_fetch_transcript)

    if not transcript:
        print_colored("❌ Error: Could not fetch transcript for this video.", "red")
        print_colored("This might be because:", "yellow")
        print_colored("  - The video doesn't have captions/transcripts available", "white")
        print_colored("  - The video has disabled transcripts", "white")
        print_colored("  - The video is private or unavailable", "white")
        sys.exit(1)

    def _process_transcript():
        if args.verbose:
            transcript_tokens = estimate_tokens(transcript)
            print_colored(f"✅ Transcript fetched: {len(transcript)} characters, ~{transcript_tokens} tokens", "green")

            chunks = chunk_transcript(transcript)
            print_colored(f"📦 Split into {len(chunks)} chunks for processing", "blue")

        return summarize_transcript(transcript)

    # Generate summaries
    print_colored("🤖 Generating summaries with OpenAI...", "magenta")

    try:
        summaries = db_manager.execute_in_context(_process_transcript)

        if args.verbose:
            print_colored("✅ Summaries generated successfully", "green")

        # Display results
        format_summary_output(summaries, video_id)

        print_colored("✨ Summary complete! Video data saved to database.", "green")

    except Exception as e:
        print_colored(f"❌ Error generating summaries: {str(e)}", "red")
        if args.verbose:
            import traceback
            print_colored(traceback.format_exc(), "red")
        sys.exit(1)


if __name__ == "__main__":
    main()
