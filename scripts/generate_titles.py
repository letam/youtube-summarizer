#!/usr/bin/env python3
"""
Generate titles for video records based on their transcript/summary content.

Usage:
    python scripts/generate_titles.py          # Generate titles for records without one
    python scripts/generate_titles.py --all    # Regenerate titles for all records
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from app import app
from models import Transcript, db

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"  # Use smaller model for title generation


def generate_title(transcript: Transcript) -> str:
    """Generate a title for a video based on its summary or transcript."""
    # Prefer concise summary if available, otherwise use transcript
    concise_summary = next(
        (s.content for s in transcript.summaries if s.summary_type == "concise"),
        None,
    )

    if concise_summary:
        content = concise_summary
        prompt = "Based on this video summary, generate a concise, descriptive title (max 100 characters). Return only the title, no quotes or extra text."
    else:
        # Use first 2000 chars of transcript if no summary
        content = transcript.transcript_text[:2000]
        prompt = "Based on this video transcript excerpt, generate a concise, descriptive title (max 100 characters). Return only the title, no quotes or extra text."

    response = client.responses.create(
        model=MODEL,
        instructions=prompt,
        input=content,
        temperature=0.7,
        max_output_tokens=50,
    )

    return response.output_text.strip().strip("\"'")


def main():
    parser = argparse.ArgumentParser(description="Generate titles for video records")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerate titles for all records, not just those without titles",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    with app.app_context():
        if args.all:
            transcripts = Transcript.query.all()
        else:
            transcripts = Transcript.query.filter(
                (Transcript.generated_title.is_(None))
                | (Transcript.generated_title == "")
            ).all()

        if not transcripts:
            print("No records to process.")
            return

        print(f"Processing {len(transcripts)} record(s)...\n")

        for transcript in transcripts:
            print(f"Video ID: {transcript.video_id}")

            try:
                title = generate_title(transcript)
                print(f"  Generated: {title}")

                if not args.dry_run:
                    transcript.generated_title = title
                    db.session.commit()
                    print("  Saved.")
                else:
                    print("  (dry run - not saved)")

            except Exception as e:
                print(f"  Error: {e}")

            print()

        print("Done.")


if __name__ == "__main__":
    main()
