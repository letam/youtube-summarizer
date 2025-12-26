# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YouTube Transcript Summarizer - A Python application that fetches YouTube video transcripts and generates AI-powered summaries using OpenAI's GPT models. Available as both a Flask web app and CLI tool.

## Common Commands

```bash
just install          # Create venv and install production dependencies
just install-dev      # Install development dependencies
just run              # Run Flask development server
just cli <url>        # Summarize a YouTube video via CLI
just cli --list       # List previously processed videos
just fmt              # Format code with Ruff
just check            # Lint code without fixing
just test             # Run pytest
just clean            # Remove cache files
```

## Architecture

**Entry Points:**
- `app.py` - Flask web application with inline HTML template
- `cli.py` - Command-line interface that reuses core functions from app.py

**Database Layer:**
- `models.py` - SQLAlchemy models: `Transcript` (stores video transcripts) and `Summary` (stores generated summaries with types: concise, detailed, key_points)
- SQLite database at `instance/app.db`

**Core Processing Flow:**
1. `extract_video_id()` - Parse YouTube URL to get video ID
2. `fetch_transcript()` - Get transcript from DB cache or YouTube API
3. `chunk_transcript()` - Split long transcripts for API token limits
4. `summarize_transcript()` - Generate three summary types via OpenAI

**Key Dependencies:**
- `youtube-transcript-api` for fetching transcripts
- `openai` SDK using `client.responses.create()` method
- Flask-SQLAlchemy for persistence

## Configuration

- Environment: Copy `.env.example` to `.env` and set `OPENAI_API_KEY`
- Model: Currently uses `gpt-4o` (configurable via `MODEL` constant in app.py)
- Linting: Ruff configured in `pyproject.toml` with comprehensive rules
