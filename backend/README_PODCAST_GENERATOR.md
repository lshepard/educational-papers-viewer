# Standalone Podcast Generator

Generate podcast episodes (script + MP3 audio) from text prompts using Google's Gemini AI.

## Features

- ✅ Simple command-line interface
- ✅ No database or research tools - just direct text-to-podcast
- ✅ Multi-speaker audio with natural-sounding voices (Gemini 2.5 Pro TTS)
- ✅ Outputs both script (.txt) and audio (.mp3) files
- ✅ Customizable speakers, models, and output directories

## Requirements

- Python 3.11+
- GEMINI_API_KEY environment variable (in `.env` file or exported)

## Installation

```bash
# Already installed with the backend dependencies
cd backend
uv sync
```

## Usage

### Quick Start

```bash
# Generate from command line
uv run python generate_podcast_from_prompt.py "Create a podcast about the history of jazz music"

# The script will create:
# - output/podcast_YYYYMMDD_HHMMSS.txt (script)
# - output/podcast_YYYYMMDD_HHMMSS.mp3 (audio)
```

### Advanced Usage

```bash
# Read prompt from file
uv run python generate_podcast_from_prompt.py --file my_prompt.txt

# Interactive mode (multi-line prompt)
uv run python generate_podcast_from_prompt.py --interactive

# Custom output directory
uv run python generate_podcast_from_prompt.py "Your prompt" --output my_podcasts/

# Custom speaker names
uv run python generate_podcast_from_prompt.py "Your prompt" --speakers Alice Bob

# Use different Gemini model
uv run python generate_podcast_from_prompt.py "Your prompt" --model gemini-2.0-flash-exp
```

### Help

```bash
uv run python generate_podcast_from_prompt.py --help
```

## Writing Good Prompts

The script works best with clear, specific prompts. Here are some examples:

### Good Prompts

```
"Create a 3-minute podcast about why the sky is blue. Make it fun and educational for kids aged 8-12."

"Generate a podcast discussing the benefits of meditation. Target audience is busy professionals who are new to meditation. Keep it under 5 minutes."

"Create an entertaining podcast about the history of video games, focusing on the 1980s arcade era. Include some fun facts and trivia."
```

### Prompt Structure

For best results, include:
1. **Topic**: What the podcast is about
2. **Audience**: Who it's for (optional but helpful)
3. **Length**: How long it should be (e.g., "3-5 minutes", "brief")
4. **Tone**: Educational, entertaining, conversational, etc.

## Output

The script generates two files:

1. **Script (.txt)**: The podcast dialogue in this format:
   ```
   Alex: Welcome to the show!
   Sam: Thanks for having me!
   ...
   ```

2. **Audio (.mp3)**: High-quality MP3 audio with:
   - Natural-sounding multi-speaker voices
   - Proper pacing and intonation
   - Ready to play or distribute

## Examples

### Example 1: Educational Content

```bash
uv run python generate_podcast_from_prompt.py \
  "Create a 4-minute educational podcast about how computers work. \
   Explain binary, CPU, and memory in simple terms for beginners."
```

### Example 2: From File

Create a file `prompt.txt`:
```
Create an engaging podcast about the importance of sleep.

Target audience: College students who often sacrifice sleep.

Include:
- Why sleep matters for learning and memory
- Recommended sleep duration
- Tips for better sleep habits
- Keep it conversational and relatable

Length: 3-5 minutes
```

Then run:
```bash
uv run python generate_podcast_from_prompt.py --file prompt.txt
```

### Example 3: Custom Speakers

```bash
uv run python generate_podcast_from_prompt.py \
  "Create a podcast about space exploration" \
  --speakers Neil Buzz
```

## Troubleshooting

### "GEMINI_API_KEY not found in environment"

Make sure you have a `.env` file in the backend directory with:
```
GEMINI_API_KEY=your_api_key_here
```

Or export it:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Script too short/long

Adjust your prompt to specify the desired length:
- "Create a brief 2-minute podcast..."
- "Create a detailed 10-minute podcast..."
- "Keep it under 5 minutes..."

### Audio quality issues

The script uses Gemini 2.5 Pro TTS with high-quality voices. If you experience issues:
- Check your internet connection
- Ensure you have the latest version of dependencies: `uv sync`

## Technical Details

- **Script Generation**: Uses Gemini 2.0 Flash for fast, high-quality script generation
- **Audio Generation**: Uses Gemini 2.5 Pro TTS with multi-speaker support
- **Voices**:
  - Default: Kore (Alex) and Puck (Sam)
  - Natural pacing and intonation
  - No robotic artifacts
- **Output Format**: 128kbps MP3, mono

## Limitations

- Requires internet connection (calls Gemini API)
- Generation time: 30 seconds to 2 minutes depending on length
- Maximum length: ~10 minutes (longer scripts may be cut off)
- Two speakers only (can be customized via --speakers)

## Integration with Main Backend

This is a standalone script that uses shared library functions from `lib/podcast_generator.py`.

If you want to integrate custom podcast generation into the main backend API, see the `/podcast/generate-custom` endpoint in `main.py`.
