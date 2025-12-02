#!/usr/bin/env python3
"""
Standalone Podcast Generator from Prompt

Generate a podcast episode (script + MP3 audio) from a text prompt.
No research tools, no database - just direct text-to-podcast conversion.

Usage:
    python generate_podcast_from_prompt.py "Your prompt here"
    python generate_podcast_from_prompt.py --file prompt.txt
    python generate_podcast_from_prompt.py --interactive
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

from lib.podcasts.audio import generate_audio_from_script, convert_to_mp3

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_script(prompt: str, genai_client: genai.Client, model: str = "gemini-2.0-flash-exp") -> str:
    """
    Generate a podcast script from a prompt using Gemini.

    Args:
        prompt: The text prompt describing what the podcast should be about
        genai_client: Gemini AI client
        model: Gemini model to use

    Returns:
        Generated script text
    """
    logger.info("Generating podcast script...")

    # Build the full prompt with podcast formatting instructions
    full_prompt = f"""You are creating a podcast script for an engaging conversation between two hosts.

**Your Task:**
{prompt}

**Format Requirements:**
- Create a natural, conversational dialogue between Alex and Sam
- Use this exact format:
  Alex: [dialogue]
  Sam: [dialogue]
  Alex: [dialogue]
  ...
- Make it engaging and fun
- Target 3-5 minutes when spoken (roughly 450-750 words)
- Use natural, conversational language - NO "um", "like", or filler words
- Be informative but accessible

Generate the podcast script now:"""

    try:
        response = genai_client.models.generate_content(
            model=model,
            contents=full_prompt
        )

        script = response.text.strip()
        logger.info(f"Generated script: {len(script)} characters")

        return script

    except Exception as e:
        logger.error(f"Failed to generate script: {e}", exc_info=True)
        raise


def save_files(script: str, audio_data: bytes, output_dir: str = "output") -> tuple[str, str]:
    """
    Save script and audio to files.

    Args:
        script: The podcast script text
        audio_data: MP3 audio bytes
        output_dir: Directory to save files to

    Returns:
        Tuple of (script_path, audio_path)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"podcast_{timestamp}"

    # Save script
    script_path = output_path / f"{base_filename}.txt"
    with open(script_path, 'w') as f:
        f.write(script)
    logger.info(f"💾 Script saved to: {script_path}")

    # Save audio
    audio_path = output_path / f"{base_filename}.mp3"
    with open(audio_path, 'wb') as f:
        f.write(audio_data)
    logger.info(f"🎵 Audio saved to: {audio_path}")

    return str(script_path), str(audio_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a podcast from a text prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate from command line:
    %(prog)s "Create a podcast about the history of jazz music"

  Generate from file:
    %(prog)s --file my_prompt.txt

  Interactive mode:
    %(prog)s --interactive

  Specify output directory:
    %(prog)s "Your prompt" --output my_podcasts/

  Use different Gemini model:
    %(prog)s "Your prompt" --model gemini-2.0-flash-exp
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'prompt',
        nargs='?',
        help='The podcast prompt text'
    )
    input_group.add_argument(
        '--file', '-f',
        help='Read prompt from a text file'
    )
    input_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Enter prompt interactively'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        default='output',
        help='Output directory for generated files (default: output/)'
    )

    # Model options
    parser.add_argument(
        '--model', '-m',
        default='gemini-2.0-flash-exp',
        help='Gemini model to use (default: gemini-2.0-flash-exp)'
    )

    # Speaker options
    parser.add_argument(
        '--speakers',
        nargs=2,
        default=['Alex', 'Sam'],
        help='Names of the two podcast hosts (default: Alex Sam)'
    )

    args = parser.parse_args()

    # Get prompt from appropriate source
    if args.file:
        logger.info(f"Reading prompt from file: {args.file}")
        with open(args.file, 'r') as f:
            prompt = f.read().strip()
    elif args.interactive:
        print("Enter your podcast prompt (press Ctrl+D or Ctrl+Z when done):")
        print("-" * 60)
        prompt = sys.stdin.read().strip()
        print("-" * 60)
    else:
        prompt = args.prompt

    if not prompt:
        logger.error("Empty prompt provided")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("STANDALONE PODCAST GENERATOR")
    logger.info("=" * 80)
    logger.info(f"\nPrompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Speakers: {args.speakers[0]} and {args.speakers[1]}")
    logger.info(f"Output directory: {args.output}")

    # Check for API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not found in environment")
        logger.error("Please set it in .env file or export it")
        sys.exit(1)

    try:
        # Initialize Gemini client
        logger.info("\n1. Initializing Gemini AI client...")
        genai_client = genai.Client(api_key=gemini_api_key)
        logger.info("   ✅ Client initialized")

        # Generate script
        logger.info("\n2. Generating podcast script...")
        script = generate_script(prompt, genai_client, model=args.model)
        logger.info(f"   ✅ Script generated ({len(script)} characters)")

        # Generate audio
        logger.info("\n3. Generating audio with Gemini 2.0 Flash TTS...")
        audio_data = generate_audio_from_script(
            script=script,
            genai_client=genai_client,
            speaker_names=args.speakers
        )
        logger.info(f"   ✅ Audio generated ({len(audio_data)} bytes)")

        # Convert to MP3
        logger.info("\n4. Converting to MP3...")
        mp3_data = convert_to_mp3(audio_data)
        logger.info(f"   ✅ MP3 conversion complete ({len(mp3_data)} bytes)")

        # Save files
        logger.info("\n5. Saving files...")
        script_path, audio_path = save_files(script, mp3_data, args.output)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("✅ PODCAST GENERATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nFiles created:")
        logger.info(f"  📝 Script: {script_path}")
        logger.info(f"  🎵 Audio:  {audio_path}")
        logger.info(f"\nYou can now play the podcast with:")
        logger.info(f"  mpg123 {audio_path}")
        logger.info(f"  or open it in your favorite audio player")

        return 0

    except KeyboardInterrupt:
        logger.info("\n\nCancelled by user")
        return 130
    except Exception as e:
        logger.error(f"\n❌ Failed to generate podcast: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
