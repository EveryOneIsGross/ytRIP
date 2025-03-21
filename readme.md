# YouTube Video Processor

A Python tool for downloading, transcribing, and analyzing YouTube videos.

## Features

- Download YouTube videos
- Transcribe audio using local models or API services
- Extract and analyze video frames
- Generate markdown summaries with transcriptions and frame descriptions

## Requirements

- Python 3.7+
- FFmpeg
- Various Python libraries (see requirements.txt)

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install FFmpeg if not already present on your system

## Usage

```bash
python ytrip.py <input> <output_folder> [options]
```

- `<input>`: YouTube URL or path to a text file containing YouTube URLs (one per line)
- `<output_folder>`: Directory to store processed files

### Options

```
--min_pause FLOAT      Minimum pause duration (in seconds) to trigger frame capture (default: 0.5)
--prompt_config PATH   Path to YAML file containing prompt configurations
--content_string TEXT  Custom prompt for image description
--vision_api {ollama}  Vision API to use for image processing (default: ollama)
--transcript_window FLOAT  Time window in seconds for surrounding transcript context (default: 10.0)
--ollama_model TEXT    Ollama model to use for image processing (default: minicpm-v:latest)
```

## Examples

### Process a single YouTube video

```bash
python ytrip.py https://www.youtube.com/watch?v=dQw4w9WgXcQ output_folder
```

### Process multiple videos from a file

```bash
python ytrip.py videos.txt output_folder
```

Where `videos.txt` contains:
```
https://www.youtube.com/watch?v=dQw4w9WgXcQ
https://www.youtube.com/watch?v=9bZkp7q19f0
```

### Use custom pause duration for frame capture

```bash
python ytrip.py https://www.youtube.com/watch?v=dQw4w9WgXcQ output_folder --min_pause 1.0
```

### Use specific Ollama model

```bash
python ytrip.py https://www.youtube.com/watch?v=dQw4w9WgXcQ output_folder --ollama_model llava:latest
```

## Output

For each video, the tool generates:
- Transcription (JSON)
- Frame descriptions (JSON) 
- Summary markdown file with theatrical-style scene descriptions
