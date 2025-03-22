import os
import re
import json
import argparse
import time
import tempfile
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import logging
import traceback
import base64
from colorama import Fore, Style, init
import yaml
import logging.handlers
from datetime import datetime
from pydantic import BaseModel, Field

# Core processing libraries
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pytube import YouTube
from yt_dlp import YoutubeDL
import cv2
from pydub import AudioSegment
import ollama

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Literal, Any
from enum import Enum

# Configure basic setup
load_dotenv()
init(autoreset=True)

# Constants
SIGNIFICANT_PAUSE_DURATION = 0.5  # seconds

class VisionAPIType(str, Enum):
    OLLAMA = "ollama"

class OllamaInference:
    """Handles Ollama model inference calls"""
    def __init__(self, model_name: str = "minicpm-v:latest"):
        self.model_name = model_name

    def __call__(self, prompt: str, image_path: str, format_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Ollama inference call with the given prompt and image"""
        return ollama.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }],
            format=format_schema,
            options={"temperature": 0}
        )

class ImageContent(BaseModel):
    """Content type for image-based messages"""
    type: Literal["image_url"] = "image_url"
    image_url: Dict[str, str] = Field(..., description="Image URL or base64 data")

class TextContent(BaseModel):
    """Content type for text-based messages"""
    type: Literal["text"] = "text"
    text: str = Field(..., description="Text content")

class Message(BaseModel):
    """Message structure for API requests"""
    role: str = Field(..., description="Role of the message sender")
    content: Union[str, List[Union[TextContent, ImageContent]]] = Field(..., description="Message content")
    images: Optional[List[str]] = Field(None, description="List of image paths (for Ollama API)")

class FrameDescription(BaseModel):
    """Frame analysis using theatrical direction concepts"""
    
    # Core elements (required)
    scene_description: str = Field("", description="Complete stage direction paragraph integrating all elements")
    location: str = Field("", description="Setting in theatrical terms")
    
    # Scene context
    time_of_day: Optional[str] = Field(None)
    atmosphere: Optional[str] = Field(None, description="Overall emotional tone/mood")
    
    # Character and action elements
    characters: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Characters with appearance, position, emotion")
    actions: Optional[List[str]] = Field(default_factory=list, description="Key physical actions and movements")
    
    # Visual elements
    visuals: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Technical elements including graphics, props, framing")
    ocr_text: Optional[str] = Field(None, description="Text visible in frame")

    @classmethod
    def error_state(cls, error_msg: str) -> Dict[str, Any]:
        """Generate a consistent error state for failed analysis"""
        return {
            "scene_description": f"Error processing image: {error_msg}",
            "location": "Error",
            "time_of_day": None,
            "atmosphere": None,
            "characters": [],
            "actions": [],
            "visuals": {},
            "ocr_text": None
        }

class FrameAnalysis(FrameDescription):
    """Frame analysis with metadata"""
    timestamp: float = Field(..., description="Timestamp in seconds")
    video_title: str = Field(..., description="Video title")
    
    @classmethod
    def error_state(cls, error_msg: str, timestamp: float = 0.0, video_title: str = "") -> Dict[str, Any]:
        """Generate a consistent error state for failed analysis with metadata"""
        base_error = FrameDescription.error_state(error_msg)
        base_error.update({
            'timestamp': timestamp,
            'video_title': video_title
        })
        return base_error

class ImageRequest(BaseModel):
    """Request structure for image analysis"""
    image_path: str
    frame_info: Dict[str, Union[float, str]]
    video_title: str
    previous_description: Optional[str] = None
    api_type: VisionAPIType
    system_prompt: str
    user_prompt: str

class ImageResponse(BaseModel):
    """Response structure for image analysis"""
    frame_analysis: FrameAnalysis
    raw_response: Optional[str] = None
    error: Optional[str] = None
    processing_time: float 

def setup_logging(output_folder: str) -> None:
    """Configure logging with both file and console handlers"""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(output_folder, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    json_formatter = logging.Formatter('%(message)s')
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # JSON Lines handler for structured logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_handler = logging.handlers.RotatingFileHandler(
        os.path.join(logs_dir, f'frame_analysis_{timestamp}.jsonl'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    json_handler.setFormatter(json_formatter)
    json_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(json_handler)
    
    # Log setup completion
    logging.info(json.dumps({
        'timestamp': datetime.now().isoformat(),
        'event': 'logging_initialized',
        'output_folder': output_folder
    }))

def log_frame_analysis(frame_info: Dict[str, Any], response: 'ImageResponse') -> None:
    """Log frame analysis in JSON Lines format with complete output capture"""
    try:
        # Convert frame analysis to dict with all fields
        analysis_dict = response.frame_analysis.model_dump(exclude_none=False)
        
        # Create comprehensive log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'frame_timestamp': frame_info.get('timestamp'),
            'frame_path': frame_info.get('path'),
            'processing_time': response.processing_time,
            'analysis': analysis_dict
        }
        
        # Add error information if present
        if response.error:
            log_entry['error'] = response.error
            
        # Add raw response if available (useful for debugging)
        if response.raw_response:
            log_entry['raw_response'] = response.raw_response
            
        # Log the entry as JSON
        logging.info(json.dumps(log_entry))
    except Exception as e:
        # Ensure logging itself doesn't fail
        error_msg = f"Error logging frame analysis: {str(e)}"
        logging.error(error_msg)
        # Attempt simplified logging
        simplified_entry = {
            'timestamp': datetime.now().isoformat(),
            'frame_timestamp': frame_info.get('timestamp'),
            'logging_error': str(e)
        }
        logging.info(json.dumps(simplified_entry))

class PromptManager:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path) if config_path else {}
        self.default_content_string = """
            The following image is a frame from a YouTube video. 
            Describe the image in detail, noting relevant visual elements 
            and how it relates to the video context.
        """
        self.default_system_prompt = """
            You are an AI assistant specialized in analyzing and describing 
            images from video frames. Focus on visual elements and their context.
        """

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            return {}
            
    def truncate_middle(self, text: str, max_length: int = 128) -> str:
        if not text or len(text) <= max_length:
            return text
        
        # Keep equal parts from start and end
        half_length = (max_length - 3) // 2
        return f"{text[:half_length]}...{text[-(max_length - half_length - 3):]}"

    def get_frame_prompt(self, frame_info: Dict[str, Any], 
                        video_title: str, 
                        previous_description: str = None,
                        surrounding_transcript: str = None,
                        video_description: str = None) -> str:
        content_template = self.config.get('content_string', self.default_content_string)
        
        # Truncate long descriptions
        max_length = self.config.get('max_description_length', 128)
        if video_description:
            video_description = self.truncate_middle(video_description, max_length)
        
        return content_template.format(
            video_title=video_title,
            timestamp=frame_info['timestamp'],
            previous_description=previous_description if previous_description else 'Not available.',
            surrounding_transcript=surrounding_transcript if surrounding_transcript else 'No transcript available.',
            video_description=video_description if video_description else 'No description available.'
        )

    def get_system_prompt(self, api: str) -> str:
        return self.config.get('system_prompt', self.default_system_prompt)

class AudioTranscriber:
    def __init__(self, model_id="openai/whisper-large-v3"):
        logging.info(f"Loading model {model_id}...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            
        )

    def transcribe(self, audio_file: str) -> Dict[str, Any]:
        """Transcribe audio file using the Whisper model with pause detection"""
        result = self.pipe(
            audio_file,
            return_timestamps=True,
            generate_kwargs={
                "task": "transcribe",
                "language": "en"
            }
        )
        
        # Format the result with pause detection
        previous_end = 0.0
        segments = []
        for chunk in result["chunks"]:
            start, end = chunk["timestamp"]
            pause_duration = start - previous_end if previous_end is not None else 0.0
            
            segments.append({
                "start": start,
                "end": end,
                "text": chunk["text"].strip(),
                "pause_before": pause_duration
            })
            previous_end = end
        
        transcription = {
            "text": result["text"],
            "segments": segments
        }
        
        print(f"{Fore.CYAN}Local Transcription:{Style.RESET_ALL} {transcription['text']}...")
        return transcription

def safe_filename(name: str, max_segment_length: int = 64) -> str:
    """Create a safe filename from the given string with length limits per segment"""
    # Replace spaces and invalid characters with underscores
    safe_name = re.sub(r'[<>:"/\\|?*\s,]', '_', name)
    # Remove or replace other problematic characters
    safe_name = re.sub(r'[\x00-\x1f]', '', safe_name)
    # Remove consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    safe_name = safe_name.strip('._')  # Remove leading/trailing dots and underscores
    
    # Split into segments and limit each segment's length
    segments = safe_name.split('_')
    shortened_segments = [seg[:max_segment_length] for seg in segments if seg]
    
    # Rejoin with underscores
    safe_name = '_'.join(shortened_segments)
    
    # Final length check (200 chars total)
    if len(safe_name) > 200:
        safe_name = safe_name[:200]
    
    return safe_name if safe_name else 'untitled'

def get_video_info(video_url: str) -> Dict[str, Any]:
    """Get video information with fallback methods"""
    # First try yt-dlp as it's more reliable
    try:
        with YoutubeDL() as ydl:
            info = ydl.extract_info(video_url, download=False)
            return {
                'title': info['title'],
                'description': info.get('description', 'No description available'),
                'duration': info.get('duration', 0),
                'channel': info.get('channel', 'Unknown'),
                'video_id': info.get('id', '')
            }
    except Exception as e:
        logging.warning(f"yt-dlp info extraction failed: {str(e)}, trying pytube...")
        
        # Fallback to pytube
        try:
            yt = YouTube(video_url)
            return {
                'title': yt.title,
                'description': yt.description,
                'duration': yt.length,
                'channel': yt.author,
                'video_id': yt.video_id
            }
        except Exception as e:
            logging.warning(f"pytube info extraction failed: {str(e)}, using minimal info...")
            
            # Last resort - extract minimal info from URL
            video_id = video_url.split('v=')[-1].split('&')[0]
            return {
                'title': f"video_{video_id}",
                'description': 'No description available',
                'duration': 0,
                'channel': 'Unknown',
                'video_id': video_id
            }

def download_video(video_url: str, output_folder: str) -> tuple:
    """Download video with improved FFmpeg handling"""
    info = get_video_info(video_url)
    
    # Use same shortened naming convention
    video_id = info.get('video_id', video_url.split('v=')[-1].split('&')[0])
    short_title = safe_filename(info['title'][:30])
    base_name = f"{short_title}_{video_id}"
    
    # First try downloading with simpler options
    ydl_opts = {
        'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]',
        'outtmpl': f'{output_folder}/{base_name}.%(ext)s',
        'keepvideo': True,
        'merge_output_format': 'mp4',  # Force MP4 output
    }
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        # Define expected output paths using base_name
        video_path = os.path.join(output_folder, f"{base_name}.mp4")
        audio_path = os.path.join(output_folder, f"{base_name}.mp3")
        
        # If video file exists in another format, convert it to mp4
        if not os.path.exists(video_path):
            for ext in ['.webm', '.mkv']:
                temp_path = os.path.join(output_folder, f"{base_name}{ext}")
                if os.path.exists(temp_path):
                    # Convert to mp4 using ffmpeg
                    os.system(f'ffmpeg -i "{temp_path}" -c copy "{video_path}"')
                    os.remove(temp_path)  # Remove original file
                    break
        
        # Extract audio
        if os.path.exists(video_path):
            try:
                from moviepy.editor import VideoFileClip
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(audio_path)
                video.close()
                logging.info("Audio extracted successfully using moviepy")
            except Exception as e:
                logging.warning(f"Moviepy audio extraction failed: {e}, trying pydub...")
                try:
                    audio = AudioSegment.from_file(video_path)
                    audio.export(audio_path, format="mp3")
                    logging.info("Audio extracted successfully using pydub")
                except Exception as e2:
                    logging.error(f"All audio extraction methods failed: {e2}")
                    raise

    except Exception as e:
        # Fallback to simpler format if initial download fails
        logging.warning(f"Initial download failed: {e}, trying simpler format...")
        ydl_opts = {
            'format': 'best[height<=480]/best',
            'outtmpl': f'{output_folder}/{base_name}.%(ext)s',
            'merge_output_format': 'mp4',  # Force MP4 output
            'postprocessors': [],
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    
    return info['duration'], base_name, info['description'], info['channel']


def process_video(video_url: str, output_folder: str, args):
    """Main video processing function with improved error handling"""
    audio_segments = []
    try:
        logging.info(f"Starting video processing: {video_url}")
        
        print(f"\n{Fore.BLUE}Processing video:{Style.RESET_ALL} {video_url}")
        
        # Get video info and create safe paths
        info = get_video_info(video_url)
        
        # Create a shorter base name using video ID or first few words
        video_id = info.get('video_id', video_url.split('v=')[-1].split('&')[0])
        short_title = safe_filename(info['title'][:30])  # Take first 30 chars of title
        base_name = f"{short_title}_{video_id}"  # Combine short title with video ID
        
        video_folder = os.path.join(output_folder, base_name)
        frames_folder = os.path.join(video_folder, "frames")
        os.makedirs(video_folder, exist_ok=True)
        os.makedirs(frames_folder, exist_ok=True)

        # Define file paths using shorter base_name
        video_path = os.path.join(video_folder, f"{base_name}.mp4")
        audio_path = os.path.join(video_folder, f"{base_name}.mp3")

        try:
            duration, _, video_description, channel = download_video(video_url, video_folder)
        except Exception as e:
            logging.error(f"Video download failed: {str(e)}")
            duration = info['duration']
            video_description = info['description']
            channel = info['channel']
            # If download failed, we can't continue
            raise

        # Verify files exist before processing
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Initialize processing components
        transcriber = AudioTranscriber(model_id="openai/whisper-large-v3")
        
        # Process audio with pause detection
        transcription = transcriber.transcribe(audio_path)
        
        # Extract frames based on pauses
        frame_data = extract_frames(video_path, transcription, frames_folder, min_pause=args.min_pause)
        
        # Initialize prompt manager
        prompt_manager = PromptManager(args.prompt_config if hasattr(args, 'prompt_config') else None)
        
        # Process frames and maintain timestamp mapping
        frame_descriptions = []
        previous_description = None
        
        for frame in frame_data:
            # Get surrounding transcript for this frame
            surrounding_transcript = get_surrounding_transcript(
                transcription=transcription, 
                timestamp=frame['timestamp'],
                window_seconds=args.transcript_window
            )
            
            frame_prompt = prompt_manager.get_frame_prompt(
                frame_info=frame,
                video_title=info['title'],
                previous_description=previous_description,
                surrounding_transcript=surrounding_transcript,
                video_description=video_description
            )
            system_prompt = prompt_manager.get_system_prompt(args.vision_api)
            
            response = process_image(
                image_path=frame['path'],
                frame_info=frame,
                video_title=info['title'],
                frame_prompt=frame_prompt,
                system_prompt=system_prompt,
                api=args.vision_api,
                previous_description=previous_description,
                ollama_model=args.ollama_model
            )
            
            # Store structured analysis
            frame_descriptions.append({
                "timestamp": frame['timestamp'],
                "pause_before": frame['pause_before'],
                "surrounding_transcript": surrounding_transcript,
                "processing_time": response.processing_time,
                "error": response.error,
                "analysis": response.frame_analysis.model_dump(exclude={"video_title", "timestamp"})  # Exclude metadata fields
            })
            
            previous_description = response.frame_analysis.scene_description

        # Generate output files
        metadata = {
            "title": info['title'],
            "channel": channel,
            "url": video_url,
            "description": video_description
        }
        
        # Save transcription
        with open(os.path.join(video_folder, f"{base_name}_trans.json"), "w") as f:
            json.dump(transcription, f, indent=2)
            
        # Save frame descriptions
        with open(os.path.join(video_folder, f"{base_name}_frames.json"), "w") as f:
            json.dump(frame_descriptions, f, indent=2)
            
        # Generate markdown with improved formatting
        markdown_path = os.path.join(video_folder, f"{base_name}_analysis.md")
        generate_markdown_analysis(
            metadata=metadata,
            transcription=transcription,
            frame_descriptions=frame_descriptions,
            output_path=markdown_path
        )

        logging.info(f"Processing completed for {info['title']}")
        logging.info(f"Output files saved in: {video_folder}")

    except Exception as e:
        print(f"{Fore.RED}Error processing video {video_url}: {str(e)}{Style.RESET_ALL}")
        error_details = {
            'timestamp': datetime.now().isoformat(),
            'event': 'video_processing_error',
            'video_url': video_url,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        logging.error(json.dumps(error_details))
    finally:
        # Clean up temporary files
        for segment_file in audio_segments:
            if os.path.exists(segment_file):
                os.remove(segment_file)

def extract_frames(video_path: str, transcription: Dict[str, Any], output_folder: str, min_pause: float = SIGNIFICANT_PAUSE_DURATION) -> List[Dict[str, Any]]:
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_data = []
    os.makedirs(output_folder, exist_ok=True)
    
    # Add start frame (2 seconds in)
    start_frame_index = int(2 * fps)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    ret, frame = video.read()
    if ret:
        frame_data.append({
            "filename": f"frame_2.00.jpg",
            "timestamp": 2.0,
            "path": os.path.join(output_folder, f"frame_2.00.jpg"),
            "pause_before": 0.0
        })
        cv2.imwrite(frame_data[-1]["path"], frame)
    
    # Extract frames at significant pauses
    for segment in transcription["segments"]:
        pause_duration = segment.get("pause_before", 0)
        if pause_duration >= min_pause:
            frame_index = int(segment["start"] * fps)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = video.read()
            
            if ret:
                frame_filename = f"frame_{segment['start']:.2f}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                frame_data.append({
                    "filename": frame_filename,
                    "timestamp": segment["start"],
                    "path": frame_path,
                    "pause_before": pause_duration
                })
    
    # Add end frame (2 seconds from end)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame_index = total_frames - int(2 * fps)
    video.set(cv2.CAP_PROP_POS_FRAMES, end_frame_index)
    ret, frame = video.read()
    if ret:
        end_time = (total_frames / fps) - 2
        frame_data.append({
            "filename": f"frame_{end_time:.2f}.jpg",
            "timestamp": end_time,
            "path": os.path.join(output_folder, f"frame_{end_time:.2f}.jpg"),
            "pause_before": 0.0
        })
        cv2.imwrite(frame_data[-1]["path"], frame)
    
    video.release()
    return frame_data

def process_image(
    image_path: str,
    frame_info: Dict[str, Any],
    video_title: str,
    frame_prompt: str,
    system_prompt: str,
    api: str,
    previous_description: Optional[str] = None,
    ollama_model: str = "minicpm-v:latest"  # Add default model parameter
) -> ImageResponse:
    """Process image using specified vision API with enhanced prompting and structured output"""
    start_time = time.time()
    
    # Verify file exists before processing
    if not os.path.exists(image_path):
        error_msg = f"File not found: {image_path}"
        processing_time = time.time() - start_time
        
        # Create error response
        frame_analysis = FrameAnalysis(
            **FrameAnalysis.error_state(
                error_msg,
                timestamp=frame_info["timestamp"],
                video_title=video_title
            )
        )
        
        response = ImageResponse(
            frame_analysis=frame_analysis,
            raw_response=None,
            error=error_msg,
            processing_time=processing_time
        )
        log_frame_analysis(frame_info, response)
        return response

    # Create structured prompt using Pydantic schema
    schema = FrameDescription.model_json_schema()
    formatted_schema = json.dumps(schema, indent=2)
    
    structured_prompt = f"""
    {frame_prompt}

    Analyze the image according to the following schema, using theatrical stage direction concepts:
    {formatted_schema}
    
    Focus on describing the scene as if it were a theatrical production with stage directions.
    """
    
    request = ImageRequest(
        image_path=image_path,
        frame_info=frame_info,
        video_title=video_title,
        previous_description=previous_description,
        api_type=VisionAPIType(api),
        system_prompt=system_prompt,
        user_prompt=structured_prompt
    )
    
    print(f"\n{Fore.GREEN}═══════════════════ API REQUEST DETAILS ═══════════════════{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}Image Path:{Style.RESET_ALL} {request.image_path}")
    print(f"{Fore.CYAN}Timestamp:{Style.RESET_ALL} {frame_info.get('timestamp', 0.0):.2f}s")
    print(f"{Fore.CYAN}API Type:{Style.RESET_ALL} {request.api_type.value}")
    
    print(f"\n{Fore.YELLOW}Complete API Request Message:{Style.RESET_ALL}")
    api_request = {
        'messages': [{
            'role': 'system',
            'content': system_prompt
        }, {
            'role': 'user',
            'content': structured_prompt,
            'images': [image_path]
        }],
        'format': schema,
        'options': {'temperature': 0}
    }
    print(json.dumps(api_request, indent=2))
    print(f"\n{Fore.GREEN}═══════════════════════════════════════════════════════════{Style.RESET_ALL}\n")
    
    max_retries = 8
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            if request.api_type == VisionAPIType.OLLAMA:
                ollama_client = OllamaInference(model_name=ollama_model)
                response = ollama_client(
                    prompt=request.user_prompt,
                    image_path=request.image_path,
                    format_schema=schema
                )
                raw_response = response['message']['content']
                
                print(f"\n{Fore.GREEN}═══════════════════ API RESPONSE DETAILS ═══════════════════{Style.RESET_ALL}")
                print(f"\n{Fore.YELLOW}Raw API Response:{Style.RESET_ALL}")
                # Convert response to dict for JSON serialization
                response_dict = {
                    'message': response['message'],
                    'model': response.get('model', ''),
                    'created_at': response.get('created_at', ''),
                    'done': response.get('done', False),
                    'total_duration': response.get('total_duration', 0),
                    'load_duration': response.get('load_duration', 0),
                    'prompt_eval_duration': response.get('prompt_eval_duration', 0)
                }
                print(json.dumps(response_dict, indent=2, default=str))
                print(f"\n{Fore.GREEN}═══════════════════════════════════════════════════════════{Style.RESET_ALL}\n")
                
                try:
                    frame_analysis = FrameDescription.model_validate_json(raw_response)
                    frame_data = frame_analysis.model_dump()
                    if "timestamp" in frame_data:
                        del frame_data["timestamp"]
                    if "video_title" in frame_data:
                        del frame_data["video_title"]
                        
                    frame_analysis = FrameAnalysis(
                        **frame_data,
                        timestamp=frame_info['timestamp'],
                        video_title=video_title
                    )
                except Exception as e:
                    raise ValueError(f"Failed to validate model response: {str(e)}")
            else:
                raise ValueError(f"Unsupported vision API: {request.api_type}")
            
            processing_time = time.time() - start_time
            
            response = ImageResponse(
                frame_analysis=frame_analysis,
                raw_response=raw_response,
                error=None,
                processing_time=processing_time
            )
            log_frame_analysis(frame_info, response)
            
            print(f"\n{Fore.GREEN}Analysis Complete{Style.RESET_ALL}")
            print(f"Processing Time: {response.processing_time:.2f}s\n")
            print(f"{Fore.CYAN}Validated Frame Analysis:{Style.RESET_ALL}")
            print(json.dumps(frame_analysis.model_dump(), indent=2))
            
            return response
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n{Fore.RED}Attempt {attempt + 1} failed:{Style.RESET_ALL} {str(e)}")
                print(f"Retrying in {retry_delay} seconds...\n")
                time.sleep(retry_delay)
            else:
                error_msg = f"Failed after {max_retries} attempts: {str(e)}"
                print(f"\n{Fore.RED}Error:{Style.RESET_ALL} {error_msg}\n")
                processing_time = time.time() - start_time
                
                frame_analysis = FrameAnalysis(
                    **FrameAnalysis.error_state(
                        str(e), 
                        timestamp=frame_info["timestamp"],
                        video_title=video_title
                    )
                )
                
                response = ImageResponse(
                    frame_analysis=frame_analysis,
                    raw_response=None,
                    error=str(e),
                    processing_time=processing_time
                )
                log_frame_analysis(frame_info, response)
                return response

def process_video(video_url: str, output_folder: str, args):
    """Main video processing function with improved error handling"""
    audio_segments = []
    try:
        logging.info(f"Starting video processing: {video_url}")
        
        print(f"\n{Fore.BLUE}Processing video:{Style.RESET_ALL} {video_url}")
        
        # Get video info and create safe paths
        info = get_video_info(video_url)
        
        # Create a shorter base name using video ID or first few words
        video_id = info.get('video_id', video_url.split('v=')[-1].split('&')[0])
        short_title = safe_filename(info['title'][:30])  # Take first 30 chars of title
        base_name = f"{short_title}_{video_id}"  # Combine short title with video ID
        
        video_folder = os.path.join(output_folder, base_name)
        frames_folder = os.path.join(video_folder, "frames")
        os.makedirs(video_folder, exist_ok=True)
        os.makedirs(frames_folder, exist_ok=True)

        # Define file paths using shorter base_name
        video_path = os.path.join(video_folder, f"{base_name}.mp4")
        audio_path = os.path.join(video_folder, f"{base_name}.mp3")

        try:
            duration, _, video_description, channel = download_video(video_url, video_folder)
        except Exception as e:
            logging.error(f"Video download failed: {str(e)}")
            duration = info['duration']
            video_description = info['description']
            channel = info['channel']
            # If download failed, we can't continue
            raise

        # Verify files exist before processing
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Initialize processing components
        transcriber = AudioTranscriber(model_id="openai/whisper-large-v3")
        
        # Process audio with pause detection
        transcription = transcriber.transcribe(audio_path)
        
        # Extract frames based on pauses
        frame_data = extract_frames(video_path, transcription, frames_folder, min_pause=args.min_pause)
        
        # Initialize prompt manager
        prompt_manager = PromptManager(args.prompt_config if hasattr(args, 'prompt_config') else None)
        
        # Process frames and maintain timestamp mapping
        frame_descriptions = []
        previous_description = None
        
        for frame in frame_data:
            # Get surrounding transcript for this frame
            surrounding_transcript = get_surrounding_transcript(
                transcription=transcription, 
                timestamp=frame['timestamp'],
                window_seconds=args.transcript_window
            )
            
            frame_prompt = prompt_manager.get_frame_prompt(
                frame_info=frame,
                video_title=info['title'],
                previous_description=previous_description,
                surrounding_transcript=surrounding_transcript,
                video_description=video_description
            )
            system_prompt = prompt_manager.get_system_prompt(args.vision_api)
            
            response = process_image(
                image_path=frame['path'],
                frame_info=frame,
                video_title=info['title'],
                frame_prompt=frame_prompt,
                system_prompt=system_prompt,
                api=args.vision_api,
                previous_description=previous_description
            )
            
            # Store structured analysis
            frame_descriptions.append({
                "timestamp": frame['timestamp'],
                "pause_before": frame['pause_before'],
                "surrounding_transcript": surrounding_transcript,
                "processing_time": response.processing_time,
                "error": response.error,
                "analysis": response.frame_analysis.model_dump(exclude={"video_title", "timestamp"})  # Exclude metadata fields
            })
            
            previous_description = response.frame_analysis.scene_description

        # Generate output files
        metadata = {
            "title": info['title'],
            "channel": channel,
            "url": video_url,
            "description": video_description
        }
        
        # Save transcription
        with open(os.path.join(video_folder, f"{base_name}_trans.json"), "w") as f:
            json.dump(transcription, f, indent=2)
            
        # Save frame descriptions
        with open(os.path.join(video_folder, f"{base_name}_frames.json"), "w") as f:
            json.dump(frame_descriptions, f, indent=2)
            
        # Generate markdown with improved formatting
        markdown_path = os.path.join(video_folder, f"{base_name}_analysis.md")
        generate_markdown_analysis(
            metadata=metadata,
            transcription=transcription,
            frame_descriptions=frame_descriptions,
            output_path=markdown_path
        )

        logging.info(f"Processing completed for {info['title']}")
        logging.info(f"Output files saved in: {video_folder}")

    except Exception as e:
        print(f"{Fore.RED}Error processing video {video_url}: {str(e)}{Style.RESET_ALL}")
        error_details = {
            'timestamp': datetime.now().isoformat(),
            'event': 'video_processing_error',
            'video_url': video_url,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        logging.error(json.dumps(error_details))
    finally:
        # Clean up temporary files
        for segment_file in audio_segments:
            if os.path.exists(segment_file):
                os.remove(segment_file)

def generate_markdown_analysis(
    metadata: Dict[str, str],
    transcription: Dict[str, Any],
    frame_descriptions: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Generate a markdown file that weaves together transcript and frame descriptions.
    Uses a simple, direct approach to convert JSON to markdown.
    """
    try:
        # Create a simple timeline of transcript segments and frames
        timeline = []
        
        # Group transcript segments into chunks based on significant pauses
        if transcription and "segments" in transcription:
            current_chunk = []
            chunk_start = None
            
            for segment in transcription["segments"]:
                if isinstance(segment, dict) and "start" in segment and "text" in segment:
                    # Start new chunk if this is first segment or follows significant pause
                    if chunk_start is None or segment.get("pause_before", 0) >= SIGNIFICANT_PAUSE_DURATION:
                        # Add previous chunk to timeline if it exists
                        if current_chunk:
                            timeline.append({
                                "type": "transcript_chunk",
                                "timestamp": chunk_start,
                                "content": " ".join([seg["text"].strip() for seg in current_chunk]),
                                "pause_after": segment.get("pause_before", 0)
                            })
                        # Start new chunk
                        current_chunk = [segment]
                        chunk_start = segment["start"]
                    else:
                        # Add to current chunk
                        current_chunk.append(segment)
            
            # Add final chunk if it exists
            if current_chunk:
                timeline.append({
                    "type": "transcript_chunk",
                    "timestamp": chunk_start,
                    "content": " ".join([seg["text"].strip() for seg in current_chunk]),
                    "pause_after": 0.0
                })
        
        # Add frame descriptions
        if frame_descriptions:
            for frame in frame_descriptions:
                if isinstance(frame, dict) and "timestamp" in frame:
                    timeline.append({
                        "type": "frame",
                        "timestamp": frame["timestamp"],
                        "data": frame
                    })
        
        # Sort timeline by timestamp
        timeline = sorted(timeline, key=lambda x: x.get("timestamp", 0))
        
        with open(output_path, "w", encoding="utf-8") as f:
            # Write metadata section
            f.write(f"# {metadata.get('title', 'Video Analysis')}\n\n")
            f.write(f"## Channel: {metadata.get('channel', 'Unknown')}\n")
            f.write(f"[Video URL]({metadata.get('url', '')})\n\n")
            f.write("## Description\n")
            f.write(f"{metadata.get('description', 'No description available')}\n\n")
            f.write("## Timeline\n\n")
            f.write("_" * 32 + "\n\n")
            
            # Process timeline
            for event in timeline:
                try:
                    if event["type"] == "transcript_chunk":
                        # Write timestamp and content as a single quote block
                        #f.write(f"[{event['timestamp']:.2f}s]\n")
                        f.write(f"> {event['content']}\n\n")
                    
                    elif event["type"] == "frame":
                        frame_data = event["data"]
                        
                        # Write frame timestamp
                        #f.write(f"\n[{frame_data.get('timestamp', 0):.2f}s]")
                        
                        # Add pause duration if available
                        #if "pause_before" in frame_data and frame_data["pause_before"] >= SIGNIFICANT_PAUSE_DURATION:
                        #    f.write(f" *[{frame_data['pause_before']:.1f}s pause]*")
                        
                        f.write("\n\n")
                        
                        # Convert analysis to markdown using the simplified approach
                        try:
                            if "analysis" in frame_data and frame_data["analysis"]:
                                json_to_markdown(frame_data["analysis"], f)
                                f.write("\n---\n\n")
                            elif "raw_analysis" in frame_data and frame_data["raw_analysis"]:
                                f.write(f"{frame_data['raw_analysis']}\n\n---\n\n")
                            else:
                                f.write("*No analysis available for this frame*\n\n---\n\n")
                        except Exception as e:
                            logging.error(f"Error processing frame analysis: {str(e)}")
                            f.write(f"*Error processing frame analysis: {str(e)}*\n\n---\n\n")
                except Exception as e:
                    logging.error(f"Error processing timeline event: {str(e)}")
                    f.write(f"*Error processing timeline event: {str(e)}*\n\n")
            
            # Add footer
            f.write("\n---\n")
            f.write("THE ABOVE IS A YOUTUBE VIDEO RETOLD AS A 'PLAY' FOR LLM PLAINTEXT UNDERSTANDING.\n '>' SIGNIFIES DIALOGUE WITH SCENE DESCRIPTIONS IN 'MARKDOWN' FORMAT.\n")
    except Exception as e:
        logging.error(f"Error generating markdown: {str(e)}\n{traceback.format_exc()}")
        # Create a minimal markdown file with error information
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Error Generating Video Analysis\n\n")
            f.write(f"An error occurred while generating the markdown analysis: {str(e)}\n\n")
            f.write("## Available Metadata\n\n")
            try:
                for key, value in metadata.items():
                    f.write(f"- **{key}**: {value}\n")
            except:
                f.write("*Error processing metadata*\n")
            
            f.write("\n## Error Details\n\n")
            f.write("```\n")
            f.write(traceback.format_exc())
            f.write("\n```\n")

def json_to_markdown(data, file_obj, level=0, prefix=""):
    """
    Convert any JSON structure to markdown with minimal conditional logic.
    
    Args:
        data: The data to convert (can be any JSON-serializable type)
        file_obj: The file object to write to
        level: The current nesting level (0 = top level)
        prefix: Prefix for the current line (used for nested structures)
    """
    # Skip None values entirely
    if data is None:
        return
        
    # Handle different data types
    if isinstance(data, dict):
        # Skip empty dictionaries
        if not data:
            return
            
        # Process each key-value pair
        for key, value in data.items():
            # Skip internal fields and empty values
            if key.startswith('_'):
                continue
                
            # Format the key
            formatted_key = key.replace('_', ' ').title() if isinstance(key, str) else str(key)
            
            # Handle the value based on its type and emptiness
            if value is None or value == "" or (isinstance(value, (list, dict)) and not value):
                # Skip empty values
                continue
                
            if level == 0:
                # Top level keys get bold headers
                file_obj.write(f"**{formatted_key}:**\n")
                
                # Handle different value types
                if isinstance(value, dict):
                    json_to_markdown(value, file_obj, level + 1)
                    file_obj.write("\n")
                elif isinstance(value, list):
                    json_to_markdown(value, file_obj, level + 1)
                    file_obj.write("\n")
                else:
                    file_obj.write(f"{value}\n\n")
            else:
                # Nested keys get bullet points
                indent = "  " * (level - 1)
                current_prefix = f"{prefix}{indent} "
                
                # Handle different value types
                if isinstance(value, dict):
                    file_obj.write(f"{current_prefix}**{formatted_key}:**\n")
                    json_to_markdown(value, file_obj, level + 1, prefix)
                elif isinstance(value, list):
                    file_obj.write(f"{current_prefix}*{formatted_key}:*\n")
                    json_to_markdown(value, file_obj, level + 1, prefix)
                else:
                    file_obj.write(f"{current_prefix}{formatted_key}: {value}\n")
    
    elif isinstance(data, list):
        # Skip empty lists
        if not data:
            return
            
        # Process each item in the list
        indent = "  " * level
        
        for item in data:
            # Skip None and empty values
            if item is None or item == "" or (isinstance(item, (list, dict)) and not item):
                continue
                
            # Handle different item types
            if isinstance(item, dict):
                # Special case for type/description pairs
                if "type" in item and "description" in item and item["description"]:
                    item_type = item["type"]
                    if isinstance(item_type, str):
                        item_type = item_type.title()
                    else:
                        item_type = str(item_type)
                    file_obj.write(f"{prefix}{indent} {item_type}: {item['description']}\n")
                else:
                    # For other dictionaries, create a bullet and process recursively
                    file_obj.write(f"{prefix}{indent}-\n")
                    json_to_markdown(item, file_obj, level + 1, prefix)
            elif isinstance(item, list):
                # For nested lists, create a bullet and process recursively
                file_obj.write(f"{prefix}{indent}-\n")
                json_to_markdown(item, file_obj, level + 1, prefix)
            else:
                # For simple values, create a bullet point
                file_obj.write(f"{prefix}{indent}- {item}\n")
    
    else:
        # For primitive types, just write the value
        indent = "  " * level
        file_obj.write(f"{prefix}{indent}{data}\n")

def format_section_content(content: str, max_line_length: int = 80) -> str:
    """Format content to wrap at specified line length while preserving sentences and paragraphs."""
    paragraphs = content.split('\n\n')
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        # Split into sentences (accounting for common abbreviations)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
        formatted_sentences = []
        
        for sentence in sentences:
            # If sentence is shorter than max_line_length, keep it as is
            if len(sentence) <= max_line_length:
                formatted_sentences.append(sentence)
                continue
                
            # For longer sentences, try to break at natural points (commas, semicolons)
            parts = re.split(r'(?<=[,;])\s+', sentence)
            current_line = []
            current_length = 0
            
            for part in parts:
                if current_length + len(part) + len(current_line) <= max_line_length:
                    current_line.append(part)
                    current_length += len(part)
                else:
                    if current_line:
                        formatted_sentences.append(' '.join(current_line))
                    current_line = [part]
                    current_length = len(part)
            
            if current_line:
                formatted_sentences.append(' '.join(current_line))
        
        formatted_paragraphs.append(' '.join(formatted_sentences))
    
    return '\n\n'.join(formatted_paragraphs)

def get_surrounding_transcript(transcription: Dict[str, Any], timestamp: float, window_seconds: float = 10.0) -> str:
    """Get transcript segments surrounding a given timestamp within a specified time window."""
    surrounding_segments = []
    
    # Find segments that fall within the time window
    for segment in transcription.get("segments", []):
        try:
            if (segment["start"] >= timestamp - window_seconds and 
                segment["start"] <= timestamp + window_seconds):
                surrounding_segments.append(segment["text"])
        except (KeyError, TypeError):
            continue
    
    return " ".join(surrounding_segments) if surrounding_segments else "No transcript available for this timestamp."

def main():
    parser = argparse.ArgumentParser(description="Process YouTube videos with enhanced transcription and frame analysis")
    parser.add_argument("input", help="YouTube URL or path to text file containing YouTube URLs")
    parser.add_argument("output_folder", help="Path to output folder")
    parser.add_argument("--min_pause", type=float, default=SIGNIFICANT_PAUSE_DURATION, 
                       help="Minimum pause duration (in seconds) to trigger frame capture")
    parser.add_argument("--prompt_config", help="Path to YAML file containing prompt configurations")
    parser.add_argument("--content_string", 
        default="Describe the image, focusing on key visual elements and context:", 
        help="Custom prompt for image description")
    parser.add_argument("--vision_api", choices=["ollama"], default="ollama",
                      help="Vision API to use for image processing")
    parser.add_argument("--transcript_window", type=float, default=10.0,
                      help="Time window in seconds for surrounding transcript context")
    parser.add_argument("--ollama_model", default="minicpm-v:latest",
                      help="Ollama model to use for image processing (default: minicpm-v:latest)")

    args = parser.parse_args()
    
    # Create base output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Setup logging at the base output folder level
    setup_logging(args.output_folder)
    
    # Determine if input is a URL or file
    if args.input.startswith(('http://', 'https://', 'www.', 'youtube.com', 'youtu.be')):
        video_urls = [args.input]
        logging.info(f"Processing single video URL: {args.input}")
    else:
        # Check if file exists before trying to read it
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        with open(args.input) as f:
            video_urls = [url.strip() for url in f.readlines() if url.strip()]
        logging.info(f"Processing {len(video_urls)} videos from file: {args.input}")
    
    # Process each video URL
    for url in video_urls:
        try:
            # Create video-specific output folder using safe filename
            info = get_video_info(url)
            video_id = info.get('video_id', url.split('v=')[-1].split('&')[0])
            short_title = safe_filename(info['title'][:30])
            video_folder = os.path.join(args.output_folder, f"{short_title}_{video_id}")
            os.makedirs(video_folder, exist_ok=True)
            
            # Process the video
            process_video(url, video_folder, args)
        except Exception as e:
            logging.error(f"Failed to process video {url}: {str(e)}")
            continue

if __name__ == "__main__":
    main()