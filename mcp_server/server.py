#!/usr/bin/env python3
"""
Kokoro TTS MCP Server - Text-to-Speech as an MCP Tool

Serves Kokoro TTS (built-in voices via pykokoro) as Model Context Protocol tools.
Supports both built-in voice synthesis and voice cloning from reference audio.

Based on the Boson AI Higgs Audio TTS API pattern:
  {
    "model": "kokoro",
    "input": "Hello, this is a test.",
    "voice": "af_bella",          # optional: built-in voice name
    "ref_audio": "/path/to.wav",   # optional: reference audio for cloning
    "ref_text": "...",             # optional: transcript of ref_audio
    "quality": "fp16"              # optional: fp32, fp16, q8, q4
  }

Usage:
  python -m mcp_server.server [--host 0.0.0.0] [--port 8765]
"""

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, CallToolResult, ListToolsResult, ServerResult
    from mcp.server.models import InitializationOptions
    from mcp.types import ToolsCapability, ServerCapabilities
except ImportError as e:
    print(f"ERROR: MCP package not found. Install with: pip install mcp")
    sys.exit(1)

# Optional kokoro imports for direct synthesis (preferred when available)
PYKOKORO_AVAILABLE = False
try:
    from pykokoro import KokoroPipeline, PipelineConfig
    from pykokoro.generation_config import GenerationConfig
    import numpy as np
    try:
        import soundfile as sf
        SOUNDFILE_AVAILABLE = True
    except ImportError:
        from scipy.io import wavfile
        SOUNDFILE_AVAILABLE = True  # fallback
    PYKOKORO_AVAILABLE = True
except ImportError:
    PYKOKORO_AVAILABLE = False

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("kokoro-mcp-server")

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "voice": "af_bella",              # Default built-in voice
    "device": "cpu",                   # cpu, cuda, auto, coreml
    "model_quality": "fp16",           # fp32, fp16, q8, q4
    "sample_rate": 24000,              # WAV sample rate
    "output_dir": "/tmp/kokoro-tts",   # Where to save output files
    "sampler_path": None,             # Path to sampler.py (auto-discovered)
    "clone_path": None,               # Path to clone.py (auto-discovered)
    "hf_repo": "PatnaikAshish/kokoclone",   # HuggingFace repo for kokoro.onnx (voice cloning)
    
    # Chunking configuration - prevents cutoff on large blocks of text
    "max_chunks_per_request": 30,      # Maximum number of audio chunks per request
    "silence_duration_samples": 5000,  # Silence gap between chunks (~200ms at 24kHz)
}

# Chunking constants (similar to sampler.py line 62)
MAX_CHARS_PER_CHUNK = 500             # Conservative limit for kokoro inference
DEFAULT_MAX_SENTENCES = 2            # Default: process 2 sentences per chunk

# Model name that maps to Kokoro - matches Boson API convention
MODEL_NAME = "kokoro"


# ============================================================================
# Auto-discovery of kokoro scripts
# ============================================================================

def discover_kokoro_scripts():
    """Find sampler.py and clone.py in the filesystem."""
    # Common locations to search
    search_paths = [
        Path("/app/kokoro"),              # Docker mount point  
        Path("/app/kokoro-original"),     # Docker-compose mounted kokoro (ro)
        Path.home() / "kokoro",           # Home directory
        Path(__file__).parent.parent.parent / "kokoro",  # Relative to this script
    ]

    sampler_path = None
    clone_path = None

    for base in search_paths:
        if not base.exists():
            continue
        s = base / "sampler.py"
        c = base / "clone.py"
        if s.exists():
            sampler_path = str(s)
        if c.exists():
            clone_path = str(c)

    # Update defaults if found
    if sampler_path:
        DEFAULT_CONFIG["sampler_path"] = sampler_path
    if clone_path:
        DEFAULT_CONFIG["clone_path"] = clone_path

    return DEFAULT_CONFIG


# ============================================================================
# Direct TTS Engine (when pykokoro is available)
# ============================================================================

class KokoroEngine:
    """Direct kokoro synthesis engine using pykokoro library."""

    def __init__(self, voice: str = "af_bella", device: str = "cpu", quality: str = "fp16"):
        self.voice = voice
        self.device = device
        self.quality = quality
        self.pipeline = None

    def initialize(self):
        """Initialize the kokoro pipeline."""
        if not PYKOKORO_AVAILABLE:
            raise RuntimeError("pykokoro library is not installed")

        logger.info(f"Initializing KokoroEngine: voice={self.voice}, device={self.device}, quality={self.quality}")

        generation_config = GenerationConfig()

        if self.device != "cpu":
            config = PipelineConfig(
                voice=self.voice,
                provider=self.device,
                model_quality=self.quality,
                generation=generation_config,
            )
        else:
            config = PipelineConfig(
                voice=self.voice,
                model_quality=self.quality,
                generation=generation_config,
            )

        self.pipeline = KokoroPipeline(config)
        logger.info("KokoroEngine initialized successfully")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize text to audio. Returns (audio_array, sample_rate)."""
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        result = self.pipeline.run(text)
        return result.audio, 24000  # kokoro outputs at 24kHz


# ============================================================================
# Voice Cloning Engine (in-memory caching for clone.py functionality)
# ============================================================================

KOKOCLONE_AVAILABLE = False
try:
    # Note: kokoclone uses src-layout mode, so code is in core/cloner.py not kokoclone/__init__.py
    from core.cloner import KokoClone as KokoroCloneLib
    KOKOCLONE_AVAILABLE = True
    
    # Apply patch to use local kanade model files if available (avoids HF download on every clone)
    _LOCAL_MODEL_DIR = os.environ.get("KOKORO_LOCAL_MODEL_DIR", "/app/models")
    _LOCAL_CONFIG_YAML = os.path.join(_LOCAL_MODEL_DIR, "kanade-config.yaml")
    _LOCAL_WEIGHTS_SAFETENSORS = os.path.join(_LOCAL_MODEL_DIR, "kanade.safetensors")
    _LOCAL_KOKORO_ONNX = os.path.join(_LOCAL_MODEL_DIR, "kokoro.onnx")
    
    # Check for mounted voice samples (from kokoro-original volume mount)
    _MOUNTED_VOICE_DIR = os.environ.get("KOKORO_VOICE_MOUNT", "/app/kokoro-original/voice")
    _MOUNTED_VOICES_BIN = os.path.join(_MOUNTED_VOICE_DIR, "voices-v1.0.bin")
    
    def _check_local_model():
        return os.path.exists(_LOCAL_CONFIG_YAML) and os.path.exists(_LOCAL_WEIGHTS_SAFETENSORS)
    
    # Check if local kokoro.onnx is available (prevents runtime download)
    LOCAL_KOKORO_ONNX_AVAILABLE = os.path.exists(_LOCAL_KOKORO_ONNX)
    if LOCAL_KOKORO_ONNX_AVAILABLE:
        logger.info(f"[MODEL] Local kokoro.onnx found at {_LOCAL_KOKORO_ONNX}")
    
    # Check for mounted voice samples (prevents HF download)
    VOICES_BIN_AVAILABLE = os.path.exists(_MOUNTED_VOICES_BIN)
    if VOICES_BIN_AVAILABLE:
        logger.info(f"[VOICES] Local voices-v1.0.bin found at {_MOUNTED_VOICES_BIN}")
    
    if KOKOCLONE_AVAILABLE and _check_local_model():
        try:
            from kanade_tokenizer import KanadeModel, load_vocoder
            _orig_init = KokoroCloneLib.__init__
            
            def _patched_kanade_init(self, kanade_model="frothywater/kanade-12.5hz", hf_repo="PatnaikAshish/kokoclone"):
                import torch
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(f"Initializing KokoClone on: {self.device.type.upper()}")
                self.hf_repo = hf_repo
                
                # Try local kanade model first
                if _check_local_model():
                    print(f"[LOCAL MODEL] Using local kanade model from {_LOCAL_MODEL_DIR}")
                    try:
                        self.kanade = KanadeModel.from_pretrained(
                            repo_id=None, config_path=_LOCAL_CONFIG_YAML, weights_path=_LOCAL_WEIGHTS_SAFETENSORS
                        ).to(self.device).eval()
                        self.vocoder = load_vocoder(self.kanade.config.vocoder_name).to(self.device)
                        self.sample_rate = self.kanade.config.sample_rate
                    except Exception as e:
                        print(f"[WARNING] Local model loading failed ({e}), falling back to HF...")
                else:
                    pass
                
                # Fallback to original if not loaded locally
                if not hasattr(self, 'kanade') or self.kanade is None:
                    print("[HF FALLBACK] Loading kanade from HuggingFace...")
                    _orig_init(self, kanade_model, hf_repo)
                
                self.kokoro_cache = {}
            
            KokoroCloneLib.__init__ = _patched_kanade_init
            logger.info(f"[PATCH] KokoClone patched to use local kanade model files from {_LOCAL_MODEL_DIR}")
        except Exception as e:
            logger.debug(f"Could not apply kokoclone kanade patch: {e}")

    # Patch _ensure_file to check for local kokoro.onnx before downloading from HF
    if KOKOCLONE_AVAILABLE and LOCAL_KOKORO_ONNX_AVAILABLE:
        try:
            import functools
            
            _orig_ensure_file = KokoroCloneLib._ensure_file
            
            @functools.wraps(_orig_ensure_file)
            def _patched_ensure_file(self, folder, filename):
                """Check local models directory before attempting HF download."""
                # Special handling for kokoro.onnx - check local path first
                if folder == "model" and filename == "kokoro.onnx":
                    if os.path.exists(_LOCAL_KOKORO_ONNX):
                        logger.info(f"[MODEL] Using local kokoro.onnx from {_LOCAL_KOKORO_ONNX}")
                        return _LOCAL_KOKORO_ONNX
                    else:
                        # Fallback to original behavior (download from HF)
                        logger.warning("[MODEL] Local kokoro.onnx not found, will download from HF")
                
                # Special handling for voices-v1.0.bin - check mounted voice directory
                if folder == "voice" and filename == "voices-v1.0.bin":
                    if os.path.exists(_MOUNTED_VOICES_BIN):
                        logger.info(f"[VOICES] Using local voices-v1.0.bin from {_MOUNTED_VOICES_BIN}")
                        return _MOUNTED_VOICES_BIN
                    else:
                        # Fallback to original behavior (download from HF)
                        logger.warning("[VOICES] Local voices-v1.0.bin not found, will download from HF")
                
                # Default path - check if file exists in expected location after download
                filepath = os.path.join(folder, filename)
                repo_filepath = f"{folder}/{filename}"
                
                if not os.path.exists(filepath):
                    print(f"Downloading missing file '{filename}' from {self.hf_repo}...")
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(
                        repo_id=self.hf_repo,
                        filename=repo_filepath,
                        local_dir="." # Downloads securely into local ./model or ./voice
                    )
                return filepath
            
            KokoroCloneLib._ensure_file = _patched_ensure_file
            logger.info(f"[PATCH] KokoClone patched to check for local kokoro.onnx and voices before HF download")
        except Exception as e:
            logger.debug(f"Could not apply kokoclone ensure_file patch: {e}")
        
except ImportError:
    KOKOCLONE_AVAILABLE = False


class KokoroCloneEngine:
    """In-memory voice cloning engine that caches kanade model and reference audio features.
    
    Keeps the cloned model state in memory between calls to avoid reloading on every TTS request.
    Only loads new reference audio data when clone_voice() is called with a different file.
    """

    def __init__(self, device: str = "cpu", kanade_model: str = None):
        self.device = device
        self.kanade_model = kanade_model or DEFAULT_CONFIG.get("kanade_model", "frothywater/kanade-12.5hz")
        self.hf_repo = DEFAULT_CONFIG.get("hf_repo", "PatnaikAshish/kokoclone")
        
        # Cloned voice state - cached in memory
        self._cloner: Any = None          # KokoClone instance (loaded once)
        self._ref_audio_path: Optional[str] = None  # Currently loaded reference audio
        self._lang: str = "en"            # TTS language
        
        logger.info(f"KokoroCloneEngine created (kanade={self.kanade_model}, device={self.device})")

    def _ensure_cloner_loaded(self):
        """Lazy-load the KokoClone library and kanade model (once)."""
        if not KOKOCLONE_AVAILABLE:
            raise RuntimeError("kokoclone library is not installed. Cannot perform voice cloning.")
        
        if self._cloner is None:
            logger.info(f"Loading KokoClone engine (this may take a moment on first call)...")
            try:
                # Note: We pass lang later per-call since KokoClone can handle this dynamically
                self._cloner = KokoroCloneLib(
                    kanade_model=self.kanade_model,
                    hf_repo=self.hf_repo
                )
                logger.info("KokoClone engine loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load KokoClone: {e}")
                raise RuntimeError(f"Cannot initialize voice cloning: {e}")

    def clone_voice(self, ref_audio_path: str, lang: str = "en") -> bool:
        """Load reference audio for voice cloning into memory.
        
        If called with a different file than previously loaded, the old state is discarded
        and new reference audio features are extracted and cached.
        
        Args:
            ref_audio_path: Path to WAV/MP3 file for voice cloning
            lang: Language code (en, hi, fr, it, es, pt, ja, zh)
            
        Returns:
            True if validation passed
        """
        self._ensure_cloner_loaded()
        
        # Check if this is a different reference audio than what's already loaded
        needs_reload = (self._ref_audio_path != ref_audio_path)
        
        if not os.path.exists(ref_audio_path):
            logger.error(f"Reference audio file not found: {ref_audio_path}")
            return False
        
        # Validate audio duration
        try:
            import soundfile as sf
            data, sr = sf.read(ref_audio_path)
            duration = len(data) / sr
            
            if duration < 2.0:
                logger.warning(f"Short audio file ({duration:.1f}s), recommend 3-10 seconds for best results")
            
            channels = "mono" if data.ndim == 1 else f"{data.shape[1]}ch"
            logger.info(f"Reference audio loaded: {sr}Hz, {channels}, {duration:.1f}s")
        except Exception as e:
            logger.error(f"Error reading reference audio: {e}")
            return False
        
        # If new reference audio, reload the cloned state
        if needs_reload:
            logger.info(f"Loading voice cloning model for: {ref_audio_path}")
            
            # Reset cloner to force re-initialization with new reference
            self._cloner = None  # Force reload in synthesize
            
            try:
                # Initialize fresh KokoClone instance with this reference audio
                self._cloner = KokoroCloneLib(
                    kanade_model=self.kanade_model,
                    hf_repo=self.hf_repo
                )
                
                # Validate the reference audio through the cloner
                result = self._cloner.validate_reference(ref_audio_path) if hasattr(self._cloner, 'validate_reference') else True
                
            except Exception as e:
                logger.error(f"Failed to initialize cloning with new reference: {e}")
                raise RuntimeError(f"Voice cloning initialization failed: {e}")
        
        # Cache the current state
        self._ref_audio_path = ref_audio_path
        self._lang = lang
        
        if needs_reload:
            logger.info("New voice cloned into memory - subsequent TTS calls will reuse this")
        else:
            logger.debug("Reusing existing cached voice (no reload needed)")
        
        return True

    def reset_voice(self):
        """Clear the currently loaded cloned voice from memory.
        
        Called when user wants to switch to a different voice or release resources.
        The kanade model itself remains loaded for fast re-cloning.
        """
        if self._ref_audio_path:
            logger.info(f"Resetting cached voice (was: {self._ref_audio_path})")
            # Keep cloner instance but clear reference audio - forces reload on next clone_voice()
            self._cloner = None  # Force re-initialization with new reference
            self._ref_audio_path = None
        logger.info("Voice cache cleared - next TTS call will require clone_voice() or tts_clone with ref_audio")

    def synthesize(self, text: str, output_dir: Path) -> Optional[Path]:
        """Synthesize text using the currently loaded cloned voice.
        
        Reuses the in-memory cloned state - no model reloading between calls.
        
        Args:
            text: Text to convert to speech
            output_dir: Directory to save output WAV file
            
        Returns:
            Path to saved WAV file, or None on failure
        """
        if self._cloner is None or self._ref_audio_path is None:
            raise RuntimeError("No voice cloned yet. Call clone_voice() first.")

        # Generate unique filename
        timestamp = int(time.time())
        safe_text = "".join(c if c.isalnum() else "_" for c in text)[:50]
        output_path = output_dir / f"clone_{safe_text}_{timestamp}.wav"

        logger.info(f"Synthesizing (using cached voice): '{text[:40]}...'")

        try:
            # Use KokoClone's generate method - it reuses internal state efficiently
            self._cloner.generate(
                text=text,
                lang=self._lang,
                reference_audio=self._ref_audio_path,
                output_path=str(output_path)
            )
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"Saved cloned speech: {output_path} ({file_size/1024:.0f}KB)")
                return output_path
            else:
                logger.error("Output file not created")
                return None
                
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise RuntimeError(f"TTS generation failed: {str(e)}")


# ============================================================================
# Subprocess TTS Fallback (when pykokoro is not available or for cloning)
# ============================================================================

def run_subprocess_tts(
    text: str,
    voice: str = "af_bella",
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    quality: str = "fp16",
    device: str = "cpu"
) -> Path:
    """Run sampler.py or clone.py via subprocess for TTS."""

    config = discover_kokoro_scripts()
    output_dir = Path(DEFAULT_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    filename = f"tts_{uuid.uuid4().hex[:12]}.wav"
    output_path = output_dir / filename

    if ref_audio:
        # Use clone.py for voice cloning
        if not config["clone_path"]:
            raise FileNotFoundError("clone.py not found. Cannot perform voice cloning.")

        cmd = [
            sys.executable, str(config["clone_path"]),
            "--ref-audio", ref_audio,
            "--text", text,
            "--output-dir", str(output_dir),
            "--model-quality", quality,
        ]
        if device != "cpu":
            cmd.extend(["--device", device])

    else:
        # Use sampler.py for built-in voices
        if not config["sampler_path"]:
            raise FileNotFoundError("sampler.py not found. Cannot synthesize with built-in voices.")

        cmd = [
            sys.executable, str(config["sampler_path"]),
            "--voice", voice,
            "--text", text,
            "--output-dir", str(output_dir),
            "--model-quality", quality,
        ]
        if device != "cpu":
            cmd.extend(["--device", device])

    logger.info(f"Running: {' '.join(cmd)}")

    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        logger.error(f"Subprocess failed (rc={result.returncode}): {result.stderr}")
        raise RuntimeError(f"TTS generation failed: {result.stderr[:500]}")

    return output_path


# ============================================================================
# Text Chunking Functions (based on sampler.py)
# ============================================================================

def chunk_text_by_sentences(text: str, max_sentences: int = DEFAULT_MAX_SENTENCES) -> list[str]:
    """Split text into chunks suitable for kokoro inference based on sentence boundaries.
    
    Similar to sampler.py's chunk_text() but respects configurable sentence limits.
    Prevents cutoff on large blocks of text by splitting at natural pause points.
    
    Behavior:
      - If text fits within MAX_CHARS_PER_CHUNK, return as single chunk (no unnecessary splitting)
      - If text exceeds limit, split into chunks respecting both character AND sentence limits
    
    Args:
        text: Input text to split
        max_sentences: Maximum sentences per chunk (default: 2)
        
    Returns:
        List of text chunks
    """
    import re
    
    if len(text) <= MAX_CHARS_PER_CHUNK:
        # Text fits within limit - no need to split
        return [text] if text.strip() else []
    
    # Split into individual sentences using regex that handles various terminators
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text.strip())
    
    # Filter out empty sentences
    sentences = [s for s in sentences if s and s[0].isalpha()]
    
    if not sentences:
        return []
    
    # Group sentences into chunks respecting both character AND sentence limits
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        # Flush chunk if adding this would exceed either limit
        should_flush = False
        
        # Check sentence count limit
        if len(current_chunk) >= max_sentences:
            should_flush = True
            
        # Check character count limit (with margin for joining spaces)
        estimated_length = current_length + sentence_len + (1 * len(current_chunk))  # Add space per join
        if estimated_length > MAX_CHARS_PER_CHUNK and current_chunk:
            should_flush = True
        
        if should_flush:
            chunks.append(' '.join(current_chunk).strip())
            current_chunk = []
            current_length = 0
        
        # Handle very long sentences (no punctuation)
        if sentence_len > MAX_CHARS_PER_CHUNK:
            # Split by fixed character chunks as last resort
            remaining = sentence
            while remaining:
                chunk_part = remaining[:MAX_CHARS_PER_CHUNK]
                remaining = remaining[MAX_CHARS_PER_CHUNK:]
                
                if len(current_chunk) >= max_sentences or (current_length + len(chunk_part)) > MAX_CHARS_PER_CHUNK and current_chunk:
                    chunks.append(' '.join(current_chunk).strip())
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(chunk_part)
                current_length += len(chunk_part)
            continue
        
        # Add sentence to current chunk
        current_chunk.append(sentence)
        current_length += sentence_len
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())
    
    return [c for c in chunks if c]


def merge_audio_chunks(chunks: list[np.ndarray], silence_duration_samples: int = 5000) -> np.ndarray:
    """Concatenate multiple audio chunks with silence gaps between them.
    
    Args:
        chunks: List of numpy arrays containing audio segments
        silence_duration_samples: Number of zero samples for silence gap 
                                  (default ~200ms at 24kHz)
        
    Returns:
        Concatenated audio array
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    
    # Start with first chunk
    merged = chunks[0].copy()
    
    for i in range(1, len(chunks)):
        # Add silence gap between segments
        silence = np.zeros(silence_duration_samples, dtype=chunks[i].dtype)
        merged = np.concatenate([merged, silence, chunks[i]])
    
    return merged


def _synthesize_with_chunking(pipeline: Any, text: str, output_dir: Path, 
                               max_sentences: int = DEFAULT_MAX_SENTENCES,
                               max_chunks: int = None,
                               silence_duration_samples: int = 5000) -> tuple[Path, float]:
    """Synthesize long text by splitting into chunks and merging.
    
    Similar to sampler.py's generate_long_text() method (lines 299-361).
    Only chunks when necessary - short texts are unaffected.
    
    Args:
        pipeline: KokoroPipeline instance for synthesis
        text: Text to convert to speech  
        output_dir: Directory to save merged WAV file
        max_sentences: Maximum sentences per chunk (default: 2)
        max_chunks: Maximum number of chunks to process (None = unlimited)
        silence_duration_samples: Silence gap between segments in samples
        
    Returns:
        Tuple of (output_path, total_duration_seconds)
        
    Raises:
        ValueError: If text would exceed max_chunks limit
    """
    if max_chunks is None:
        max_chunks = DEFAULT_CONFIG["max_chunks_per_request"]
    
    # Split text into manageable chunks
    segments = chunk_text_by_sentences(text, max_sentences=max_sentences)
    
    logger.info(f"Text split into {len(segments)} segment(s) (total: {len(text)} chars)")
    
    if len(segments) > max_chunks:
        raise ValueError(
            f"Text would require {len(segments)} chunks but limit is {max_chunks}. "
            f"Try using 'max_sentences' parameter to reduce chunk size or contact support."
        )
    
    # Generate each segment and collect audio
    all_audio = []
    
    for i, segment in enumerate(segments):
        logger.info(f"  Segment {i+1}/{len(segments)}: '{segment[:40]}...'")
        
        res = pipeline.run(segment)
        audio = res.audio
        all_audio.append(audio)
        
        # Progress logging
        if (i + 1) % 5 == 0 or i == len(segments) - 1:
            logger.info(f"  Generated {i+1}/{len(segments)} segments...")
    
    # Merge all chunks into a single file
    logger.info(f"Merging {len(all_audio)} segments with {silence_duration_samples} samples silence gaps...")
    merged = merge_audio_chunks(all_audio, silence_duration_samples)
    
    # Save final merged file
    timestamp = int(time.time())
    safe_text = "".join(c if c.isalnum() else "_" for c in text)[:50]
    output_path = output_dir / f"merged_{safe_text}_{timestamp}.wav"
    
    import soundfile as sf
    sf.write(str(output_path), merged, DEFAULT_CONFIG["sample_rate"])
    
    # Calculate duration and size
    duration = len(merged) / DEFAULT_CONFIG["sample_rate"]
    file_size = os.path.getsize(output_path)
    
    logger.info(f"Saved merged audio: {output_path}")
    logger.info(f"Segments: {len(segments)} | Duration: {duration:.1f}s | Size: {file_size/1024:.0f}KB")
    
    return output_path, duration


def wav_to_base64(filepath: str) -> Optional[str]:
    """Convert a WAV file to base64 for MCP response."""
    try:
        with open(filepath, "rb") as f:
            import base64
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to read WAV file {filepath}: {e}")
        return None


# ============================================================================
# MCP Tool Definitions
# ============================================================================

def get_tool_definitions() -> list[Tool]:
    """Return list of MCP tool definitions."""
    tools = [
        Tool(
            name="tts_synthesize",
            description=(
                "Synthesize text to speech using Kokoro TTS with built-in voices. "
                "Supports 54+ voices across English (US/GB), Spanish, French, Japanese, and Chinese. "
                "Outputs WAV audio at 24kHz sample rate."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Text to synthesize into speech"
                    },
                    "voice": {
                        "type": "string",
                        "description": (
                            "Voice name (e.g., 'af_bella', 'am_michael'). "
                            "Default: af_bella. Run list_voices tool for all options."
                        )
                    },
                    "quality": {
                        "type": "string",
                        "enum": ["fp32", "fp16", "q8", "q4"],
                        "description": (
                            "Model quality: fp32 (highest/fargest), fp16 (balanced), "
                            "q8 (good/fast), q4 (fastest/smallest). Default: fp16"
                        )
                    },
                    "device": {
                        "type": "string",
                        "enum": ["cpu", "cuda", "auto"],
                        "description": "Compute device. Default: cpu"
                    },
                    "auto_play": {
                        "type": "boolean",
                        "description": (
                            "Automatically play the generated audio after synthesis. "                            
                            "Requires system audio support (paplay/aplay/pygame). Default: false"
                        ),
                        "default": False
                    },
                    "delete_after_playback": {
                        "type": "boolean",
                        "description": (
                            "Delete the generated audio file after successful auto-playback. "
                            "Only applies when auto_play is enabled. Default: true"
                        ),
                        "default": True
                    },
                    "max_sentences": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": (
                            "Maximum number of sentences per audio chunk. "
                            "Default: 2. Use this to control how text is split for large inputs. "
                            "Lower values = shorter chunks but more segments in output."
                        ),
                        "default": DEFAULT_MAX_SENTENCES
                    },
                    "max_chunks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": (
                            "Maximum number of audio chunks to generate. "
                            "Default: 10. Requests exceeding this will return an error."
                        ),
                        "default": DEFAULT_CONFIG["max_chunks_per_request"]
                    }
                },
                "required": ["input"]
            }
        ),
        Tool(
            name="tts_clone",
            description=(
                "Synthesize text using voice cloning from a reference audio file. "
                "Uses the reference audio to clone any voice for speech synthesis. "
                "Requires a reference WAV/MP3 file and optionally its transcript."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Text to synthesize using the cloned voice"
                    },
                    "ref_audio": {
                        "type": "string",
                        "description": (
                            "Path to reference audio file (WAV/MP3) for voice cloning. "
                            "Can be local path or URL if mounted in container."
                        )
                    },
                    "ref_text": {
                        "type": "string",
                        "description": "Transcript of the reference audio (optional, helps accuracy)"
                    },
                    "quality": {
                        "type": "string",
                        "enum": ["fp32", "fp16", "q8", "q4"],
                        "description": "Model quality for TTS. Default: fp16"
                    },
                    "device": {
                        "type": "string",
                        "enum": ["cpu", "cuda", "auto"],
                        "description": "Compute device. Default: cpu (clone requires GPU for best performance)"
                    },
                    "auto_play": {
                        "type": "boolean", 
                        "description": (
                            "Automatically play the generated audio after synthesis. "                            
                            "Requires system audio support (paplay/aplay/pygame). Default: false"
                        ),
                        "default": False
                    },
                    "delete_after_playback": {
                        "type": "boolean",
                        "description": (
                            "Delete the generated audio file after successful auto-playback. "
                            "Only applies when auto_play is enabled. Default: true"
                        ),
                        "default": True
                    },
                    "max_sentences": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": (
                            "Maximum number of sentences per audio chunk. "
                            "Default: 2. Use this to control how text is split for large inputs."
                        ),
                        "default": DEFAULT_MAX_SENTENCES
                    },
                    "max_chunks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": (
                            "Maximum number of audio chunks to generate. "
                            "Default: 10."
                        ),
                        "default": DEFAULT_CONFIG["max_chunks_per_request"]
                    }
                },
                "required": ["input", "ref_audio"]
            }
        ),
        Tool(
            name="list_voices",
            description=(
                "List all available built-in Kokoro TTS voices with their language and gender."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="play_audio",
            description=(
                "Play a WAV audio file through system speakers. Supports local files and paths "                "within the container. Optionally deletes the file after playback."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Path to WAV audio file to play. Must exist on disk. "                            "Example: /tmp/kokoro-tts/output.wav or ./audio/result.wav"
                        )
                    },
                    "delete_after_play": {
                        "type": "boolean",
                        "description": (
                            "Automatically delete the audio file after successful playback. "                            "Default: true"
                        ),
                        "default": True
                    }
                },
                "required": ["file_path"]
            }
        )
    ]

    return tools


# ============================================================================
# MCP Server Implementation
# ============================================================================

class KokoroTTSServer:
    """MCP server wrapping Kokoro TTS functionality."""

    def __init__(self):
        self.app = Server("kokoro-tts")
        self.engine: Optional[KokoroEngine] = None
        self._clone_engine: Optional[KokoroCloneEngine] = None  # In-memory clone cache
        self._voice_list_cache: Optional[list[str]] = None

    def register_handlers(self, tools):
        """Register MCP tool handlers."""

        @self.app.list_tools()
        async def list_tts_tools():
            # Return ListToolsResult for modern API compatibility
            from mcp.types import ListToolsResult
            return ListToolsResult(tools=tools)

        @self.app.call_tool()
        async def handle_all_tools(name: str, arguments: dict):
            """Central tool handler that routes to specific implementations."""
            
            if name == "tts_synthesize":
                return await self._handle_synthesize(arguments)
            elif name == "tts_clone":
                return await self._handle_clone(arguments)
            elif name == "list_voices":
                return await self._handle_list_voices()
            elif name == "play_audio":
                return await self._handle_play_audio(arguments)
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                    isError=True,
                )

    async def _handle_synthesize(self, arguments: dict):
        """Handle tts_synthesize tool call.
        
        Supports automatic chunking for large text inputs to prevent cutoff.
        """
        try:
            text = arguments.get("input", "")
            voice = arguments.get("voice", DEFAULT_CONFIG["voice"])
            quality = arguments.get("quality", DEFAULT_CONFIG["model_quality"])
            device = arguments.get("device", DEFAULT_CONFIG["device"])
            
            # Extract chunking parameters
            max_sentences = arguments.get("max_sentences", DEFAULT_MAX_SENTENCES)
            max_chunks = arguments.get("max_chunks", DEFAULT_CONFIG["max_chunks_per_request"])

            if not text.strip():
                return CallToolResult(
                    content=[TextContent(type="text", text="Error: 'input' cannot be empty")],
                    isError=True,
                )

            # Check if text needs chunking (based on sampler.py MAX_CHARS_PER_CHUNK)
            needs_chunking = len(text) > MAX_CHARS_PER_CHUNK
            
            if needs_chunking and PYKOKORO_AVAILABLE:
                logger.info(f"Long text detected ({len(text)} chars), using chunked synthesis")
                
                # Initialize or reinitialize engine for this request
                if self.engine is None or self.engine.voice != voice or self.engine.quality != quality:
                    self.engine = KokoroEngine(voice=voice, device=device, quality=quality)
                    self.engine.initialize()

                output_dir = Path(DEFAULT_CONFIG["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Use chunked synthesis for long texts
                    output_path, duration = _synthesize_with_chunking(
                        pipeline=self.engine.pipeline,
                        text=text,
                        output_dir=output_dir,
                        max_sentences=max_sentences,
                        max_chunks=max_chunks,
                        silence_duration_samples=DEFAULT_CONFIG["silence_duration_samples"]
                    )
                    
                    # Build response with chunk metadata
                    import os
                    size_kb = os.path.getsize(output_path) / 1024
                    
                    content_blocks = [
                        TextContent(type="text", text=f"Audio generated successfully (chunked).\nFile: {output_path}"),
                        TextContent(type="text", text=f"Chunks: merged audio from multiple segments | Sample rate: 24kHz | Format: WAV"),
                        TextContent(type="text", text=f"Size: {size_kb:.0f} KB | Duration: ~{duration:.1f}s")
                    ]

                    # Auto-play if requested (after getting file info, before returning response)
                    auto_play_enabled = arguments.get("auto_play", False)
                    delete_after = arguments.get("delete_after_playback", True)
                    if auto_play_enabled:
                        await self._auto_play_if_enabled(output_path, True, delete_after)

                    return CallToolResult(
                        content=content_blocks,
                        isError=False,
                    )
                    
                except ValueError as e:
                    # Chunk limit exceeded - provide helpful error message
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Error: {str(e)}")],
                        isError=True,
                    )

            elif needs_chunking and not PYKOKORO_AVAILABLE:
                logger.info(f"Long text detected ({len(text)} chars), chunking not supported in subprocess mode - splitting by character limit")
            
            # Try direct synthesis first (faster when pykokoro available)
            output_path = None

            if PYKOKORO_AVAILABLE and not arguments.get("ref_audio"):
                try:
                    if self.engine is None or self.engine.voice != voice or self.engine.quality != quality:
                        self.engine = KokoroEngine(voice=voice, device=device, quality=quality)
                        self.engine.initialize()

                    audio_array, sr = self.engine.synthesize(text)

                    # Save to file for reference
                    output_dir = Path(DEFAULT_CONFIG["output_dir"])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    filepath = output_dir / f"direct_{uuid.uuid4().hex[:8]}.wav"

                    if SOUNDFILE_AVAILABLE:
                        try:
                            import soundfile as sf
                            sf.write(str(filepath), audio_array, sr)
                        except ImportError:
                            from scipy.io.wavfile import write as wav_write
                            wav_write(str(filepath), sr, audio_array.astype(np.float32))
                    else:
                        from scipy.io.wavfile import write as wav_write
                        wav_write(str(filepath), sr, audio_array.astype(np.float32))

                    output_path = str(filepath)
                except Exception as e:
                    logger.warning(f"Direct synthesis failed ({e}), falling back to subprocess")
                    self.engine = None  # Reset engine so next call reinitializes

            if not output_path:
                # Fallback to subprocess
                output_path = run_subprocess_tts(
                    text=text,
                    voice=voice,
                    quality=quality,
                    device=device,
                )

            # Build response content blocks (get size BEFORE potential deletion)
            import os
            size_kb = os.path.getsize(output_path) / 1024 if output_path else 0

            # Auto-play if requested (after getting file info, before returning response)
            auto_play_enabled = arguments.get("auto_play", False)
            delete_after = arguments.get("delete_after_playback", True)
            if auto_play_enabled:
                await self._auto_play_if_enabled(output_path, True, delete_after)

            content_blocks = [
                TextContent(type="text", text=f"Audio generated successfully.\nFile: {output_path}"),
                TextContent(type="text", text=f"Size: {size_kb:.0f} KB | Sample rate: 24kHz | Format: WAV")
            ]

            return CallToolResult(
                content=content_blocks,
                isError=False,
            )

        except Exception as e:
            logger.error(f"tts_synthesize error: {e}", exc_info=True)
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True,
            )

    async def _handle_clone(self, arguments: dict):
        """Handle tts_clone tool call.
        
        Uses in-memory caching: the cloned voice state (kanade model + reference audio features)
        is kept between calls. Only when a new ref_audio is provided does it reload into memory.
        Supports automatic chunking for large text inputs to prevent cutoff.
        """
        try:
            text = arguments.get("input", "")
            ref_audio = arguments.get("ref_audio", "")
            ref_text = arguments.get("ref_text", "")
            quality = arguments.get("quality", DEFAULT_CONFIG["model_quality"])
            device = arguments.get("device", "cpu")
            
            # Extract chunking parameters
            max_sentences = arguments.get("max_sentences", DEFAULT_MAX_SENTENCES)
            max_chunks = arguments.get("max_chunks", DEFAULT_CONFIG["max_chunks_per_request"])

            if not text.strip():
                return CallToolResult(
                    content=[TextContent(type="text", text="Error: 'input' cannot be empty")],
                    isError=True,
                )
            if not ref_audio:
                return CallToolResult(
                    content=[TextContent(type="text", text="Error: 'ref_audio' is required for voice cloning")],
                    isError=True,
                )

            # Check if text needs chunking
            needs_chunking = len(text) > MAX_CHARS_PER_CHUNK
            
            # Lazy initialize the clone engine (loads kanade model once)
            if self._clone_engine is None:
                logger.info("Initializing in-memory clone engine...")
                self._clone_engine = KokoroCloneEngine(device=device)

            # Check if we need to load a new reference audio into memory
            needs_voice_load = (self._clone_engine._ref_audio_path != ref_audio)
            
            if needs_voice_load:
                logger.info(f"Loading voice cloning model for: {ref_audio}")
                success = self._clone_engine.clone_voice(ref_audio, lang="en")
                if not success:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Error: Failed to load reference audio: {ref_audio}")],
                        isError=True,
                    )
            else:
                logger.info("Reusing cached voice cloning state (no reload needed)")

            # Synthesize using the in-memory engine (fast - no subprocess)
            output_dir = Path(DEFAULT_CONFIG["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if needs_chunking:
                logger.info(f"Long text detected ({len(text)} chars), using chunked clone synthesis")
                
                # Generate segments using the cloned voice
                all_audio = []
                segments = chunk_text_by_sentences(text, max_sentences=max_sentences)
                
                if len(segments) > max_chunks:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Error: Text would require {len(segments)} chunks but limit is {max_chunks}. Reduce 'max_sentences' or shorten input.")],
                        isError=True,
                    )
                
                for i, segment in enumerate(segments):
                    logger.info(f"  Segment {i+1}/{len(segments)}: '{segment[:40]}...'")
                    
                    # Use clone engine's synthesize method for each segment
                    output_path = self._clone_engine.synthesize(segment, output_dir)
                    
                    if not output_path:
                        return CallToolResult(
                            content=[TextContent(type="text", text=f"Error: Synthesis failed on segment {i+1}")],
                            isError=True,
                        )
                    
                    # Load audio into numpy array for merging
                    try:
                        import soundfile as sf
                        import os  # Explicitly import os before using it
                        audio_data, sr = sf.read(output_path)
                        all_audio.append(audio_data)
                        os.remove(output_path)  # Clean up temp file
                    except Exception as e:
                        logger.error(f"Failed to read segment {i+1}: {e}")
                        return CallToolResult(
                            content=[TextContent(type="text", text=f"Error reading audio segment: {e}")],
                            isError=True,
                        )
                
                # Merge all chunks into a single file
                logger.info(f"Merging {len(all_audio)} segments...")
                merged = merge_audio_chunks(all_audio, DEFAULT_CONFIG["silence_duration_samples"])
                
                # Save final merged file
                timestamp = int(time.time())
                safe_text = "".join(c if c.isalnum() else "_" for c in text)[:50]
                output_path = output_dir / f"clone_merged_{safe_text}_{timestamp}.wav"
                
                import soundfile as sf
                sf.write(str(output_path), merged, DEFAULT_CONFIG["sample_rate"])
                
                # Calculate duration and size
                duration = len(merged) / DEFAULT_CONFIG["sample_rate"]
                file_size = os.path.getsize(output_path)
                
                logger.info(f"Saved merged cloned audio: {output_path}")
                logger.info(f"Segments: {len(segments)} | Duration: {duration:.1f}s | Size: {file_size/1024:.0f}KB")

                # Build response with chunk metadata
                size_kb = os.path.getsize(output_path) / 1024
                
                content_blocks = [
                    TextContent(type="text", text=f"Voice cloned speech generated successfully (chunked).\nFile: {output_path}"),
                    TextContent(type="text", text=f"Chunks: merged audio from multiple segments | Sample rate: 24kHz | Format: WAV"),
                    TextContent(type="text", text=f"Size: {size_kb:.0f} KB | Duration: ~{duration:.1f}s")
                ]

            else:
                # Normal synthesis without chunking
                output_path = self._clone_engine.synthesize(text, output_dir)
                
                if not output_path:
                    return CallToolResult(
                        content=[TextContent(type="text", text="Error: Synthesis failed - no output generated")],
                        isError=True,
                    )

            # Build response content blocks (get size BEFORE potential deletion)
            import os
            size_kb = os.path.getsize(output_path) / 1024 if output_path else 0

            # Auto-play if requested (after getting file info, before returning response)
            auto_play_enabled = arguments.get("auto_play", False)
            delete_after = arguments.get("delete_after_playback", True)
            if auto_play_enabled:
                await self._auto_play_if_enabled(output_path, True, delete_after)

            content_blocks = [
                TextContent(type="text", text=f"Voice cloned speech generated successfully.\nFile: {output_path}"),
                TextContent(type="text", text=f"Size: {size_kb:.0f} KB | Sample rate: 24kHz | Format: WAV")
            ]

            return CallToolResult(
                content=content_blocks,
                isError=False,
            )

        except Exception as e:
            logger.error(f"tts_clone error: {e}", exc_info=True)
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True,
            )

    async def _handle_list_voices(self):
        """Handle list_voices tool call."""
        try:
            if PYKOKORO_AVAILABLE and self.engine is None:
                # Initialize engine briefly to get voice list
                temp_engine = KokoroEngine(voice=DEFAULT_CONFIG["voice"], device="cpu", quality="fp16")
                temp_engine.initialize()

                try:
                    km = temp_engine.pipeline._kokoro
                    voices = km._voice_manager.get_voices()
                    self._voice_list_cache = voices
                except Exception:
                    # First run may not have voices loaded yet
                    pass
            elif PYKOKORO_AVAILABLE and self.engine:
                try:
                    km = self.engine.pipeline._kokoro
                    self._voice_list_cache = km._voice_manager.get_voices()
                except Exception:
                    pass

            if not self._voice_list_cache:
                # Generate basic list from presets
                voice_presets = {
                    "af_": ("English US", "Female"),
                    "am_": ("English US", "Male"),
                    "bf_": ("English GB", "Female"),
                    "bm_": ("English GB", "Male"),
                    "ef_": ("Spanish", "Female"),
                    "em_": ("Spanish", "Male"),
                    "ff_": ("French", "Female"),
                    "jf_": ("Japanese", "Female"),
                    "jm_": ("Japanese", "Male"),
                }

                lines = ["Built-in voice presets (run with pykokoro installed for full list):"]
                for prefix, (lang, gender) in voice_presets.items():
                    # Common voices per category
                    examples = {
                        "af_": ["af_bella", "af_sarah", "af_alloy"],
                        "am_": ["am_michael", "am_adam"],
                        "bf_": ["bf_alice", "bf_emma"],
                        "bm_": ["bm_george", "bm_lewis"],
                        "ef_": ["ef_dora"],
                        "em_": ["em_alex"],
                        "ff_": ["ff_siwis"],
                        "jf_": ["jf_alpha"],
                        "jm_": ["jm_kumo"],
                    }
                    for v in examples.get(prefix, []):
                        lines.append(f"  {v} ({lang}, {gender})")

                return CallToolResult(
                    content=[TextContent(type="text", text="\n".join(lines))],
                    isError=False,
                )

            # Format voice list with categories
            categorized = {}
            for v in self._voice_list_cache:
                prefix = v[:2] if len(v) >= 2 else ""
                lang, gender = "", ""
                if prefix.startswith("af"): lang, gender = "English US", "Female"
                elif prefix.startswith("am"): lang, gender = "English US", "Male"
                elif prefix.startswith("bf"): lang, gender = "English GB", "Female"
                elif prefix.startswith("bm"): lang, gender = "English GB", "Male"
                elif prefix.startswith("ef"): lang, gender = "Spanish", "Female"
                elif prefix.startswith("em"): lang, gender = "Spanish", "Male"
                elif prefix.startswith("ff"): lang, gender = "French", "Female"
                elif prefix.startswith("jf"): lang, gender = "Japanese", "Female"
                elif prefix.startswith("jm"): lang, gender = "Japanese", "Male"

                cat_key = f"{lang} ({gender})"
                if cat_key not in categorized:
                    categorized[cat_key] = []
                categorized[cat_key].append(v)

            lines = [f"Available voices ({len(self._voice_list_cache)} total):"]
            for category, voice_list in sorted(categorized.items()):
                lines.append(f"\n  {category}:")
                for v in sorted(voice_list):
                    lines.append(f"    - {v}")

            return CallToolResult(
                content=[TextContent(type="text", text="\n".join(lines))],
                isError=False,
            )

        except Exception as e:
            logger.error(f"list_voices error: {e}", exc_info=True)
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error listing voices: {str(e)}")],
                isError=True,
            )

    async def _handle_play_audio(self, arguments: dict):
        """Handle play_audio tool call. Plays WAV file via paplay/aplay/pygame and optionally deletes it."""
        try:
            import asyncio
            from pathlib import Path
            
            file_path = arguments.get("file_path", "")
            delete_after = arguments.get("delete_after_play", True)
            
            if not file_path:
                return CallToolResult(
                    content=[TextContent(type="text", text="Error: 'file_path' is required")],
                    isError=True,
                )
            
            # Check if file exists
            filepath = Path(file_path)
            if not filepath.exists():
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: File not found: {file_path}")],
                    isError=True,
                )
            
            # Verify it's a WAV file
            if not str(filepath).lower().endswith(".wav"):
                logger.warning(f"Non-WAV file requested for playback: {filepath}")
            
            # Play audio using paplay (PulseAudio/PipeWire) first, then fallback to aplay or pygame
            play_success = False
            try:
                import subprocess
                
                # Try paplay first (works with PulseAudio/PipeWire socket mounts)
                try:
                    result = subprocess.run(
                        ["paplay", str(filepath)],
                        capture_output=False,
                        timeout=60  # 1 minute max for long files
                    )
                    if result.returncode == 0:
                        play_success = True
                        logger.info(f"Played audio via paplay: {filepath}")
                except FileNotFoundError:
                    logger.info("paplay not found, trying aplay fallback...")
                    
                    # Try aplay as second fallback (direct ALSA access)
                    try:
                        result = subprocess.run(
                            ["aplay", "-v", str(filepath)],
                            capture_output=False,
                            timeout=60
                        )
                        if result.returncode == 0:
                            play_success = True
                            logger.info(f"Played audio via aplay: {filepath}")
                    except FileNotFoundError:
                        # Try pygame as final fallback
                        try:
                            import pygame.mixer
                            pygame.init()
                            pygame.mixer.music.load(str(filepath))
                            pygame.mixer.music.play()
                            # Wait for playback to complete
                            while pygame.mixer.music.get_busy():
                                await asyncio.sleep(0.1)
                            play_success = True
                            logger.info(f"Played audio via pygame: {filepath}")
                            pygame.quit()
                        except ImportError:
                            logger.error("Neither paplay, aplay nor pygame available for audio playback")
            except subprocess.TimeoutExpired:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: Audio playback timed out (>60s)")],
                    isError=True,
                )
            
            # Delete file if requested and playback succeeded
            deleted = False
            if play_success and delete_after:
                try:
                    filepath.unlink()  # Delete the file
                    deleted = True
                    logger.info(f"Deleted audio file after playback: {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to delete file after playback: {e}")
            
            if play_success:
                status_parts = [f"Audio played successfully from: {file_path}"]
                if deleted:
                    status_parts.append("File deleted after playback.")
                
                return CallToolResult(
                    content=[TextContent(type="text", text="\n".join(status_parts))],
                    isError=False,
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: Failed to play audio from {file_path}")],
                    isError=True,
                )
                
        except Exception as e:
            logger.error(f"play_audio error: {e}", exc_info=True)
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error playing audio: {str(e)}")],
                isError=True,
            )

    async def _auto_play_if_enabled(self, file_path: str, auto_play: bool, delete_after: bool = True):
        """Helper to play audio if auto_play is enabled and optionally delete after.
        
        Tries paplay (PulseAudio/PipeWire) first for containerized deployments,
        then falls back to aplay or pygame as needed.
        
        Args:
            file_path: Path to the WAV file
            auto_play: Whether to play the audio automatically
            delete_after: Whether to delete the file after successful playback (default: True)
        """
        if not auto_play or not file_path:
            return
        
        try:
            import subprocess
            
            # Try paplay first (works with PulseAudio/PipeWire socket mounts)
            try:
                result = subprocess.run(
                    ["paplay", str(file_path)],
                    capture_output=False,
                    timeout=60
                )
                if result.returncode == 0:
                    logger.info(f"Auto-played audio (paplay): {file_path}")
                    # Delete after successful playback only if enabled
                    if delete_after:
                        import os
                        os.remove(file_path)
                        logger.info(f"Deleted auto-played file: {file_path}")
            except FileNotFoundError:
                # Try aplay as second fallback (direct ALSA access)
                try:
                    result = subprocess.run(
                        ["aplay", "-v", str(file_path)],
                        capture_output=False,
                        timeout=60
                    )
                    if result.returncode == 0:
                        logger.info(f"Auto-played audio (aplay): {file_path}")
                        # Delete after successful playback only if enabled
                        if delete_after:
                            import os
                            os.remove(file_path)
                            logger.info(f"Deleted auto-played file: {file_path}")
                except FileNotFoundError:
                    # Try pygame as final fallback
                    try:
                        import asyncio
                        import pygame.mixer
                        pygame.init()
                        pygame.mixer.music.load(str(file_path))
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            await asyncio.sleep(0.1)
                        logger.info(f"Auto-played audio (pygame): {file_path}")
                        # Delete after playback only if enabled
                        if delete_after:  
                            import os  
                            os.remove(file_path)
                        pygame.quit()
                    except ImportError:
                        pass
                    
        except Exception as e:
            logger.warning(f"Auto-play failed for {file_path}: {e}")

    def run(self, host: str = "0.0.0.0", port: int = 8765):
        """Start the MCP server."""
        logger.info("Starting Kokoro TTS MCP Server")
        logger.info(f"Host: {host}, Port: {port}")

        config = discover_kokoro_scripts()
        logger.info(f"sampler.py: {config['sampler_path']}")
        logger.info(f"clone.py: {config['clone_path']}")
        logger.info(f"pykokoro available: {PYKOKORO_AVAILABLE}")

        # Initialize engine if pykokoro is available
        if PYKOKORO_AVAILABLE:
            try:
                self.engine = KokoroEngine(voice=DEFAULT_CONFIG["voice"], device="cpu", quality=DEFAULT_CONFIG["model_quality"])
                self.engine.initialize()
            except Exception as e:
                logger.warning(f"Failed to initialize pykokoro engine (will use subprocess): {e}")

        # For MCP over stdio, run the server
        async def main():
            # Create proper initialization options for the MCP protocol
            capabilities = ServerCapabilities(
                tools=ToolsCapability(),  # Enable tool support
                extra_data=None
            )
            init_options = InitializationOptions(
                server_name="kokoro-tts",
                server_version="1.0.0",
                capabilities=capabilities,
                instructions="Kokoro TTS MCP Server - Text-to-Speech synthesis with built-in voices and voice cloning"
            )
            
            async with stdio_server() as (read_stream, write_stream):
                await self.app.run(read_stream, write_stream, init_options)

        import asyncio
        import signal
        
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully")
        
        # Register signal handlers for graceful shutdown
        try:
            signal.signal(signal.SIGTERM, handle_signal)
            signal.signal(signal.SIGINT, handle_signal)
        except (ValueError, OSError):
            pass  # Signal handling not available in all contexts
        
        # Keep trying to run the MCP server - if stdin closes (no client), retry
        while True:
            try:
                asyncio.run(main())
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received, shutting down")
                break
            except SystemExit as e:
                logger.warning(f"MCP server exited with code {e.code}, restarting in 1s...")
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
            
            # Brief pause before retrying (only if not a clean shutdown)
            import os
            if os.environ.get("MCP_SERVER_FOREVER", "true").lower() == "true":
                time.sleep(1)


# ============================================================================
# Entry Point
# ============================================================================

def create_app(**kwargs) -> Server:
    """Create and configure the MCP server app. Used for testing."""
    if "host" in kwargs:
        DEFAULT_CONFIG["output_dir"] = Path(kwargs.get("output_dir", "/tmp/kokoro-tts"))

    return get_tool_definitions()


def main():
    """Main entry point for the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Kokoro TTS MCP Server")
    parser.add_argument("--voice", default=None, help="Default voice")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    parser.add_argument("--quality", choices=["fp32", "fp16", "q8", "q4"], default=None)
    # TCP mode options
    parser.add_argument("--tcp", action="store_true", help="Enable TCP transport mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host address for TCP mode")
    parser.add_argument("--port", type=int, default=8765, help="Port number for TCP mode")
    args = parser.parse_args()

    # Apply CLI overrides
    if args.voice:
        DEFAULT_CONFIG["voice"] = args.voice
    if args.device:
        DEFAULT_CONFIG["device"] = args.device
    if args.quality:
        DEFAULT_CONFIG["model_quality"] = args.quality

    server = KokoroTTSServer()
    tools = get_tool_definitions()
    server.register_handlers(tools)
    
    # Use TCP mode or stdio mode
    if args.tcp:
        from mcp_server import tcp_mcp
        
        async def run_tcp_mode():
            """Run in TCP transport mode."""
            logger.info(f"Starting MCP-over-TCP server on {args.host}:{args.port}")
            
            # Initialize engine if pykokoro is available (same as stdio mode)
            if PYKOKORO_AVAILABLE:
                try:
                    server.engine = KokoroEngine(voice=DEFAULT_CONFIG["voice"], device="cpu", quality=DEFAULT_CONFIG["model_quality"])
                    server.engine.initialize()
                except Exception as e:
                    logger.warning(f"Failed to initialize pykokoro engine (will use subprocess): {e}")
            
            # Create TCP server instance
            tcp_server = tcp_mcp.TcpMcpServer(host=args.host, port=args.port)
            tcp_server.register_tools(tools)
            
            # Register tool handlers from the main server's tool dispatch logic
            async def handle_tts_synthesize(arguments: dict):
                return await server._handle_synthesize(arguments)
            
            async def handle_tts_clone(arguments: dict):
                return await server._handle_clone(arguments)
            
            async def handle_list_voices(arguments: dict = None):
                # Ignore arguments - this tool takes none
                return await server._handle_list_voices()
            
            async def handle_play_audio(arguments: dict):
                return await server._handle_play_audio(arguments)
            
            # Map tool names to their specific handlers (each handler only takes arguments)
            tcp_server.register_tool_handler("tts_synthesize", handle_tts_synthesize)
            tcp_server.register_tool_handler("tts_clone", handle_tts_clone)
            tcp_server.register_tool_handler("list_voices", handle_list_voices)
            tcp_server.register_tool_handler("play_audio", handle_play_audio)
            
            await tcp_server.start()
            
            # Keep running until interrupted (no retry loop for TCP - server accepts connections on demand)
            import signal
            
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, shutting down...")
                asyncio.create_task(tcp_server.stop())
            
            try:
                signal.signal(signal.SIGTERM, signal_handler)
                signal.signal(signal.SIGINT, signal_handler)
            except (ValueError, OSError):
                pass
            
            while True:
                await asyncio.sleep(1)
        
        import asyncio
        try:
            asyncio.run(run_tcp_mode())
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down")
    else:
        server.run()


if __name__ == "__main__":
    main()
