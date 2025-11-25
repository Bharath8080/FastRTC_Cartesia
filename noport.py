import argparse
import io
import time
import re
from typing import Generator, Tuple, Union
import numpy as np
import soundfile as sf
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
)
from cartesia import Cartesia
from loguru import logger
from dotenv import load_dotenv
import os
load_dotenv()
from websearch_agent import agent, agent_config

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

# Initialize Cartesia with API key
cartesia_client = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))

# Cartesia Sonic 3 TTS Configuration
logger.info("🎤 Initializing Cartesia Sonic 3 TTS...")
CARTESIA_TTS_CONFIG = {
    "model_id": "sonic-3",  # Latest streaming TTS model
    "voice": {
        "mode": "id",
        "id": "f786b574-daa5-4673-aa0c-cbe3e8534c02",  # Katie - stable, realistic voice for voice agents
    },
    "output_format": {
        "container": "raw",
        "sample_rate": 24000,
        "encoding": "pcm_f32le",
    },
}
logger.info("✅ Cartesia Sonic 3 TTS configured successfully")


def response(
    audio: tuple[int, np.ndarray],
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Process audio input, transcribe it, generate a response using LangGraph, and deliver TTS audio.
    Optimized for ultra-low latency with Cartesia STT and Sonic 3 TTS.

    Args:
        audio: Tuple containing sample rate and audio data

    Yields:
        Tuples of (sample_rate, audio_array) for audio playback
    """
    start_time = time.time()
    logger.info("🎙️ Received audio input")

    # ============ STT (Speech-to-Text) with Cartesia ============
    stt_start = time.time()
    logger.debug("🔄 Transcribing audio with Cartesia...")
    sample_rate, audio_data = audio
    
    # Convert audio to PCM format for Cartesia
    # Cartesia expects 16kHz, 16-bit PCM
    target_sample_rate = 16000
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        import librosa
        # Convert to float32 for resampling
        if audio_data.dtype != np.float32:
            audio_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        else:
            audio_float = audio_data
        
        # Resample
        audio_resampled = librosa.resample(
            audio_float.T.flatten() if audio_float.ndim > 1 else audio_float,
            orig_sr=sample_rate,
            target_sr=target_sample_rate
        )
        audio_data = audio_resampled
        sample_rate = target_sample_rate
    
    # Convert to 16-bit PCM bytes
    if audio_data.dtype == np.float32:
        audio_int16 = (audio_data * 32767).astype(np.int16)
    else:
        audio_int16 = audio_data.astype(np.int16)
    
    audio_bytes = audio_int16.tobytes()
    
    # Create websocket connection with optimized endpointing
    ws = cartesia_client.stt.websocket(
        model="ink-whisper",
        language="en",
        encoding="pcm_s16le",
        sample_rate=target_sample_rate,
        min_volume=0.1,  # Low threshold for voice detection
        max_silence_duration_secs=0.3,  # Quick endpointing
    )
    
    # Send audio in chunks (20ms chunks for streaming)
    chunk_size = int(target_sample_rate * 0.02 * 2)  # 20ms chunks
    for i in range(0, len(audio_bytes), chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        if chunk:
            ws.send(chunk)
    
    # Finalize transcription
    ws.send("finalize")
    ws.send("done")
    
    # Receive transcription results
    transcript = ""
    for result in ws.receive():
        if result['type'] == 'transcript':
            if result['is_final']:
                transcript = result['text']
                break
        elif result['type'] == 'done':
            break
    
    ws.close()
    
    stt_time = time.time() - stt_start
    logger.info(f'👂 Transcribed in {stt_time:.2f}s: "{transcript}"')

    # ============ LLM (Language Model) ============
    llm_start = time.time()
    logger.debug("🧠 Running agent...")
    agent_response = agent.invoke(
        {"messages": [{"role": "user", "content": transcript}]}, config=agent_config
    )
    response_text = agent_response["messages"][-1].content
    llm_time = time.time() - llm_start
    logger.info(f'💬 Response in {llm_time:.2f}s: "{response_text}"')

    # ============ TTS (Text-to-Speech) with Cartesia Sonic 3 ============
    tts_start = time.time()
    logger.debug("🔊 Generating speech with Cartesia Sonic 3...")
    
    # Clean markdown formatting for better TTS output
    clean_text = response_text
    # Remove asterisks (bold/italic markdown)
    clean_text = re.sub(r'\*+', '', clean_text)
    # Remove other common markdown symbols (including table separators)
    clean_text = re.sub(r'[#_`]', '', clean_text)
    # Remove dashes/hyphens used in tables and horizontal rules
    clean_text = re.sub(r'-{2,}', ' ', clean_text)  # Replace multiple dashes with space
    # Remove pipe symbols used in markdown tables
    clean_text = re.sub(r'\|', ' ', clean_text)
    # Remove extra whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    if clean_text != response_text:
        logger.debug(f"Cleaned text for TTS: {clean_text}")
    
    try:
        # Generate speech using Cartesia Sonic 3 TTS (streaming)
        chunk_count = 0
        chunk_iter = cartesia_client.tts.bytes(
            model_id=CARTESIA_TTS_CONFIG["model_id"],
            transcript=clean_text,
            voice=CARTESIA_TTS_CONFIG["voice"],
            output_format=CARTESIA_TTS_CONFIG["output_format"],
        )
        
        # Buffer to accumulate partial chunks
        buffer = b""
        element_size = 4  # float32 is 4 bytes
        
        # Stream audio chunks and convert to FastRTC format
        for chunk in chunk_iter:
            # Accumulate chunks in buffer
            buffer += chunk
            
            # Process complete float32 samples
            num_complete_samples = len(buffer) // element_size
            if num_complete_samples > 0:
                # Extract complete samples
                complete_bytes = num_complete_samples * element_size
                complete_buffer = buffer[:complete_bytes]
                buffer = buffer[complete_bytes:]  # Keep remainder for next iteration
                
                # Convert to numpy array
                audio_array = np.frombuffer(complete_buffer, dtype=np.float32)
                chunk_count += 1
                
                # Yield in FastRTC format: (sample_rate, audio_array)
                yield (CARTESIA_TTS_CONFIG["output_format"]["sample_rate"], audio_array)
        
        # Process any remaining bytes in buffer
        if len(buffer) > 0:
            # Pad to complete sample if needed
            remainder = len(buffer) % element_size
            if remainder != 0:
                buffer += b'\x00' * (element_size - remainder)
            
            if len(buffer) >= element_size:
                audio_array = np.frombuffer(buffer, dtype=np.float32)
                chunk_count += 1
                yield (CARTESIA_TTS_CONFIG["output_format"]["sample_rate"], audio_array)
        
        tts_time = time.time() - tts_start
        total_time = time.time() - start_time
        logger.info(f'⚡ Performance: STT={stt_time:.2f}s | LLM={llm_time:.2f}s | TTS={tts_time:.2f}s | Total={total_time:.2f}s | Chunks={chunk_count}')
            
    except Exception as e:
        logger.error(f"Error in Cartesia TTS generation: {str(e)}")
        raise


def create_stream() -> Stream:
    """
    Create and configure a Stream instance with audio capabilities.
    Optimized for low latency.

    Returns:
        Stream: Configured FastRTC Stream instance
    """
    return Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(
            response,
            algo_options=AlgoOptions(
                speech_threshold=0.4,  # Slightly lower for faster detection
            ),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastRTC Cartesia Voice Agent (Ultra-Low Latency)")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (get a temp phone number)",
    )
    args = parser.parse_args()

    stream = create_stream()
    logger.info("🎧 Stream handler configured")

    if args.phone:
        logger.info("🌈 Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info("🌈 Launching with Gradio UI...")

        stream.ui.launch()
