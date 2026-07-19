import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\dataset_tools\vad_split_genuine.py"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

old_block = """    # Load and resample using pydub (bypasses numba DLL block)
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(VAD_SAMPLE_RATE).set_channels(1)
        samples = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            y = samples.astype(np.float32) / 32768.0
        elif audio.sample_width == 4:
            y = samples.astype(np.float32) / 2147483648.0
        else:
            return 0, 0, "Unsupported sample width"
    except Exception as e:
        return 0, 0, str(e)"""

new_block = """    # Load and resample using raw ffmpeg (bypasses numba DLL block and missing ffprobe)
    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        # We output raw 16-bit PCM (s16le) at VAD_SAMPLE_RATE Hz, mono
        cmd = [
            ffmpeg_exe,
            "-i", input_path,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(VAD_SAMPLE_RATE),
            "-ac", "1",
            "-"
        ]
        
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            return 0, 0, f"ffmpeg error: {proc.stderr.decode('utf-8', 'ignore')}"
            
        raw_audio = proc.stdout
        # Convert bytes to numpy float32
        samples = np.frombuffer(raw_audio, dtype=np.int16)
        y = samples.astype(np.float32) / 32768.0
    except Exception as e:
        return 0, 0, str(e)"""

if old_block in text:
    text = text.replace(old_block, new_block)
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("Fixed VAD script to use raw ffmpeg")
else:
    print("Could not find block to replace")
