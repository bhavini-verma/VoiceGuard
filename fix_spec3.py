import codecs
import re

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

# Use regex to replace the freqBins line with the new frequency-capped version
old_lines = "  const freqBins=fftSize/2;\r\n  const frameW=w/frames,binH=h/freqBins;"

new_lines = """  // Cap display at 8000 Hz (covers all speech). Fixes 48kHz live mic showing blank (speech was in bottom 12%).
  const hzPerBin = sampleRate / fftSize;
  const maxDisplayHz = 8000;
  const freqBins = Math.min(fftSize/2, Math.max(1, Math.ceil(maxDisplayHz / hzPerBin)));
  const frameW=w/frames,binH=h/freqBins;"""

if old_lines in text:
    text = text.replace(old_lines, new_lines)
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("SUCCESS")
else:
    # Try with \r\r\n (double carriage return)
    old_lines2 = "  const freqBins=fftSize/2;\r\r\n  const frameW=w/frames,binH=h/freqBins;"
    if old_lines2 in text:
        text = text.replace(old_lines2, new_lines)
        with codecs.open(path, "w", "utf-8") as f:
            f.write(text)
        print("SUCCESS (double CR)")
    else:
        print("NOT FOUND - trying regex")
        new_text = re.sub(r'const freqBins=fftSize/2;\s*const frameW=w/frames,binH=h/freqBins;',
            '// Cap display at 8000 Hz (covers all speech). Fixes 48kHz live mic showing blank.\n  const hzPerBin = sampleRate / fftSize;\n  const maxDisplayHz = 8000;\n  const freqBins = Math.min(fftSize/2, Math.max(1, Math.ceil(maxDisplayHz / hzPerBin)));\n  const frameW=w/frames,binH=h/freqBins;',
            text)
        if new_text != text:
            with codecs.open(path, "w", "utf-8") as f:
                f.write(new_text)
            print("SUCCESS (regex)")
        else:
            print("REGEX ALSO FAILED")
