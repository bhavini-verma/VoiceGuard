import codecs
import re

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

pattern = r"waveData=wd;drawWave\(waveData,0\);\s*drawSpectrogram\(ch,buf\.sampleRate,Math\.min\(150,Math\.floor\(ch\.length/256\)\)\);\s*document\.getElementById\('wv-stat'\)\.textContent='SIGNAL OK';\s*document\.getElementById\('wv-stat'\)\.style\.color='var\(--success\)';"

replacement = """waveData=wd;drawWave(waveData,0);
          drawSpectrogram(ch,buf.sampleRate,Math.min(150,Math.floor(ch.length/256)));
          
          document.getElementById('wv-dur').textContent=buf.duration.toFixed(1)+'s';
          document.getElementById('wv-sr').textContent=buf.sampleRate+'Hz';
          document.getElementById('wv-ch').textContent=buf.numberOfChannels;
          const specDurLabel = document.getElementById('spec-dur-label');
          if(specDurLabel) specDurLabel.textContent=buf.duration.toFixed(1)+'s';
          
          document.getElementById('wv-stat').textContent='STAGED - READY';
          document.getElementById('wv-stat').style.color='var(--success)';
"""

new_text = re.sub(pattern, replacement, text)

if new_text != text:
    with codecs.open(path, "w", "utf-8") as f:
        f.write(new_text)
    print("SUCCESS")
else:
    print("Regex failed to match")
