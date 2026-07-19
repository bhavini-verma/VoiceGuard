import codecs
import re

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

target = """          const wd=[];for(let i=0;i<pts;i++){let mx=0;for(let j=0;j<step;j++){const v=Math.abs(ch[i*step+j]||0);if(v>mx)mx=v;}wd.push(mx*2-1);}
          waveData=wd;drawWave(waveData,0);
          drawSpectrogram(ch,buf.sampleRate,Math.min(150,Math.floor(ch.length/256)));
          document.getElementById('wv-stat').textContent='SIGNAL OK';
          document.getElementById('wv-stat').style.color='var(--success)';"""

replacement = """          const wd=[];for(let i=0;i<pts;i++){let mx=0;for(let j=0;j<step;j++){const v=Math.abs(ch[i*step+j]||0);if(v>mx)mx=v;}wd.push(mx*2-1);}
          waveData=wd;drawWave(waveData,0);
          drawSpectrogram(ch,buf.sampleRate,Math.min(150,Math.floor(ch.length/256)));
          
          document.getElementById('wv-dur').textContent=buf.duration.toFixed(1)+'s';
          document.getElementById('wv-sr').textContent=buf.sampleRate+'Hz';
          document.getElementById('wv-ch').textContent=buf.numberOfChannels;
          document.getElementById('spec-dur-label').textContent=buf.duration.toFixed(1)+'s';
          document.getElementById('spec-stat').textContent='SIGNAL OK';
          document.getElementById('spec-stat').style.color='var(--success)';
          
          document.getElementById('wv-stat').textContent='STAGED - READY';
          document.getElementById('wv-stat').style.color='var(--success)';
"""

if target in text:
    text = text.replace(target, replacement)
    with codecs.open(path, "w", "utf-8") as f:
        f.write(text)
    print("SUCCESS")
else:
    print("Target not found")
