import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

import re

# Find the exact text where it auto-switches
old_pattern = r'var navIntel=document\.getElementById\(\'nav-intel\'\);\s*switchPage\(\'page-intel\',navIntel\);\s*// Ad[^\n]*'
# We will just replace it with the hidden rendering trick
new_block = """
          // To draw the spectrogram while on Dashboard, we must temporarily render page-intel 
          // off-screen so offsetWidth is not 0.
          const pageIntel = document.getElementById('page-intel');
          const wasHidden = pageIntel && pageIntel.style.display === 'none';
          if (wasHidden) {
            pageIntel.style.visibility = 'hidden';
            pageIntel.style.display = 'block';
            pageIntel.style.position = 'absolute';
            pageIntel.style.top = '-99999px';
          }
          
          resizeSpec();
          resizeWave();
          drawWave(waveData, 0);
          drawSpectrogram(ch, buf.sampleRate, Math.min(150, Math.floor(ch.length/256)));
          
          if (wasHidden) {
            pageIntel.style.display = 'none';
            pageIntel.style.visibility = '';
            pageIntel.style.position = '';
            pageIntel.style.top = '';
          }
"""

text, count = re.subn(old_pattern, new_block, text)
print("Replaced:", count)

# Also let's check for any other tryDraw timeouts that might be causing issues
old_trydraw = """          var _ch=ch,_sr=buf.sampleRate,_nf=Math.min(150,Math.floor(ch.length/256));
          var _tries=0;
          function tryDraw(){
            resizeSpec();resizeWave();
            if(specCanvas.offsetWidth>0&&specCanvas.offsetHeight>0){
              drawWave(waveData,0);
              drawSpectrogram(_ch,_sr,_nf);
            } else if(_tries++<30){
              setTimeout(tryDraw,50);
            }
          }
          setTimeout(tryDraw,60);"""

text, count2 = re.subn(re.escape(old_trydraw), "", text)
print("Removed old tryDraw:", count2)


with codecs.open(path, "w", "utf-8") as f:
    f.write(text)
