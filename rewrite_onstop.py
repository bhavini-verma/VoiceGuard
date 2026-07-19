import codecs

path = r"c:\Users\rajas\.gemini\antigravity-ide\scratch\VoiceGuard_DL_Exp\static\voiceguard_uco_bank_platform.html"
with codecs.open(path, "r", "utf-8") as f:
    text = f.read()

idx_start = text.find("mr.onstop=()")
idx_end = text.find("function stopMic()", idx_start)

# Replace the entire mr.onstop block with a clean rewrite
new_onstop = """mr.onstop=()=>{
      const blob=new Blob(micChunks,{type:'audio/webm'});
      const reader=new FileReader();
      reader.onload=ev=>{
        const tmpCtx=new(window.AudioContext||window.webkitAudioContext)();
        tmpCtx.decodeAudioData(ev.target.result.slice(0),function(buf){
          // ---- Store audio data ----
          storedAudioBuffer=buf;
          audioDuration=buf.duration;
          const ch=buf.getChannelData(0);
          const pts=300;const step=Math.floor(ch.length/pts)||1;
          const wd=[];
          for(let i=0;i<pts;i++){let mx=0;for(let j=0;j<step;j++){const v=Math.abs(ch[i*step+j]||0);if(v>mx)mx=v;}wd.push(mx*2-1);}
          waveData=wd;

          // ---- Update metadata labels ----
          document.getElementById('wv-dur').textContent=buf.duration.toFixed(1)+'s';
          document.getElementById('wv-sr').textContent=buf.sampleRate+'Hz';
          document.getElementById('wv-ch').textContent=buf.numberOfChannels;
          var sdl=document.getElementById('spec-dur-label');if(sdl)sdl.textContent=buf.duration.toFixed(1)+'s';
          document.getElementById('wv-stat').textContent='STAGED - READY';
          document.getElementById('wv-stat').style.color='var(--success)';

          // ---- Convert to WAV and stage for analysis ----
          var wavBlob=audioBufferToWav(buf);
          var f=new File([wavBlob],'MIC_'+Date.now()+'.wav',{type:'audio/wav'});
          STAGED_FILE=f;
          setUploadZoneState('loaded',f.name,formatSize(f.size)+' \xb7 PCM/WAV \xb7 '+buf.duration.toFixed(1)+'s');
          setHdrChip('ready');
          document.getElementById('clear-btn').style.display='flex';
          document.getElementById('start-btn').style.display='flex';

          // ---- Wire up audio playback ----
          if(currentAudio){currentAudio.pause();currentAudio.src='';}
          currentAudio=new Audio(URL.createObjectURL(blob));
          currentAudio.crossOrigin='anonymous';
          // Wire playback WebAudio (fresh context each time to avoid stale node errors)
          try{
            var pCtx=new(window.AudioContext||window.webkitAudioContext)();
            var pAna=pCtx.createAnalyser();pAna.fftSize=256;
            var pSrc=pCtx.createMediaElementSource(currentAudio);
            pSrc.connect(pAna);pAna.connect(pCtx.destination);
            playbackCtx=pCtx;playbackAnalyser=pAna;playbackSource=pSrc;
          }catch(e){console.warn('Playback WebAudio setup:',e);}

          currentAudio.addEventListener('ended',function(){
            isPlaying=false;document.getElementById('play-icon').className='ti ti-player-play';
            wavePos=0;document.getElementById('prog-fill').style.width='0%';
            drawWave(waveData,0);
          });
          currentAudio.addEventListener('timeupdate',function(e){
            var a=e.target;
            var dur=(a.duration&&a.duration!==Infinity)?a.duration:buf.duration;
            if(dur>0){
              wavePos=a.currentTime/dur;
              document.getElementById('prog-fill').style.width=(wavePos*100)+'%';
              drawWave(waveData,wavePos);
              drawSpecPlayhead(wavePos);
            }
          });

          // ---- Draw spectrogram ----
          // Navigate to page-intel first (canvas lives there), then draw once visible.
          var navIntel=document.getElementById('nav-intel');
          switchPage('page-intel',navIntel);
          // Additionally, poll until canvas has valid dimensions (belt-and-suspenders).
          var _ch=ch,_sr=buf.sampleRate,_nf=Math.min(150,Math.floor(ch.length/256));
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
          setTimeout(tryDraw,60);

        },function(){document.getElementById('wv-stat').textContent='DECODE ERR';});
      };
      reader.readAsArrayBuffer(blob);
    };
    mr.start(100);
  }).catch(function(){setHdrChip('ready');});
}
"""

text = text[:idx_start] + new_onstop + "\n" + text[idx_end:]

with codecs.open(path, "w", "utf-8") as f:
    f.write(text)
print("SUCCESS - mr.onstop cleanly rewritten")
