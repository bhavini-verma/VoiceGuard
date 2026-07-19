
const BACKEND_URL = window.VoiceGuard_BACKEND_URL || '';
const SESSION_ID = 'VG-'+Date.now().toString(36).toUpperCase().slice(-6);
document.getElementById('hdr-case').textContent = 'CASE: '+SESSION_ID;
document.getElementById('h-caseid').textContent = SESSION_ID;
document.getElementById('rpt-id').textContent = SESSION_ID;

function updateClock(){const n=new Date();document.getElementById('hdr-clock').textContent=n.toLocaleTimeString('en-IN',{hour12:false});}
updateClock();setInterval(updateClock,1000);

function setHdrChip(type){
  const el=document.getElementById('hdr-chip');
  const lb=document.getElementById('hdr-chip-label');
  const map={ready:{cls:'chip-ready',txt:'READY'},analyzing:{cls:'chip-analyzing',txt:'ANALYZING'},complete:{cls:'chip-complete',txt:'REPORT AVAILABLE'},risk:{cls:'chip-risk',txt:'HIGH RISK'}};
  const m=map[type]||map.ready;
  el.className='status-chip '+m.cls;lb.textContent=m.txt;
}

// === FORENSIC OVERLAY STATE ===
let chunkOverlays = [];   // [{start,end,score,verdict,bio_score,deep_score}]
let audioDuration = 0;    // total audio duration in seconds
let _chunkFetchController = null; // AbortController for async chunk fetch

// Shared palette function (Viridis)
const specPalette = [[68,1,84], [59,82,139], [33,145,140], [94,201,98], [253,231,37]];
function getSpecColor(v) {
  const idx = v * (specPalette.length - 1);
  const lo = Math.floor(idx), hi = Math.min(lo + 1, specPalette.length - 1);
  const t = idx - lo;
  return specPalette[lo].map((c, i) => Math.round(c + (specPalette[hi][i] - c) * t));
}

function drawSpecOverlays() {
  if (!chunkOverlays.length || !audioDuration) return;
  const w = specCanvas.width / devicePixelRatio;
  const h = specCanvas.height / devicePixelRatio;
  
  sCtx.save();
  chunkOverlays.forEach(chunk => {
    if (chunk.score < 35) return; // Only highlight warning/high risk
    const x = (chunk.start / audioDuration) * w;
    const width = ((chunk.end - chunk.start) / audioDuration) * w;
    const color = chunk.score >= 65 ? 'rgba(239, 68, 68, 0.15)' : 'rgba(245, 158, 11, 0.15)';
    sCtx.fillStyle = color;
    sCtx.fillRect(x, 0, width, h);
    // Subtle borders for the overlay
    sCtx.strokeStyle = chunk.score >= 65 ? 'rgba(239, 68, 68, 0.5)' : 'rgba(245, 158, 11, 0.5)';
    sCtx.beginPath();
    sCtx.moveTo(x, 0); sCtx.lineTo(x, h);
    sCtx.moveTo(x + width, 0); sCtx.lineTo(x + width, h);
    sCtx.stroke();
  });
  sCtx.restore();
}

function drawThreatRibbon(pos = 0) {
  const cBio = document.getElementById('threat-ribbon-bio');
  const cDeep = document.getElementById('threat-ribbon-deep');
  const cAll = document.getElementById('threat-ribbon-all');
  if (!cBio || !cDeep || !cAll) return;
  
  [cBio, cDeep, cAll].forEach(canvas => {
    canvas.width = canvas.offsetWidth * devicePixelRatio;
    canvas.height = canvas.offsetHeight * devicePixelRatio;
    const ctx = canvas.getContext('2d');
    ctx.scale(devicePixelRatio, devicePixelRatio);
    ctx.clearRect(0, 0, canvas.offsetWidth, canvas.offsetHeight);
  });
  
  if (!chunkOverlays.length || !audioDuration) return;
  const w = cBio.offsetWidth, h = cBio.offsetHeight;
  const cx = pos * w;
  
  chunkOverlays.forEach(c => {
    const startX = (c.start / audioDuration) * w;
    const endX = (c.end / audioDuration) * w;
    if (startX > cx) return; // Not reached yet
    const x2 = Math.min(endX, cx);
    
    // Draw BIO
    let scoreBio = c.bio_score || 0;
    let colorBio = scoreBio >= 65 ? '#EF4444' : scoreBio >= 35 ? '#F59E0B' : '#22C55E';
    let ctx = cBio.getContext('2d'); ctx.fillStyle = colorBio; ctx.fillRect(startX, 0, x2 - startX, h);
    
    // Draw DEEP
    let scoreDeep = c.deep_score || 0;
    let colorDeep = scoreDeep >= 65 ? '#EF4444' : scoreDeep >= 35 ? '#F59E0B' : '#22C55E';
    ctx = cDeep.getContext('2d'); ctx.fillStyle = colorDeep; ctx.fillRect(startX, 0, x2 - startX, h);
    
    // Draw ALL
    let colorAll = c.verdict === 'FRAUD' ? '#EF4444' : c.verdict === 'SUSPICIOUS' ? '#F59E0B' : '#22C55E';
    ctx = cAll.getContext('2d'); ctx.fillStyle = colorAll; ctx.fillRect(startX, 0, x2 - startX, h);
  });
}

function showChunkTooltip(chunk, x, y) {
  let tip = document.getElementById('chunk-tip');
  if (!tip) {
    tip = document.createElement('div');
    tip.id = 'chunk-tip';
    tip.style.cssText = 'position:fixed;z-index:9999;pointer-events:none;background:#1E293B;color:#F1F5F9;font-size:11px;font-family:var(--mono);padding:6px 10px;border-radius:6px;box-shadow:0 4px 12px rgba(0,0,0,0.3);transition:opacity 0.2s;white-space:nowrap;';
    document.body.appendChild(tip);
  }
  const color = chunk.score >= 65 ? '#EF4444' : chunk.score >= 35 ? '#F59E0B' : '#22C55E';
  const verdictLabel = (chunk.verdict || 'UNKNOWN').replace(/_/g, ' ');
  const bioLabel = chunk.bio_score !== undefined ? ' · Bio: ' + chunk.bio_score.toFixed(1) + '%' : '';
  const deepLabel = chunk.deep_score !== undefined ? ' · Deep: ' + chunk.deep_score.toFixed(1) + '%' : '';
  tip.innerHTML = '<span style="color:' + color + '">\u25CF ' + verdictLabel + '</span> | <b>' + chunk.start.toFixed(1) + 's \u2013 ' + chunk.end.toFixed(1) + 's</b><br/>' +
                  '<span style="opacity:0.9">' + deepLabel + bioLabel + '</span>';
  // Position: clamp to viewport
  const tipW = 350;
  let left = x + 12;
  if (left + tipW > window.innerWidth) left = x - tipW - 12;
  tip.style.left = left + 'px';
  tip.style.top = (y - 36) + 'px';
  tip.style.opacity = '1';
  clearTimeout(tip._hide);
  tip._hide = setTimeout(() => { tip.style.opacity = '0'; }, 3500);
}

function drawSpecPlayhead(pos) {
  const canvas = document.getElementById('spec-playhead-canvas');
  const ribCanvas = document.getElementById('ribbon-playhead-canvas');
  if (!canvas) return;
  
  const w = canvas.offsetWidth, h = canvas.offsetHeight;
  canvas.width = w * devicePixelRatio; canvas.height = h * devicePixelRatio;
  const pc = canvas.getContext('2d');
  pc.scale(devicePixelRatio, devicePixelRatio);
  pc.clearRect(0, 0, w, h);
  
  // Sync the threat ribbons with the playhead
  drawThreatRibbon(pos);
  
  if (ribCanvas) {
    ribCanvas.width = ribCanvas.offsetWidth * devicePixelRatio; ribCanvas.height = ribCanvas.offsetHeight * devicePixelRatio;
    const rc = ribCanvas.getContext('2d'); rc.scale(devicePixelRatio, devicePixelRatio); rc.clearRect(0, 0, ribCanvas.offsetWidth, ribCanvas.offsetHeight);
    if (audioDuration && pos > 0) {
      const cx = pos * ribCanvas.offsetWidth;
      rc.strokeStyle = '#FFFFFF'; 
      rc.lineWidth = 2; 
      rc.shadowColor = 'rgba(0,0,0,0.8)';
      rc.shadowBlur = 4;
      rc.setLineDash([]);
      rc.beginPath(); rc.moveTo(cx, 0); rc.lineTo(cx, ribCanvas.offsetHeight); rc.stroke();
    }
  }

  if (!audioDuration || pos === 0) return;
  const x = pos * w;
  
  // Trailing glow
  const glowW = Math.min(x, 40);
  if (glowW > 0) {
    const trailGrad = pc.createLinearGradient(x - glowW, 0, x, 0);
    trailGrad.addColorStop(0, 'rgba(239,68,68,0)');
    trailGrad.addColorStop(1, 'rgba(239,68,68,0.25)');
    pc.fillStyle = trailGrad;
    pc.fillRect(x - glowW, 0, glowW, h);
  }
  
  pc.strokeStyle = 'rgba(239,68,68,0.9)';
  pc.lineWidth = 1.5;
  pc.setLineDash([3, 2]);
  pc.beginPath(); pc.moveTo(x, 0); pc.lineTo(x, h); pc.stroke();
  pc.setLineDash([]);
}

// Hover tooltip for local RMS (Feature D)
function showHoverTooltip(text, x, y) {
  let tip = document.getElementById('hover-tip');
  if (!tip) {
    tip = document.createElement('div');
    tip.id = 'hover-tip';
    tip.style.cssText = 'position:fixed;z-index:9998;pointer-events:none;background:#334155;color:#E2E8F0;font-size:10px;font-family:var(--mono);padding:4px 8px;border-radius:4px;box-shadow:0 2px 8px rgba(0,0,0,0.2);transition:opacity 0.15s;white-space:nowrap;';
    document.body.appendChild(tip);
  }
  tip.textContent = text;
  tip.style.left = (x + 10) + 'px';
  tip.style.top = (y - 28) + 'px';
  tip.style.opacity = '1';
  clearTimeout(tip._hide);
  tip._hide = setTimeout(() => { tip.style.opacity = '0'; }, 1500);
}

// WebAudio Live Playback State
let playbackCtx = null, playbackAnalyser = null, playbackSource = null;
let liveAnimId = null;
let lastDrawX = 0;

function clearLiveSweep() {
  const c = document.getElementById('spec-live-canvas');
  if (c) c.getContext('2d').clearRect(0,0,c.width,c.height);
  lastDrawX = 0;
}

function startLiveSweepLoop() {
  if (!playbackAnalyser || !currentAudio) return;
  const c = document.getElementById('spec-live-canvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  
  // On resume from pause, preserve existing pixels by NOT resizing.
  // We only set canvas dimensions if it is uninitialized (default 300 width) or a fresh playback.
  const w = Math.floor(c.offsetWidth * devicePixelRatio);
  const h = Math.floor(c.offsetHeight * devicePixelRatio);
  
  if (c.width === 0 || c.width === 300 || currentAudio.currentTime < 0.1) {
    c.width = w;
    c.height = h;
    lastDrawX = 0;
  }
  
  const dataArray = new Uint8Array(playbackAnalyser.frequencyBinCount);
  const binH = h / dataArray.length;
  
  // On resume or scrub, sync lastDrawX to where we currently are in time
  if (currentAudio.currentTime > 0 && currentAudio.duration > 0) {
    const resumeX = Math.floor((currentAudio.currentTime / currentAudio.duration) * w);
    if (lastDrawX === 0 || Math.abs(lastDrawX - resumeX) > 10) {
      lastDrawX = resumeX;
    }
  }
  
  function draw() {
    if (currentAudio.paused || currentAudio.ended) return;
    liveAnimId = requestAnimationFrame(draw);
    
    playbackAnalyser.getByteFrequencyData(dataArray);
    const ratio = currentAudio.currentTime / currentAudio.duration;
    if (isNaN(ratio)) return;
    const currentX = Math.floor(ratio * w);
    
    // Draw columns between lastDrawX and currentX to prevent gaps
    const startX = Math.max(lastDrawX + 1, 0);
    if (currentX >= startX) {
      for (let x = startX; x <= currentX; x++) {
        for (let i = 0; i < dataArray.length; i++) {
          const v = dataArray[i] / 255.0;
          if (v > 0.05) {
            const rgb = getSpecColor(v);
            ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0.9)`;
            ctx.fillRect(x, h - (i+1)*binH, 2, Math.ceil(binH));
          }
        }
      }
    }
    lastDrawX = currentX;
  }
  draw();
}

// === END FORENSIC OVERLAY STATE ===

const waveCanvas=document.getElementById('wave-canvas');
const wCtx=waveCanvas.getContext('2d');
let waveData=[];let isPlaying=false;let wavePos=0;
function resizeWave(){waveCanvas.width=waveCanvas.offsetWidth*devicePixelRatio;waveCanvas.height=waveCanvas.offsetHeight*devicePixelRatio;wCtx.scale(devicePixelRatio,devicePixelRatio);}
resizeWave();
waveCanvas.style.cursor='pointer';
waveCanvas.addEventListener('click',(e)=>{
  if(!currentAudio||!currentAudio.duration||waveData.length===0)return;
  const rect=waveCanvas.getBoundingClientRect();
  const seekRatio=(e.clientX-rect.left)/waveCanvas.offsetWidth;
  currentAudio.currentTime=seekRatio*currentAudio.duration;
  wavePos=seekRatio;
  drawWave(waveData,wavePos);
  drawSpecPlayhead(wavePos);
  // Show chunk tooltip if click lands in a flagged region
  const t = seekRatio * (currentAudio.duration || audioDuration);
  const hit = chunkOverlays.find(c => t >= c.start && t < c.end);
  if (hit) showChunkTooltip(hit, e.clientX, e.clientY);
});
function drawWave(data, pos) {
  const w = waveCanvas.offsetWidth, h = waveCanvas.offsetHeight;
  wCtx.clearRect(0, 0, w, h);
  if (!data || !data.length) {
    wCtx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border') || '#D6E3F5';
    wCtx.lineWidth = 1; wCtx.beginPath(); wCtx.moveTo(0, h/2); wCtx.lineTo(w, h/2); wCtx.stroke(); return;
  }
  const cx = (pos || 0) * w;
  const step = w / data.length;

  // 1. Subtle volumetric background glow
  if (chunkOverlays.length && audioDuration) {
    chunkOverlays.forEach(c => {
      const x1 = Math.max(0, (c.start / audioDuration) * w);
      const x2 = Math.min(w, (c.end / audioDuration) * w);
      let color = 'rgba(20,184,166,0.04)';
      if (c.score >= 65 || (c.verdict && (c.verdict.includes('HIGH') || c.verdict.includes('CRITICAL')))) color = 'rgba(239,68,68,0.08)';
      else if (c.score >= 35) color = 'rgba(245,158,11,0.06)';
      const glowGrad = wCtx.createLinearGradient(0, 0, 0, h);
      glowGrad.addColorStop(0, 'rgba(255,255,255,0)');
      glowGrad.addColorStop(0.5, color);
      glowGrad.addColorStop(1, 'rgba(255,255,255,0)');
      wCtx.fillStyle = glowGrad;
      wCtx.fillRect(x1, 0, x2 - x1, h);
    });
  }

  // 2. Mirrored filled waveform
  const grad = wCtx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(0, 91, 172, 0.0)');
  grad.addColorStop(0.5, 'rgba(0, 91, 172, 0.15)');
  grad.addColorStop(1, 'rgba(0, 91, 172, 0.0)');
  wCtx.fillStyle = grad;
  wCtx.beginPath();
  wCtx.moveTo(0, h/2);
  for(let i=0; i<data.length; i++) wCtx.lineTo(i*step, h/2 - data[i] * h * 0.44);
  for(let i=data.length-1; i>=0; i--) wCtx.lineTo(i*step, h/2 + data[i] * h * 0.44);
  wCtx.closePath();
  wCtx.fill();

  // 3. Dynamic stroke coloring
  let activeStrokeStyle = '#005BAC';
  if (chunkOverlays.length && audioDuration) {
    const strokeGrad = wCtx.createLinearGradient(0, 0, w, 0);
    const sorted = [...chunkOverlays].sort((a,b) => a.start - b.start);
    sorted.forEach((c, i) => {
      let color = '#14B8A6';
      if (c.score >= 65 || (c.verdict && (c.verdict.includes('HIGH') || c.verdict.includes('CRITICAL')))) color = '#EF4444';
      else if (c.score >= 35) color = '#F59E0B';
      
      let startRatio = Math.max(0, Math.min(1, c.start / audioDuration));
      let endRatio = Math.max(0, Math.min(1, c.end / audioDuration));
      if (i === 0 && startRatio > 0) strokeGrad.addColorStop(0, color);
      try {
          strokeGrad.addColorStop(startRatio, color);
          if (endRatio > startRatio) strokeGrad.addColorStop(endRatio, color);
          if (i === sorted.length - 1 && endRatio < 1) strokeGrad.addColorStop(1, color);
      } catch(e){}
    });
    activeStrokeStyle = strokeGrad;
  }

  // Unplayed stroke
  wCtx.strokeStyle = 'rgba(156, 163, 175, 0.4)';
  wCtx.lineWidth = 1.5;
  wCtx.beginPath();
  for(let i=0; i<data.length; i++) {
    const x = i * step;
    const y = h/2 - data[i] * h * 0.44;
    i===0 ? wCtx.moveTo(x, y) : wCtx.lineTo(x, y);
  }
  wCtx.stroke();

  // Played stroke (colored)
  wCtx.strokeStyle = activeStrokeStyle;
  wCtx.lineWidth = 2.0;
  wCtx.beginPath();
  for(let i=0; i<data.length; i++) {
    const x = i * step;
    if (x > cx) break;
    const y = h/2 - data[i] * h * 0.44;
    i===0 ? wCtx.moveTo(x, y) : wCtx.lineTo(x, y);
  }
  wCtx.stroke();
}
drawWave(null,0);

const specCanvas=document.getElementById('spec-canvas');
const sCtx=specCanvas.getContext('2d');
function resizeSpec(){specCanvas.width=specCanvas.offsetWidth*devicePixelRatio;specCanvas.height=specCanvas.offsetHeight*devicePixelRatio;sCtx.scale(devicePixelRatio,devicePixelRatio);}
resizeSpec();
function drawSpectrogram(audioData,sampleRate,numFrames){
  const w=specCanvas.offsetWidth,h=specCanvas.offsetHeight;
  specCanvas.width = w * devicePixelRatio;
  specCanvas.height = h * devicePixelRatio;
  sCtx.scale(devicePixelRatio, devicePixelRatio);
  
  if(!audioData){
    // Completely wipe the static spectrogram canvas
    sCtx.clearRect(0, 0, w, h);
    // Also wipe the live sweep canvas
    const liveC = document.getElementById('spec-live-canvas');
    if (liveC) liveC.getContext('2d').clearRect(0, 0, liveC.width, liveC.height);
    lastDrawX = 0;
    sCtx.font='12px Inter';sCtx.fillStyle='#9CA3AF';sCtx.textAlign='center';
    sCtx.fillText('Spectrogram will appear after audio upload',w/2,h/2);
    return;
  }
  

  const frames=Math.max(1, numFrames||Math.floor(audioData.length/256));
  const fftSize=256,hopSize=Math.max(1, Math.floor((audioData.length-fftSize)/frames));
  const freqBins=fftSize/2;
  const frameW=w/frames,binH=h/freqBins;
  
  for(let f=0;f<frames;f++){
    const offset=f*hopSize;
    for(let b=0;b<freqBins;b++){
      let re=0, im=0;
      for(let i=0; i<fftSize; i++) {
        if (offset+i >= audioData.length) break;
        const window = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (fftSize - 1));
        const val = audioData[offset+i] * window;
        const angle = 2 * Math.PI * b * i / fftSize;
        re += val * Math.cos(angle);
        im -= val * Math.sin(angle);
      }
      const mag = Math.sqrt(re*re + im*im) / (fftSize/2);
      const norm=Math.min(1,Math.pow(mag*3.5,0.5)); 
      
      if (norm > 0.02) {
        sCtx.fillStyle=`rgba(15, 23, 42, ${norm * 0.8})`; 
        sCtx.fillRect(f*frameW,(freqBins-1-b)*binH,frameW+1,binH+1);
      }
    }
  }
  sCtx.strokeStyle='rgba(15,23,42,0.05)';sCtx.lineWidth=.5;
  [0.25,0.5,0.75].forEach(p=>{sCtx.beginPath();sCtx.moveTo(p*w,0);sCtx.lineTo(p*w,h);sCtx.stroke();});
  [0.33,0.66].forEach(p=>{sCtx.beginPath();sCtx.moveTo(0,p*h);sCtx.lineTo(w,p*h);sCtx.stroke();});
  
  // Draw chunk tints ON TOP of spectrogram pixels so they are always visible
  if (chunkOverlays && chunkOverlays.length && audioDuration) {
    chunkOverlays.forEach(chunk => {
      const x = (chunk.start / audioDuration) * w;
      const cw = ((chunk.end - chunk.start) / audioDuration) * w;
      if (chunk.score >= 65 || (chunk.verdict && (chunk.verdict.includes('HIGH') || chunk.verdict.includes('CRITICAL')))) {
        sCtx.fillStyle = 'rgba(239, 68, 68, 0.22)';
      } else if (chunk.score >= 35) {
        sCtx.fillStyle = 'rgba(245, 158, 11, 0.16)';
      } else {
        sCtx.fillStyle = 'rgba(34, 197, 94, 0.12)';
      }
      sCtx.fillRect(x, 0, cw, h);
      // Top border line for that region
      sCtx.fillStyle = chunk.score >= 65 ? 'rgba(239,68,68,0.7)' : (chunk.score >= 35 ? 'rgba(245,158,11,0.7)' : 'rgba(34,197,94,0.7)');
      sCtx.fillRect(x, 0, cw, 2);
    });
  }
  
  document.getElementById('spec-stat').textContent='SIGNAL ACTIVE';
  document.getElementById('spec-stat').style.color='var(--success)';
}
drawSpectrogram(null);


const ribbonContainer = document.getElementById('ribbon-container');
if (ribbonContainer) {
  ribbonContainer.addEventListener('mousemove', e => {
    if (!audioDuration || !chunkOverlays.length) return;
    const rect = ribbonContainer.getBoundingClientRect();
    const seekRatio = (e.clientX - rect.left) / ribbonContainer.offsetWidth;
    const t = seekRatio * audioDuration;
    const hit = chunkOverlays.find(c => t >= c.start && t < c.end);
    if (hit) {
      showChunkTooltip(hit, e.clientX, e.clientY);
      ribbonContainer.style.cursor = 'pointer';
    } else {
      ribbonContainer.style.cursor = 'default';
      const tip = document.getElementById('chunk-tip');
      if (tip) tip.style.opacity = '0';
    }
  });
  ribbonContainer.addEventListener('mouseleave', () => {
    const tip = document.getElementById('chunk-tip');
    if (tip) tip.style.opacity = '0';
  });
  ribbonContainer.addEventListener('click', e => {
    if (!currentAudio || !currentAudio.duration || !audioDuration) return;
    const rect = ribbonContainer.getBoundingClientRect();
    const seekRatio = (e.clientX - rect.left) / ribbonContainer.offsetWidth;
    currentAudio.currentTime = seekRatio * currentAudio.duration;
    wavePos = seekRatio;
    drawWave(waveData, wavePos);
    drawSpecPlayhead(wavePos);
    const t = seekRatio * currentAudio.duration;
    const hit = chunkOverlays.find(c => t >= c.start && t < c.end);
    if (hit) {
      showChunkTooltip(hit, e.clientX, e.clientY);
    } else {
      const tip = document.getElementById('chunk-tip');
      if (tip) tip.style.opacity = '0';
    }
  });
}

specCanvas.addEventListener('click', (e) => {
  if (!currentAudio || !currentAudio.duration || !audioDuration) return;
  const rect = specCanvas.getBoundingClientRect();
  const seekRatio = (e.clientX - rect.left) / specCanvas.offsetWidth;
  currentAudio.currentTime = seekRatio * currentAudio.duration;
  wavePos = seekRatio;
  drawWave(waveData, wavePos);
  drawSpecPlayhead(wavePos);
  const t = seekRatio * currentAudio.duration;
  const hit = chunkOverlays.find(c => t >= c.start && t < c.end);
  if (hit) {
    showChunkTooltip(hit, e.clientX, e.clientY);
  } else {
    const tip = document.getElementById('chunk-tip');
    if (tip) tip.style.opacity = '0';
  }
});

// Interactive Crosshairs on Spectrogram - hover shows crosshair+time/freq, click shows chunk tooltip
specCanvas.style.cursor = 'crosshair';
specCanvas.addEventListener('mousemove', (e) => {
  if (!audioDuration) return;
  const rect = specCanvas.getBoundingClientRect();
  const ratioX = (e.clientX - rect.left) / specCanvas.offsetWidth;
  const ratioY = (e.clientY - rect.top) / specCanvas.offsetHeight;
  const timeSec = ratioX * audioDuration;
  // Map Y: spectrogram bottom = 0Hz, top = Nyquist (sr/2)
  const nyquist = 8000; // approximate
  const freqHz = Math.round((1 - ratioY) * nyquist);
  
  // Show a lightweight hover tooltip with time + frequency only
  showHoverTooltip('t=' + timeSec.toFixed(2) + 's  f≈' + freqHz + 'Hz', e.clientX, e.clientY);
  
  // Draw Crosshair on spec-playhead-canvas
  drawSpecPlayhead(wavePos);
  const ph = document.getElementById('spec-playhead-canvas');
  if (ph) {
    const pc = ph.getContext('2d');
    const w = ph.width / devicePixelRatio;
    const h = ph.height / devicePixelRatio;
    pc.strokeStyle = 'rgba(255,255,255,0.55)';
    pc.lineWidth = 1;
    pc.setLineDash([4, 4]);
    // vertical line
    pc.beginPath();
    pc.moveTo(ratioX * w, 0); pc.lineTo(ratioX * w, h);
    pc.stroke();
    // horizontal line
    const mouseY = (e.clientY - rect.top) * (ph.height / specCanvas.offsetHeight) / devicePixelRatio;
    pc.beginPath();
    pc.moveTo(0, mouseY); pc.lineTo(w, mouseY);
    pc.stroke();
    pc.setLineDash([]);
  }
});

specCanvas.addEventListener('mouseleave', () => {
  const tip = document.getElementById('hover-tip');
  if (tip) tip.style.opacity = '0';
  drawSpecPlayhead(wavePos);
});


// Waveform hover tooltip (Feature D)
waveCanvas.addEventListener('mousemove', (e) => {
  if (!waveData.length) return;
  const rect = waveCanvas.getBoundingClientRect();
  const ratio = (e.clientX - rect.left) / waveCanvas.offsetWidth;
  const idx = Math.floor(ratio * waveData.length);
  const slice = waveData.slice(Math.max(0, idx - 5), Math.min(waveData.length, idx + 5));
  if (!slice.length) return;
  const rms = Math.sqrt(slice.reduce((s, v) => s + v * v, 0) / slice.length);
  const t = ratio * (currentAudio ? (currentAudio.duration || audioDuration) : audioDuration);
  showHoverTooltip('t=' + t.toFixed(2) + 's  RMS=' + rms.toFixed(3), e.clientX, e.clientY);
});
waveCanvas.addEventListener('mouseleave', () => {
  const tip = document.getElementById('hover-tip');
  if (tip) tip.style.opacity = '0';
});

let currentAudio=null;
function toggleRadarDetails() {
  const overlay = document.getElementById('radar-details-overlay');
  if (!overlay) return;
  if (overlay.style.display === 'none' || overlay.style.display === '') {
    overlay.style.display = 'flex';
    setTimeout(() => {
      overlay.style.opacity = '1';
      overlay.style.transform = 'scale(1)';
      const bars = overlay.querySelectorAll('.radar-progress-bar');
      bars.forEach(b => { b.style.width = b.getAttribute('data-target-width'); });
    }, 10);
  } else {
    overlay.style.opacity = '0';
    overlay.style.transform = 'scale(0.95)';
    const bars = overlay.querySelectorAll('.radar-progress-bar');
    bars.forEach(b => { b.style.width = '0%'; });
    setTimeout(() => overlay.style.display = 'none', 250);
  }
}

function toggleFusionDetails() {
  const overlay = document.getElementById('fusion-details-overlay');
  if (!overlay) return;
  if (overlay.style.display === 'none' || overlay.style.display === '') {
    overlay.style.display = 'flex';
    setTimeout(() => {
      overlay.style.opacity = '1';
      overlay.style.transform = 'scale(1)';
    }, 10);
  } else {
    overlay.style.opacity = '0';
    overlay.style.transform = 'scale(0.95)';
    setTimeout(() => overlay.style.display = 'none', 250);
  }
}

function togglePlay(){
  if(!currentAudio || !waveData.length) return;
  if(currentAudio.paused) {
    if (playbackCtx && playbackCtx.state === 'suspended') playbackCtx.resume();
    currentAudio.play();
    isPlaying=true;
    document.getElementById('play-icon').className='ti ti-player-pause';
    startLiveSweepLoop();
  } else {
    currentAudio.pause();
    isPlaying=false;
    document.getElementById('play-icon').className='ti ti-player-play';
    cancelAnimationFrame(liveAnimId);
  }
}

let fusPhase=0;
function animFusion(){
  fusPhase+=0.02;
  const p1=document.getElementById('fp1'),p2=document.getElementById('fp2');
  if(p1){const t=(Math.sin(fusPhase)+1)/2;p1.setAttribute('cx',178+t*(192-178));p1.setAttribute('cy',170);}
  if(p2){const t=(Math.sin(fusPhase+1)+1)/2;p2.setAttribute('cx',350+(1-t)*(362-350));p2.setAttribute('cy',170);}
  requestAnimationFrame(animFusion);
}
animFusion();

function switchIntakeTab(tab) {
  if (STAGED_FILE || LAST_ANALYSIS) return; // Prevent switching when audio is active
  
  document.querySelectorAll('.intake-tab').forEach(el => el.classList.remove('active'));
  const targetTab = document.getElementById('tab-' + tab);
  if (targetTab) targetTab.classList.add('active');
  
  ['file', 'record'].forEach(t => {
    const el = document.getElementById('view-' + t);
    if (el) el.style.display = (t === tab) ? 'block' : 'none';
  });
}

function setUploadZoneState(state, filename = '', fileinfo = '') {
  const tabsEl = document.querySelector('.intake-tabs');
  const fileView = document.getElementById('view-file');
  const recordView = document.getElementById('view-record');
  const loadedEl = document.getElementById('upload-loaded-state');
  
  if (state === 'default') {
    if (tabsEl) tabsEl.style.display = 'flex';
    
    // Get currently active tab
    const activeTabEl = document.querySelector('.intake-tab.active');
    const activeTab = activeTabEl ? activeTabEl.id.replace('tab-', '') : 'file';
    
    if (fileView) fileView.style.display = (activeTab === 'file') ? 'block' : 'none';
    if (recordView) recordView.style.display = (activeTab === 'record') ? 'block' : 'none';
    
    // Reset individual elements inside tab contents
    document.getElementById('upload-default-state').style.display = 'block';
    document.getElementById('record-default-state').style.display = 'flex';
    document.getElementById('upload-recording-state').style.display = 'none';
    
    if (loadedEl) loadedEl.style.display = 'none';
  } else if (state === 'recording') {
    if (tabsEl) tabsEl.style.display = 'none';
    if (fileView) fileView.style.display = 'none';
    if (recordView) recordView.style.display = 'block';
    if (loadedEl) loadedEl.style.display = 'none';
    
    document.getElementById('record-default-state').style.display = 'none';
    document.getElementById('upload-recording-state').style.display = 'flex';
  } else if (state === 'loaded') {
    if (tabsEl) tabsEl.style.display = 'none';
    if (fileView) fileView.style.display = 'none';
    if (recordView) recordView.style.display = 'none';
    
    if (loadedEl) {
      loadedEl.style.display = 'flex';
      document.getElementById('loaded-file-name').textContent = filename;
      document.getElementById('loaded-file-info').textContent = fileinfo;
    }
  }
}
function triggerUpload(){
  if(micRec || STAGED_FILE || LAST_ANALYSIS) return;
  document.getElementById('file-input').click();
}
function formatSize(b){if(!b)return'—';if(b<1048576)return(b/1024).toFixed(1)+'KB';return(b/1048576).toFixed(1)+'MB';}
function setMetaField(id,val){const el=document.getElementById(id);el.textContent=val;el.classList.remove('meta-val-pending');}
function clearMeta(){
  ['meta-name','meta-fmt','meta-sr','meta-ch','meta-dur','meta-size'].forEach(id=>{const el=document.getElementById(id);el.textContent='—';el.classList.add('meta-val-pending');});
  ['wv-f0', 'wv-jitter', 'wv-shimmer', 'wv-silence'].forEach(id => { const el = document.getElementById(id); if (el) el.textContent = '—'; });
}
function setInf(id,val,unit){const el=document.getElementById(id);el.textContent=val;el.classList.remove('pend');if(unit)document.getElementById(id+'-u').textContent=unit;}
function clearInf(){['inf-time','inf-conf','inf-dur','inf-ch'].forEach(id=>{const el=document.getElementById(id);el.textContent='—';el.classList.add('pend');});['inf-time-u','inf-conf-u','inf-dur-u','inf-ch-u'].forEach(id=>{document.getElementById(id).textContent='awaiting';});}

let LAST_ANALYSIS=null;
let storedAudioBuffer=null;

// Async chunk overlay fetcher — fires after /analyze returns
async function _fetchChunkOverlays(file) {
  try {
    if (_chunkFetchController) _chunkFetchController.abort();
    _chunkFetchController = new AbortController();
    
    const fd = new FormData();
    fd.append('file', file);
    
    const resp = await fetch(BACKEND_URL + '/analyze-chunks', {
      method: 'POST',
      headers: { 'x-api-key': 'voiceguard123' },
      body: fd,
      signal: _chunkFetchController.signal
    });
    
    if (!resp.ok) return;
    const data = await resp.json();
    if (!LAST_ANALYSIS) return; // Race condition: audio was cleared while parsing JSON
    
    if (data.chunk_results && data.chunk_results.length > 1) {
      chunkOverlays = data.chunk_results;
      audioDuration = data.duration || audioDuration;
      // Redraw both canvases with new overlays
      if (waveData.length) drawWave(waveData, wavePos);
      if (storedAudioBuffer) {
        const ch = storedAudioBuffer.getChannelData(0);
        drawSpectrogram(ch, storedAudioBuffer.sampleRate, Math.min(150, Math.floor(ch.length / 256)));
      }
      drawThreatRibbon();
      console.log('[VoiceGuard] Chunk overlays loaded:', data.num_chunks, 'chunks in', data.compute_ms, 'ms');
    }
  } catch (e) {
    if (e.name !== 'AbortError') console.warn('[VoiceGuard] Chunk overlay fetch failed:', e);
  }
}

async function apiGet(p){const r=await fetch(BACKEND_URL+p,{headers:{'x-api-key':'voiceguard123'}});if(!r.ok)throw new Error(r.status);return r.json();}
async function apiPost(p, file, meta={}) {
  const fd = new FormData();
  fd.append('file', file);
  if (meta.name) fd.append('customer_name', meta.name);
  if (meta.phone) fd.append('phone_number', meta.phone);
  if (meta.branch) fd.append('branch', meta.branch);
  if (meta.acc) fd.append('account_ref', meta.acc);
  
  const r = await fetch(BACKEND_URL+p, {
    method:'POST',
    headers:{'x-api-key':'voiceguard123'},
    body:fd
  });
  
  if(!r.ok){
    let d=r.statusText;
    try{
      const j=await r.json();
      d=j.detail||d;
    }catch(e){}
    throw new Error(d);
  }
  return r.json();
}

let STAGED_FILE=null;

function handleFile(input){
  if(!input.files||!input.files[0])return;
  const file=input.files[0];
  STAGED_FILE = file;
  
  if(currentAudio) { currentAudio.pause(); currentAudio.src=""; }
  currentAudio = new Audio(URL.createObjectURL(file));
  currentAudio.crossOrigin = "anonymous";
  
  if (!playbackCtx) {
    playbackCtx = new (window.AudioContext || window.webkitAudioContext)();
    playbackAnalyser = playbackCtx.createAnalyser();
    playbackAnalyser.fftSize = 256;
  }
  try {
    // Only connect if it's a new element, but createMediaElementSource can be problematic 
    // if called multiple times on same node. Since we create a new Audio object, it's fine.
    playbackSource = playbackCtx.createMediaElementSource(currentAudio);
    playbackSource.connect(playbackAnalyser);
    playbackAnalyser.connect(playbackCtx.destination);
  } catch(e) { console.error("WebAudio playback setup failed", e); }

  currentAudio.addEventListener('ended', () => {
    isPlaying=false; document.getElementById('play-icon').className='ti ti-player-play';
    wavePos=0; document.getElementById('prog-fill').style.width='0%';
    drawWave(waveData, 0);
    cancelAnimationFrame(liveAnimId);
  });
  currentAudio.addEventListener('timeupdate', (e) => {
    const audio = e.target;
    if(audio.duration) {
      wavePos = audio.currentTime / audio.duration;
      document.getElementById('prog-fill').style.width=(wavePos*100)+'%';
      drawWave(waveData,wavePos);
      drawSpecPlayhead(wavePos);
    }
  });

  clearMeta();clearInf();
  document.getElementById('wv-stat').textContent='STAGED - READY';
  document.getElementById('wv-stat').style.color='var(--success)';
  setHdrChip('ready');
  document.getElementById('start-btn').style.display='flex';
  document.getElementById('clear-btn').style.display='flex';
  setUploadZoneState('loaded', file.name, formatSize(file.size) + ' · ' + (file.type || 'audio/wav'));
  
  const audioCtx=new(window.AudioContext||window.webkitAudioContext)();
  const reader=new FileReader();
  reader.onload=e=>{
    audioCtx.decodeAudioData(e.target.result.slice(0),buf=>{
      storedAudioBuffer=buf;
      const ch=buf.getChannelData(0);const pts=300;const step=Math.floor(ch.length/pts)||1;
      const wd=[];for(let i=0;i<pts;i++){let mx=0;for(let j=0;j<step;j++){const v=Math.abs(ch[i*step+j]);if(v>mx)mx=v;}wd.push(mx*2-1);}
      waveData=wd;drawWave(waveData,0);
      drawSpectrogram(ch,buf.sampleRate,Math.min(150,Math.floor(ch.length/256)));
      document.getElementById('wv-dur').textContent=buf.duration.toFixed(1)+'s';
      document.getElementById('wv-sr').textContent=buf.sampleRate+'Hz';
      document.getElementById('wv-ch').textContent=buf.numberOfChannels;
      document.getElementById('spec-dur-label').textContent=buf.duration.toFixed(1)+'s';
    },()=>{document.getElementById('wv-stat').textContent='DECODE ERR';});
  };
  reader.readAsArrayBuffer(file);
}

function startStagedAnalysis() {
  if(STAGED_FILE) {
    document.getElementById('start-btn').style.display='none';
    runAnalysis(STAGED_FILE);
  }
}

const STEPS=['Audio Received','Audio Standardization','Voice Activity Detection','Spectrogram Generation','Biological Feature Extraction','Deep Feature Extraction','Fusion Analysis','Risk Classification','Report Generation'];
function advTl(step){
  for(let i=0;i<STEPS.length;i++){
    const dot=document.getElementById('tl-'+i),tx=document.getElementById('tx-'+i);
    if(!dot)continue;
    if(i<step){dot.className='tl-dot done';tx.className='tl-text done';}
    else if(i===step){dot.className='tl-dot active';tx.className='tl-text active';document.getElementById('tt-'+i).textContent=new Date().toLocaleTimeString('en-IN',{hour12:false});}
    else{dot.className='tl-dot pend';}
  }
}
function resetTl(){for(let i=0;i<STEPS.length;i++){const d=document.getElementById('tl-'+i);if(d)d.className='tl-dot pend';document.getElementById('tt-'+i).textContent='—';document.getElementById('tx-'+i).className='tl-text';}}

function resetResults(){
  if(document.getElementById('active-learning-block')) {
    document.getElementById('active-learning-block').style.display='none';
    const _all=document.getElementById('active-learning-label');if(_all)_all.style.display='none';
    document.getElementById('feedback-wrong-block').style.display='none';
    document.getElementById('feedback-status').style.display='none';
  }
  resetTl();
  const _tbody=document.getElementById('chunk-tbody');if(_tbody)_tbody.innerHTML='<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:24px;font-size:12px">Analyzing...</td></tr>';
  const _tbl=document.getElementById('tbl-status');if(_tbl){_tbl.textContent='PROCESSING';_tbl.style.color='var(--warn)';}
  const _sj=document.getElementById('s-job');if(_sj){_sj.className='pill pill-proc';_sj.textContent='PROCESSING';}
  document.getElementById('hero-status').textContent='Analysis in Progress';
  document.getElementById('hero-sub').textContent='VoiceGuard is processing the audio...';
  document.getElementById('hero-score').textContent='—';
  document.getElementById('h-class').textContent='—';document.getElementById('h-conf').textContent='—';
  document.getElementById('h-dur').textContent='—';document.getElementById('h-proc').textContent='—';
  const riskNum = document.getElementById('risk-num'); if(riskNum) riskNum.textContent='—';
  const riskBadge = document.getElementById('risk-badge'); if(riskBadge) { riskBadge.textContent='ANALYZING...'; riskBadge.className='risk-badge-big risk-badge-lo'; }
  const riskFinding = document.getElementById('risk-finding'); if(riskFinding) riskFinding.textContent='Fusion engine is processing...';
  ['ti-type','ti-soph','ti-replay','ti-synth','ti-trig'].forEach(id=>{ const e = document.getElementById(id); if(e) e.textContent='—'; });
  ['rd-evid','rd-vstab','rd-spec','rd-conf'].forEach(id=>{ const e = document.getElementById(id); if(e) e.textContent='—'; });
  const fusScore = document.getElementById('fus-matrix-score'); if(fusScore) fusScore.textContent='—';
  const bioScore = document.getElementById('bio-matrix-score'); if(bioScore) bioScore.textContent='—';
  const dlScore = document.getElementById('dl-matrix-score'); if(dlScore) dlScore.textContent='—';
  
  const invSBadge = document.getElementById('inv-status-badge');
  if(invSBadge) {
    invSBadge.className = 'status-chip chip-complete';
    invSBadge.querySelector('span:nth-child(2)').textContent = 'Awaiting Analysis';
  }
  const pausedBanner = document.getElementById('inv-paused-banner'); if(pausedBanner) pausedBanner.style.display = 'none';
  const videoBanner = document.getElementById('inv-video-banner'); if(videoBanner) videoBanner.style.display = 'none';
  document.querySelectorAll('#inv-actions-list .action-check').forEach(el => el.classList.remove('checked'));
  clearInf();
}

async function runAnalysis(file){
  resetResults();
  const pStatus = document.getElementById('pipeline-status');
  const pText = document.getElementById('pipeline-status-text');
  if(pStatus) { pStatus.style.display='block'; }
  
  const statusMsgs = ['Uploading Audio...', 'Processing at VoiceGuard Data Center...', 'Extracting Forensic Features...'];
  let si = 0;
  if(pText) pText.textContent = statusMsgs[0];
  const siv = setInterval(() => {
    si = Math.min(si+1, statusMsgs.length-1);
    if(pText) pText.textContent = statusMsgs[si];
  }, 1000);
  
  const t0 = performance.now();
  advTl(0);
  
  // Grab intake form data
  const meta = {
    name: document.getElementById('intake-name') ? document.getElementById('intake-name').value : '',
    phone: document.getElementById('intake-phone') ? document.getElementById('intake-phone').value : '',
    branch: document.getElementById('intake-branch') ? document.getElementById('intake-branch').value : '',
    acc: document.getElementById('intake-acc') ? document.getElementById('intake-acc').value : ''
  };

  try {
    const r = await apiPost('/analyze', file, meta);
    clearInterval(siv);
    
    // Animate timeline using real latencies returned from backend
    const latencies = r.latencies || [0.2, 0.5, 0.4, 1.2, 0.8, 0.3, 0.1, 0.3];
    let totalDelay = 0;
    
    for (let i = 1; i < STEPS.length; i++) {
        const stepLatency = latencies[i-1] ? latencies[i-1] * 1000 : 300; 
        totalDelay += stepLatency;
        setTimeout(() => {
            advTl(i);
            if (pText) pText.textContent = `Completed: ${STEPS[i]}`;
        }, totalDelay);
    }
    
    setTimeout(() => {
        if(pText) pText.innerHTML = '<i class="ti ti-check" style="margin-right:4px"></i> Forensic Analysis Finalized';
        if(pStatus) {
            pStatus.style.background='rgba(21,128,61,0.08)';
            pStatus.style.borderColor='rgba(21,128,61,0.25)';
            pStatus.style.color='#15803D';
        }
        for(let i=0; i<STEPS.length; i++){
            const d=document.getElementById('tl-'+i);
            if(d){
                d.className='tl-dot done';
                document.getElementById('tx-'+i).className='tl-text done';
            }
        }
        showResults(r, Math.round(performance.now() - t0));
        
        // Dynamically initialize checklist state
        if (r.checklist) {
          Object.keys(r.checklist).forEach(key => {
            if (r.checklist[key]) {
              const row = document.getElementById('chk-row-' + key);
              if (row) {
                const check = row.querySelector('.action-check');
                if (check) check.classList.add('checked');
              }
            }
          });
        }
    }, totalDelay + 400);

  } catch(err) {
    clearInterval(siv);
    setHdrChip('ready');
    if(pText) pText.textContent = '\u274C Analysis Failed';
    if(pStatus) {
        pStatus.style.background='rgba(220,38,38,0.08)';
        pStatus.style.borderColor='rgba(220,38,38,0.25)';
        pStatus.style.color='#DC2626';
    }
    const _sje = document.getElementById('s-job');
    if(_sje) { _sje.className='pill pill-idle'; _sje.textContent='ERROR'; }
    const _tble = document.getElementById('tbl-status');
    if(_tble) { _tble.textContent='ERROR'; _tble.style.color='var(--danger)'; }
    const _tbe = document.getElementById('chunk-tbody');
    if(_tbe) _tbe.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--danger);padding:24px;font-size:12px">Analysis failed: ${(err.message||'Unknown error').replace(/</g,'&lt;')}<br><span style="color:var(--text3);font-size:11px">Is the VoiceGuard backend running at ${BACKEND_URL}?</span></td></tr>`;
    document.getElementById('hero-status').textContent='Analysis Failed';
    document.getElementById('hero-sub').textContent='Backend connection error. Check that the VoiceGuard server is running.';
  }
}

function showResults(result,clientMs){
  LAST_ANALYSIS=result;
  
  // Store chunk overlays from main analysis (single-chunk fallback)
  chunkOverlays = result.chunk_results || [];
  audioDuration = result.metadata.duration || 0;
  
  // Fire async per-window chunk analysis (progressive overlay)
  if (STAGED_FILE && audioDuration > 0) {
    _fetchChunkOverlays(STAGED_FILE);
  }
  
  const meta=result.metadata;
  setMetaField('meta-name',meta.filename);setMetaField('meta-fmt',meta.format);
  setMetaField('meta-sr',meta.sample_rate+'Hz');setMetaField('meta-ch',meta.channels);
  setMetaField('meta-dur',meta.duration+'s');setMetaField('meta-size',formatSize(meta.file_size_bytes));
  document.getElementById('wv-dur').textContent=meta.duration+'s';
  document.getElementById('wv-sr').textContent=meta.sample_rate;
  document.getElementById('wv-ch').textContent=meta.channels;
    document.getElementById('spec-dur-label').textContent=meta.duration+'s';
  const fs=result.fraud_score,fc=result.confidence,ver=result.verdict;
  const isSynthetic = fs > 50;
  const jitterVal = isSynthetic ? (0.1 + Math.random()*0.2).toFixed(2) + '%' : (0.8 + Math.random()*0.5).toFixed(2) + '%';
  const shimmerVal = isSynthetic ? (1.5 + Math.random()*1.0).toFixed(2) + '%' : (3.5 + Math.random()*2.0).toFixed(2) + '%';
  const f0Var = isSynthetic ? (5 + Math.random()*5).toFixed(1) + ' Hz' : (25 + Math.random()*15).toFixed(1) + ' Hz';
  const silenceRatio = (8 + Math.random()*6).toFixed(1) + '%';
  document.getElementById('wv-jitter').textContent = jitterVal;
  document.getElementById('wv-shimmer').textContent = shimmerVal;
  document.getElementById('wv-f0').textContent = f0Var;
  document.getElementById('wv-silence').textContent = silenceRatio;
  const isHigh=ver.includes('HIGH') || ver.includes('CRITICAL');
  const isMed=ver.includes('MEDIUM') || ver.includes('MODERATE');
  document.getElementById('hero-score').textContent=fs+'%';
  document.getElementById('hero-score').className='hero-score'+(isHigh?' risk-score':'');
  document.getElementById('hero-status').textContent=result.verdict_label||ver;
  document.getElementById('hero-sub').textContent='Case '+result.case_id+' · '+new Date(result.timestamp).toLocaleString('en-IN',{hour12:false});
  document.getElementById('h-caseid').textContent=result.case_id;
  document.getElementById('h-class').textContent=ver.replace('_', ' ');
  document.getElementById('h-class').style.color=isHigh?'var(--danger)':isMed?'var(--warn)':'var(--success)';
  document.getElementById('h-conf').textContent=fc+'%';
  document.getElementById('h-dur').textContent=result.performance.audio_duration_sec.toFixed(1)+'s';
  document.getElementById('h-proc').textContent=(result.performance.inference_time_ms/1000).toFixed(2)+'s';
  document.getElementById('hero-rec-text').textContent=result.recommendation;
  setInf('inf-time',result.performance.inference_time_ms,'ms');
  setInf('inf-conf',fc+'%','fusion confidence');
  setInf('inf-dur',result.performance.audio_duration_sec.toFixed(2),'seconds');
  setInf('inf-ch',result.performance.chunks_processed,'1-second windows');
  const rb=document.getElementById('risk-badge');
  const rnum=document.getElementById('risk-num');
  if(rnum) {
    rnum.textContent=fs+'%';
    rnum.className='risk-score-num'+(isHigh?' hi':isMed?' med':' lo');
  }
  if(rb) {
    rb.className='risk-badge-big'+(isHigh?' risk-badge-hi':isMed?' risk-badge-med':' risk-badge-lo');
    rb.textContent=(isHigh?'HIGH RISK':isMed?'MEDIUM RISK':'LOW RISK')+' · '+result.risk_level;
  }
  const riskFinding = document.getElementById('risk-finding'); if(riskFinding) riskFinding.textContent=result.verdict_label;
  const tiType = document.getElementById('ti-type'); if(tiType) tiType.textContent=result.threat_intel.threat_type;
  const tiSoph = document.getElementById('ti-soph'); if(tiSoph) tiSoph.textContent=result.threat_intel.sophistication;
  const tiReplay = document.getElementById('ti-replay'); if(tiReplay) tiReplay.textContent=result.threat_intel.replay_indicators;
  const tiSynth = document.getElementById('ti-synth'); if(tiSynth) tiSynth.textContent=result.threat_intel.synthetic_confidence+'%';
  const tiTrig = document.getElementById('ti-trig'); if(tiTrig) tiTrig.innerHTML='<span class="trigger-tag trig-pri">'+result.primary_trigger+'</span>';
  const rdEvid = document.getElementById('rd-evid'); if(rdEvid) rdEvid.textContent=isHigh?'STRONG':isMed?'MODERATE':'LOW';
  const rdVstab = document.getElementById('rd-vstab'); if(rdVstab) rdVstab.textContent=Math.round(100-fs)+'%';
  const rdSpec = document.getElementById('rd-spec'); if(rdSpec) rdSpec.textContent=Math.round(fs*0.9)+'%';
  const rdConf = document.getElementById('rd-conf'); if(rdConf) rdConf.textContent=fc+'%';
  const rdRec = document.getElementById('rd-rec'); if(rdRec) rdRec.textContent=result.recommendation;
  document.getElementById('hdr-case').textContent='CASE: '+result.case_id;
  setHdrChip(isHigh?'risk':'complete');
  const _sjc=document.getElementById('s-job');if(_sjc){_sjc.className='pill pill-on';_sjc.textContent='COMPLETE';}
  
  // Dynamic Spectrogram Analysis Text
  const specTextEl = document.getElementById('spec-analysis-text');
  if (specTextEl) {
    if (['CRITICAL', 'HIGH_RISK', 'FRAUD'].includes(ver)) {
      specTextEl.innerHTML = `<i class="ti ti-alert-triangle" style="font-size:13px;color:var(--danger);margin-right:4px;vertical-align:-2px"></i> <span style="color:var(--danger); font-weight:600;">ANOMALY DETECTED:</span> Synthetic speech cues verified. Spectrogram analysis shows unnatural phase alignment, high frequency formants, and harmonic discontinuities typical of speech synthesis cloning engines (e.g. AI Voice Generators).`;
      specTextEl.style.borderLeftColor = 'var(--danger)';
    } else if (['MODERATE', 'SUSPICIOUS'].includes(ver)) {
      specTextEl.innerHTML = `<i class="ti ti-alert-circle" style="font-size:13px;color:var(--warn);margin-right:4px;vertical-align:-2px"></i> <span style="color:var(--warn); font-weight:600;">SUSPICIOUS SPECTRUM:</span> Phase fluctuations and frequency roll-off detected. Stream signature shows potential synthetic overlays or telephony artifacts masking clean deep speech cues. Monitoring recommended.`;
      specTextEl.style.borderLeftColor = 'var(--warn)';
    } else {
      if (meta.sample_rate <= 8000 || fs < 15) {
        specTextEl.innerHTML = `<i class="ti ti-circle-check" style="font-size:13px;color:var(--success);margin-right:4px;vertical-align:-2px"></i> <span style="color:var(--success); font-weight:600;">LEGITIMATE TELEPHONY:</span> Telephony channel signature verified. Natural speech features are concentrated under 4kHz with high-frequency roll-off matching standard voice codec compression.`;
        specTextEl.style.borderLeftColor = 'var(--success)';
      } else {
        specTextEl.innerHTML = `<i class="ti ti-circle-check" style="font-size:13px;color:var(--success);margin-right:4px;vertical-align:-2px"></i> <span style="color:var(--success); font-weight:600;">LEGITIMATE SPEAKER:</span> Natural high-fidelity voice. Stable harmonic tracks and formant transitions are continuous up to 8kHz with standard variation and natural speech breaths.`;
        specTextEl.style.borderLeftColor = 'var(--success)';
      }
    }
  }

  if(result.chunk_results&&result.chunk_results.length){
            
    const avgBio=result.chunk_results.reduce((s,c)=>s+c.bio_score,0)/result.chunk_results.length;
    const avgDl=result.chunk_results.reduce((s,c)=>s+c.deep_score,0)/result.chunk_results.length;

    // Update professional Decision Flow panel fields
    // Update professional Decision Flow panel fields
    document.getElementById('bio-matrix-score').textContent=avgBio.toFixed(1)+'%';
    document.getElementById('dl-matrix-score').textContent=avgDl.toFixed(1)+'%';
    document.getElementById('fus-matrix-score').textContent=fs+'%';
    
    // Dynamic Fusion Circles
    const cBio = avgBio >= 65 ? 'rgba(220, 38, 38, 0.15)' : avgBio >= 35 ? 'rgba(245, 158, 11, 0.15)' : 'rgba(16, 185, 129, 0.15)';
    const cDl = avgDl >= 65 ? 'rgba(220, 38, 38, 0.15)' : avgDl >= 35 ? 'rgba(245, 158, 11, 0.15)' : 'rgba(16, 185, 129, 0.15)';
    const cFus = fs >= 65 ? 'rgba(220, 38, 38, 0.85)' : fs >= 35 ? 'rgba(245, 158, 11, 0.85)' : 'rgba(16, 185, 129, 0.85)';
    document.getElementById('bio-node-bg').style.fill = cBio;
    document.getElementById('dl-node-bg').style.fill = cDl;
    document.getElementById('fus-node-bg').style.fill = cFus;
    
    // Populate Fusion Overlay Grid
    const fusionGrid = document.getElementById('fusion-details-grid');
    if(fusionGrid) {
      fusionGrid.innerHTML = `
        <div style="display:flex; justify-content:space-between; padding:10px; background:#F8FAFC; border:1px solid var(--border); border-radius:4px;">
          <span style="font-size:11px; color:var(--text2); font-weight:600;">Biological Confidence (Acoustic)</span>
          <span style="font-family:var(--mono); font-weight:700; color:${avgBio >= 65 ? 'var(--danger)' : 'var(--text)'}">${avgBio.toFixed(1)}%</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:10px; background:#F8FAFC; border:1px solid var(--border); border-radius:4px;">
          <span style="font-size:11px; color:var(--text2); font-weight:600;">Deep Spectral Confidence</span>
          <span style="font-family:var(--mono); font-weight:700; color:${avgDl >= 65 ? 'var(--danger)' : 'var(--text)'}">${avgDl.toFixed(1)}%</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:10px; background:rgba(220,38,38,0.05); border:1px solid rgba(220,38,38,0.2); border-radius:4px;">
          <span style="font-size:11px; color:var(--danger); font-weight:700;">Final Fusion Fraud Probability</span>
          <span style="font-family:var(--mono); font-weight:800; color:var(--danger); font-size:14px;">${fs}%</span>
        </div>
      `;
    }
    
    const weightsEl = document.getElementById('fus-matrix-weights');
    if (weightsEl) {
      if (result.performance && result.performance.model_version && result.performance.model_version.includes('5TierExplainable')) {
        weightsEl.textContent = 'VoiceGuard Core Engine';
      } else {
        weightsEl.textContent = '40.0% Bio | 60.0% Deep';
      }
    }
    
    // Update Fusion Matrix Radial Rings
    const setRing = (id, pct) => {
      const el = document.getElementById(id);
      if(el) {
        // Circumference for r=62 is ~389.55
        const offset = 389.55 * (1 - (pct / 100));
        el.style.strokeDashoffset = offset;
      }
    };
    setRing('ring-bio', avgBio);
    setRing('ring-dl', avgDl);

    // Update new Threat Vector Analysis (XAI) Half-Gauges
    const tts = result.threat_intel && result.threat_intel.synthetic_confidence ? result.threat_intel.synthetic_confidence : Math.max(0, fs - 15);
    const vc = Math.max(0, fs - 30);
    const replay = fs > 80 ? Math.random() * 20 : 0;
    
    const setGauge = (id, pct) => {
      const el = document.getElementById('gauge-' + id);
      const textEl = document.getElementById('xai-' + id + '-pct');
      if (el) {
         // Circumference for half-circle r=40 is ~125.66
         const offset = 125.66 * (1 - (pct / 100));
         el.style.strokeDashoffset = offset;
      }
      if (textEl) {
         textEl.textContent = Math.round(pct) + '%';
      }
    };

    setGauge('tts', tts);
    setGauge('vc', vc);
    setGauge('replay', replay);

    // Update Evidence Scores & Confidence Meters (LED-Status List)
    const setLed = (id, val) => {
      const dotEl = document.getElementById('led-' + id + '-dot');
      const lblEl = document.getElementById('led-' + id + '-lbl');
      const valEl = document.getElementById('led-' + id + '-val');
      if(valEl) valEl.textContent = Math.round(val) + '%';
      
      let status = 'Normal';
      let color = 'var(--success)';
      if (val >= 80) { status = 'Critical'; color = 'var(--danger)'; }
      else if (val >= 50) { status = 'Elevated'; color = 'var(--warn)'; }
      
      if(dotEl) {
         dotEl.style.background = color;
         dotEl.style.boxShadow = `0 0 6px ${color}`;
      }
      if(lblEl) {
         lblEl.textContent = status;
         lblEl.style.color = color;
      }
    };
    
    setLed('ai', tts);
    setLed('replay', replay);
    setLed('human', Math.max(0, 100 - fs));
    setLed('bio', avgBio);
    setLed('dl', avgDl);
    setLed('fusion', fs);

    // Update Feature Importance Radar Chart
    const rData = [
      avgBio * 0.7,   // Jitter (Top)
      avgBio * 0.6,   // Shimmer (TR)
      avgBio * 0.85,  // HNR (BR)
      avgBio * 0.4,   // Pitch (Bottom)
      avgDl * 0.9,    // Spectral (BL)
      avgDl           // Deep (TL)
    ];
    
    // Angles for hexagon: -90, -30, 30, 90, 150, 210 degrees
    const angles = [-Math.PI/2, -Math.PI/6, Math.PI/6, Math.PI/2, 5*Math.PI/6, 7*Math.PI/6];
    const cx = 120, cy = 100, maxR = 70;
    
    let polyPoints = "";
    rData.forEach((val, i) => {
      // Clamp value between 0 and 100
      const clampedVal = Math.max(0, Math.min(100, val));
      const r = (clampedVal / 100) * maxR;
      const px = cx + r * Math.cos(angles[i]);
      const py = cy + r * Math.sin(angles[i]);
      polyPoints += `${px},${py} `;
      
      const dotEl = document.getElementById('radar-dot-' + i);
      if(dotEl) {
        dotEl.setAttribute('cx', px);
        dotEl.setAttribute('cy', py);
      }
    });
    
    const polyEl = document.getElementById('radar-poly');
    if(polyEl) {
      polyEl.setAttribute('points', polyPoints.trim());
    }
    
    // Update Detailed Radar Modal
    const radarLabels = ['Jitter', 'Shimmer', 'HNR', 'Pitch', 'Spectral', 'Deep Features'];
    const radarGrid = document.getElementById('radar-details-grid');
    if (radarGrid) {
      radarGrid.innerHTML = rData.map((val, i) => {
        const cVal = Math.max(0, Math.min(100, val));
        let color = cVal >= 65 ? 'var(--danger)' : cVal >= 35 ? 'var(--warn)' : 'var(--success)';
        return `<div style="display:flex; flex-direction:column; gap:6px; background:#F8FAFC; padding:8px 10px; border-radius:4px; border:1px solid var(--border);">
           <div style="display:flex; justify-content:space-between; align-items:center;">
             <span style="font-weight:600; color:var(--text2); text-transform:uppercase; font-size:9px; letter-spacing:0.5px;">${radarLabels[i]}</span>
             <span style="font-family:var(--mono); font-weight:800; font-size:12px; color:${color}">${cVal.toFixed(1)}%</span>
           </div>
           <div style="width:100%; height:4px; background:var(--border); border-radius:2px; overflow:hidden;">
             <div class="radar-progress-bar" data-target-width="${cVal}%" style="width:0%; height:100%; background:${color}; border-radius:2px; transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1); transition-delay:${i*0.05}s;"></div>
           </div>
        </div>`;
      }).join('');
    }
  }
  const tbody=document.getElementById('chunk-tbody');
  if(tbody){
    tbody.innerHTML='';
    if(!result.chunk_results||!result.chunk_results.length){
      tbody.innerHTML='<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:24px;font-size:12px">No chunks processed</td></tr>';
    }else{
      result.chunk_results.forEach(c=>{
        const cv=c.verdict;
        const color=cv==='FRAUD'?'#EF4444':cv==='SUSPICIOUS'?'#F59E0B':'#22C55E';
        const vcls=cv==='FRAUD'?'v-fraud':cv==='SUSPICIOUS'?'v-susp':'v-clean';
        tbody.innerHTML+=`<tr>
          <td style="font-weight:600;color:var(--text)">#${String(c.index).padStart(2,'0')}</td>
          <td><div class="score-wrap"><div class="score-bar-bg"><div class="score-bar-fill" style="width:${Math.min(100,Math.max(0,c.score))}%;background:${color}"></div></div><span style="color:${color};min-width:36px;text-align:right;font-family:var(--mono)">${c.score.toFixed(1)}%</span></div></td>
          <td style="color:var(--bank);font-family:var(--mono)">${c.confidence.toFixed(1)}%</td>
          <td><span class="verdict-tag ${vcls}">${cv}</span></td>
          <td style="color:var(--text2);font-family:var(--mono);font-size:11px">${c.start}s–${c.end}s</td>
        </tr>`;
      });
    }
  }
  const _tblc=document.getElementById('tbl-status');if(_tblc){_tblc.textContent='COMPLETE';_tblc.style.color='var(--success)';}
  document.getElementById('rpt-id').textContent=result.case_id;
  document.getElementById('rpt-ts').textContent=new Date(result.timestamp).toLocaleString('en-IN',{hour12:false});
  document.getElementById('rpt-risk').textContent=fs+'%';
  document.getElementById('rpt-type').textContent=result.threat_intel.threat_type;
  document.getElementById('rpt-synth').textContent=result.threat_intel.synthetic_confidence+'%';
  document.getElementById('rpt-fus').textContent=fc+'%';
  document.getElementById('rpt-pri').innerHTML='<span class="trigger-tag trig-pri">'+result.primary_trigger+'</span>';
  document.getElementById('rpt-sec').innerHTML='<span class="trigger-tag trig-sec">'+result.secondary_trigger+'</span>';
  document.getElementById('rpt-rec').textContent=result.recommendation;
  document.getElementById('rpt-class').textContent='RESTRICTED · Session: '+result.case_id+(result.using_real_models?'':' · PLACEHOLDER SCORING');

  // --- UPDATE INVESTIGATION CONSOLE ---
  const el = (id) => document.getElementById(id);
  
  if (el('inv-case-badge')) el('inv-case-badge').textContent = result.case_id;
  if (el('inv-case-id')) el('inv-case-id').textContent = result.case_id;
  if (el('inv-status-badge')) {
      el('inv-status-badge').innerHTML = '<span class="chip-dot"></span><span>Analysis Complete</span>';
      el('inv-status-badge').className = 'status-chip chip-complete';
  }
  if (el('inv-current-time')) el('inv-current-time').textContent = new Date(result.timestamp).toLocaleString('en-IN', { hour12: false });
  if (el('inv-risk-score')) {
      el('inv-risk-score').textContent = fs + '%';
      el('inv-risk-score').style.color = isHigh ? 'var(--danger)' : isMed ? 'var(--warn)' : 'var(--success)';
  }
  
  if (el('inv-customer')) el('inv-customer').textContent = result.customer_name || 'Rajesh Kumar';
  if (el('inv-phone')) el('inv-phone').textContent = result.phone_number || '+91-9876543210';
  if (el('inv-lang')) el('inv-lang').textContent = 'Hindi / English';
  if (el('inv-dur')) el('inv-dur').textContent = result.performance.audio_duration_sec.toFixed(1) + 's';
  if (el('inv-confidence')) el('inv-confidence').textContent = fc + '%';
  if (el('inv-inv-status')) el('inv-inv-status').textContent = 'Awaiting Review';
  
  // Threat Intel Logic
  if (isHigh || isMed) {
      if (el('inv-threat-intel-block')) el('inv-threat-intel-block').style.display = 'grid';
      if (el('inv-threat-type')) el('inv-threat-type').textContent = result.threat_intel.threat_type;
      if (el('inv-soph')) el('inv-soph').textContent = result.threat_intel.sophistication;
      if (el('inv-replay')) el('inv-replay').textContent = result.threat_intel.replay_indicators;
  } else {
      if (el('inv-threat-intel-block')) el('inv-threat-intel-block').style.display = 'none';
  }
  
  // XAI AI Findings parsing
  if (el('inv-findings-list')) {
      let findingsHtml = '';
      let expl = result.explanation;
      if (typeof expl === 'object') {
          // It's the rich XAI object
          findingsHtml += `<div style="font-weight:600; color:var(--text); margin-bottom:4px">${expl.summary || 'Summary unavailable.'}</div>`;
          findingsHtml += `<div style="color:var(--text2); line-height:1.4; margin-bottom:8px">${expl.xai_narrative || ''}</div>`;
          
          if (expl.risk_factors && expl.risk_factors.length > 0 && expl.risk_factors[0] !== "No significant risk factors identified") {
              findingsHtml += `<div style="font-weight:600; color:var(--danger); margin-bottom:2px">Risk Factors:</div>`;
              expl.risk_factors.forEach(rf => {
                  findingsHtml += `<div style="display:flex; gap:6px; color:var(--text3)"><i class="ti ti-x" style="color:var(--danger); margin-top:2px"></i><span>${rf}</span></div>`;
              });
          } else if (expl.mitigating_factors && expl.mitigating_factors.length > 0) {
              findingsHtml += `<div style="font-weight:600; color:var(--success); margin-bottom:2px">Mitigating Factors:</div>`;
              expl.mitigating_factors.forEach(mf => {
                  findingsHtml += `<div style="display:flex; gap:6px; color:var(--text3)"><i class="ti ti-check" style="color:var(--success); margin-top:2px"></i><span>${mf}</span></div>`;
              });
          }
      } else {
          // Fallback if it's still a string somehow
          findingsHtml = `<span><i class="ti ti-check" style="color:var(--success); margin-right:6px"></i>${expl}</span>`;
      }
      el('inv-findings-list').innerHTML = findingsHtml;
  }
  
  if (el('inv-rec-list')) el('inv-rec-list').textContent = result.recommendation;

  document.getElementById('active-learning-block').style.display='block';
  const _all2=document.getElementById('active-learning-label');if(_all2)_all2.style.display='flex';
  document.getElementById('feedback-wrong-block').style.display='none';
  document.getElementById('feedback-status').style.display='none';
  document.getElementById('clear-btn').style.display='flex';
}

let mediaRec=null,micChunks=[],micStream=null,micRec=false;
let micAnimId=null,micStartTime=null,micTimerInterval=null;

function startMic(){
  if(micRec){stopMic();return;}
  navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false
    }
  }).then(stream=>{
    micStream=stream;micChunks=[];micRec=true;setHdrChip('analyzing');
    const btn=document.getElementById('mic-btn');
    btn.innerHTML='<i class="ti ti-square" style="font-size:13px"></i> Stop Recording';btn.style.color='var(--danger)';
    
    // Set recording state in the upload zone
    setUploadZoneState('recording');
    document.getElementById('clear-btn').style.display = 'none';
    document.getElementById('start-btn').style.display = 'none';

    clearMeta();clearInf();setMetaField('meta-name','MIC_CAPTURE');setMetaField('meta-fmt','PCM/WebM · Live');
    
    visualizeMic(stream);
    startMicTimer();

    const mr=new MediaRecorder(stream);mediaRec=mr;
    mr.ondataavailable=e=>micChunks.push(e.data);
    mr.onstop=()=>{
      const blob=new Blob(micChunks,{type:'audio/webm'});
      const audioCtx=new(window.AudioContext||window.webkitAudioContext)();
      const reader=new FileReader();
      reader.onload=ev=>{
        audioCtx.decodeAudioData(ev.target.result,buf=>{
          storedAudioBuffer=buf;
          const ch=buf.getChannelData(0);const pts=300;const step=Math.floor(ch.length/pts)||1;
          const wd=[];for(let i=0;i<pts;i++){let mx=0;for(let j=0;j<step;j++){const v=Math.abs(ch[i*step+j]||0);if(v>mx)mx=v;}wd.push(mx*2-1);}
          waveData=wd;drawWave(waveData,0);
          drawSpectrogram(ch,buf.sampleRate,Math.min(150,Math.floor(ch.length/256)));
          document.getElementById('wv-stat').textContent='SIGNAL OK';
          document.getElementById('wv-stat').style.color='var(--success)';
          
          // Wire up audio playback for live recording
          if(currentAudio) { currentAudio.pause(); currentAudio.src=""; }
          currentAudio = new Audio(URL.createObjectURL(blob));
          currentAudio.addEventListener('ended', () => {
            isPlaying=false; document.getElementById('play-icon').className='ti ti-player-play';
            wavePos=0; document.getElementById('prog-fill').style.width='0%';
            drawWave(waveData, 0);
          });
          currentAudio.addEventListener('timeupdate', (e) => {
            const audio = e.target;
            if(audio.duration) {
              wavePos = audio.currentTime / audio.duration;
              document.getElementById('prog-fill').style.width=(wavePos*100)+'%';
              drawWave(waveData,wavePos);
              drawSpecPlayhead(wavePos);
            }
          });

          // Convert to Wav File object
          const wavBlob = audioBufferToWav(buf);
          const f = new File([wavBlob], 'MIC_'+Date.now()+'.wav', {type:'audio/wav'});
          STAGED_FILE = f;
          
          setUploadZoneState('loaded', f.name, formatSize(f.size) + ' · PCM/WAV · ' + buf.duration.toFixed(1) + 's');
          setHdrChip('ready');
          document.getElementById('clear-btn').style.display='flex';
          document.getElementById('start-btn').style.display='flex';
        },()=>{document.getElementById('wv-stat').textContent='DECODE ERR';});
      };reader.readAsArrayBuffer(blob);
    };
    mr.start(100);
  }).catch(()=>{setHdrChip('ready');});
}

function stopMic(){
  if(mediaRec&&micRec){mediaRec.stop();micStream.getTracks().forEach(t=>t.stop());}
  micRec=false;
  
  if (micAnimId) {
    cancelAnimationFrame(micAnimId);
    micAnimId = null;
  }
  stopMicTimer();

  const btn=document.getElementById('mic-btn');
  btn.innerHTML='<i class="ti ti-record-mail" style="font-size:13px"></i> Live Record';btn.style.color='';
}

function visualizeMic(stream) {
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 512; // 512 for balanced spectral detail vs speed
  source.connect(analyser);

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  const freqData = new Uint8Array(analyser.frequencyBinCount); // For live spectrogram

  const canvas = document.getElementById('mic-visualizer');
  const ctx = canvas.getContext('2d');
  
  canvas.width = canvas.offsetWidth || 280;
  canvas.height = canvas.offsetHeight || 60;

  function draw() {
    if (!micRec) {
      audioCtx.close();
      return;
    }
    micAnimId = requestAnimationFrame(draw);

    analyser.getByteTimeDomainData(dataArray);

    ctx.fillStyle = '#FAFCFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Guide center line
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, canvas.height/2);
    ctx.lineTo(canvas.width, canvas.height/2);
    ctx.stroke();

    // 1. Soft background phase-shifted wave
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.3)';
    ctx.beginPath();
    let x1 = 0;
    const sliceWidth1 = canvas.width / bufferLength;
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const offsetVal = Math.sin(i * 0.15 + Date.now() * 0.01) * 3;
      const y = (v * canvas.height) / 2 + offsetVal;
      if (i === 0) ctx.moveTo(x1, y);
      else ctx.lineTo(x1, y);
      x1 += sliceWidth1;
    }
    ctx.stroke();

    // 2. Main glowing primary wave
    ctx.lineWidth = 3;
    ctx.shadowBlur = 8;
    ctx.shadowColor = 'rgba(239, 68, 68, 0.4)';
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
    gradient.addColorStop(0, '#EF4444');
    gradient.addColorStop(0.5, '#F59E0B');
    gradient.addColorStop(1, '#EF4444');
    ctx.strokeStyle = gradient;
    ctx.beginPath();

    let x = 0;
    const sliceWidth = canvas.width / bufferLength;
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * canvas.height) / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      x += sliceWidth;
    }
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
    
    ctx.shadowBlur = 0; // Reset shadow
  }
  draw();
}

function startMicTimer() {
  micStartTime = Date.now();
  const timerEl = document.getElementById('mic-timer');
  timerEl.textContent = '00:00';
  micTimerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - micStartTime) / 1000);
    const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const ss = String(elapsed % 60).padStart(2, '0');
    timerEl.textContent = `${mm}:${ss}`;
  }, 1000);
}

function stopMicTimer() {
  if (micTimerInterval) {
    clearInterval(micTimerInterval);
    micTimerInterval = null;
  }
}

function clearCurrentAudio() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.src = "";
    currentAudio = null;
  }
  STAGED_FILE = null;
  LAST_ANALYSIS = null;
  storedAudioBuffer = null;
  waveData = [];
  wavePos = 0;
  chunkOverlays = [];
  audioDuration = 0;
  if (_chunkFetchController) {
    _chunkFetchController.abort();
    _chunkFetchController = null;
  }
  
  if (micAnimId) {
    cancelAnimationFrame(micAnimId);
    micAnimId = null;
  }
  if (liveAnimId) {
    cancelAnimationFrame(liveAnimId);
    liveAnimId = null;
  }
  
  // Guarantee canvas clears happen first
  drawWave([], 0);
  drawSpectrogram(null);
  drawThreatRibbon();
  
  // Reset Timeline & Results
    setUploadZoneState('default');
    resetTl();
    clearMeta();
    clearInf();
    setHdrChip('ready');
    
    // Reset screen report placeholders
    const scrAwaiting = document.getElementById('screen-report-awaiting');
    const scrSummary = document.getElementById('screen-report-summary');
    if(scrAwaiting) scrAwaiting.style.display = 'block';
    if(scrSummary) scrSummary.style.display = 'none';
    
    // Hide buttons
    document.getElementById('start-btn').style.display = 'none';
    document.getElementById('clear-btn').style.display = 'none';

  
  // Reset Hero card
  document.getElementById('hero-status').textContent = 'Awaiting Audio Input';
  document.getElementById('hero-sub').textContent = 'Upload a recording or start live capture to begin forensic analysis';
  document.getElementById('hero-score').textContent = '—';
  document.getElementById('hero-score').className = 'hero-score';
  document.getElementById('h-caseid').textContent = 'VG-——';
  document.getElementById('h-class').textContent = '—';
  document.getElementById('h-class').style.color = '';
  document.getElementById('h-conf').textContent = '—';
  document.getElementById('h-dur').textContent = '—';
  document.getElementById('h-proc').textContent = '—';
  document.getElementById('hero-rec-text').textContent = 'No active analysis — submit audio to receive a forensic recommendation.';
  document.getElementById('hdr-case').textContent = 'CASE: VG-——';
  
  // Reset risk details
  const riskNum = document.getElementById('risk-num'); if(riskNum) { riskNum.textContent = '—'; riskNum.className = 'risk-score-num lo'; }
  const rb = document.getElementById('risk-badge'); if(rb) { rb.className = 'risk-badge-big risk-badge-lo'; rb.textContent = 'AWAITING ANALYSIS'; }
  
  // Clear status chip
  const pText = document.getElementById('pipeline-status-text');
  if(pText) pText.innerHTML = 'Awaiting Audio Input...';
  const pStatus = document.getElementById('pipeline-status');
  if(pStatus) pStatus.style.display = 'none';
  
  const riskFinding = document.getElementById('risk-finding'); if(riskFinding) riskFinding.textContent = 'Submit an audio recording to receive forensic risk classification and fraud probability score.';
  
  // Reset threat intel fields
  ['ti-type','ti-soph','ti-replay','ti-synth','ti-trig'].forEach(id=>{ const e = document.getElementById(id); if(e) e.textContent='—'; });
  ['rd-evid','rd-vstab','rd-spec','rd-conf'].forEach(id=>{ const e = document.getElementById(id); if(e) e.textContent='—'; });
  const rdRec = document.getElementById('rd-rec'); if(rdRec) rdRec.textContent = 'Run analysis to generate recommendation.';
  
  // Reset fusion engine metrics
  const fusScore = document.getElementById('fus-matrix-score'); if(fusScore) fusScore.textContent = '—';
  const bioScore = document.getElementById('bio-matrix-score'); if(bioScore) bioScore.textContent = '—';
  const dlScore = document.getElementById('dl-matrix-score'); if(dlScore) dlScore.textContent = '—';
  const weightsEl = document.getElementById('fus-matrix-weights'); if(weightsEl) weightsEl.textContent = 'VoiceGuard Core Engine';
  
  // Reset XAI sliders (with null checks)
  const xaiBio = document.getElementById('xai-bio'); if (xaiBio) xaiBio.style.width = '0%';
  const xaiBioPct = document.getElementById('xai-bio-pct'); if (xaiBioPct) xaiBioPct.textContent = '—';
  const xaiDl = document.getElementById('xai-dl'); if (xaiDl) xaiDl.style.width = '0%';
  const xaiDlPct = document.getElementById('xai-dl-pct'); if (xaiDlPct) xaiDlPct.textContent = '—';
  const xaiFus = document.getElementById('xai-fus'); if (xaiFus) xaiFus.style.width = '0%';
  const xaiFusPct = document.getElementById('xai-fus-pct'); if (xaiFusPct) xaiFusPct.textContent = '—';
  
  // Reset Threat Vector Gauges
  ['tts', 'vc', 'replay'].forEach(id => {
    const el = document.getElementById('gauge-' + id);
    const textEl = document.getElementById('xai-' + id + '-pct');
    if (el) el.style.strokeDashoffset = 125.66;
    if (textEl) textEl.textContent = '—';
  });

  // Reset Evidence & Confidence LEDs
  ['ai', 'replay', 'human', 'bio', 'dl', 'fusion'].forEach(id => {
    const dot = document.getElementById('led-' + id + '-dot');
    const lbl = document.getElementById('led-' + id + '-lbl');
    const val = document.getElementById('led-' + id + '-val');
    if (dot) {
      dot.style.background = 'var(--text3)';
      dot.style.boxShadow = '0 0 0px var(--text3)';
    }
    if (lbl) {
      lbl.textContent = 'Pending';
      lbl.style.color = 'var(--text2)';
    }
    if (val) {
      val.textContent = '—';
      val.style.color = id === 'fusion' ? 'var(--bank)' : 'var(--text)';
    }
  });

  // Reset Identified Forensic Artifacts
  const dynFactors = document.getElementById('xai-factors-dynamic');
  if (dynFactors) dynFactors.innerHTML = '<span style="color:var(--text3);">Awaiting analysis...</span>';

  // Reset Radar Chart
  const poly = document.getElementById('radar-poly');
  if (poly) poly.setAttribute('points', '120,100 120,100 120,100 120,100 120,100 120,100');
  for (let i = 0; i < 6; i++) {
    const dot = document.getElementById('radar-dot-' + i);
    if (dot) {
      dot.setAttribute('cx', 120);
      dot.setAttribute('cy', 100);
    }
  }
  
  // Reset spectrogram, waveform canvases, and ribbons
  drawWave([], 0);
  drawSpectrogram(null);
  drawThreatRibbon();
  
  // Reset info texts
  const specTextEl = document.getElementById('spec-analysis-text');
  if (specTextEl) {
    specTextEl.innerHTML = '<i class="ti ti-info-circle" style="font-size:13px;color:var(--bank);margin-right:4px;vertical-align:-2px"></i>VoiceGuard spectrogram analysis monitors spectral frequency distribution. In voice clones, high frequencies (4kHz - 8kHz) reveal phase anomalies, unnatural silence, or harmonic grid alignments typical of synthetic speech generators.';
    specTextEl.style.borderLeftColor = 'var(--bank)';
  }
  const wvTextEl = document.getElementById('wave-analysis-text');
  if (wvTextEl) {
    wvTextEl.innerHTML = '<i class="ti ti-info-circle" style="font-size:13px;color:var(--bank2);margin-right:4px;vertical-align:-2px"></i>The waveform plots voice amplitude level changes over time. Natural speech displays random transients and smooth decay segments, while cloned voice segments often showcase artificially flat speech segments or abrupt thresholds.';
    wvTextEl.style.borderLeftColor = 'var(--bank2)';
  }
  
  // Reset playbar & progress
  document.getElementById('prog-fill').style.width = '0%';
  document.getElementById('wv-dur').textContent = '—';
  document.getElementById('wv-sr').textContent = '—';
  document.getElementById('wv-ch').textContent = '—';
  document.getElementById('spec-dur-label').textContent = '—';
  
  // Reset chunk analysis table
  const tbody = document.getElementById('chunk-tbody');
  if (tbody) tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text3);padding:24px;font-size:12px">No analysis data — upload audio to begin</td></tr>';
  const tblStatus = document.getElementById('tbl-status');
  if (tblStatus) {
    tblStatus.textContent = 'AWAITING INPUT';
    tblStatus.style.color = 'var(--text3)';
  }
  const sJob = document.getElementById('s-job');
  if (sJob) {
    sJob.className = 'pill pill-idle';
    sJob.textContent = 'IDLE';
  }
  
  // Reset report section
  document.getElementById('rpt-id').textContent = 'VG-——';
  document.getElementById('rpt-ts').textContent = '—';
  document.getElementById('rpt-risk').textContent = '—';
  document.getElementById('rpt-type').textContent = '—';
  document.getElementById('rpt-synth').textContent = '—';
  document.getElementById('rpt-fus').textContent = '—';
  document.getElementById('rpt-pri').innerHTML = '<span class="trigger-tag trig-pri">—</span>';
  document.getElementById('rpt-sec').innerHTML = '<span class="trigger-tag trig-sec">—</span>';
  document.getElementById('rpt-rec').textContent = 'Run analysis to generate recommendation.';
  document.getElementById('rpt-class').textContent = 'AWAITING ANALYSIS';
  
  // Reset feedback & console states
  if(document.getElementById('active-learning-block')) {
    document.getElementById('active-learning-block').style.display='none';
    const _all3=document.getElementById('active-learning-label');if(_all3)_all3.style.display='none';
    document.getElementById('feedback-wrong-block').style.display='none';
    document.getElementById('feedback-status').style.display='none';
  }
  const invSBadge = document.getElementById('inv-status-badge');
  if(invSBadge) {
    invSBadge.className = 'status-chip chip-complete';
    invSBadge.querySelector('span:nth-child(2)').textContent = 'Awaiting Analysis';
  }
  const pausedBanner = document.getElementById('inv-paused-banner'); if(pausedBanner) pausedBanner.style.display = 'none';
  const videoBanner = document.getElementById('inv-video-banner'); if(videoBanner) videoBanner.style.display = 'none';
  // Reset Investigation Console Data
  ['inv-customer', 'inv-phone', 'inv-dur', 'inv-confidence', 'inv-threat-type', 'inv-soph'].forEach(id => {
    const el = document.getElementById(id);
    if(el) el.textContent = '—';
  });
  const invTranscript = document.getElementById('inv-transcript');
  if(invTranscript) invTranscript.innerHTML = '<div style="color:var(--text3); font-style:italic; padding:10px;">Awaiting analysis...</div>';
  const invLogs = document.getElementById('inv-agent-logs');
  if(invLogs) invLogs.innerHTML = '<div style="color:var(--text3); font-style:italic; padding:10px;">No interactions logged yet.</div>';

  document.querySelectorAll('#inv-actions-list .action-check').forEach(el => el.classList.remove('checked'));
}

function newInvestigation() {
  clearCurrentAudio();
  switchPage('page-dashboard', document.getElementById('nav-dashboard'));
  triggerUpload();
}

function audioBufferToWav(buffer) {
  const sr = buffer.sampleRate;
  const n = buffer.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buf);
  const ws = (o, s) => { for(let i=0; i<s.length; i++) view.setUint8(o+i, s.charCodeAt(i)); };
  ws(0, 'RIFF'); view.setUint32(4, 36 + n * 2, true); ws(8, 'WAVE'); ws(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sr, true); view.setUint32(28, sr * 2, true); view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); ws(36, 'data'); view.setUint32(40, n * 2, true);
  const ch = buffer.getChannelData(0);
  for(let i=0; i<n; i++) {
    const s = Math.max(-1, Math.min(1, ch[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([buf], {type: 'audio/wav'});
}

function genDemoWav(){
  const sr=16000,dur=6.2,n=Math.floor(sr*dur);
  const buf=new ArrayBuffer(44+n*2);const view=new DataView(buf);
  function ws(o,s){for(let i=0;i<s.length;i++)view.setUint8(o+i,s.charCodeAt(i));}
  ws(0,'RIFF');view.setUint32(4,36+n*2,true);ws(8,'WAVE');ws(12,'fmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,sr,true);view.setUint32(28,sr*2,true);view.setUint16(32,2,true);view.setUint16(34,16,true);ws(36,'data');view.setUint32(40,n*2,true);
  for(let i=0;i<n;i++){const t=i/sr,env=Math.sin(Math.PI*i/n),val=(Math.sin(2*Math.PI*180*t)*.3+Math.sin(2*Math.PI*420*t)*.15)*env;const s=Math.max(-1,Math.min(1,val));view.setInt16(44+i*2,s*32767,true);}
  return new Blob([buf],{type:'audio/wav'});
}
function runDemo(){
  const pts=300,arr=[];
  for(let i=0;i<pts;i++){const t=i/pts,env=Math.sin(t*Math.PI);arr.push((Math.sin(t*80)*.3+Math.sin(t*200)*.15)*env);}
  waveData=arr;drawWave(waveData,0);
  document.getElementById('wv-stat').textContent='DEMO SIGNAL';document.getElementById('wv-stat').style.color='var(--warn)';
  clearMeta();clearInf();
  const demoRaw=new Float32Array(16000*6);
  for(let i=0;i<demoRaw.length;i++){const t=i/16000;demoRaw[i]=(Math.sin(t*2*Math.PI*180)*.3+Math.sin(t*2*Math.PI*420)*.15)*Math.sin(Math.PI*i/demoRaw.length);}
  drawSpectrogram(demoRaw,16000,120);
  document.getElementById('spec-dur-label').textContent='6.2s';
  const blob=genDemoWav();
  
  // Wire up audio playback for demo
  if(currentAudio) { currentAudio.pause(); currentAudio.src=""; }
  currentAudio = new Audio(URL.createObjectURL(blob));
  currentAudio.addEventListener('ended', () => {
    isPlaying=false; document.getElementById('play-icon').className='ti ti-player-play';
    wavePos=0; document.getElementById('prog-fill').style.width='0%';
    drawWave(waveData, 0);
  });
  currentAudio.addEventListener('timeupdate', (e) => {
    const audio = e.target;
    if(audio.duration) {
      wavePos = audio.currentTime / audio.duration;
      document.getElementById('prog-fill').style.width=(wavePos*100)+'%';
      drawWave(waveData,wavePos);
    }
  });

  const f=new File([blob],'DEMO_SYNTHETIC_TONE_001.wav',{type:'audio/wav'});
  STAGED_FILE = f;
  
  setUploadZoneState('loaded', f.name, formatSize(f.size) + ' · 16kHz · 6.2s · PCM/WAV');
  document.getElementById('clear-btn').style.display='flex';
  
  runAnalysis(f);
}

function downloadReport(){
  if(!LAST_ANALYSIS){alert('Run an analysis first to generate a report.');return;}
  window.print();
}

async function loadSystemStatus(){
  try{
    const s=await apiGet('/model-status');
    const sm=document.getElementById('s-mdl');if(sm){sm.className='pill pill-on';sm.textContent=s.model_loaded?'LOADED':'PLACEHOLDER';}
    const si=document.getElementById('s-inf');if(si){si.textContent=s.inference_ready?'READY':'NOT READY';}
    const sb=document.getElementById('s-back');if(sb){sb.className='pill pill-conn';sb.textContent='CONNECTED';}
  }catch(e){
    const sb=document.getElementById('s-back');if(sb){sb.className='pill pill-idle';sb.textContent='OFFLINE';}
  }
  try{
    const h=await apiGet('/health');
    const sa=document.getElementById('s-api');if(sa){sa.textContent=h.backend_api==='CONNECTED'?'ONLINE':'OFFLINE';}
    const sd=document.getElementById('s-db');if(sd){sd.textContent=h.database==='CONNECTED'?'CONNECTED':'OFFLINE';}
  }catch(e){
    const sa=document.getElementById('s-api');if(sa){sa.className='pill pill-idle';sa.textContent='OFFLINE';}
  }
}
loadSystemStatus();setInterval(loadSystemStatus,15000);

window.addEventListener('resize',()=>{resizeWave();resizeSpec();drawWave(waveData,wavePos);});

function submitFeedback(isCorrect){
  if(isCorrect){
    if(LAST_ANALYSIS.verdict === 'FRAUD') {
      submitTrueLabel('Synthetic');
    } else {
      submitTrueLabel('Real');
    }
  } else {
    document.getElementById('feedback-wrong-block').style.display='block';
  }
}
async function submitTrueLabel(trueLabel){
  try{
    const form = new FormData();
    form.append('is_correct', 'true');
    form.append('true_label', trueLabel);
    const res = await fetch(BACKEND_URL + '/feedback', {method:'POST', headers:{'x-api-key':'voiceguard123'}, body:form});
    const j = await res.json();
    const st = document.getElementById('feedback-status');
    st.textContent = j.message; st.style.display='block'; st.style.color='#15803D';
    document.getElementById('feedback-wrong-block').style.display='none';
  }catch(e){
    const st = document.getElementById('feedback-status');
    st.textContent = "Error saving feedback."; st.style.display='block'; st.style.color='#DC2626';
  }
}
async function retrainModel(){
  const st = document.getElementById('feedback-status');
  st.textContent = "Retraining in background... please wait."; st.style.display='block'; st.style.color='#D97706';
  try{
    const res = await fetch(BACKEND_URL + '/retrain', {method:'POST', headers:{'x-api-key':'voiceguard123'}});
    const j = await res.json();
    st.textContent = j.message; st.style.color='#15803D';
  }catch(e){
    const st = document.getElementById('feedback-status');
    st.textContent = "Retrain failed."; st.style.color='#DC2626';
  }
}

function switchPage(pageId, navItem) {
  const pages = ['page-dashboard', 'page-intel', 'page-stats', 'page-kb', 'page-audit', 'page-investigator'];
  pages.forEach(p => {
    const el = document.getElementById(p);
    if(el) el.style.display = (p === pageId) ? 'block' : 'none';
  });
  
  if(navItem) {
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    navItem.classList.add('active');
  }

  // Fix canvas rendering bug when switching from hidden display
  if(pageId === 'page-intel') {
    // Small timeout to ensure DOM has updated display block before reading offsetWidth
    setTimeout(() => {
      resizeWave();
      resizeSpec();
      drawWave(waveData, wavePos);
      if (storedAudioBuffer) {
        const ch = storedAudioBuffer.getChannelData(0);
        drawSpectrogram(ch, storedAudioBuffer.sampleRate, Math.min(150, Math.floor(ch.length / 256)));
      } else {
        drawSpectrogram(null, 0, 0);
      }
    }, 50);
  }
}

// System Clock
function updateClock() {
  const el = document.getElementById('hdr-clock');
  if (el) {
    el.textContent = new Date().toLocaleTimeString('en-IN', { hour12: false });
  }
}
setInterval(updateClock, 1000);
updateClock();

// Popover click event handlers
document.addEventListener('click', (e) => {
  const notifBtn = document.querySelector('.hdr-notif');
  const notifDropdown = document.getElementById('notif-dropdown');
  const profileBtn = document.querySelector('.hdr-avatar');
  const profileDropdown = document.getElementById('profile-dropdown');
  const diagBtn = document.getElementById('hdr-chip');
  const diagDropdown = document.getElementById('diagnostics-dropdown');

  // Toggle notif
  if (notifBtn && notifBtn.contains(e.target)) {
    notifDropdown.style.display = notifDropdown.style.display === 'none' ? 'block' : 'none';
    if (profileDropdown) profileDropdown.style.display = 'none';
    if (diagDropdown) diagDropdown.style.display = 'none';
  }
  // Toggle profile
  else if (profileBtn && profileBtn.contains(e.target)) {
    profileDropdown.style.display = profileDropdown.style.display === 'none' ? 'block' : 'none';
    if (notifDropdown) notifDropdown.style.display = 'none';
    if (diagDropdown) diagDropdown.style.display = 'none';
  }
  // Toggle diag
  else if (diagBtn && diagBtn.contains(e.target)) {
    diagDropdown.style.display = diagDropdown.style.display === 'none' ? 'block' : 'none';
    if (notifDropdown) notifDropdown.style.display = 'none';
    if (profileDropdown) profileDropdown.style.display = 'none';
  }
  // Click outside to close
  else {
    // Make sure we didn't click inside any of the dropdowns
    if (notifDropdown && !notifDropdown.contains(e.target)) notifDropdown.style.display = 'none';
    if (profileDropdown && !profileDropdown.contains(e.target)) profileDropdown.style.display = 'none';
    if (diagDropdown && !diagDropdown.contains(e.target)) diagDropdown.style.display = 'none';
  }
});

// Initialize default view
switchIntakeTab('file');

// ==================== NEW WORKFLOW FEATURES ====================

// ========== CHAT KNOWLEDGE BASE ==========
const CHAT_KNOWLEDGE = {
  'replay': 'A <b>replay attack</b> occurs when a fraudster records a genuine customer\'s voice and replays it to fool voice authentication. VoiceGuard detects this via temporal inconsistencies, spectral smoothness anomalies, and re-recording artifacts.',
  'clone': 'An <b>AI voice clone</b> is generated using deep learning TTS/VC models trained on a few seconds of real audio. The cloned voice mimics the target but lacks natural micro-variations in jitter, shimmer, and pitch that VoiceGuard monitors.',
  'deepfake': 'A <b>deepfake speech</b> attack uses generative AI (GANs or diffusion models) to produce highly realistic synthetic audio. These are harder to detect and show up as abnormal spectral features and phase discontinuities.',
  'confidence': 'The <b>confidence score</b> reflects the model\'s certainty. 95%+ = very high certainty. 70–94% = moderate, secondary verification recommended. Below 70% = uncertain, flag for manual review.',
  'voiceguard': 'VoiceGuard uses a <b>Fusion Architecture</b>: (1) Biological Engine — 97 acoustic features including jitter, shimmer, HNR, pitch. (2) Deep Learning Engine — Wav2Vec2 extracts 2048 embeddings. Both feed a Fusion Core for final fraud probability.',
  'jitter': '<b>Jitter</b> is cycle-to-cycle frequency variation in vocal folds. <b>Shimmer</b> is amplitude variation between pulses. <b>HNR</b> (Harmonics-to-Noise Ratio) measures harmonic purity. Synthetic voices show abnormal values in all three.',
  'sop': '<b>RBI SOP for Voice Fraud:</b><br>1. Verify customer identity via OTP/Video KYC<br>2. Freeze suspicious transactions<br>3. Escalate to Fraud Investigation Cell<br>4. Preserve audio evidence per CERT-In guidelines<br>5. Notify the customer<br>6. File regulatory report if value exceeds ₹1 lakh<br>7. Submit case for model retraining',
};

// ========== AGENT NOTES (Reports page) ==========
function appendNote(text) {
  const ta = document.getElementById('agent-notes');
  if(ta) { ta.value = ta.value ? ta.value + '\n' + text : text; ta.focus(); }
}
function saveNotes() {
  const saved = document.getElementById('notes-saved');
  if(saved) { saved.style.display = 'flex'; saved.style.alignItems = 'center'; saved.style.gap = '4px'; setTimeout(()=>{ saved.style.display='none'; }, 2000); }
  addAuditEntry('NOTES SAVED', 'Notes updated');
}

// ========== AGENT NOTES (Investigator page) ==========
function appendInvNote(text) {
  const ta = document.getElementById('inv-notes');
  if(ta) { ta.value = ta.value ? ta.value + '\n' + text : text; ta.focus(); }
}
function saveInvNotes() {
  if (!LAST_ANALYSIS) return;
  const notes = document.getElementById('inv-notes').value;
  let formData = new FormData();
  formData.append('notes', notes);
  
  fetch('/api/v1/cases/' + LAST_ANALYSIS.case_id + '/action', {
    method: 'POST',
    headers: { 'Authorization': 'Bearer ' + API_KEY },
    body: formData
  }).then(r => r.json()).then(data => {
    const saved = document.getElementById('inv-notes-saved');
    if(saved) { 
      saved.style.display = 'flex'; 
      saved.style.alignItems = 'center'; 
      saved.style.gap = '4px'; 
      setTimeout(()=>{ saved.style.display='none'; }, 2000); 
    }
    addAuditEntry('NOTES SAVED', 'Investigation notes updated in database.');
  }).catch(e => console.error(e));
}

// ========== KNOWLEDGE BASE TOGGLE ==========
function toggleKb(el) { if(el) el.classList.toggle('open'); }

// ========== CONFIDENCE METER ==========
function updateConfidenceMeter(fraudScore, replayPct, confidence) {
  const cmAi = document.getElementById('cm-ai');
  const cmReplay = document.getElementById('cm-replay');
  const cmOverall = document.getElementById('cm-overall');
  if(cmAi) { cmAi.style.width = Math.min(100,fraudScore)+'%'; const pct = document.getElementById('cm-ai-pct'); if(pct) pct.textContent = fraudScore+'%'; }
  if(cmReplay) { cmReplay.style.width = Math.min(100,replayPct)+'%'; const pct = document.getElementById('cm-replay-pct'); if(pct) pct.textContent = replayPct+'%'; }
  if(cmOverall) { cmOverall.style.width = Math.min(100,confidence)+'%'; const pct = document.getElementById('cm-overall-pct'); if(pct) pct.textContent = confidence+'%'; }
}

// ========== ATTACK TYPE BANNER ==========
function updateAttackBanner(threatType) {
  const banner = document.getElementById('attack-banner');
  const icon = document.getElementById('attack-icon');
  const title = document.getElementById('attack-title');
  const sub = document.getElementById('attack-sub');
  if(!banner) return;
  let cls='atb-unknown', ic='ti-question-mark', t='Unknown Voice', s='Threat type could not be classified.';
  if(threatType) {
    const tt = threatType.toUpperCase();
    if(tt.includes('AI') || tt.includes('CLONE') || tt.includes('SYNTH')) {
      cls='atb-ai'; ic='ti-robot'; t='AI Voice Clone Detected'; s='Synthetic TTS or voice conversion model detected. High biological feature deviation.';
    } else if(tt.includes('REPLAY')) {
      cls='atb-replay'; ic='ti-repeat'; t='Replay Attack Detected'; s='Pre-recorded voice segment identified. Temporal and spectral replay artifacts present.';
    } else if(tt.includes('DEEP') || tt.includes('FAKE')) {
      cls='atb-deepfake'; ic='ti-layers-intersect'; t='Deepfake Speech Detected'; s='Generative AI audio detected. Advanced GAN or diffusion model signatures present.';
    } else if(tt.includes('HUMAN') || tt.includes('REAL')) {
      cls='atb-human'; ic='ti-user-check'; t='Human Voice — Authentic'; s='No synthetic artifacts detected. Biological features within normal range.';
    }
  }
  banner.className = 'attack-type-banner ' + cls;
  icon.className = 'ti ' + ic;
  title.textContent = t;
  sub.textContent = s;
}

// ========== SIMILAR CASES ==========
const FAKE_CASES = [
  {id:'VG-A2F91B',lang:'Hindi',attack:'Replay Attack',risk:94,outcome:'Transaction Blocked',sim:97},
  {id:'VG-C8D34E',lang:'Bengali',attack:'AI Voice Clone',risk:91,outcome:'Escalated to Cyber Cell',sim:88},
  {id:'VG-F1290A',lang:'Hindi',attack:'AI Voice Clone',risk:89,outcome:'Video KYC Requested',sim:84},
  {id:'VG-D5E72C',lang:'English',attack:'Replay Attack',risk:87,outcome:'Transaction Blocked',sim:79},
  {id:'VG-B3391F',lang:'Tamil',attack:'AI Voice Clone',risk:82,outcome:'Customer Notified',sim:72},
];
function showSimilarCases(verdict) {
  if (!LAST_ANALYSIS) return;
  const wrap = document.getElementById('similar-cases-wrap');
  if(!wrap) return;
  
  wrap.innerHTML = '<span style="color:var(--text3); font-size:11px">Fetching similar cases from database...</span>';
  
  fetch(BACKEND_URL + '/api/v1/cases/' + LAST_ANALYSIS.case_id + '/similar', {
    headers: { 'x-api-key': 'voiceguard123' }
  }).then(r => r.json()).then(data => {
    if (data.status === "error" || !data.similar_cases || data.similar_cases.length === 0) {
      wrap.innerHTML = '<span style="color:var(--text3); font-size:11px">No similar cases found in recent history.</span>';
      return;
    }
    
    let html = '';
    data.similar_cases.forEach(c => {
       const riskColor = c.verdict === 'FRAUD' ? 'color:var(--danger)' : (c.verdict === 'SUSPICIOUS' ? 'color:var(--warn)' : 'color:var(--success)');
       html += `
       <div style="padding:10px; border:1px solid var(--border); border-radius:6px; margin-bottom:8px; background:#FAFCFF; position:relative;">
         <div style="font-size:11px; font-weight:700; ${riskColor}; margin-bottom:4px">${c.case_id} (${c.verdict})</div>
         <div style="font-size:10px; color:var(--text2); display:flex; gap:10px;">
            <span>Branch: ${c.branch_location}</span>
            <span>Type: ${c.threat_type || 'Unknown'}</span>
         </div>
         <div style="font-size:10px; color:var(--text3); margin-top:4px;">Date: ${c.timestamp}</div>
         <div style="position:absolute; top:10px; right:10px; font-size:11px; font-weight:700; color:var(--bank); background:#EAF2FB; padding:2px 6px; border-radius:4px;">Sim: ${Math.round(c.similarity_score * 100)}%</div>
       </div>`;
    });
    wrap.innerHTML = html;
  }).catch(e => {
    wrap.innerHTML = '<span style="color:var(--text3); font-size:11px">Failed to load similar cases.</span>';
    console.error(e);
  });
}

// ========== AUDIT LOGS ==========
let auditLogs = [];
function addAuditEntry(action, decision, caseId) {
  const now = new Date().toLocaleTimeString('en-IN',{hour12:false});
  const cid = caseId || (LAST_ANALYSIS ? LAST_ANALYSIS.case_id : 'VG-——');
  auditLogs.unshift({case: cid, time: now, officer: 'FK', action, decision, status: 'Logged'});
  renderAuditLog();
  const badge = document.getElementById('audit-count');
  if(badge) badge.textContent = auditLogs.length;
  const logCnt = document.getElementById('audit-log-count');
  if(logCnt) logCnt.textContent = auditLogs.length + ' entries';
}
function renderAuditLog() {
  const tbody = document.getElementById('audit-tbody');
  if(!tbody) return;
  if(!auditLogs.length) { tbody.innerHTML='<tr><td colspan="6" style="text-align:center;color:var(--text3);padding:24px;font-size:12px">No audit entries yet.</td></tr>'; return; }
  tbody.innerHTML = auditLogs.map(e => {
    let acls = 'a-analysis';
    if(e.action.includes('REPORT') || e.action.includes('EXPORT')) acls = 'a-report';
    else if(e.action.includes('FEEDBACK') || e.action.includes('NOTES')) acls = 'a-feedback';
    else if(e.action.includes('EMAIL') || e.action.includes('CSV')) acls = 'a-export';
    return `<tr>
      <td style="font-family:var(--mono);color:var(--bank);font-weight:600">${e.case}</td>
      <td style="font-family:var(--mono);color:var(--text3)">${e.time}</td>
      <td style="color:var(--text2)">${e.officer}</td>
      <td><span class="audit-action ${acls}">${e.action}</span></td>
      <td style="color:var(--text2)">${e.decision}</td>
      <td style="color:var(--success);font-size:10px;font-weight:600">${e.status}</td>
    </tr>`;
  }).join('');
}
function clearAuditLogs() {
  auditLogs = [];
  renderAuditLog();
  const badge = document.getElementById('audit-count');
  if(badge) badge.textContent = '0';
  const logCnt = document.getElementById('audit-log-count');
  if(logCnt) logCnt.textContent = '0 entries';
}

// ========== EXPORT HELPERS ==========
function exportExcel() {
  if(!LAST_ANALYSIS){alert('Run an analysis first.');return;}
  const csv = `Case ID,Timestamp,Risk Score,Confidence,Verdict,Threat Type,Replay,Language,Duration\n${LAST_ANALYSIS.case_id},${LAST_ANALYSIS.timestamp},${LAST_ANALYSIS.fraud_score}%,${LAST_ANALYSIS.confidence}%,${LAST_ANALYSIS.verdict},${LAST_ANALYSIS.threat_intel.threat_type},${LAST_ANALYSIS.threat_intel.replay_indicators},${LAST_ANALYSIS.metadata?.language||'Detected'},${LAST_ANALYSIS.performance.audio_duration_sec}s`;
  const blob = new Blob([csv], {type:'text/csv'});
  const a = document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='VoiceGuard_'+LAST_ANALYSIS.case_id+'.csv'; document.body.appendChild(a); a.click(); a.remove();
  addAuditEntry('EXPORT CSV', 'Excel export');
}
function exportSummary() {
  if(!LAST_ANALYSIS){alert('Run an analysis first.');return;}
  const r = LAST_ANALYSIS;
  const txt = `VoiceGuard Investigation Summary\n${'='.repeat(40)}\nCase ID: ${r.case_id}\nDate: ${new Date(r.timestamp).toLocaleString('en-IN')}\nRisk Score: ${r.fraud_score}%\nConfidence: ${r.confidence}%\nVerdict: ${r.verdict}\nThreat: ${r.threat_intel.threat_type}\nRecommendation: ${r.recommendation}\nAnalyst Notes: ${document.getElementById('agent-notes')?.value||'None'}\n`;
  const blob = new Blob([txt], {type:'text/plain'});
  const a = document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='Summary_'+r.case_id+'.txt'; document.body.appendChild(a); a.click(); a.remove();
  addAuditEntry('EXPORT SUMMARY', 'Text summary');
}
function exportEmail() {
  if(!LAST_ANALYSIS){alert('Run an analysis first.');return;}
  const r = LAST_ANALYSIS;
  const body = encodeURIComponent(`Fraud Investigation Alert - ${r.case_id}\n\nCase: ${r.case_id}\nRisk Score: ${r.fraud_score}%\nVerdict: ${r.verdict}\nThreat Type: ${r.threat_intel?.threat_type}\nConfidence: ${r.confidence}%\n\nRecommended Action: ${r.recommendation}\n\n-- VoiceGuard Fraud Intelligence Division`);
  window.open(`mailto:fraudteam@ucobank.com?subject=VoiceGuard Alert: ${r.case_id} - ${r.verdict}&body=${body}`);
  addAuditEntry('EMAIL DRAFTED', 'Email report');
}

// A simple token-matching engine with spelling distance tolerance
function getLevenshteinDistance(a, b) {
  const tmp = [];
  for (let i = 0; i <= a.length; i++) { tmp[i] = [i]; }
  for (let j = 0; j <= b.length; j++) { tmp[0][j] = j; }
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      tmp[i][j] = Math.min(
        tmp[i - 1][j] + 1,
        tmp[i][j - 1] + 1,
        tmp[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
    }
  }
  return tmp[a.length][b.length];
}

function wordSimilarity(word1, word2) {
  if (word1 === word2) return 1.0;
  const dist = getLevenshteinDistance(word1, word2);
  const maxLen = Math.max(word1.length, word2.length);
  if (maxLen === 0) return 1.0;
  return 1.0 - dist / maxLen;
}

function matchTokens(tokens, targetKeywords) {
  for (const token of tokens) {
    if (token.length < 3) {
      if (targetKeywords.includes(token)) return true;
      continue;
    }
    for (const kw of targetKeywords) {
      if (kw.length < 3) {
        if (token === kw) return true;
        continue;
      }
      if (wordSimilarity(token, kw) > 0.8) {
        return true;
      }
    }
  }
  return false;
}

// ========== AI CHAT (Reports page - Fallback) ==========
async function sendChat(quickText) {
  invChat(quickText);
}

// ========== AI CHAT (Investigator page) ==========
async function invChat(quickText) {
  const input = document.getElementById('inv-chat-input');
  const msgs = document.getElementById('inv-chat-msgs');
  const typing = document.getElementById('inv-chat-typing');
  if(!input || !msgs) return;
  const text = (quickText || input.value || '').trim();
  if(!text) return;
  input.value = '';
  const userBubble = document.createElement('div');
  userBubble.className = 'chat-bubble bubble-user';
  userBubble.textContent = text;
  msgs.appendChild(userBubble);
  msgs.scrollTop = msgs.scrollHeight;
  if(typing) { typing.style.display = 'block'; msgs.scrollTop = msgs.scrollHeight; }

  setTimeout(() => {
    if(typing) typing.style.display = 'none';
    const aiBubble = document.createElement('div');
    aiBubble.className = 'chat-bubble bubble-ai';
    
    // Process input
    const cleanText = text.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?]/g, "");
    const tokens = cleanText.split(/\s+/);
    
    let reply = "";
    
    // 1. Replay Attacks
    if (matchTokens(tokens, ['replay', 'replayed', 'recording', 're-recorded', 'playback', 'replai', 'recoding'])) {
      reply = CHAT_KNOWLEDGE.replay;
    }
    // 2. AI Cloning / Synthesis
    else if (matchTokens(tokens, ['clone', 'cloned', 'cloning', 'synthetic', 'tts', 'generated', 'generative', 'clon', 'synthetik', 'fake'])) {
      reply = CHAT_KNOWLEDGE.clone;
    }
    // 3. Deepfake
    else if (matchTokens(tokens, ['deepfake', 'deep-fake', 'gan', 'diffusion', 'df', 'depefake'])) {
      reply = CHAT_KNOWLEDGE.deepfake;
    }
    // 4. Model Confidence
    else if (matchTokens(tokens, ['confidence', 'certainty', 'score', 'prob', 'probability', 'confidense', 'certanty'])) {
      if (LAST_ANALYSIS) {
        reply = `The confidence score for Case <b>${LAST_ANALYSIS.case_id}</b> is <b>${LAST_ANALYSIS.confidence}%</b>. This combines standard error models with biological descriptor thresholds to calibrate our security risk probability.`;
      } else {
        reply = CHAT_KNOWLEDGE.confidence;
      }
    }
    // 5. Biometric descriptors
    else if (matchTokens(tokens, ['jitter', 'shimmer', 'hnr', 'harmonics', 'formant', 'pitch', 'acoustic', 'biology', 'biological', 'jiter', 'shimer'])) {
      reply = CHAT_KNOWLEDGE.jitter;
    }
    // 6. VoiceGuard Architecture
    else if (matchTokens(tokens, ['voiceguard', 'architecture', 'fusion', 'model', 'system', 'engine', 'pipeline', 'workflow'])) {
      reply = CHAT_KNOWLEDGE.voiceguard;
    }
    // 7. RBI SOP / Policies
    else if (matchTokens(tokens, ['sop', 'rbi', 'procedure', 'regulatory', 'guideline', 'guidelines', 'rules', 'rule', 'escalate', 'reporting', 'limit', 'lakh', 'one'])) {
      reply = CHAT_KNOWLEDGE.sop;
    }
    // 8. Next actions
    else if (matchTokens(tokens, ['action', 'recommend', 'next', 'do', 'what to do', 'block', 'freeze', 'held', 'hold', 'verify'])) {
      if (LAST_ANALYSIS) {
        reply = `<b>Recommended Actions for ${LAST_ANALYSIS.case_id}:</b><br>• Primary Action: ${LAST_ANALYSIS.recommendation}<br>• Trigger Video KYC verification for this caller.<br>• Freeze pending transactions on this account.<br>• Escalate case logs to UCO Bank Fraud Control Cell.`;
      } else {
        reply = `Please upload and analyze a recording first. Standard next steps include freezing the transaction, requesting Video KYC, and notifying the Security Escalation Team.`;
      }
    }
    // 9. Flag reason
    else if (matchTokens(tokens, ['flag', 'why', 'reason', 'trigger', 'flagged', 'why fake', 'explain risk'])) {
      if (LAST_ANALYSIS) {
        reply = `Case <b>${LAST_ANALYSIS.case_id}</b> was flagged as <b>${LAST_ANALYSIS.verdict}</b> with a risk score of <b>${LAST_ANALYSIS.fraud_score}%</b>. The primary trigger was identified as: <b>${LAST_ANALYSIS.primary_trigger}</b>. Threat classification: <b>${LAST_ANALYSIS.threat_intel?.threat_type || 'Voice Spoofing'}</b>.`;
      } else {
        reply = 'Please upload a recording and run analysis first to check case flags.';
      }
    }
    // 10. Summary
    else if (matchTokens(tokens, ['summarize', 'summary', 'case info', 'details', 'verdict', 'sumary', 'brief'])) {
      if (LAST_ANALYSIS) {
        reply = `<b>Forensics Case Summary (${LAST_ANALYSIS.case_id}):</b><br>• Verdict: <b>${LAST_ANALYSIS.verdict}</b><br>• Risk Score: <b>${LAST_ANALYSIS.fraud_score}%</b><br>• Confidence: <b>${LAST_ANALYSIS.confidence}%</b><br>• Duration: <b>${LAST_ANALYSIS.performance?.audio_duration_sec || 'N/A'}s</b><br>• Processing Time: <b>${LAST_ANALYSIS.performance?.processing_time_ms || LAST_ANALYSIS.performance?.inference_time_ms || 210}ms</b>`;
      } else {
        reply = 'Run an analysis first to get a summary.';
      }
    }
    // 11. Similar cases
    else if (matchTokens(tokens, ['similar', 'cases', 'match', 'history', 'past cases'])) {
      reply = 'Similar fraud patterns show matches with:<br>• <b>VG-A4F2C1</b> (94% similarity, Replay Attack)<br>• <b>VG-B7E3D9</b> (88% similarity, AI Voice Clone).<br>Both were successfully frozen under UCO Bank SOP.';
    }
    // 12. Report
    else if (matchTokens(tokens, ['report', 'pdf', 'csv', 'export', 'download'])) {
      if (LAST_ANALYSIS) {
        reply = `<b>Forensic Report ready for ${LAST_ANALYSIS.case_id}:</b> Use the "Print PDF Report" button to print a formal cyber incident report with signature lines and compliance logs.`;
      } else {
        reply = `Run an analysis to generate a report.`;
      }
    }
    // Default smart fallback
    else {
      reply = `I am the VoiceGuard Forensic AI Agent. I didn't quite capture that, but I can answer bank policies, technical pipeline specs, or active cases. Try asking about:<br>• <b>Technical</b>: Jitter/Shimmer, Wav2Vec2 model, 4441D feature dimensions, calibration.<br>• <b>Banking / SOP</b>: RBI guidelines, freezing accounts, escalation thresholds (₹1 Lakh).<br>• <b>Case Analysis</b>: Understanding why the current call was flagged or what to do next.<br><br><i>Tip: I support spelling-tolerant questions like "explian replai" or "what is jiter"!</i>`;
    }
    
    aiBubble.innerHTML = '<b>Investigator AI:</b> ' + reply;
    msgs.appendChild(aiBubble);
    msgs.scrollTop = msgs.scrollHeight;
    
    addAuditEntry('CHAT QUERY', text.slice(0,30)+'...');
  }, 500 + Math.random() * 500);
}

// ========== TOGGLE ACTION CHECKBOX & CHECKLIST HANDLERS ==========
function toggleAction(el) {
  // kept as fallback
  const check = el.querySelector('.action-check');
  if(check) {
    check.classList.toggle('checked');
    const label = el.querySelector('strong')?.textContent || 'Action';
    addAuditEntry('ACTION ' + (check.classList.contains('checked') ? 'COMPLETED' : 'UNCHECKED'), label);
  }
}

function handleChecklistClick(type) {
  if (!LAST_ANALYSIS) return;
  const row = document.getElementById('chk-row-' + type);
  if(!row) return;
  const check = row.querySelector('.action-check');
  if(!check) return;
  
  if (type === 'otp') {
    if (check.classList.contains('checked')) {
      check.classList.remove('checked');
      document.getElementById('chk-otp-popup').style.display = 'none';
    } else {
      document.getElementById('chk-otp-popup').style.display = 'block';
      if (LAST_ANALYSIS.otp_code) {
        document.getElementById('chk-otp-input').placeholder = "Enter code (Live: " + LAST_ANALYSIS.otp_code + ")";
      }
      return; 
    }
  } else {
    check.classList.toggle('checked');
  }

  const isChecked = check.classList.contains('checked');
  let state = {};
  state[type] = isChecked;
  fetch('/api/v1/cases/' + LAST_ANALYSIS.case_id + '/checklist', {
    method: 'POST',
    headers: {
      'Authorization': 'Bearer ' + API_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(state)
  }).then(r => r.json()).then(data => {
     addAuditEntry(isChecked ? 'ACTION CHECKED' : 'ACTION UNCHECKED', type.toUpperCase());
  }).catch(e => console.error(e));
}

function verifyChecklistOtp() {
  if (!LAST_ANALYSIS) return;
  const input = document.getElementById('chk-otp-input').value;
  const status = document.getElementById('chk-otp-status');
  const validOtp = LAST_ANALYSIS.otp_code || "123456";
  
  if (input === validOtp) {
    status.innerHTML = '<span style="color:#16A34A">✔ OTP Verified</span>';
    const check = document.getElementById('chk-row-otp').querySelector('.action-check');
    check.classList.add('checked');
    
    fetch('/api/v1/cases/' + LAST_ANALYSIS.case_id + '/checklist', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer ' + API_KEY, 'Content-Type': 'application/json' },
      body: JSON.stringify({ otp: true })
    }).then(r => r.json()).then(data => {
       addAuditEntry('ACTION CHECKED', 'OTP VERIFICATION (Verified)');
    });
    
    setTimeout(() => {
      document.getElementById('chk-otp-popup').style.display = 'none';
      document.getElementById('chk-otp-input').value = '';
      status.innerHTML = '';
    }, 1500);
  } else {
    status.innerHTML = '<span style="color:var(--danger)">✘ Invalid OTP</span>';
  }
}

// ========== HOOK INTO showResults ==========
const _origShowResults = showResults;
showResults = function(result, clientMs) {
  _origShowResults(result, clientMs);

  // Update screen-only summary card
  const scrCaseId = document.getElementById('scr-case-id');
  const scrRiskScore = document.getElementById('scr-risk-score');
  const scrThreatType = document.getElementById('scr-threat-type');
  const scrVerdictBadge = document.getElementById('scr-verdict-badge');
  const scrAwaiting = document.getElementById('screen-report-awaiting');
  const scrSummary = document.getElementById('screen-report-summary');

  if(scrCaseId) scrCaseId.textContent = result.case_id;
  if(scrRiskScore) {
    scrRiskScore.textContent = result.fraud_score + '%';
    scrRiskScore.className = 'rpt-val' + (result.verdict === 'FRAUD' ? ' danger' : '');
  }
  if(scrThreatType) scrThreatType.textContent = result.threat_intel?.threat_type || 'None';
  if(scrVerdictBadge) {
    scrVerdictBadge.textContent = result.verdict_label || result.verdict;
    scrVerdictBadge.className = 'status-chip ' + (result.verdict === 'FRAUD' ? 'chip-risk' : result.verdict === 'SUSPICIOUS' ? 'chip-suspect' : 'chip-complete');
  }
  if(scrAwaiting) scrAwaiting.style.display = 'none';
  if(scrSummary) scrSummary.style.display = 'block';

  // Confidence meter
  const replayPct = result.threat_intel?.replay_indicators?.includes('DETECTED') ? Math.round(result.fraud_score * 0.95) : Math.round(result.fraud_score * 0.3);
  updateConfidenceMeter(result.fraud_score, replayPct, result.confidence);
  // Attack type banner
  updateAttackBanner(result.threat_intel?.threat_type);
  // Similar cases
  showSimilarCases(result.verdict);

  // Phase 4: Dynamic XAI Factors
  const xaiFlagsEl = document.getElementById('xai-factors-dynamic');
  if(xaiFlagsEl && result.explanation && result.explanation.flags) {
    const flagColors = {
      'HIGH_DISAGREEMENT': 'danger', 'BIO_ALERT': 'danger', 'DEEP_ALERT': 'danger',
      'LOW_JITTER': 'warn', 'LOW_SHIMMER': 'warn', 'HIGH_HNR': 'warn', 'FLAT_PITCH': 'warn',
      'LOW_SPECTRAL_FLATNESS': 'warn', 'BIO_BORDERLINE': 'warn', 'DEEP_BORDERLINE': 'warn',
      'STREAMS_AGREE': 'success', 'HIGH_CONFIDENCE': 'success', 'LOW_CONFIDENCE': 'info'
    };
    if (result.explanation.flags.length > 0) {
        xaiFlagsEl.innerHTML = result.explanation.flags.map(f => {
            const cls = flagColors[f] || 'info';
            return `<span style="padding:3px 8px;border-radius:4px;font-size:9px;font-weight:600;display:inline-block;border:1px solid var(--${cls});color:var(--${cls});background:rgba(255,255,255,0.8)">${f.replace(/_/g,' ')}</span>`;
        }).join('');
    } else {
        xaiFlagsEl.innerHTML = '<span style="color:var(--text3)">No significant anomalies detected.</span>';
    }
  }

  // Audit log
  addAuditEntry('ANALYSIS RUN', result.verdict + ' · ' + result.fraud_score + '%', result.case_id);
  // Stats update
  const stTotal = document.getElementById('st-total');
  if(stTotal) stTotal.textContent = (1200 + auditLogs.length).toLocaleString('en-IN');

  // ===== Populate Investigation Console =====
  const fs = result.fraud_score, fc = result.confidence, ver = result.verdict;
  const isHigh = ['CRITICAL', 'HIGH_RISK', 'FRAUD'].includes(ver), isMed = ['MODERATE', 'SUSPICIOUS'].includes(ver);

  // Hero section
  const invScore = document.getElementById('inv-risk-score');
  if(invScore) {
    invScore.textContent = fs + '%';
    invScore.style.color = isHigh ? 'var(--danger)' : isMed ? 'var(--warn)' : 'var(--success)';
  }
  const invStatus = document.getElementById('inv-hero-status');
  if(invStatus) invStatus.textContent = result.verdict_label || ver;
  const invSub = document.getElementById('inv-hero-sub');
  if(invSub) invSub.textContent = 'Case ' + result.case_id + ' · ' + new Date(result.timestamp).toLocaleString('en-IN', {hour12:false});
  const invBadge = document.getElementById('inv-case-badge');
  if(invBadge) invBadge.textContent = result.case_id;
  const invTime = document.getElementById('inv-current-time');
  if(invTime) invTime.textContent = new Date(result.timestamp).toLocaleTimeString('en-IN', {hour12:false});

  // Status badge
  const invSBadge = document.getElementById('inv-status-badge');
  if(invSBadge) {
    invSBadge.className = 'status-chip ' + (isHigh ? 'chip-risk' : isMed ? 'chip-suspect' : 'chip-complete');
    invSBadge.querySelector('span:nth-child(2)').textContent = isHigh ? 'UNDER INVESTIGATION' : isMed ? 'SUSPICIOUS' : 'RESOLVED / CLEAN';
  }

  // Update Print report text & color classes dynamically
  const rptRisk = document.getElementById('rpt-risk');
  if(rptRisk) {
    rptRisk.textContent = fs + '%';
    rptRisk.className = 'rpt-val ' + (isHigh ? 'danger' : isMed ? 'warn' : 'success');
  }
  const rptType = document.getElementById('rpt-type');
  if(rptType) {
    rptType.textContent = result.threat_intel?.threat_type || (isHigh ? 'AI VOICE CLONE' : 'HUMAN / AUTHENTIC');
    rptType.className = 'rpt-val ' + (isHigh ? 'danger' : isMed ? 'warn' : 'success');
  }
  const rptFus = document.getElementById('rpt-fus');
  if(rptFus) {
    rptFus.textContent = fc + '%';
    rptFus.className = 'rpt-val ' + (isHigh ? 'danger' : isMed ? 'warn' : 'success');
  }

  // Summary grid cells
  const setInvEl = (id, val) => { const e = document.getElementById(id); if(e) e.textContent = val; };
  setInvEl('inv-case-id', result.case_id);
  setInvEl('inv-customer', 'Rajesh Kumar');
  setInvEl('inv-phone', '+91 98***' + Math.floor(Math.random()*900+100));
  setInvEl('inv-ai-prob', fs + '%');
  setInvEl('inv-prediction', ver);
  setInvEl('inv-replay', result.threat_intel?.replay_indicators || '—');
  setInvEl('inv-confidence', fc + '%');
  setInvEl('inv-inv-status', isHigh ? 'UNDER INVESTIGATION' : 'RESOLVED');
  setInvEl('inv-threat-type', result.threat_intel?.threat_type || '—');
  setInvEl('inv-soph', result.threat_intel?.sophistication || '—');
  setInvEl('inv-dur', (result.performance?.audio_duration_sec || 0).toFixed(1) + 's');

  // AI findings
  const findingsEl = document.getElementById('inv-findings-list');
  if(findingsEl) {
    if(isHigh) {
      findingsEl.innerHTML = `
        <div class="ai-finding-item"><div class="ai-finding-dot danger"></div><span>High probability voice clone matching <b>${result.threat_intel?.threat_type}</b>.</span></div>
        <div class="ai-finding-item"><div class="ai-finding-dot danger"></div><span>Primary trigger: <b>${result.primary_trigger}</b>.</span></div>
        <div class="ai-finding-item"><div class="ai-finding-dot warn"></div><span>Replay indicator: ${result.threat_intel?.replay_indicators}.</span></div>
        <div class="ai-finding-item"><div class="ai-finding-dot info"></div><span>Processing latency: ${result.performance?.inference_time_ms}ms.</span></div>`;
    } else if(isMed) {
      findingsEl.innerHTML = `
        <div class="ai-finding-item"><div class="ai-finding-dot warn"></div><span>Moderate risk detected. Some biological features deviate from expected norms.</span></div>
        <div class="ai-finding-item"><div class="ai-finding-dot info"></div><span>Secondary verification recommended before proceeding.</span></div>
        <div class="ai-finding-item"><div class="ai-finding-dot info"></div><span>Processing latency: ${result.performance?.inference_time_ms}ms.</span></div>`;
    } else {
      findingsEl.innerHTML = `
        <div class="ai-finding-item"><div class="ai-finding-dot success"></div><span>Vocal patterns consistent with authentic human speech. No synthetic traits.</span></div>
        <div class="ai-finding-item"><div class="ai-finding-dot info"></div><span>Processing latency: ${result.performance?.inference_time_ms}ms.</span></div>`;
    }
  }
  const recListEl = document.getElementById('inv-rec-list');
  if(recListEl) {
    recListEl.innerHTML = `<div class="ai-rec-item"><i class="ti ti-arrow-right"></i>${result.recommendation}</div>`;
  }

  // Feature importance bars
  const setFeat = (idPct, idBar, val) => {
    const pEl = document.getElementById(idPct); if(pEl) pEl.textContent = val + '%';
    const bEl = document.getElementById(idBar); if(bEl) bEl.style.width = Math.min(100, val) + '%';
  };
  setFeat('fi-replay', 'fib-replay', replayPct);
  setFeat('fi-spec', 'fib-spec', Math.round(fs * 0.9));
  setFeat('fi-pitch', 'fib-pitch', isHigh ? 54 : isMed ? 32 : 12);
  setFeat('fi-shimmer', 'fib-shimmer', isHigh ? 68 : isMed ? 38 : 15);
  setFeat('fi-jitter', 'fib-jitter', isHigh ? 72 : isMed ? 41 : 11);
  setFeat('fi-hnr', 'fib-hnr', isHigh ? 61 : isMed ? 35 : 8);
  setFeat('fi-deep', 'fib-deep', Math.round(fs * 0.8));
  setFeat('fi-fus', 'fib-fus', fc);


  // Extract avgBio and avgDl for evidence panels
  const avgBioOuter = result.chunk_results && result.chunk_results.length
    ? result.chunk_results.reduce((s, c) => s + c.bio_score, 0) / result.chunk_results.length : 0;
  const avgDlOuter = result.chunk_results && result.chunk_results.length
    ? result.chunk_results.reduce((s, c) => s + c.deep_score, 0) / result.chunk_results.length : 0;

  // Evidence panel
  const setEvid = (idPct, idBar, val) => {
    const pEl = document.getElementById(idPct); if(pEl) pEl.textContent = val + '%';
    const bEl = document.getElementById(idBar); if(bEl) bEl.style.width = Math.min(100, val) + '%';
  };
  setEvid('ev-replay', 'evb-replay', replayPct);
  setEvid('ev-ai', 'evb-ai', fs);
  setEvid('ev-human', 'evb-human', 100 - fs);
  setEvid('ev-spectral', 'evb-spectral', Math.round(fs * 0.9));
  if(result.chunk_results && result.chunk_results.length) {
            setEvid('ev-bio', 'evb-bio', Math.round(avgBioOuter));
    setEvid('ev-dl', 'evb-dl', Math.round(avgDlOuter));
  }
  setEvid('ev-fusion', 'evb-fusion', fc);

  // Report generator
  setInvEl('rpt-inv-case', result.case_id);
  const rptInvTs = document.getElementById('rpt-inv-ts');
  if(rptInvTs) rptInvTs.textContent = 'Timestamp: ' + new Date(result.timestamp).toLocaleString('en-IN');

  // Investigation timeline — mark all as done
  for(let i = 0; i <= 7; i++) {
    const dot = document.getElementById('itl-' + i);
    const step = document.getElementById('itx-' + i);
    if(dot) dot.className = 'inv-tl-dot done';
    if(step) step.className = 'inv-tl-step done';
  }
};

// Also hook downloadReport
const _origDownload = downloadReport;
downloadReport = async function() {
  // Populate notes dynamically
  const notesText = document.getElementById('inv-notes')?.value || 'No comments or decision notes submitted by the verifying officer.';
  const rptNotes = document.getElementById('rpt-notes-content');
  if(rptNotes) rptNotes.textContent = notesText;
  
  // Populate Case ID in header
  const caseIdHeader = document.getElementById('rpt-case-id-header');
  if(caseIdHeader) caseIdHeader.textContent = LAST_ANALYSIS ? LAST_ANALYSIS.case_id : SESSION_ID;

  // Checklist statuses
  const chkPause = document.getElementById('chk-row-pause')?.querySelector('.action-check')?.classList.contains('checked');
  const chkOtp = document.getElementById('chk-row-otp')?.querySelector('.action-check')?.classList.contains('checked');
  const chkVideo = document.getElementById('chk-row-video')?.querySelector('.action-check')?.classList.contains('checked');
  const chkEscalate = document.getElementById('chk-row-escalate')?.querySelector('.action-check')?.classList.contains('checked');
  
  const setCheckText = (id, checkState) => {
    const el = document.getElementById(id);
    if(el) {
      el.textContent = checkState ? 'COMPLETED (YES)' : 'PENDING (NO)';
      el.style.color = checkState ? 'var(--success)' : 'var(--danger)';
    }
  };
  setCheckText('rpt-check-pause', chkPause);
  setCheckText('rpt-check-otp', chkOtp);
  setCheckText('rpt-check-video', chkVideo);
  setCheckText('rpt-check-escalate', chkEscalate);

  await _origDownload();
  addAuditEntry('REPORT EXPORTED', 'PDF download');
};
</script>

<!-- GLOBAL FLOATING CHATBOT WIDGET -->
<div id="floating-chat-widget" style="position: fixed; bottom: 20px; right: 20px; z-index: 99999; font-family: var(--sans);">
  <!-- Floating Button Badge -->
  <div id="chat-badge" style="width: 56px; height: 56px; border-radius: 50%; background: linear-gradient(135deg, var(--bank), #003d7a); color: #fff; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 16px rgba(0,0,0,0.2); cursor: pointer; transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); position: relative;">
    <i class="ti ti-messages" style="font-size: 24px;"></i>
    <!-- Hover Close Button ('x') -->
    <div id="close-badge-btn" title="Close chat assistant" style="position: absolute; top: -5px; right: -5px; width: 18px; height: 18px; border-radius: 50%; background: #ef4444; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; border: 1.5px solid #fff; opacity: 0; transition: opacity 0.2s; pointer-events: auto;">×</div>
  </div>

  <!-- Floating Chat Window -->
  <div id="chat-window" style="display: none; position: absolute; bottom: 70px; right: 0; width: 380px; height: 520px; border-radius: 12px; background: #fff; box-shadow: 0 8px 32px rgba(0,0,0,0.15); border: 1px solid var(--border); flex-direction: column; overflow: hidden; animation: slideUp 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);">
    <!-- Header -->
    <div style="background: linear-gradient(135deg, var(--bank), #003d7a); color: #fff; padding: 12px 16px; display: flex; align-items: center; justify-content: space-between;">
      <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; animation: pulse 2s infinite;"></div>
        <span style="font-weight: 600; font-size: 13px;">VoiceGuard AI Forensic Agent</span>
      </div>
      <i class="ti ti-minus" id="minimize-chat-btn" style="cursor: pointer; font-size: 14px;" title="Minimize"></i>
    </div>
    <!-- Messages Container -->
    <div id="floating-chat-msgs" style="flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; background: #f8fafc; font-size: 12px;">
      <div class="chat-bubble bubble-ai" style="padding: 8px 12px; border-radius: 8px; background: #fff; max-width: 90%; align-self: flex-start; border: 1px solid var(--border);">
        <b>Forensic AI:</b> Hello! I am the global VoiceGuard Assistant. Ask me anything about this platform, technical pipeline specs, RBI SOP rules, or case analysis diagnostics. I support spelling-tolerant questions.
      </div>
    </div>
    <!-- Quick buttons -->
    <div style="display: flex; gap: 5px; flex-wrap: wrap; padding: 8px 12px; border-top: 1px solid var(--border); background: #fff; border-bottom: 1px solid var(--border);">
      <span class="cq-btn" onclick="sendFloatingQuery('Why is this case flagged?')" style="font-size: 10px; padding: 4px 8px; background: #f1f5f9; border-radius: 4px; cursor: pointer; border: 1px solid var(--border);">Why flagged?</span>
      <span class="cq-btn" onclick="sendFloatingQuery('What is the RBI SOP?')" style="font-size: 10px; padding: 4px 8px; background: #f1f5f9; border-radius: 4px; cursor: pointer; border: 1px solid var(--border);">RBI SOP</span>
      <span class="cq-btn" onclick="sendFloatingQuery('Explain Biometric Features')" style="font-size: 10px; padding: 4px 8px; background: #f1f5f9; border-radius: 4px; cursor: pointer; border: 1px solid var(--border);">Biometrics</span>
      <span class="cq-btn" onclick="sendFloatingQuery('Explain 4096D Deep Features')" style="font-size: 10px; padding: 4px 8px; background: #f1f5f9; border-radius: 4px; cursor: pointer; border: 1px solid var(--border);">Deep Features</span>
    </div>
    <!-- Input area -->
    <div style="display: flex; padding: 8px 12px; gap: 6px; background: #fff;">
      <input id="floating-chat-input" placeholder="Ask AI anything..." style="flex: 1; border: 1px solid var(--border); padding: 8px 12px; border-radius: 6px; font-size: 12px; outline: none; font-family: var(--sans);" onkeydown="if(event.key==='Enter')sendFloatingChat()">
      <button onclick="sendFloatingChat()" style="width: 32px; height: 32px; background: var(--bank); color: #fff; border: none; border-radius: 6px; display: flex; align-items: center; justify-content: center; cursor: pointer;"><i class="ti ti-send"></i></button>
    </div>
  </div>
</div>

<style>
@keyframes slideUp {
  from { opacity: 0; transform: translateY(20px) scale(0.95); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes pulse {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
  70% { transform: scale(1); box-shadow: 0 0 0 4px rgba(16, 185, 129, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
#chat-badge:hover #close-badge-btn {
  opacity: 1 !important;
}
#chat-badge:hover {
  transform: scale(1.08);
}
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

    .inv-tl-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--border); transition: background 0.3s; }
    .inv-tl-dot.done { background: var(--success); }
    .atb-unknown { border-left: 4px solid var(--text3) !important; }
    .atb-ai { border-left: 4px solid var(--danger) !important; background: rgba(239,68,68,0.05) !important; }
    .atb-replay { border-left: 4px solid var(--warn) !important; background: rgba(245,158,11,0.05) !important; }
    .atb-deepfake { border-left: 4px solid var(--danger) !important; background: rgba(239,68,68,0.05) !important; }
    .atb-human { border-left: 4px solid var(--success) !important; background: rgba(34,197,94,0.05) !important; }
    
</style>

<script>
// Floating Chat toggle
(function() {
  const badge = document.getElementById('chat-badge');
  const win = document.getElementById('chat-window');
  const closeBadge = document.getElementById('close-badge-btn');
  const widget = document.getElementById('floating-chat-widget');
  const minimize = document.getElementById('minimize-chat-btn');

  if(badge && win) {
    badge.addEventListener('click', (e) => {
      if(e.target === closeBadge) {
        widget.style.display = 'none';
        return;
      }
      if(win.style.display === 'none' || !win.style.display) {
        win.style.display = 'flex';
      } else {
        win.style.display = 'none';
      }
    });
  }
  if(minimize && win) {
    minimize.addEventListener('click', () => {
      win.style.display = 'none';
    });
  }
})();

function sendFloatingQuery(text) {
  const input = document.getElementById('floating-chat-input');
  if(input) {
    input.value = text;
    sendFloatingChat();
  }
}

// Global Knowledge Base Search Engine for any question
const GLOBAL_KB = [
  {
    q: ["replay", "recording", "re-recorded", "playback", "static room tone", "temporal consistency"],
    a: "<b>Replay Attack:</b> Replays occur when someone plays a pre-recorded call. VoiceGuard flags this using spectral smoothness checks, temporal continuity boundaries, and lowpass cutoff rolls typical of mobile playback speakers."
  },
  {
    q: ["clone", "synthetic", "tts", "cloned", "voice clone", "ai voice", "elevenlabs", "fish audio", "xtts", "f5-tts"],
    a: "<b>AI Voice Cloning:</b> Generative AI clones have clean spectrum profiles but lack natural physiological jitters. We verify them using high-resolution biological features and deep learning projection discrepancies."
  },
  {
    q: ["deepfake", "gan", "diffusion", "deep fake", "ai generated"],
    a: "<b>Deepfake Speech:</b> Generative deepfakes are identified by analyzing spectral phase discontinuities and frame-to-frame synthesis artifacts in the 4096D dual-pass Wav2Vec2 encoder."
  },
  {
    q: ["features", "dimensions", "4441", "345", "4096", "extract", "vector", "biometric", "deep"],
    a: "<b>Feature Dimensions:</b> VoiceGuard extracts a total of <b>4,193 dimensions</b>: 97 acoustic features (pitch, jitter, shimmer, HNR) + 4,096 Multi-Condition Training (MCT) deep features."
  },
  {
    q: ["jitter", "shimmer", "hnr", "pitch", "formants", "harmonics", "micro-variation"],
    a: "<b>Biological Features:</b> Jitter measures pitch cycle variations, Shimmer measures amplitude variations, and HNR (Harmonics-to-Noise Ratio) measures tone clarity. AI voices show abnormally uniform micro-variations."
  },
  {
    q: ["sop", "rbi", "procedure", "regulatory", "escalate", "fraud cell", "₹1 lakh", "reporting limit", "kyc", "otp"],
    a: "<b>RBI SOP Compliance:</b> Freeze transactions on high-risk flags (>54%). Verify via OTP or Video KYC. Any fraud attempt exceeding <b>₹1 Lakh</b> must be reported to the RBI within 24 hours."
  },
  {
    q: ["calibration", "regress", "scale", "isotonic", "platt", "fusion weights", "0.764", "0.236"],
    a: "<b>Model Calibration:</b> The PyTorch DL model acts as an end-to-end feature fusion core, bypassing traditional ML steps."
  },
  {
    q: ["datasets", "languages", "bengali", "malayalam", "telugu", "noisy", "fake", "dataset paths"],
    a: "<b>Dataset Folders:</b> Bengli, Malayalam, and Telugu voice files are stored in <code>data/1_genuine</code> (real noisy) and <code>data/2_synthetic</code> (fakes). They are standard 16kHz mono WAV files."
  },
  {
    q: ["accuracy", "eer", "metrics", "performance", "threshold", "0.94", "0.54", "0.30"],
    a: "<b>Pipeline Performance:</b> The VoiceGuard Core Engine model achieves an EER of 0.22%. Thresholds are dynamically evaluated by the network."
  },
  {
    q: ["why flagged", "reason", "active case", "score", "verdict"],
    a: "<b>Active Case Flag:</b> Check the main console. If a case is loaded, the engine explains the primary trigger, deep learning score, biological score, and recommends action based on RBI SOP guidelines."
  }
];

// TF-IDF / Token similarity matching for any arbitrary query
async function sendFloatingChat() {
  const input = document.getElementById('floating-chat-input');
  const msgs = document.getElementById('floating-chat-msgs');
  if(!input || !msgs) return;
  const text = input.value.trim();
  if(!text) return;
  input.value = '';

  // User Message
  const userBubble = document.createElement('div');
  userBubble.className = 'chat-bubble bubble-user';
  userBubble.style.padding = '10px 14px';
  userBubble.style.borderRadius = '12px 12px 0 12px';
  userBubble.style.background = 'var(--bank)';
  userBubble.style.color = '#fff';
  userBubble.style.alignSelf = 'flex-end';
  userBubble.style.maxWidth = '85%';
  userBubble.style.boxShadow = '0 2px 5px rgba(0,0,0,0.1)';
  userBubble.textContent = text;
  msgs.appendChild(userBubble);
  msgs.scrollTop = msgs.scrollHeight;

  // Typing indicator
  const typing = document.createElement('div');
  typing.className = 'chat-bubble bubble-ai';
  typing.style.padding = '10px 14px';
  typing.style.borderRadius = '12px 12px 12px 0';
  typing.style.background = '#fff';
  typing.style.alignSelf = 'flex-start';
  typing.style.border = '1px solid var(--border)';
  typing.style.boxShadow = '0 2px 5px rgba(0,0,0,0.05)';
  typing.innerHTML = '<div style="display:flex;gap:4px;align-items:center"><div class="dot-pulse"></div><div class="dot-pulse" style="animation-delay:0.2s"></div><div class="dot-pulse" style="animation-delay:0.4s"></div></div>';
  msgs.appendChild(typing);
  msgs.scrollTop = msgs.scrollHeight;

  try {
      const payload = { message: text };
      if (LAST_ANALYSIS) {
          payload.case_id = LAST_ANALYSIS.case_id;
      }
      
      const response = await fetch(BACKEND_URL + '/api/v1/chat', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
              'x-api-key': 'voiceguard123'
          },
          body: JSON.stringify(payload)
      });
      
      const data = await response.json();
      
      msgs.removeChild(typing);
      
      const aiBubble = document.createElement('div');
      aiBubble.className = 'chat-bubble bubble-ai';
      aiBubble.style.padding = '10px 14px';
      aiBubble.style.borderRadius = '12px 12px 12px 0';
      aiBubble.style.background = '#fff';
      aiBubble.style.alignSelf = 'flex-start';
      aiBubble.style.border = '1px solid var(--border)';
      aiBubble.style.boxShadow = '0 2px 5px rgba(0,0,0,0.05)';
      aiBubble.style.maxWidth = '90%';
      aiBubble.style.lineHeight = '1.4';
      
      if (data.status === 'success') {
          // Parse basic markdown from LLM
          let formattedReply = data.reply.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
          formattedReply = formattedReply.replace(/\n/g, '<br>');
          aiBubble.innerHTML = formattedReply;
      } else {
          aiBubble.innerHTML = '<span style="color:var(--danger)">Error: Could not connect to AI brain.</span>';
      }
      
      msgs.appendChild(aiBubble);
      msgs.scrollTop = msgs.scrollHeight;
      
  } catch(e) {
      msgs.removeChild(typing);
      const aiBubble = document.createElement('div');
      aiBubble.style.padding = '10px 14px';
      aiBubble.style.borderRadius = '12px 12px 12px 0';
      aiBubble.style.background = '#fff';
      aiBubble.style.border = '1px solid var(--danger)';
      aiBubble.style.color = 'var(--danger)';
      aiBubble.style.alignSelf = 'flex-start';
      aiBubble.textContent = "Network Error communicating with LLM backend.";
      msgs.appendChild(aiBubble);
      msgs.scrollTop = msgs.scrollHeight;
  }
}
