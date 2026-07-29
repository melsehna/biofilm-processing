"""
Self-contained interactive HTML viewer for the UMAP grid: hover tooltips,
nn/min_dist toggle buttons, search, pan/zoom/box-zoom, and click-to-play
overlay MP4 in a sidebar.

Uses a generic categorical label rather than the original viewer's dataset-specific
gene-ontology highlight sets.

Labels, plate ids and MP4 paths are untrusted (user-supplied `*layout*.csv`,
directory names): escaped on the way in by `_jsonForScript` and again in the
template by `esc()` before every innerHTML write.
"""
import html as htmlLib
import json
import os


def _jsonForScript(obj):
    """json.dumps hardened for embedding in a <script> block.

    json.dumps does not escape '<', so a label containing '</script>' would close
    the script element. The \\uXXXX forms stay valid JSON but are inert to the HTML
    tokenizer. U+2028/29 are raw line terminators in JS source.
    """
    return (json.dumps(obj)
            .replace('<', '\\u003c')
            .replace('>', '\\u003e')
            .replace('&', '\\u0026')
            .replace('\u2028', '\\u2028')
            .replace('\u2029', '\\u2029'))


def _buildPointsByParam(embeddings, labels, mp4Column):
    """Group embeddings by (nn, md) and serialize to a dict-of-lists for JS."""
    paramCombos = []
    for col in embeddings.columns:
        if not col.startswith('umap_x_nn'):
            continue
        # 'umap_x_nn20_md0.2'
        rest = col[len('umap_x_nn'):]
        nnStr, _, mdStr = rest.partition('_md')
        try:
            nn, md = int(nnStr), float(mdStr)
        except ValueError:
            continue
        if (nn, md) not in paramCombos:
            paramCombos.append((nn, md))
    paramCombos.sort()

    pointsByParam = {}
    for nn, md in paramCombos:
        xCol = f'umap_x_nn{nn}_md{md}'
        yCol = f'umap_y_nn{nn}_md{md}'
        pts = []
        for _, row in embeddings.iterrows():
            wellId = str(row['wellId'])
            plateId = str(row['plateId'])
            # tuple key preferred for multi-plate runs; fall back to bare wellId
            label = labels.get((plateId, wellId)) or labels.get(wellId, '')
            mp4 = str(row[mp4Column]) if mp4Column in embeddings.columns else ''
            pts.append({
                'x': round(float(row[xCol]), 4),
                'y': round(float(row[yCol]), 4),
                'plate': str(row['plateId']),
                'well': wellId,
                'label': label,
                'mp4': mp4,
            })
        pointsByParam[f'{nn}_{md}'] = pts
    return pointsByParam, paramCombos


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><title>__TITLE__</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Helvetica Neue', Arial, sans-serif; background: #1a1a2e; color: #eee; display: flex; flex-direction: column; height: 100vh; }
  #controls { display: flex; align-items: center; gap: 20px; padding: 12px 20px; background: #16213e; border-bottom: 1px solid #334; flex-wrap: wrap; flex-shrink: 0; }
  #controls label { font-size: 14px; color: #99a; }
  .param-group { display: flex; align-items: center; gap: 8px; }
  .param-btn { background: #2a2a4a; border: 1px solid #445; border-radius: 5px; color: #ccc; padding: 6px 14px; font-size: 14px; cursor: pointer; }
  .param-btn:hover { background: #3a3a5a; }
  .param-btn.active { background: #4a6fa5; border-color: #6a9fd5; color: #fff; }
  #search-box { display: flex; align-items: center; gap: 8px; margin-left: auto; }
  #search-box input { background: #2a2a4a; border: 1px solid #445; border-radius: 5px; color: #eee; font-size: 14px; padding: 6px 12px; width: 220px; outline: none; }
  #search-box input::placeholder { color: #667; }
  #search-box .count { font-size: 13px; color: #889; }
  #main { flex: 1; display: flex; overflow: hidden; }
  #plot-container { flex: 1; position: relative; overflow: hidden; }
  canvas { display: block; width: 100%; height: 100%; }
  #tooltip { display: none; position: absolute; pointer-events: none; background: rgba(20,20,40,0.95); border: 1px solid #556; border-radius: 6px; padding: 8px 12px; font-size: 13px; line-height: 1.5; max-width: 360px; z-index: 10; }
  #tooltip .label { font-weight: bold; font-size: 15px; color: #7ecfff; }
  #tooltip .meta { color: #aab; font-size: 12px; }
  #legend { display: none; position: absolute; bottom: 14px; left: 14px; z-index: 5; background: rgba(20,20,40,0.9); border: 1px solid #556; border-radius: 6px; padding: 10px 14px; max-height: 50vh; overflow-y: auto; font-size: 13px; }
  .legend-item { display: flex; align-items: center; gap: 8px; padding: 2px 0; }
  .swatch { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
  #info { position: absolute; top: 10px; right: 14px; font-size: 12px; color: #556; z-index: 5; }
  #sidebar { width: 440px; background: #16213e; border-left: 1px solid #334; display: flex; flex-direction: column; overflow: hidden; }
  #sidebar.hidden { display: none; }
  #video-header { padding: 12px 16px; border-bottom: 1px solid #334; font-size: 13px; position: relative; }
  #video-header .label { font-weight: bold; font-size: 15px; color: #7ecfff; }
  #video-header .meta { color: #aab; font-size: 12px; }
  #video-close { position: absolute; top: 8px; right: 12px; background: none; border: none; color: #889; font-size: 20px; cursor: pointer; padding: 4px; }
  #video-close:hover { color: #eee; }
  #video-wrap { flex: 1; display: flex; align-items: center; justify-content: center; background: #000; }
  #video-wrap video { max-width: 100%; max-height: 100%; }
  #no-video { color: #667; font-size: 14px; text-align: center; padding: 20px; }
</style></head><body>
<div id="controls">
  <div class="param-group"><label>n_neighbors:</label><div id="nn-btns"></div></div>
  <div class="param-group"><label>min_dist:</label><div id="md-btns"></div></div>
  <div id="search-box">
    <input id="search" type="text" placeholder="Search label, well, or plate..." />
    <span class="count" id="search-count"></span>
  </div>
</div>
<div id="main">
  <div id="plot-container">
    <canvas id="canvas"></canvas>
    <div id="tooltip"></div>
    <div id="legend"></div>
    <div id="info">scroll=zoom · drag=pan · shift-drag=box zoom · dblclick=reset · click=video</div>
  </div>
  <div id="sidebar" class="hidden">
    <div id="video-header"></div>
    <button id="video-close">&times;</button>
    <div id="video-wrap"><div id="no-video">Click a point to view overlay</div></div>
  </div>
</div>
<script>
const allData = __DATA__;
const paramCombos = __PARAMS__;
// Data is untrusted — escape before every innerHTML write.
const ESC_MAP = {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'};
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ESC_MAP[c]);
// relative overlay paths only; reject scheme-like or protocol-relative
const safeMp4 = s => {
  const v = String(s ?? '');
  if (!v || /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(v) || v.startsWith('//')) return '';
  return encodeURI(v).replace(/"/g, '%22');
};
const WT_LABEL = 'WT';
const LABEL_COLOR_LIMIT = 80;

const nnValues = [...new Set(paramCombos.map(p => p[0]))].sort((a,b) => a-b);
const mdValues = [...new Set(paramCombos.map(p => p[1]))].sort((a,b) => a-b);
let currentNn = nnValues[Math.floor(nnValues.length / 2)] || nnValues[0];
let currentMd = mdValues[Math.floor(mdValues.length / 2)] || mdValues[0];
let points = [];
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const sidebar = document.getElementById('sidebar');
const videoHeader = document.getElementById('video-header');
const videoWrap = document.getElementById('video-wrap');
const searchInput = document.getElementById('search');
const searchCount = document.getElementById('search-count');
let dpr = window.devicePixelRatio || 1;
let W, H, vx, vy, vw, vh;
let searchTerm = '', highlightSet = null;

function labelColor(idx, total, alpha) {
  const hue = (idx * 360 / total) % 360;
  return `hsla(${hue},${70+(idx%3)*10}%,${55+(idx%2)*10}%,${alpha})`;
}
function loadPoints() {
  points = allData[currentNn + '_' + currentMd] || [];
  resetView(); applySearch(); draw();
}
function resetView() {
  if (!points.length) return;
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
  const dw = (xMax-xMin)||1, dh = (yMax-yMin)||1, pad = 0.05;
  vx = xMin-pad*dw; vy = yMin-pad*dh; vw = dw*(1+2*pad); vh = dh*(1+2*pad);
}
function worldToScreen(wx, wy) { return [(wx-vx)/vw*W, (1-(wy-vy)/vh)*H]; }
function screenToWorld(sx, sy) { return [sx/W*vw+vx, (1-sy/H)*vh+vy]; }

function draw() {
  if (!W) return;
  ctx.clearRect(0, 0, W, H);
  const r = Math.max(1.8, Math.min(5, 2400/Math.sqrt(points.length || 1)));
  const searchActive = highlightSet !== null;
  const visibleLabels = new Set(), visibleIdx = [];
  for (let i = 0; i < points.length; i++) {
    const [sx, sy] = worldToScreen(points[i].x, points[i].y);
    if (sx >= -5 && sx <= W+5 && sy >= -5 && sy <= H+5) { visibleIdx.push(i); if (points[i].label) visibleLabels.add(points[i].label); }
  }
  const colorByLabel = !searchActive && visibleLabels.size > 0 && visibleLabels.size <= LABEL_COLOR_LIMIT;
  const labelList = colorByLabel ? [...visibleLabels].filter(l => l !== WT_LABEL).sort() : [];
  const labelToIdx = {};
  labelList.forEach((l, i) => labelToIdx[l] = i);

  for (const i of visibleIdx) {
    const p = points[i];
    if (searchActive && highlightSet.has(i)) continue;
    const [sx, sy] = worldToScreen(p.x, p.y);
    ctx.beginPath(); ctx.arc(sx, sy, r, 0, Math.PI*2);
    if (searchActive) ctx.fillStyle = 'rgba(120,120,160,0.18)';
    else if (p.label === WT_LABEL) ctx.fillStyle = 'rgba(0,0,0,0.85)';
    else if (colorByLabel && p.label) ctx.fillStyle = labelColor(labelToIdx[p.label], Math.max(labelList.length, 1), 0.78);
    else if (p.label) ctx.fillStyle = 'rgba(100,180,255,0.55)';
    else ctx.fillStyle = 'rgba(160,160,170,0.45)';
    ctx.fill();
  }
  if (searchActive) {
    for (const i of highlightSet) {
      const p = points[i]; const [sx, sy] = worldToScreen(p.x, p.y);
      ctx.beginPath(); ctx.arc(sx, sy, r*1.6, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(50,220,120,0.95)'; ctx.fill();
    }
  }
  const legend = document.getElementById('legend');
  if (colorByLabel && !searchActive) {
    let h = '';
    if (visibleLabels.has(WT_LABEL)) h += `<div class="legend-item"><span class="swatch" style="background:#000"></span>${WT_LABEL}</div>`;
    for (let i = 0; i < labelList.length; i++) h += `<div class="legend-item"><span class="swatch" style="background:${labelColor(i,Math.max(labelList.length,1),1)}"></span>${esc(labelList[i])}</div>`;
    legend.innerHTML = h; legend.style.display = 'block';
  } else legend.style.display = 'none';
}
function findNearest(sx, sy, maxDist) {
  let best = -1, bestD2 = maxDist*maxDist;
  for (let i = 0; i < points.length; i++) {
    const [px, py] = worldToScreen(points[i].x, points[i].y);
    const d2 = (px-sx)**2+(py-sy)**2;
    if (d2 < bestD2) { bestD2 = d2; best = i; }
  }
  return best;
}
function applySearch() {
  if (!searchTerm) { highlightSet = null; searchCount.textContent = ''; return; }
  highlightSet = new Set();
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    if ((p.label && p.label.toLowerCase().includes(searchTerm))
        || p.well.toLowerCase().includes(searchTerm)
        || p.plate.toLowerCase().includes(searchTerm)) highlightSet.add(i);
  }
  searchCount.textContent = highlightSet.size + ' hits';
}
function tooltipHtml(p) {
  let h = '';
  if (p.label) h += `<div class="label">${esc(p.label)}</div>`;
  h += `<div class="meta">Plate: ${esc(p.plate)} · Well: ${esc(p.well)}</div>`;
  return h;
}
function showVideo(p) {
  sidebar.classList.remove('hidden');
  videoHeader.innerHTML = tooltipHtml(p);
  const mp4 = safeMp4(p.mp4);
  if (mp4) videoWrap.innerHTML = `<video controls autoplay loop muted><source src="${mp4}" type="video/mp4">Cannot load video</video>`;
  else videoWrap.innerHTML = '<div id="no-video">No overlay available</div>';
  resize();
}
document.getElementById('video-close').addEventListener('click', () => { sidebar.classList.add('hidden'); resize(); });
function buildButtons() {
  const nnC = document.getElementById('nn-btns'), mdC = document.getElementById('md-btns');
  nnValues.forEach(nn => {
    const btn = document.createElement('button');
    btn.className = 'param-btn' + (nn === currentNn ? ' active' : '');
    btn.textContent = nn; btn.onclick = () => { currentNn = nn; updateButtons(); loadPoints(); };
    nnC.appendChild(btn);
  });
  mdValues.forEach(md => {
    const btn = document.createElement('button');
    btn.className = 'param-btn' + (md === currentMd ? ' active' : '');
    btn.textContent = md; btn.onclick = () => { currentMd = md; updateButtons(); loadPoints(); };
    mdC.appendChild(btn);
  });
}
function updateButtons() {
  document.querySelectorAll('#nn-btns .param-btn').forEach((btn,i) => btn.classList.toggle('active', nnValues[i]===currentNn));
  document.querySelectorAll('#md-btns .param-btn').forEach((btn,i) => btn.classList.toggle('active', mdValues[i]===currentMd));
}
let mode = 'none', dragX, dragY, boxX0, boxY0, boxX1, boxY1;
canvas.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX-rect.left, sy = e.clientY-rect.top;
  if (mode === 'boxzoom') {
    boxX1 = sx; boxY1 = sy; draw();
    ctx.strokeStyle = 'rgba(120,210,255,0.8)'; ctx.lineWidth = 1.5; ctx.setLineDash([6,3]);
    ctx.strokeRect(Math.min(boxX0,boxX1), Math.min(boxY0,boxY1), Math.abs(boxX1-boxX0), Math.abs(boxY1-boxY0));
    ctx.fillStyle = 'rgba(120,210,255,0.08)';
    ctx.fillRect(Math.min(boxX0,boxX1), Math.min(boxY0,boxY1), Math.abs(boxX1-boxX0), Math.abs(boxY1-boxY0));
    ctx.setLineDash([]); return;
  }
  if (mode === 'pan') { vx -= (e.clientX-dragX)/W*vw; vy += (e.clientY-dragY)/H*vh; dragX = e.clientX; dragY = e.clientY; draw(); return; }
  const idx = findNearest(sx, sy, 12);
  if (idx >= 0) {
    tooltip.innerHTML = tooltipHtml(points[idx]); tooltip.style.display = 'block';
    let tx = sx+14, ty = sy-10;
    if (tx+tooltip.offsetWidth > W) tx = sx-tooltip.offsetWidth-10;
    if (ty < 0) ty = 4;
    if (ty+tooltip.offsetHeight > H) ty = H-tooltip.offsetHeight-4;
    tooltip.style.left = tx+'px'; tooltip.style.top = ty+'px'; canvas.style.cursor = 'pointer';
  } else { tooltip.style.display = 'none'; canvas.style.cursor = e.shiftKey ? 'crosshair' : 'default'; }
});
canvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX-rect.left, sy = e.clientY-rect.top;
  if (e.shiftKey) { mode = 'boxzoom'; boxX0=sx; boxY0=sy; boxX1=sx; boxY1=sy; }
  else { mode = 'pan'; dragX = e.clientX; dragY = e.clientY; }
});
window.addEventListener('mouseup', e => {
  if (mode === 'boxzoom') {
    const bw = Math.abs(boxX1-boxX0), bh = Math.abs(boxY1-boxY0);
    if (bw > 5 && bh > 5) {
      const [wx0,wy0] = screenToWorld(Math.min(boxX0,boxX1), Math.max(boxY0,boxY1));
      const [wx1,wy1] = screenToWorld(Math.max(boxX0,boxX1), Math.min(boxY0,boxY1));
      vx=wx0; vy=wy0; vw=wx1-wx0; vh=wy1-wy0;
    }
    mode = 'none'; draw(); return;
  }
  if (mode === 'pan') {
    const moved = Math.abs(e.clientX-dragX) + Math.abs(e.clientY-dragY);
    if (moved < 4) {
      const rect = canvas.getBoundingClientRect();
      const idx = findNearest(e.clientX-rect.left, e.clientY-rect.top, 12);
      if (idx >= 0) showVideo(points[idx]);
    }
    mode = 'none';
  }
});
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX-rect.left, sy = e.clientY-rect.top;
  const [wx,wy] = screenToWorld(sx, sy);
  const factor = e.deltaY > 0 ? 1.15 : 1/1.15;
  vw *= factor; vh *= factor;
  vx = wx-sx/W*vw; vy = wy-(1-sy/H)*vh; draw();
}, { passive: false });
canvas.addEventListener('dblclick', () => { resetView(); draw(); });
searchInput.addEventListener('input', e => { searchTerm = e.target.value.trim().toLowerCase(); applySearch(); draw(); });
function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  W = rect.width; H = rect.height;
  canvas.width = W*dpr; canvas.height = H*dpr;
  canvas.style.width = W+'px'; canvas.style.height = H+'px';
  ctx.setTransform(dpr,0,0,dpr,0,0); draw();
}
window.addEventListener('resize', resize);
buildButtons(); loadPoints(); resize();
</script></body></html>
"""


def writeInteractiveHtml(embeddings, labels, outPath, title='UMAP', mp4Column='mp4'):
    """Write a self-contained interactive HTML viewer for the UMAP grid.

    Parameters
    ----------
    embeddings : pd.DataFrame
        Output of fitUmapGrid — must have plateId, wellId, and one
        umap_x_nn{nn}_md{md} / umap_y_nn{nn}_md{md} pair per fitted (nn, md).
    labels : dict[str, str]
        wellId -> label, from loadLabels(). Empty dict is fine.
    outPath : str
    title : str
        HTML <title>.
    mp4Column : str
        Column in `embeddings` containing per-row overlay MP4 paths
        (relative to outPath's directory). If absent, no videos play.
    """
    pointsByParam, paramCombos = _buildPointsByParam(embeddings, labels, mp4Column)
    html = (_HTML_TEMPLATE
            .replace('__TITLE__', htmlLib.escape(str(title), quote=True))
            .replace('__DATA__', _jsonForScript(pointsByParam))
            .replace('__PARAMS__', _jsonForScript(paramCombos)))
    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    with open(outPath, 'w') as f:
        f.write(html)
    return outPath
