/**
 * MoCoLUS POCUS Trainer — Web Frontend
 * Professional clinical interface for point-of-care lung ultrasound training.
 */

// ---------------------------------------------------------------------------
// WebSocket Manager
// ---------------------------------------------------------------------------

class WebSocketManager {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.onMessage = null;
    this.reconnectDelay = 1000;
    this.maxReconnect = 10000;
    this.connect();
  }

  connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    this.ws = new WebSocket(`${proto}//${location.host}${this.url}`);

    this.ws.onopen = () => {
      document.getElementById("ws-status").className = "status-dot connected";
      this.reconnectDelay = 1000;
    };

    this.ws.onclose = () => {
      document.getElementById("ws-status").className = "status-dot disconnected";
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, this.maxReconnect);
    };

    this.ws.onerror = () => {};

    this.ws.onmessage = (evt) => {
      if (this.onMessage) {
        try { this.onMessage(JSON.parse(evt.data)); }
        catch (e) { console.error("WS parse error:", e); }
      }
    };
  }

  send(msg) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }
}

// ---------------------------------------------------------------------------
// B-Mode Renderer — high-quality canvas rendering
// ---------------------------------------------------------------------------

class BmodeRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");
    this.img = new window.Image();
    this.currentData = null;
    // Probe orientation for image transform
    this.probeYaw = 0;
    this.probePitch = 0;
    this.probeRoll = 0;
  }

  setProbeOrientation(yaw, pitch, roll) {
    this.probeYaw = yaw;
    this.probePitch = pitch;
    this.probeRoll = roll;
  }

  drawFrame(base64Data, meta) {
    this.currentData = meta;
    this.img.onload = () => {
      const c = this.canvas;
      const ctx = this.ctx;
      const w = c.width, h = c.height;

      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, w, h);

      const m = { l: 40, r: 8, t: 22, b: 30 };
      const iw = w - m.l - m.r;
      const ih = h - m.t - m.b;
      const icx = m.l + iw / 2;
      const icy = m.t + ih / 2;

      // Apply probe orientation transform to the US image
      // Yaw: slight rotation of the scan plane
      // Pitch: vertical shift (probe tilt toward/away from skin)
      // Roll: lateral shift (rocking the probe)
      ctx.save();
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";

      const yawRad = this.probeYaw * Math.PI / 180 * 0.015;  // subtle rotation
      const pitchShift = this.probePitch * ih * 0.002;         // vertical shift
      const rollShift = this.probeRoll * iw * 0.003;           // lateral shift

      // Clip to image region
      ctx.beginPath();
      ctx.rect(m.l, m.t, iw, ih);
      ctx.clip();

      ctx.translate(icx + rollShift, icy + pitchShift);
      ctx.rotate(yawRad);
      ctx.translate(-icx, -icy);
      ctx.drawImage(this.img, m.l, m.t, iw, ih);
      ctx.restore();

      // Sector outline (curvilinear probe shape)
      ctx.strokeStyle = "rgba(120, 180, 220, 0.7)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      const ti = iw * 0.12;
      ctx.moveTo(m.l + ti, m.t);
      ctx.lineTo(m.l + iw - ti, m.t);
      ctx.lineTo(m.l + iw, m.t + ih);
      ctx.lineTo(m.l, m.t + ih);
      ctx.closePath();
      ctx.stroke();

      // Depth markers
      ctx.fillStyle = "#8898a8";
      ctx.font = "13px 'SF Mono', Consolas, monospace";
      ctx.textAlign = "right";
      for (let d = 0; d <= 12; d += 2) {
        const y = m.t + (d / 12) * ih;
        ctx.fillText(`${d}`, m.l - 4, y + 3);
        ctx.strokeStyle = "#1e2838";
        ctx.lineWidth = 0.4;
        ctx.beginPath();
        ctx.moveTo(m.l - 2, y);
        ctx.lineTo(m.l, y);
        ctx.stroke();
      }
      ctx.textAlign = "left";
      ctx.fillText("cm", m.l - 14, m.t - 5);

      if (meta) {
        ctx.fillStyle = "#506070";
        ctx.textAlign = "right";
        ctx.font = "12px 'SF Mono', Consolas, monospace";
        ctx.fillText(`${meta.frame_idx}/${meta.n_frames}`, w - 4, 14);
      }
    };
    this.img.src = "data:image/jpeg;base64," + base64Data;
  }
}

// ---------------------------------------------------------------------------
// M-Mode Renderer
// ---------------------------------------------------------------------------

class MmodeRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");
    this.img = new window.Image();
  }

  drawFrame(base64Data) {
    this.img.onload = () => {
      const ctx = this.ctx;
      const w = this.canvas.width, h = this.canvas.height;
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, w, h);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(this.img, 0, 0, w, h);
    };
    this.img.src = "data:image/jpeg;base64," + base64Data;
  }
}

// ---------------------------------------------------------------------------
// Probe Overlay Renderer
// ---------------------------------------------------------------------------

class ProbeOverlayRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");

    this.activeZone = null;
    this.yaw = 0;
    this.pitch = 0;
    this.roll = 0;
    this.imuConnected = false;
    this.zones = {};
    this.examinedZones = new Set();
    this._dirty = true;

    // Free probe position (normalized 0-1)
    this.probeNx = 0.5;
    this.probeNy = 0.25;
    this.probeVisible = false; // shown after first click/zone select

    // Beam-target: where the angled scan plane projects onto the chest surface.
    // Equals (probeNx, probeNy) when the probe is held perpendicular; offset by
    // tan(tilt) * scan_depth when pitched/rolled. Drives which zone(s) the
    // server-side Gaussian blender weights heaviest.
    this.beamTargetNx = this.probeNx;
    this.beamTargetNy = this.probeNy;

    this.isDragging = false;
    this.dragMode = null; // 'move' or 'rotate'
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.dragStartYaw = 0;
    this.dragStartPitch = 0;
    this.dragStartNx = 0;
    this.dragStartNy = 0;
    this.onOrientationChange = null; // callback(yaw, pitch, roll)
    this.onProbeMove = null; // callback(nx, ny) for free positioning

    this.canvas.addEventListener("mousedown", (e) => this._onMouseDown(e));
    this.canvas.addEventListener("mousemove", (e) => this._onMouseMove(e));
    this.canvas.addEventListener("mouseup", () => this._onMouseUp());
    this.canvas.addEventListener("mouseleave", () => this._onMouseUp());
    this.canvas.addEventListener("contextmenu", (e) => e.preventDefault());

    this._ro = new ResizeObserver(() => { this._resize(); this._dirty = true; });
    this._ro.observe(this.canvas.parentElement);
    this._resize();
    this._animate();
  }

  _resize() {
    const parent = this.canvas.parentElement;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    if (w < 1 || h < 1) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.canvas.style.width = w + "px";
    this.canvas.style.height = h + "px";
    this._w = w;
    this._h = h;
    this._dpr = dpr;
  }

  setZones(z) { this.zones = z; this._dirty = true; }
  setActiveZone(z) { this.activeZone = z; this._dirty = true; }
  setExamined(z) { this.examinedZones = new Set(z); this._dirty = true; }

  setProbePosition(nx, ny) {
    this.probeNx = nx;
    this.probeNy = ny;
    this.probeVisible = true;
    this._dirty = true;
  }

  setBeamTarget(nx, ny) {
    this.beamTargetNx = nx;
    this.beamTargetNy = ny;
    this._dirty = true;
  }

  snapToZone(zoneKey) {
    if (this.zones[zoneKey]) {
      this.probeNx = this.zones[zoneKey].nx;
      this.probeNy = this.zones[zoneKey].ny;
      this.beamTargetNx = this.probeNx;
      this.beamTargetNy = this.probeNy;
      this.probeVisible = true;
      this.yaw = 0; this.pitch = 0; this.roll = 0;
      this._updateAngleDisplay();
      this._dirty = true;
    }
  }

  setIMU(y, p, r) {
    this.yaw = y; this.pitch = p; this.roll = r; this._dirty = true;
  }

  setIMUConnected(c) {
    this.imuConnected = c;
    const el = document.getElementById("probe-mode-label");
    if (el) el.textContent = c ? "BLE Live" : "Manual";
  }

  resetOrientation() {
    this.yaw = 0; this.pitch = 0; this.roll = 0;
    this._updateAngleDisplay(); this._dirty = true;
  }

  _animate() {
    if (this._dirty) { this._render(); this._dirty = false; }
    requestAnimationFrame(() => this._animate());
  }

  // Coordinate mapping: normalized [0,1] → canvas px
  _zx(nx, w) { return 22 + nx * (w - 44); }
  _zy(ny, h) { return 18 + ny * (h - 40); }

  _render() {
    const ctx = this.ctx;
    const w = this._w || 1;
    const h = this._h || 1;
    const dpr = this._dpr || 1;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const bw = w * 0.38; // half-body width
    const top = h * 0.04;
    const bot = h * 0.92;
    const bodyH = bot - top;

    // Draw layers back-to-front
    this._drawTorsoOutline(ctx, cx, bw, top, bot, bodyH, w, h);
    this._drawLungFields(ctx, cx, bw, top, bodyH);
    this._drawSpine(ctx, cx, top, bodyH);
    this._drawRibs(ctx, cx, bw, top, bodyH, w);
    this._drawSternum(ctx, cx, bw, top, bodyH);
    this._drawDiaphragm(ctx, cx, bw, top, bodyH);
    this._drawZoneMarkers(ctx, w, h);

    // Probe position indicator (small crosshair — 3D probe is in WebGL overlay)
    if (this.probeVisible) {
      const px = this._zx(this.probeNx, w);
      const py = this._zy(this.probeNy, h);
      const bx = this._zx(this.beamTargetNx, w);
      const by = this._zy(this.beamTargetNy, h);

      // Tether: physical probe → beam target (shows how tilt projects the beam)
      const tiltDist = Math.hypot(bx - px, by - py);
      if (tiltDist > 1.5) {
        ctx.strokeStyle = "rgba(120, 200, 140, 0.5)";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(bx, by);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Beam target (green) — the effective scanning point given tilt
      ctx.strokeStyle = "rgba(120, 200, 140, 0.75)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(bx, by, 5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = "rgba(120, 200, 140, 0.35)";
      ctx.fill();

      // Physical probe (orange crosshair)
      ctx.strokeStyle = "rgba(255, 128, 48, 0.6)";
      ctx.lineWidth = 1.5;
      const cs = 8;
      ctx.beginPath();
      ctx.moveTo(px - cs, py); ctx.lineTo(px + cs, py);
      ctx.moveTo(px, py - cs); ctx.lineTo(px, py + cs);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 128, 48, 0.8)";
      ctx.fill();
    }

    // Side labels
    ctx.fillStyle = "#7090a8";
    ctx.font = `bold ${Math.max(14, w * 0.045)}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText("R", 14, h * 0.48);
    ctx.fillText("L", w - 14, h * 0.48);
    ctx.fillStyle = "#607888";
    ctx.font = "13px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Anterior view", 6, h - 4);

    ctx.restore();
  }

  _drawTorsoOutline(ctx, cx, bw, top, bot, bodyH) {
    ctx.beginPath();
    ctx.moveTo(cx - bw * 0.15, top);
    ctx.quadraticCurveTo(cx - bw * 0.4, top + bodyH * 0.02, cx - bw, top + bodyH * 0.07);
    ctx.quadraticCurveTo(cx - bw * 1.15, top + bodyH * 0.12, cx - bw * 1.05, top + bodyH * 0.18);
    ctx.quadraticCurveTo(cx - bw * 0.95, top + bodyH * 0.35, cx - bw * 0.88, top + bodyH * 0.55);
    ctx.quadraticCurveTo(cx - bw * 0.78, top + bodyH * 0.78, cx - bw * 0.65, top + bodyH * 0.92);
    ctx.quadraticCurveTo(cx - bw * 0.4, bot, cx, bot);
    ctx.quadraticCurveTo(cx + bw * 0.4, bot, cx + bw * 0.65, top + bodyH * 0.92);
    ctx.quadraticCurveTo(cx + bw * 0.78, top + bodyH * 0.78, cx + bw * 0.88, top + bodyH * 0.55);
    ctx.quadraticCurveTo(cx + bw * 0.95, top + bodyH * 0.35, cx + bw * 1.05, top + bodyH * 0.18);
    ctx.quadraticCurveTo(cx + bw * 1.15, top + bodyH * 0.12, cx + bw, top + bodyH * 0.07);
    ctx.quadraticCurveTo(cx + bw * 0.4, top + bodyH * 0.02, cx + bw * 0.15, top);
    ctx.closePath();
    ctx.fillStyle = "#0c1219";
    ctx.fill();
    ctx.strokeStyle = "#4a6a80";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  _drawLungFields(ctx, cx, bw, top, bodyH) {
    // Semi-transparent lung fields showing aerated lung tissue
    const lungAlpha = 0.18;

    // Left lung (viewer's right)
    ctx.beginPath();
    ctx.moveTo(cx + bw * 0.12, top + bodyH * 0.10);
    ctx.quadraticCurveTo(cx + bw * 0.7, top + bodyH * 0.08, cx + bw * 0.85, top + bodyH * 0.15);
    ctx.quadraticCurveTo(cx + bw * 0.9, top + bodyH * 0.35, cx + bw * 0.82, top + bodyH * 0.55);
    ctx.quadraticCurveTo(cx + bw * 0.65, top + bodyH * 0.72, cx + bw * 0.2, top + bodyH * 0.70);
    ctx.quadraticCurveTo(cx + bw * 0.1, top + bodyH * 0.50, cx + bw * 0.12, top + bodyH * 0.10);
    ctx.closePath();
    ctx.fillStyle = `rgba(70, 130, 160, ${lungAlpha})`;
    ctx.fill();
    ctx.strokeStyle = "rgba(90, 160, 200, 0.35)";
    ctx.lineWidth = 1.0;
    ctx.stroke();

    // Right lung (viewer's left)
    ctx.beginPath();
    ctx.moveTo(cx - bw * 0.12, top + bodyH * 0.10);
    ctx.quadraticCurveTo(cx - bw * 0.7, top + bodyH * 0.08, cx - bw * 0.85, top + bodyH * 0.15);
    ctx.quadraticCurveTo(cx - bw * 0.9, top + bodyH * 0.35, cx - bw * 0.82, top + bodyH * 0.55);
    ctx.quadraticCurveTo(cx - bw * 0.70, top + bodyH * 0.75, cx - bw * 0.2, top + bodyH * 0.75);
    ctx.quadraticCurveTo(cx - bw * 0.1, top + bodyH * 0.50, cx - bw * 0.12, top + bodyH * 0.10);
    ctx.closePath();
    ctx.fillStyle = `rgba(70, 130, 160, ${lungAlpha})`;
    ctx.fill();
    ctx.strokeStyle = "rgba(90, 160, 200, 0.35)";
    ctx.lineWidth = 1.0;
    ctx.stroke();

    // Lung labels
    ctx.fillStyle = "rgba(90, 160, 200, 0.45)";
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("L lung", cx + bw * 0.45, top + bodyH * 0.38);
    ctx.fillText("R lung", cx - bw * 0.45, top + bodyH * 0.38);
  }

  _drawSpine(ctx, cx, top, bodyH) {
    // Vertebral bodies as small rounded rectangles
    ctx.fillStyle = "rgba(60, 90, 120, 0.3)";
    ctx.strokeStyle = "rgba(60, 90, 120, 0.25)";
    ctx.lineWidth = 0.4;
    const vw = 6, vh = 5;
    for (let i = 0; i < 12; i++) {
      const vy = top + bodyH * (0.08 + i * 0.058);
      ctx.beginPath();
      ctx.roundRect(cx - vw / 2, vy, vw, vh, 1);
      ctx.fill();
      ctx.stroke();
    }
  }

  _drawSternum(ctx, cx, bw, top, bodyH) {
    // Manubrium
    ctx.fillStyle = "rgba(70, 100, 130, 0.35)";
    ctx.strokeStyle = "rgba(70, 100, 130, 0.3)";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(cx - bw * 0.08, top + bodyH * 0.04);
    ctx.lineTo(cx + bw * 0.08, top + bodyH * 0.04);
    ctx.lineTo(cx + bw * 0.06, top + bodyH * 0.12);
    ctx.lineTo(cx - bw * 0.06, top + bodyH * 0.12);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Body of sternum
    ctx.beginPath();
    ctx.moveTo(cx - bw * 0.05, top + bodyH * 0.13);
    ctx.lineTo(cx + bw * 0.05, top + bodyH * 0.13);
    ctx.lineTo(cx + bw * 0.04, top + bodyH * 0.58);
    ctx.lineTo(cx - bw * 0.04, top + bodyH * 0.58);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Xiphoid process
    ctx.beginPath();
    ctx.moveTo(cx - bw * 0.025, top + bodyH * 0.59);
    ctx.lineTo(cx + bw * 0.025, top + bodyH * 0.59);
    ctx.lineTo(cx, top + bodyH * 0.66);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  _drawRibs(ctx, cx, bw, top, bodyH, w) {
    // 12 rib pairs with proper curvature and ICS numbering
    const ribColor = "rgba(80, 115, 145, 0.35)";
    const ribStroke = "rgba(80, 115, 145, 0.50)";

    for (let i = 1; i <= 10; i++) {
      const frac = 0.06 + i * 0.058;
      const ry = top + bodyH * frac;
      const spread = bw * (0.30 + i * 0.055);
      const droop = bodyH * (0.005 + i * 0.004);
      const thick = Math.max(1.5, 2.5 - i * 0.1);

      ctx.strokeStyle = ribStroke;
      ctx.lineWidth = thick;

      // Left rib
      ctx.beginPath();
      ctx.moveTo(cx + bw * 0.06, ry);
      ctx.quadraticCurveTo(cx + spread * 0.55, ry + droop * 0.4, cx + spread, ry + droop);
      ctx.stroke();

      // Right rib
      ctx.beginPath();
      ctx.moveTo(cx - bw * 0.06, ry);
      ctx.quadraticCurveTo(cx - spread * 0.55, ry + droop * 0.4, cx - spread, ry + droop);
      ctx.stroke();

      // ICS number labels (only for visible ones)
      if (i <= 7 && i >= 2) {
        ctx.fillStyle = "rgba(100, 140, 175, 0.45)";
        ctx.font = "12px Inter, sans-serif";
        ctx.textAlign = "right";
        ctx.fillText(`${i}`, cx - bw * 0.08, ry + 3);
      }
    }

    // Costal cartilage connections (ribs 1-7 to sternum)
    ctx.strokeStyle = "rgba(70, 100, 130, 0.25)";
    ctx.lineWidth = 0.4;
    for (let i = 1; i <= 7; i++) {
      const ry = top + bodyH * (0.06 + i * 0.058);
      ctx.beginPath();
      ctx.moveTo(cx + bw * 0.06, ry);
      ctx.lineTo(cx + bw * 0.04, ry + 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - bw * 0.06, ry);
      ctx.lineTo(cx - bw * 0.04, ry + 2);
      ctx.stroke();
    }
  }

  _drawDiaphragm(ctx, cx, bw, top, bodyH) {
    // Diaphragm dome
    ctx.strokeStyle = "rgba(200, 150, 90, 0.5)";
    ctx.lineWidth = 1.8;
    ctx.setLineDash([3, 2]);

    // Right dome (slightly higher)
    ctx.beginPath();
    ctx.moveTo(cx - bw * 0.05, top + bodyH * 0.68);
    ctx.quadraticCurveTo(cx - bw * 0.45, top + bodyH * 0.62, cx - bw * 0.80, top + bodyH * 0.72);
    ctx.stroke();

    // Left dome
    ctx.beginPath();
    ctx.moveTo(cx + bw * 0.05, top + bodyH * 0.70);
    ctx.quadraticCurveTo(cx + bw * 0.45, top + bodyH * 0.64, cx + bw * 0.80, top + bodyH * 0.74);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label
    ctx.fillStyle = "rgba(200, 150, 90, 0.5)";
    ctx.font = "13px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("diaphragm", cx, top + bodyH * 0.76);
  }

  _drawZoneMarkers(ctx, w, h) {
    const abbrevs = {
      "UPPER_BLUE_L": "UB-L", "UPPER_BLUE_R": "UB-R",
      "LOWER_BLUE_L": "LB-L", "LOWER_BLUE_R": "LB-R",
      "PLAPS_L": "PL-L", "PLAPS_R": "PL-R",
      "DIAPHRAGM_L": "Di-L", "DIAPHRAGM_R": "Di-R",
    };

    for (const [key, pos] of Object.entries(this.zones)) {
      const zx = this._zx(pos.nx, w);
      const zy = this._zy(pos.ny, h);
      const r = Math.max(7, Math.min(12, w * 0.035));

      let fill, stroke, sw;
      if (key === this.activeZone) {
        fill = "rgba(90, 158, 192, 0.75)"; stroke = "#5a9ec0"; sw = 2;
        // Glow ring for active zone
        ctx.beginPath();
        ctx.arc(zx, zy, r + 4, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(90, 158, 192, 0.2)";
        ctx.lineWidth = 3;
        ctx.stroke();
      } else if (this.examinedZones.has(key)) {
        fill = "rgba(76, 175, 122, 0.5)"; stroke = "#4caf7a"; sw = 1.5;
      } else {
        fill = "rgba(26, 58, 92, 0.5)"; stroke = "#2a5a8c"; sw = 1;
      }

      ctx.beginPath();
      ctx.arc(zx, zy, r, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = sw;
      ctx.stroke();

      ctx.fillStyle = "#c0ccd8";
      ctx.font = `bold ${Math.max(9, r * 0.7)}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(abbrevs[key] || key.slice(0, 4), zx, zy);
      ctx.textBaseline = "alphabetic";
    }
  }

  _drawProbe(ctx, px, py, bw) {
    ctx.save();
    ctx.translate(px, py);

    // --- 3D projection helpers ---
    // Convert pitch/roll/yaw to radians
    const yawR = this.yaw * Math.PI / 180;
    const pitchR = this.pitch * Math.PI / 180;
    const rollR = this.roll * Math.PI / 180;

    // Scale probe to be large and prominent (~40% of body width)
    const S = Math.max(1.8, bw * 0.016);

    // 3D rotation matrix (simplified for canvas 2D projection)
    // We project 3D points (x,y,z) onto screen (sx,sy)
    const cosY = Math.cos(yawR), sinY = Math.sin(yawR);
    const cosP = Math.cos(pitchR), sinP = Math.sin(pitchR);
    const cosR = Math.cos(rollR), sinR = Math.sin(rollR);

    // Combined rotation: Yaw(Z) * Pitch(X) * Roll(Y)
    function project3D(x, y, z) {
      // Roll (around Y-axis)
      let x1 = x * cosR + z * sinR;
      let y1 = y;
      let z1 = -x * sinR + z * cosR;
      // Pitch (around X-axis)
      let x2 = x1;
      let y2 = y1 * cosP - z1 * sinP;
      let z2 = y1 * sinP + z1 * cosP;
      // Yaw (around Z-axis)
      let x3 = x2 * cosY - y2 * sinY;
      let y3 = x2 * sinY + y2 * cosY;
      let z3 = z2;
      // Perspective projection (mild)
      const persp = 1.0 + z3 * 0.0015;
      return { x: x3 * persp * S, y: y3 * persp * S, z: z3 };
    }

    // Draw a 3D polygon given array of [x,y,z] points
    function drawPoly(pts, fill, stroke, lw) {
      const projected = pts.map(p => project3D(p[0], p[1], p[2]));
      ctx.beginPath();
      ctx.moveTo(projected[0].x, projected[0].y);
      for (let i = 1; i < projected.length; i++) ctx.lineTo(projected[i].x, projected[i].y);
      ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1; ctx.stroke(); }
    }

    // --- Probe geometry in 3D local coords ---
    // Probe points upward (negative Y = up on screen before rotation)
    // Handle top, body, and scanning face
    const hw = 8;    // handle half-width
    const hh = 35;   // handle height
    const bwp = 14;  // body half-width at shoulder
    const faceW = 26; // face half-width
    const bodyLen = 30; // body length
    const faceD = 6;  // face depth/curvature
    const thick = 12; // probe thickness (Z-axis)

    // --- Drop shadow (offset, darker) ---
    ctx.save();
    ctx.translate(4 * S, 4 * S);
    ctx.globalAlpha = 0.25;
    const shadowPts = [
      [-hw, -hh - bodyLen, 0], [hw, -hh - bodyLen, 0],
      [bwp, -bodyLen, 0], [faceW, 0, 0],
      [-faceW, 0, 0], [-bwp, -bodyLen, 0]
    ];
    drawPoly(shadowPts, "#000", null);
    ctx.globalAlpha = 1.0;
    ctx.restore();

    // --- Selection glow ---
    ctx.shadowColor = "rgba(90, 158, 192, 0.5)";
    ctx.shadowBlur = 20 * S;

    // --- Compute face normal for lighting ---
    const faceNorm = project3D(0, 0, 1);
    const lightFactor = 0.5 + 0.5 * Math.max(0, faceNorm.z / S);  // 0-1 lighting

    // --- Back face (Z = -thick/2) — drawn first ---
    const backZ = -thick / 2;
    const backPts = [
      [-hw, -hh - bodyLen, backZ], [hw, -hh - bodyLen, backZ],
      [bwp, -bodyLen, backZ], [faceW, 0, backZ],
      [-faceW, 0, backZ], [-bwp, -bodyLen, backZ]
    ];
    drawPoly(backPts, "#1a2835", "#2a4050", 1);
    ctx.shadowBlur = 0;

    // --- Side faces (connect front and back) ---
    const frontZ = thick / 2;
    const frontPtsRaw = [
      [-hw, -hh - bodyLen, frontZ], [hw, -hh - bodyLen, frontZ],
      [bwp, -bodyLen, frontZ], [faceW, 0, frontZ],
      [-faceW, 0, frontZ], [-bwp, -bodyLen, frontZ]
    ];
    const backPtsRaw = [
      [-hw, -hh - bodyLen, backZ], [hw, -hh - bodyLen, backZ],
      [bwp, -bodyLen, backZ], [faceW, 0, backZ],
      [-faceW, 0, backZ], [-bwp, -bodyLen, backZ]
    ];

    // Draw side panels (connecting front/back edges)
    const sideColors = ["#253848", "#2a4050", "#2d4458", "#2a4050", "#253848", "#2a3a4a"];
    for (let i = 0; i < 6; i++) {
      const j = (i + 1) % 6;
      const sidePts = [frontPtsRaw[i], frontPtsRaw[j], backPtsRaw[j], backPtsRaw[i]];
      // Check if this face is visible (simple backface culling)
      const p0 = project3D(...sidePts[0]);
      const p1 = project3D(...sidePts[1]);
      const p2 = project3D(...sidePts[2]);
      const cross = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
      if (cross > 0) {
        drawPoly(sidePts, sideColors[i], "#3a5568", 0.8);
      }
    }

    // --- Front face (Z = thick/2) ---
    // Dynamic color based on lighting
    const r1 = Math.round(40 + 40 * lightFactor);
    const g1 = Math.round(70 + 40 * lightFactor);
    const b1 = Math.round(90 + 40 * lightFactor);
    drawPoly(frontPtsRaw, `rgb(${r1},${g1},${b1})`, "#5a8098", 1.5);

    // --- Scanning face (bottom edge, curved, bright cyan) ---
    const faceSteps = 12;
    const facePtsF = [];
    const facePtsB = [];
    for (let i = 0; i <= faceSteps; i++) {
      const t = i / faceSteps;
      const x = -faceW + 2 * faceW * t;
      const curve = faceD * Math.sin(t * Math.PI); // convex bulge
      facePtsF.push([x, curve, frontZ]);
      facePtsB.push([x, curve, backZ]);
    }
    // Front scanning face strip
    const scanFace = [
      [-faceW, 0, frontZ], ...facePtsF, [faceW, 0, frontZ]
    ];
    const faceLight = 0.6 + 0.4 * lightFactor;
    const fr = Math.round(90 * faceLight);
    const fg = Math.round(158 * faceLight);
    const fb = Math.round(192 * faceLight);
    drawPoly(scanFace, `rgb(${fr},${fg},${fb})`, "#7ab8d6", 1.5);

    // Curved bottom surface (connect front and back curves)
    for (let i = 0; i < faceSteps; i++) {
      const quad = [facePtsF[i], facePtsF[i + 1], facePtsB[i + 1], facePtsB[i]];
      const p0 = project3D(...quad[0]);
      const p1 = project3D(...quad[1]);
      const p2 = project3D(...quad[2]);
      const cross = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
      if (cross > 0) {
        drawPoly(quad, `rgba(${fr},${fg},${fb},0.8)`, "#5a9ec0", 0.5);
      }
    }

    // --- Crystal element lines on front face ---
    ctx.strokeStyle = `rgba(122, 184, 214, ${0.3 + 0.3 * lightFactor})`;
    ctx.lineWidth = 0.8;
    for (let i = 1; i < 6; i++) {
      const t = i / 6;
      const x = -faceW + 2 * faceW * t;
      const curve = faceD * Math.sin(t * Math.PI);
      const p1 = project3D(x, 0, frontZ);
      const p2 = project3D(x, curve * 0.8, frontZ);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // --- Probe marker notch (orange, on right side of handle) ---
    const notchP = project3D(hw + 2, -hh - bodyLen + 10, frontZ + 1);
    const notchR = Math.max(4, 5 * S);
    ctx.fillStyle = "#e0a040";
    ctx.beginPath();
    ctx.arc(notchP.x, notchP.y, notchR, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#0b0f14";
    ctx.font = `bold ${Math.max(11, 12 * S)}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("M", notchP.x, notchP.y);

    // --- Handle grip lines ---
    ctx.strokeStyle = "rgba(90, 140, 180, 0.35)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const gy = -bodyLen - hh + 8 + i * (hh - 12) / 4;
      const p1 = project3D(-hw + 2, gy, frontZ + 0.5);
      const p2 = project3D(hw - 2, gy, frontZ + 0.5);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // --- Cable (top of handle) ---
    ctx.strokeStyle = "#3a5568";
    ctx.lineWidth = Math.max(3, 4 * S);
    ctx.lineCap = "round";
    const cableStart = project3D(0, -hh - bodyLen, 0);
    const cableMid = project3D(-3, -hh - bodyLen - 18, 2);
    const cableEnd = project3D(0, -hh - bodyLen - 32, 5);
    ctx.beginPath();
    ctx.moveTo(cableStart.x, cableStart.y);
    ctx.quadraticCurveTo(cableMid.x, cableMid.y, cableEnd.x, cableEnd.y);
    ctx.stroke();
    ctx.lineCap = "butt";

    // --- RGB Orientation Axes (rotate with probe) ---
    const axisLen = 40;
    const arrowSz = 7;
    const axO = project3D(0, 5, 0); // axis origin at probe center

    // Helper to draw a 3D axis arrow
    function drawAxis(dx, dy, dz, color, label) {
      const tip = project3D(dx * axisLen, 5 + dy * axisLen, dz * axisLen);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3 * S;
      ctx.beginPath();
      ctx.moveTo(axO.x, axO.y);
      ctx.lineTo(tip.x, tip.y);
      ctx.stroke();
      // Arrowhead
      const dir = Math.atan2(tip.y - axO.y, tip.x - axO.x);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(tip.x + Math.cos(dir) * arrowSz * S, tip.y + Math.sin(dir) * arrowSz * S);
      ctx.lineTo(tip.x + Math.cos(dir + 2.5) * arrowSz * 0.6 * S, tip.y + Math.sin(dir + 2.5) * arrowSz * 0.6 * S);
      ctx.lineTo(tip.x + Math.cos(dir - 2.5) * arrowSz * 0.6 * S, tip.y + Math.sin(dir - 2.5) * arrowSz * 0.6 * S);
      ctx.closePath();
      ctx.fill();
      // Label
      ctx.fillStyle = color;
      ctx.font = `bold ${Math.max(12, 14 * S)}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, tip.x + Math.cos(dir) * 14 * S, tip.y + Math.sin(dir) * 14 * S);
    }

    drawAxis(0, -1, 0, "#4488ff", "Z");  // Blue — up
    drawAxis(1, 0, 0, "#44cc44", "X");   // Green — right
    drawAxis(0, 0, 1, "#dd3333", "Y");   // Red — forward

    ctx.textBaseline = "alphabetic";
    ctx.restore();
  }

  _drawBeamFan(ctx, px, py, canvasH, bw) {
    const fanAngle = 55 * Math.PI / 180;
    const fanDepth = Math.min(canvasH * 0.22, bw * 0.55);

    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(this.yaw * Math.PI / 180);

    const ca = Math.PI / 2 + this.pitch * Math.PI / 180;
    const sa = ca - fanAngle / 2;
    const ea = ca + fanAngle / 2;

    // Beam fill with gradient
    const grad = ctx.createRadialGradient(0, 0, 1, 0, 0, fanDepth);
    grad.addColorStop(0, "rgba(90, 158, 192, 0.22)");
    grad.addColorStop(0.4, "rgba(90, 158, 192, 0.10)");
    grad.addColorStop(1, "rgba(90, 158, 192, 0.02)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, fanDepth, sa, ea);
    ctx.closePath();
    ctx.fill();

    // Beam edge lines
    ctx.strokeStyle = "rgba(90, 158, 192, 0.20)";
    ctx.lineWidth = 0.6;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(Math.cos(sa) * fanDepth, Math.sin(sa) * fanDepth);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(Math.cos(ea) * fanDepth, Math.sin(ea) * fanDepth);
    ctx.stroke();

    // Center line (beam axis)
    ctx.strokeStyle = "rgba(90, 158, 192, 0.12)";
    ctx.lineWidth = 0.4;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(Math.cos(ca) * fanDepth, Math.sin(ca) * fanDepth);
    ctx.stroke();
    ctx.setLineDash([]);

    // Depth markers along center axis
    ctx.fillStyle = "rgba(90, 158, 192, 0.25)";
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "left";
    for (let d = 2; d <= 10; d += 2) {
      const frac = d / 12;
      const mx = Math.cos(ca) * fanDepth * frac;
      const my = Math.sin(ca) * fanDepth * frac;
      ctx.beginPath();
      ctx.arc(mx, my, 1, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillText(`${d}cm`, mx + 3, my + 1);
    }

    ctx.restore();
  }

  _drawCrossSectionSlice(ctx, px, py, canvasH, bodyCx, bw, top, bodyH) {
    const fanDepth = Math.min(canvasH * 0.20, 75);

    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(this.yaw * Math.PI / 180);

    const rollOff = this.roll * 0.35;

    // Cross-section imaging plane
    const depths = [0.25, 0.45, 0.65]; // multiple depth lines for layered anatomy
    const labels = ["skin/fat", "pleura", "lung"];
    const colors = ["rgba(180,160,140,0.3)", "rgba(220,200,100,0.4)", "rgba(100,160,200,0.3)"];

    for (let i = 0; i < depths.length; i++) {
      const d = fanDepth * depths[i];
      const hw = fanDepth * (0.18 + depths[i] * 0.18);

      ctx.strokeStyle = colors[i];
      ctx.lineWidth = i === 1 ? 1.5 : 0.8;
      ctx.setLineDash(i === 1 ? [] : [2, 2]);
      ctx.beginPath();
      ctx.moveTo(-hw + rollOff, d);
      ctx.lineTo(hw + rollOff, d);
      ctx.stroke();
      ctx.setLineDash([]);

      // Label
      ctx.fillStyle = colors[i];
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(labels[i], hw + rollOff + 3, d + 2);
    }

    // Main cut plane indicator (at pleura depth)
    const csD = fanDepth * 0.45;
    const csW = fanDepth * 0.36;
    ctx.fillStyle = "rgba(212, 146, 58, 0.4)";
    ctx.beginPath();
    ctx.arc(-csW + rollOff, csD, 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(csW + rollOff, csD, 2, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  _updateAngleDisplay() {
    const el = document.getElementById("probe-angle-display");
    if (el) el.textContent = `Yaw: ${this.yaw.toFixed(0)}° Pitch: ${this.pitch.toFixed(0)}° Roll: ${this.roll.toFixed(0)}°`;
  }

  // Inverse of _zx/_zy: canvas px → normalized [0,1]
  _invZx(px, w) { return (px - 22) / (w - 44); }
  _invZy(py, h) { return (py - 18) / (h - 40); }

  _onMouseDown(e) {
    if (this.imuConnected) return;
    const r = this.canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const w = this._w, h = this._h;

    // Check if clicking near current probe position (within probe radius)
    const probePx = this._zx(this.probeNx, w);
    const probePy = this._zy(this.probeNy, h);
    const bw = w * 0.38;
    const hitRadius = Math.max(25, bw * 0.15);
    const distToProbe = Math.sqrt((mx - probePx) ** 2 + (my - probePy) ** 2);

    if (this.probeVisible && distToProbe < hitRadius && (e.shiftKey || e.button === 2)) {
      // Shift+click or right-click on probe → rotate
      this.isDragging = true;
      this.dragMode = 'rotate';
      this.dragStartX = mx; this.dragStartY = my;
      this.dragStartYaw = this.yaw; this.dragStartPitch = this.pitch;
    } else {
      // Click anywhere → place/move probe
      // First check if clicking a zone marker to snap
      let snapped = false;
      for (const [key, pos] of Object.entries(this.zones)) {
        const zx = this._zx(pos.nx, w);
        const zy = this._zy(pos.ny, h);
        const zoneR = Math.max(7, Math.min(12, w * 0.035));
        if (Math.sqrt((mx - zx) ** 2 + (my - zy) ** 2) < zoneR + 5) {
          this.snapToZone(key);
          snapped = true;
          this._dirty = true;
          if (this.onProbeMove) this.onProbeMove(this.probeNx, this.probeNy, key);
          break;
        }
      }
      if (!snapped) {
        // Free position
        const nx = Math.max(0.05, Math.min(0.95, this._invZx(mx, w)));
        const ny = Math.max(0.02, Math.min(0.95, this._invZy(my, h)));
        this.probeNx = nx;
        this.probeNy = ny;
        this.probeVisible = true;
        this._dirty = true;
      }
      this.isDragging = true;
      this.dragMode = 'move';
      this.dragStartX = mx; this.dragStartY = my;
      this.dragStartNx = this.probeNx;
      this.dragStartNy = this.probeNy;
      if (this.onProbeMove) this.onProbeMove(this.probeNx, this.probeNy, null);
    }
  }

  _onMouseMove(e) {
    if (!this.isDragging) return;
    const r = this.canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const w = this._w, h = this._h;

    if (this.dragMode === 'rotate') {
      const dx = mx - this.dragStartX;
      const dy = my - this.dragStartY;
      this.yaw = this.dragStartYaw + dx * 0.8;
      this.pitch = Math.max(-30, Math.min(30, this.dragStartPitch + dy * 0.5));
      this._updateAngleDisplay();
      this._dirty = true;
      if (this.onOrientationChange) {
        this.onOrientationChange(this.yaw, this.pitch, this.roll);
      }
    } else if (this.dragMode === 'move') {
      const dxPx = mx - this.dragStartX;
      const dyPx = my - this.dragStartY;
      const nx = Math.max(0.05, Math.min(0.95, this.dragStartNx + dxPx / (w - 44)));
      const ny = Math.max(0.02, Math.min(0.95, this.dragStartNy + dyPx / (h - 40)));
      this.probeNx = nx;
      this.probeNy = ny;
      this._dirty = true;
      if (this.onProbeMove) this.onProbeMove(nx, ny, null);
    }
  }

  _onMouseUp() {
    this.isDragging = false;
    this.dragMode = null;
  }
}

// ---------------------------------------------------------------------------
// Chest Zone Map (SVG)
// ---------------------------------------------------------------------------

class ChestZoneMap {
  constructor(svgId) {
    this.svg = document.getElementById(svgId);
    this.zones = {};
    this.activeZone = null;
    this.examinedZones = new Set();
    this.onZoneClick = null;
  }

  setZones(zonesData) {
    this.svg.querySelectorAll(".zone-circle, .zone-label-text").forEach(el => el.remove());
    this.zones = zonesData;
    const abbrevs = {
      "UPPER_BLUE_L": "UB-L", "UPPER_BLUE_R": "UB-R",
      "LOWER_BLUE_L": "LB-L", "LOWER_BLUE_R": "LB-R",
      "PLAPS_L": "PL-L", "PLAPS_R": "PL-R",
      "DIAPHRAGM_L": "Di-L", "DIAPHRAGM_R": "Di-R",
    };

    for (const [key, pos] of Object.entries(zonesData)) {
      const cx = 24 + pos.nx * 112;
      const cy = 14 + pos.ny * 162;

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("cx", cx);
      circle.setAttribute("cy", cy);
      circle.setAttribute("r", "10");
      circle.setAttribute("fill", "#1a3a5c");
      circle.setAttribute("stroke", "#2a5a8c");
      circle.setAttribute("stroke-width", "1");
      circle.classList.add("zone-circle");
      circle.dataset.zone = key;
      circle.addEventListener("click", () => { if (this.onZoneClick) this.onZoneClick(key); });
      this.svg.appendChild(circle);

      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", cx);
      label.setAttribute("y", cy);
      label.classList.add("zone-label-text");
      label.textContent = abbrevs[key] || key.slice(0, 4);
      this.svg.appendChild(label);
    }
  }

  setActive(zoneKey) { this.activeZone = zoneKey; this._update(); }
  setExamined(zoneKeys) {
    this.examinedZones = new Set(zoneKeys);
    this._update();
    document.getElementById("zone-progress").textContent = `${this.examinedZones.size}/8`;
  }

  _update() {
    this.svg.querySelectorAll(".zone-circle").forEach(c => {
      const k = c.dataset.zone;
      if (k === this.activeZone) {
        c.setAttribute("fill", "#5a9ec0"); c.setAttribute("stroke", "#5a9ec0"); c.setAttribute("stroke-width", "2");
      } else if (this.examinedZones.has(k)) {
        c.setAttribute("fill", "#4caf7a"); c.setAttribute("stroke", "#4caf7a"); c.setAttribute("stroke-width", "1.5");
      } else {
        c.setAttribute("fill", "#1a3a5c"); c.setAttribute("stroke", "#2a5a8c"); c.setAttribute("stroke-width", "1");
      }
    });
  }

  reset() {
    this.activeZone = null;
    this.examinedZones.clear();
    this._update();
    document.getElementById("zone-progress").textContent = "0/8";
  }
}

// ---------------------------------------------------------------------------
// BLE Probe Manager
// ---------------------------------------------------------------------------

class BLEProbeManager {
  /**
   * Connects to the BNO085 IMU probe via Web Bluetooth.
   *
   * Supports two BLE service configurations:
   *   1. Tianyun's BNO085BLE firmware (custom UUID, primary)
   *   2. Nordic UART fallback (development/generic probes)
   *
   * Data format from BNO085BLE firmware:
   *   "counter,seconds,loops,transfers,calStatus,YPR=yaw,pitch,roll,Q=qr,qi,qj,qk[,LA=ax,ay,az]"
   *
   * Translation (nx, ny) source priority:
   *   1. Pressure mat — absolute, drift-free. Wins whenever a mat sample arrived
   *      within MAT_TIMEOUT_MS.
   *   2. IMU linear acceleration — body-frame accel rotated to world frame using
   *      current yaw, double-integrated to position with ZUPT and velocity damping
   *      to bound the inevitable drift. Used only when the mat is silent.
   */
  constructor() {
    this.device = null;
    this.connected = false;
    this.onIMU = null;
    this.onPosition = null;  // (nx, ny, confidence, source) — source ∈ {"mat","imu"}
    this.onStatus = null;
    this.yawOffset = 0;
    this._lastRawYaw = 0;

    // Tianyun's BNO085BLE firmware UUIDs (primary)
    this.IMU_SERVICE_UUID = "9a48ecba-2e92-082f-c079-9e75aae428b1";
    this.IMU_CHAR_UUID = "00000000-0000-0000-0000-0000001234dd";

    // Nordic UART fallback (for development/generic BLE probes)
    this.UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e";
    this.UART_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e";

    // Pressure mat grid (16x16 Velostat via CD74HC4067 mux)
    this.GRID_ROWS = 16;
    this.GRID_COLS = 16;

    // Translation state (used in IMU-fallback mode)
    this.posNx = 0.5;
    this.posNy = 0.5;
    this.velX = 0;            // m/s, world-frame (patient coronal plane)
    this.velY = 0;
    this._lastAccelTs = null;
    this._stillSamples = 0;
    this._lastAccelMag = 0;
    this._sawLA = false;
    this._sawRawAccel = false;
    this._warnedNoAccel = false;

    // Adaptive accelerometer-bias tracker — the actual reason "the probe
    // doesn't move when I move it." Even gravity-removed linear accel from
    // the BNO085 carries a few hundredths of a m/s² of residual bias; without
    // subtraction it integrates into a wall-clock-paced phantom drift that
    // ZUPT then snaps away, masking real motion. We re-learn the bias every
    // time the probe is stationary so it stays accurate as the sensor warms
    // up and shifts.
    this.biasX = 0;
    this.biasY = 0;
    this.BIAS_ALPHA = 0.01;            // EMA factor during stillness

    // Mat-vs-IMU arbitration
    this._lastMatMs = 0;
    this.MAT_TIMEOUT_MS = 500;

    // Sensitivity / drift control. Now that bias is corrected, we can run
    // a far less aggressive damping and a tighter "motion-to-chest" mapping
    // without the probe wandering on a still desk.
    this.METERS_PER_NORM = 0.15;       // 15cm physical sweep ≈ full chest map
    this.ZUPT_ACCEL_THRESHOLD = 0.06;  // m/s² above bias = "moving"
    this.ZUPT_STILL_FRAMES = 20;       // ~0.4s at 50Hz before declaring stationary
    this.VEL_DAMPING = 0.992;          // ~30% loss/sec at 50Hz — preserves
                                       // sweep momentum, bias-correction handles drift
  }

  // World gravity in the probe's body frame given current pitch/roll.
  // Used to remove gravity from raw accelerometer when firmware doesn't
  // emit the BNO085's gravity-removed linear-accel report (LA=).
  // Convention: Z-Y-X intrinsic, probe upright = body Z is along -world-Z.
  _gravityInBodyFrame(pitchDeg, rollDeg) {
    const G = 9.81;
    const p = pitchDeg * Math.PI / 180;
    const r = rollDeg * Math.PI / 180;
    return {
      x: G * Math.sin(p),
      y: -G * Math.cos(p) * Math.sin(r),
      z: -G * Math.cos(p) * Math.cos(r),
    };
  }

  async connect() {
    if (!navigator.bluetooth) {
      this.onStatus?.("Web Bluetooth not supported — use Chrome/Edge", false);
      return;
    }
    try {
      this.onStatus?.("Scanning for probe...", false);

      // Try Tianyun's BNO085BLE service first, fall back to Nordic UART
      this.device = await navigator.bluetooth.requestDevice({
        filters: [
          { services: [this.IMU_SERVICE_UUID] },
          { services: [this.UART_SERVICE_UUID] },
        ],
        optionalServices: [this.IMU_SERVICE_UUID, this.UART_SERVICE_UUID],
      });

      const server = await this.device.gatt.connect();

      // Try primary IMU service first
      let char;
      try {
        const service = await server.getPrimaryService(this.IMU_SERVICE_UUID);
        char = await service.getCharacteristic(this.IMU_CHAR_UUID);
        this.onStatus?.("Connected (BNO085): " + this.device.name, true);
      } catch {
        // Fall back to Nordic UART
        const service = await server.getPrimaryService(this.UART_SERVICE_UUID);
        char = await service.getCharacteristic(this.UART_CHAR_UUID);
        this.onStatus?.("Connected (UART): " + this.device.name, true);
      }

      await char.startNotifications();
      char.addEventListener("characteristicvaluechanged", (e) => this._handle(e));
      this.connected = true;

      this.device.addEventListener("gattserverdisconnected", () => {
        this.connected = false;
        this.onStatus?.("Disconnected", false);
      });
    } catch (err) {
      this.onStatus?.(err.name === "NotFoundError" ? "Cancelled" : "Failed: " + err.message, false);
    }
  }

  disconnect() {
    if (this.device?.gatt?.connected) this.device.gatt.disconnect();
    this.connected = false;
    this.onStatus?.("Disconnected", false);
  }

  calibrateYaw() { this.yawOffset = -(this._lastRawYaw || 0); }

  /**
   * Reset IMU-derived translation to a known anchor. Call after the user
   * physically places the probe at a known landmark — otherwise the integrator
   * starts at (0.5, 0.5) and drifts from there.
   */
  calibratePosition(nx = 0.5, ny = 0.5) {
    this.posNx = nx;
    this.posNy = ny;
    this.velX = 0;
    this.velY = 0;
    this._stillSamples = 0;
    this._lastAccelTs = null;
    // Fresh bias estimate — keep the probe still for ~0.4s after calibration
    // and the EMA will lock onto the resting accel signature.
    this.biasX = 0;
    this.biasY = 0;
  }

  /**
   * Map pressure mat grid coordinates to normalized chest position.
   * Grid is 16x16, origin top-left.
   * Chest coordinates: nx=0 (patient right) to 1 (patient left),
   *                    ny=0 (head/clavicle) to 1 (lower abdomen).
   */
  gridToChestPosition(row, col) {
    const nx = col / (this.GRID_COLS - 1);
    const ny = row / (this.GRID_ROWS - 1);
    return { nx, ny };
  }

  _parseLabeled(parts, prefixes) {
    const re = new RegExp("^(" + prefixes.join("|") + ")=", "i");
    const idx = parts.findIndex(p => re.test(p));
    if (idx < 0 || parts.length < idx + 3) return null;
    const head = parts[idx].split("=")[1];
    const a = parseFloat(head);
    const b = parseFloat(parts[idx + 1]);
    const c = parseFloat(parts[idx + 2]);
    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) return null;
    return [a, b, c];
  }

  _handle(event) {
    const text = new TextDecoder().decode(event.target.value).trim();
    const parts = text.split(",").map(s => s.trim());

    let yaw, pitch, roll;
    const ypr = this._parseLabeled(parts, ["YPR"]);
    if (ypr) {
      [yaw, pitch, roll] = ypr;
    } else if (parts.length >= 4) {
      // Fallback: "counter,yaw,pitch,roll"
      yaw = parseFloat(parts[1]);
      pitch = parseFloat(parts[2]);
      roll = parseFloat(parts[3]);
    } else return;

    if (!Number.isFinite(yaw) || !Number.isFinite(pitch) || !Number.isFinite(roll)) return;
    this._lastRawYaw = yaw;
    const adjYaw = ((yaw + this.yawOffset + 180) % 360) - 180;
    this.onIMU?.(adjYaw, pitch, roll);

    // Two acceleration sources, in order of preference:
    //   1. LA= / LINACC= — gravity already removed by BNO085 onboard fusion.
    //      Cleanest input for double-integration.
    //   2. ACC= / A= — raw accelerometer (includes 9.81 m/s² of gravity).
    //      We remove gravity in software using the current pitch/roll.
    let ax, ay, az;
    const la = this._parseLabeled(parts, ["LA", "LINACC"]);
    if (la) {
      [ax, ay, az] = la;
      if (!this._sawLA) {
        this._sawLA = true;
        console.log("[BLE] Linear-accel stream detected — driving probe translation");
      }
    } else {
      const raw = this._parseLabeled(parts, ["ACC", "A"]);
      if (!raw) {
        if (!this._warnedNoAccel) {
          this._warnedNoAccel = true;
          console.warn(
            "[BLE] Probe firmware emits neither LA= nor ACC= — accelerometer " +
            "cannot drive translation. Sample: " + text.slice(0, 120)
          );
        }
        return;
      }
      [ax, ay, az] = raw;
      const g = this._gravityInBodyFrame(pitch, roll);
      ax -= g.x;
      ay -= g.y;
      az -= g.z;
      if (!this._sawRawAccel) {
        this._sawRawAccel = true;
        console.log("[BLE] Raw accel detected — applying software gravity removal");
      }
    }

    if (Number.isFinite(ax) && Number.isFinite(ay)) {
      this._integrateAccel(ax, ay, az, adjYaw);
    }
  }

  _integrateAccel(ax, ay, _az, yawDeg) {
    const now = performance.now();

    // Mat takes precedence — when a recent mat sample exists, suppress the
    // integrator entirely so it doesn't fight the absolute reference.
    if (now - this._lastMatMs < this.MAT_TIMEOUT_MS) {
      this._lastAccelTs = now;
      this.velX = 0;
      this.velY = 0;
      return;
    }

    if (this._lastAccelTs == null) { this._lastAccelTs = now; return; }
    const dt = Math.min(0.1, (now - this._lastAccelTs) / 1000);
    this._lastAccelTs = now;
    if (dt <= 0) return;

    // Bias-corrected accel: subtract the slow EMA of the at-rest signal.
    // Bias is what makes the probe drift on a still desk — once it's gone,
    // any honest hand motion shows up cleanly in (ax_c, ay_c).
    const ax_c = ax - this.biasX;
    const ay_c = ay - this.biasY;
    const mag = Math.hypot(ax_c, ay_c);
    this._lastAccelMag = mag;

    if (mag < this.ZUPT_ACCEL_THRESHOLD) {
      // The probe is sitting still. Snap velocity to zero (ZUPT) and
      // re-learn bias from the raw accel — this is what keeps the integrator
      // honest across temperature drift and orientation changes.
      this.biasX = this.biasX * (1 - this.BIAS_ALPHA) + ax * this.BIAS_ALPHA;
      this.biasY = this.biasY * (1 - this.BIAS_ALPHA) + ay * this.BIAS_ALPHA;
      if (++this._stillSamples >= this.ZUPT_STILL_FRAMES) {
        this.velX = 0;
        this.velY = 0;
      }
    } else {
      this._stillSamples = 0;
    }

    // Rotate body-frame accel into world frame using yaw only — pitch/roll
    // tilt the probe but the chest surface itself is what defines nx/ny.
    const yawRad = yawDeg * Math.PI / 180;
    const cos = Math.cos(yawRad), sin = Math.sin(yawRad);
    const wx = ax_c * cos - ay_c * sin;
    const wy = ax_c * sin + ay_c * cos;

    this.velX = (this.velX + wx * dt) * this.VEL_DAMPING;
    this.velY = (this.velY + wy * dt) * this.VEL_DAMPING;

    const newNx = Math.max(0, Math.min(1, this.posNx + (this.velX * dt) / this.METERS_PER_NORM));
    const newNy = Math.max(0, Math.min(1, this.posNy + (this.velY * dt) / this.METERS_PER_NORM));
    // Bounce off the chest-map edges by zeroing velocity into the wall, so
    // the probe sticks at the boundary instead of building up unbounded vel.
    if (newNx === 0 || newNx === 1) this.velX = 0;
    if (newNy === 0 || newNy === 1) this.velY = 0;
    this.posNx = newNx;
    this.posNy = newNy;

    this.onPosition?.(this.posNx, this.posNy, mag, "imu");
  }

  /**
   * Process pressure mat data to determine probe position.
   * Called externally when aggregator data is received (e.g., via serial WebSocket).
   * @param {number} row - Hot spot row (0-15)
   * @param {number} col - Hot spot column (0-15)
   * @param {number} pressure - Peak pressure value (0-4095)
   */
  updatePressurePosition(row, col, pressure) {
    if (pressure < 200) return; // No contact
    const { nx, ny } = this.gridToChestPosition(row, col);
    this._lastMatMs = performance.now();
    this.posNx = nx;
    this.posNy = ny;
    this.velX = 0;
    this.velY = 0;
    this.onPosition?.(nx, ny, pressure / 4095, "mat");
  }
}

// ---------------------------------------------------------------------------
// Training Panel
// ---------------------------------------------------------------------------

class TrainingPanel {
  constructor() { this.losses = []; }

  startTraining() {
    const params = {
      epochs: parseInt(document.getElementById("train-epochs").value),
      batch_size: parseInt(document.getElementById("train-batch").value),
      n_per_class: parseInt(document.getElementById("train-npc").value),
      image_size: parseInt(document.getElementById("train-imgsize").value),
      from_scratch: document.getElementById("train-scratch").checked,
    };
    fetch("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params) })
      .then(r => r.json()).then(data => {
        if (data.error) { alert(data.error); return; }
        document.getElementById("train-progress").classList.remove("hidden");
        document.getElementById("train-loss-chart").classList.remove("hidden");
        this.losses = [];
        this._pollSSE(data.job_id);
      }).catch(e => alert("Failed: " + e));
  }

  _pollSSE(jobId) {
    const es = new EventSource(`/api/train/${jobId}/status`);
    es.onmessage = (evt) => {
      const d = JSON.parse(evt.data);
      if (d.status === "completed" || d.status === "failed") {
        es.close();
        document.getElementById("train-progress-text").textContent = d.status === "completed" ? "Complete." : `Failed: ${d.error}`;
        return;
      }
      if (d.epoch !== undefined) {
        const pct = ((d.epoch + 1) / d.total_epochs * 100).toFixed(0);
        document.getElementById("train-progress-fill").style.width = pct + "%";
        document.getElementById("train-progress-text").textContent = `Epoch ${d.epoch + 1}/${d.total_epochs} | Loss: ${d.loss?.toFixed(5) || "--"}`;
        if (d.loss) { this.losses.push(d.loss); this._drawChart(); }
      }
    };
    es.onerror = () => es.close();
  }

  _drawChart() {
    const c = document.getElementById("train-loss-chart");
    const ctx = c.getContext("2d");
    const w = c.width, h = c.height;
    ctx.fillStyle = "#182030";
    ctx.fillRect(0, 0, w, h);
    if (this.losses.length < 2) return;
    const mx = Math.max(...this.losses), mn = Math.min(...this.losses), rng = mx - mn || 1;
    ctx.strokeStyle = "#5a9ec0";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < this.losses.length; i++) {
      const x = (i / (this.losses.length - 1)) * w;
      const y = h - ((this.losses[i] - mn) / rng) * (h - 14) - 7;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  startDataset() {
    const params = { n_per_class: parseInt(document.getElementById("ds-npc").value), output_path: document.getElementById("ds-output").value };
    fetch("/api/dataset", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params) })
      .then(r => r.json()).then(data => {
        if (data.error) { alert(data.error); return; }
        document.getElementById("ds-progress").classList.remove("hidden");
        const es = new EventSource(`/api/dataset/${data.job_id}/status`);
        es.onmessage = (evt) => {
          const d = JSON.parse(evt.data);
          if (d.status === "completed" || d.status === "failed") { es.close(); document.getElementById("ds-progress-text").textContent = d.status === "completed" ? "Complete." : `Failed: ${d.error}`; return; }
          if (d.progress !== undefined) {
            const pct = (d.progress * 100).toFixed(0);
            document.getElementById("ds-progress-fill").style.width = pct + "%";
            document.getElementById("ds-progress-text").textContent = `${pct}% — ${d.message || ""}`;
          }
        };
        es.onerror = () => es.close();
      }).catch(e => alert("Failed: " + e));
  }

  generatePreview() {
    const type = document.getElementById("preview-type").value;
    const scenario = document.getElementById("preview-scenario").value;
    fetch("/api/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ type, scenario }) })
      .then(r => r.blob()).then(blob => {
        document.getElementById("preview-img").src = URL.createObjectURL(blob);
        document.getElementById("preview-container").classList.remove("hidden");
      }).catch(e => alert("Failed: " + e));
  }

  loadCheckpoints() {
    fetch("/api/checkpoints").then(r => r.json()).then(data => {
      const el = document.getElementById("checkpoints-list");
      if (!data.checkpoints?.length) { el.textContent = "No checkpoints."; return; }
      el.innerHTML = data.checkpoints.map(c => `<div>${c.name} (${c.size_mb} MB)</div>`).join("");
    });
  }
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  const ws = new WebSocketManager("/ws/simulator");
  const bmode = new BmodeRenderer("bmode-canvas");
  const mmode = new MmodeRenderer("mmode-canvas");
  const zoneMap = new ChestZoneMap("zone-svg");
  const probeOverlay = new ProbeOverlayRenderer("probe-canvas");
  const bleManager = new BLEProbeManager();
  let training = null;

  // Probe pose → effective chest-surface scan target.
  //
  // Pitch and roll tilt the beam axis off-vertical; at scan depth D the beam
  // meets the lung at D*tan(theta) away from the contact point. Normalized
  // against a ~40cm chest, that's ~0.005 per degree for small angles. Yaw
  // rotates which body-frame axis the tilt projects onto.
  //
  // The result is sent as `probe_position` so the server-side Gaussian zone
  // blender (sigma=0.14, in `get_interpolated_frame_data`) reweights toward
  // whichever lung field the angled beam is aimed at — the image content
  // changes, not just the canvas transform.
  const TILT_K = 0.005;

  function computeBeamOffset(yaw, pitch, roll) {
    const bx = roll  * TILT_K;  // probe-body lateral tilt → chest x
    const by = pitch * TILT_K;  // probe-body fore/aft tilt → chest y
    const yawRad = yaw * Math.PI / 180;
    const c = Math.cos(yawRad), s = Math.sin(yawRad);
    return { dx: bx * c - by * s, dy: bx * s + by * c };
  }

  let _probeUpdateThrottle = 0;
  let _probeMovingTimer = null;
  function sendProbeUpdate(immediate = false) {
    if (!probeOverlay.probeVisible) return;
    const now = Date.now();
    if (!immediate && now - _probeUpdateThrottle < 80) return;
    _probeUpdateThrottle = now;

    const { dx, dy } = computeBeamOffset(probeOverlay.yaw, probeOverlay.pitch, probeOverlay.roll);
    const effNx = Math.max(0, Math.min(1, probeOverlay.probeNx + dx));
    const effNy = Math.max(0, Math.min(1, probeOverlay.probeNy + dy));
    probeOverlay.setBeamTarget(effNx, effNy);

    ws.send({
      type: "probe_position",
      nx: effNx, ny: effNy,
      raw_nx: probeOverlay.probeNx, raw_ny: probeOverlay.probeNy,
      yaw: probeOverlay.yaw, pitch: probeOverlay.pitch, roll: probeOverlay.roll,
    });

    // Dim the play button briefly while we're actively reframing — the server
    // is pushing a fresh blended stack and resuming playback mid-burst looks jumpy.
    const playBtn = document.getElementById("play-btn");
    playBtn.disabled = true;
    playBtn.style.opacity = "0.3";
    clearTimeout(_probeMovingTimer);
    _probeMovingTimer = setTimeout(() => {
      playBtn.disabled = false;
      playBtn.style.opacity = "1";
    }, 400);
  }

  // Free probe drag on chest map
  probeOverlay.onProbeMove = (nx, ny, snappedZone) => {
    if (window.probe3d) window.probe3d.setPosition(nx, ny);
    if (snappedZone) {
      ws.send({ type: "select_zone", zone: snappedZone });
      // Auto-play on probe snap to zone
      if (!isPlaying) {
        isPlaying = true; isFrozen = false;
        ws.send({ type: "unfreeze" });
        ws.send({ type: "play" });
        document.getElementById("play-btn").innerHTML = "&#9646;&#9646; Pause";
        document.getElementById("play-btn").classList.add("playing");
        document.getElementById("freeze-btn").textContent = "Freeze";
        document.getElementById("freeze-overlay").classList.add("hidden");
      }
    } else {
      sendProbeUpdate();
    }
  };

  let isPlaying = false, isFrozen = false, testingMode = false;
  let scenariosData = {};

  // ----- Training Studio: check GPU availability -----
  fetch("/api/system-info").then(r => r.json()).then(info => {
    const container = document.getElementById("training-content");
    if (info.gpu_available) {
      container.innerHTML = buildTrainingHTML();
      training = new TrainingPanel();
      document.getElementById("train-start-btn").addEventListener("click", () => training.startTraining());
      document.getElementById("ds-start-btn").addEventListener("click", () => training.startDataset());
      document.getElementById("preview-btn").addEventListener("click", () => training.generatePreview());
      document.getElementById("refresh-ckpts-btn").addEventListener("click", () => training.loadCheckpoints());
      // Populate preview scenario dropdown
      for (const [key, info2] of Object.entries(scenariosData)) {
        document.getElementById("preview-scenario")?.appendChild(new Option(info2.name, key));
      }
    } else {
      container.innerHTML = `
        <div class="training-unavailable">
          <h2>Training Studio</h2>
          <p>Training requires GPU acceleration. Connect to the ThinkStation PGX via SSH to use Training Studio.</p>
          <p><code>ssh ahastava@thinkstationpgx-9c7e</code></p>
          <p><code>cd ~/moculus && ./run.sh --web</code></p>
        </div>`;
    }
  }).catch(() => {
    document.getElementById("training-content").innerHTML = `
      <div class="training-unavailable">
        <h2>Training Studio</h2>
        <p>Could not determine system capabilities. Training may not be available in offline mode.</p>
      </div>`;
  });

  // Tab switching
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
      if (btn.dataset.tab === "training" && training) training.loadCheckpoints();
    });
  });

  // Load scenarios
  fetch("/api/scenarios").then(r => r.json()).then(data => {
    scenariosData = data;
    const sel = document.getElementById("scenario-select");
    for (const [key, info] of Object.entries(data)) {
      sel.appendChild(new Option(info.name, key));
    }
  });

  // WebSocket handler
  ws.onMessage = (msg) => {
    if (msg.type === "case_loaded") handleCaseLoaded(msg);
    else if (msg.type === "frame") handleFrame(msg);
    else if (msg.type === "diagnosis_result") handleDiagnosisResult(msg);
  };

  function handleCaseLoaded(msg) {
    const p = msg.patient;
    document.getElementById("patient-demo").textContent = `${p.age}${p.sex ? " " + p.sex : ""}`;
    document.getElementById("patient-cc").textContent = p.chief_complaint || "--";
    document.getElementById("patient-hx").textContent = p.history || "";
    document.getElementById("patient-vitals").textContent = p.vitals_text || "";
    if (msg.scenario_key) document.getElementById("scenario-select").value = msg.scenario_key;
    document.getElementById("scenario-desc").textContent = msg.scenario_description || "";
    if (msg.zones) { zoneMap.setZones(msg.zones); probeOverlay.setZones(msg.zones); }
    zoneMap.reset(); probeOverlay.setActiveZone(null); probeOverlay.setExamined([]);
    probeOverlay.probeVisible = false; // hide probe until user clicks
    if (msg.diagnosis_options) {
      const sel = document.getElementById("diagnosis-select");
      sel.innerHTML = '<option value="">-- Select diagnosis --</option>';
      msg.diagnosis_options.forEach(dx => sel.appendChild(new Option(dx, dx)));
    }
    document.getElementById("zone-label").textContent = "Select a zone";
    document.getElementById("sliding-badge").className = "hidden";
    document.getElementById("mmode-pattern").textContent = "--";
    ["finding-zone", "finding-pathology", "finding-sliding", "finding-mmode"].forEach(id => document.getElementById(id).textContent = "--");
    document.getElementById("diagnosis-result").textContent = "";
    document.getElementById("frame-slider").value = 0;
    document.getElementById("frame-counter").textContent = "--";
    document.getElementById("playback-label").textContent = "--";
    bmode.ctx.fillStyle = "#000"; bmode.ctx.fillRect(0, 0, bmode.canvas.width, bmode.canvas.height);
    mmode.ctx.fillStyle = "#000"; mmode.ctx.fillRect(0, 0, mmode.canvas.width, mmode.canvas.height);
  }

  // Clinical prose names for pathologies and zones
  const CLINICAL_NAMES = {
    "normal_a_profile": "Normal lung (A-profile)",
    "pneumothorax": "Pneumothorax (absent sliding)",
    "b_lines_focal": "Focal B-lines (1\u20132 per field)",
    "b_lines_diffuse": "Diffuse B-lines (B-profile)",
    "consolidation": "Pulmonary consolidation",
    "pleural_effusion": "Pleural effusion",
    "ards_white_lung": "ARDS / white lung syndrome",
    "lung_point": "Lung point (pathognomonic for PTX)",
    "pleural_thickening": "Pleural thickening / irregularity",
    "interstitial_syndrome": "Interstitial syndrome",
  };
  const ZONE_NAMES = {
    "UPPER_BLUE_L": "Upper BLUE Point \u2014 Left",
    "UPPER_BLUE_R": "Upper BLUE Point \u2014 Right",
    "LOWER_BLUE_L": "Lower BLUE Point \u2014 Left",
    "LOWER_BLUE_R": "Lower BLUE Point \u2014 Right",
    "PLAPS_L": "PLAPS Point \u2014 Left",
    "PLAPS_R": "PLAPS Point \u2014 Right",
    "DIAPHRAGM_L": "Diaphragm \u2014 Left",
    "DIAPHRAGM_R": "Diaphragm \u2014 Right",
  };

  function handleFrame(msg) {
    // Pass current probe orientation to B-mode renderer
    bmode.setProbeOrientation(probeOverlay.yaw, probeOverlay.pitch, probeOverlay.roll);
    bmode.drawFrame(msg.bmode, msg);
    mmode.drawFrame(msg.mmode);

    const zoneProse = ZONE_NAMES[msg.zone] || msg.zone.replace(/_/g, " ");
    document.getElementById("zone-label").textContent = zoneProse;
    document.getElementById("frame-counter").textContent = `${msg.frame_idx}/${msg.n_frames}`;
    document.getElementById("playback-label").textContent = `${msg.frame_idx}/${msg.n_frames}`;
    document.getElementById("frame-slider").value = msg.frame_idx;
    document.getElementById("frame-slider").max = msg.n_frames - 1;

    const badge = document.getElementById("sliding-badge");
    badge.classList.remove("hidden");
    badge.textContent = msg.sliding ? "LUNG SLIDING PRESENT" : "LUNG SLIDING ABSENT";
    badge.className = msg.sliding ? "present" : "absent";

    const pat = document.getElementById("mmode-pattern");
    pat.textContent = msg.mmode_pattern === "seashore" ? "Seashore sign" : "Stratosphere sign";
    pat.className = msg.mmode_pattern === "seashore" ? "green mono" : "red mono";

    // Clinical prose in findings
    document.getElementById("finding-zone").textContent = zoneProse;
    const pathProse = CLINICAL_NAMES[msg.pathology] || msg.pathology;
    document.getElementById("finding-pathology").textContent = pathProse;
    document.getElementById("finding-pathology").className = "orange";
    const sl = document.getElementById("finding-sliding");
    sl.textContent = msg.sliding
      ? "Pleural sliding: Present \u2014 visceral-parietal interface mobile"
      : "Pleural sliding: Absent \u2014 no pleural motion detected";
    sl.className = msg.sliding ? "green" : "red";
    const mm = document.getElementById("finding-mmode");
    mm.textContent = msg.mmode_pattern === "seashore"
      ? "M-mode: Seashore sign (granular below pleural line)"
      : "M-mode: Stratosphere sign (parallel lines \u2014 barcode pattern)";
    mm.className = msg.mmode_pattern === "seashore" ? "green" : "red";

    zoneMap.setActive(msg.zone);
    zoneMap.setExamined(msg.examined_zones || []);
    probeOverlay.setActiveZone(msg.zone);
    probeOverlay.setExamined(msg.examined_zones || []);

    const ov = document.getElementById("freeze-overlay");
    if (msg.frozen) ov.classList.remove("hidden"); else ov.classList.add("hidden");
  }

  function handleDiagnosisResult(msg) {
    const el = document.getElementById("diagnosis-result");
    if (msg.correct === true) {
      el.style.color = "var(--green)";
      el.textContent = `CORRECT\n\n${msg.answer}\n\n${msg.explanation}\n\nBLUE: ${msg.blue_profile}`;
    } else if (msg.correct === false) {
      el.style.color = "var(--red)";
      el.textContent = `INCORRECT\n\nYours: ${msg.selected}\nCorrect: ${msg.answer}\n\n${msg.explanation}\n\nBLUE: ${msg.blue_profile}`;
    } else {
      el.style.color = "var(--orange)";
      el.textContent = `${msg.answer}\n\n${msg.explanation}\n\nBLUE: ${msg.blue_profile}`;
    }
  }

  // Zone click — snap probe to zone and auto-play (mimics real ultrasound: always live)
  zoneMap.onZoneClick = (z) => {
    probeOverlay.snapToZone(z);
    if (window.probe3d) window.probe3d.setPosition(probeOverlay.probeNx, probeOverlay.probeNy);
    ws.send({ type: "select_zone", zone: z });
    // Always start/restart cine loop on zone select (real US is always live)
    isFrozen = false;
    document.getElementById("freeze-btn").textContent = "Freeze";
    document.getElementById("freeze-overlay").classList.add("hidden");
    isPlaying = true;
    const btn = document.getElementById("play-btn");
    ws.send({ type: "unfreeze" });
    ws.send({ type: "play" });
    btn.innerHTML = "&#9646;&#9646; Pause";
    btn.classList.add("playing");
  };

  // Playback
  document.getElementById("play-btn").addEventListener("click", () => {
    isPlaying = !isPlaying;
    const btn = document.getElementById("play-btn");
    if (isPlaying) { ws.send({ type: "play" }); btn.innerHTML = "&#9646;&#9646; Pause"; btn.classList.add("playing"); }
    else { ws.send({ type: "pause" }); btn.innerHTML = "&#9654; Play"; btn.classList.remove("playing"); }
  });
  document.getElementById("frame-slider").addEventListener("input", (e) => ws.send({ type: "set_frame", frame: parseInt(e.target.value) }));

  // Freeze
  document.getElementById("freeze-btn").addEventListener("click", () => {
    isFrozen = !isFrozen;
    const btn = document.getElementById("freeze-btn");
    if (isFrozen) {
      ws.send({ type: "freeze" }); btn.textContent = "Unfreeze"; isPlaying = false;
      document.getElementById("play-btn").innerHTML = "&#9654;"; document.getElementById("play-btn").classList.remove("playing");
    } else { ws.send({ type: "unfreeze" }); btn.textContent = "Freeze"; }
  });

  // Mode
  document.getElementById("mode-practice").addEventListener("click", () => {
    testingMode = false;
    document.getElementById("mode-practice").classList.add("active"); document.getElementById("mode-test").classList.remove("active");
    document.getElementById("scenario-card").classList.remove("hidden"); document.getElementById("diagnosis-card").classList.add("hidden");
    ws.send({ type: "set_mode", mode: "practice", scenario: document.getElementById("scenario-select").value || "normal" });
  });
  document.getElementById("mode-test").addEventListener("click", () => {
    testingMode = true;
    document.getElementById("mode-test").classList.add("active"); document.getElementById("mode-practice").classList.remove("active");
    document.getElementById("scenario-card").classList.add("hidden"); document.getElementById("diagnosis-card").classList.remove("hidden");
    ws.send({ type: "set_mode", mode: "test" });
  });

  document.getElementById("scenario-select").addEventListener("change", (e) => {
    if (!testingMode && e.target.value) {
      // Stop playback and reset on scenario change
      if (isPlaying) {
        isPlaying = false;
        ws.send({ type: "pause" });
        document.getElementById("play-btn").innerHTML = "&#9654; Play";
        document.getElementById("play-btn").classList.remove("playing");
      }
      isFrozen = false;
      document.getElementById("freeze-btn").textContent = "Freeze";
      document.getElementById("freeze-overlay").classList.add("hidden");
      ws.send({ type: "set_scenario", scenario: e.target.value });
    }
  });

  // New case
  document.getElementById("new-case-btn").addEventListener("click", () => {
    ws.send({ type: "new_case", mode: testingMode ? "test" : "practice", scenario: testingMode ? undefined : (document.getElementById("scenario-select").value || "normal") });
    isFrozen = false; document.getElementById("freeze-btn").textContent = "Freeze";
    document.getElementById("freeze-overlay").classList.add("hidden");
    isPlaying = false; document.getElementById("play-btn").innerHTML = "&#9654;"; document.getElementById("play-btn").classList.remove("playing");
  });

  // Diagnosis
  document.getElementById("submit-diagnosis-btn").addEventListener("click", () => {
    const dx = document.getElementById("diagnosis-select").value;
    if (dx) ws.send({ type: "submit_diagnosis", diagnosis: dx });
  });
  document.getElementById("reveal-btn").addEventListener("click", () => ws.send({ type: "reveal_answer" }));

  // BLE
  bleManager.onIMU = (y, p, r) => {
    probeOverlay.setIMU(y, p, r);
    bmode.setProbeOrientation(y, p, r);
    if (window.probe3d) window.probe3d.setOrientation(y, p, r);
    document.getElementById("ble-imu").textContent = `Yaw(Z): ${y.toFixed(1)}° Pitch(Y): ${p.toFixed(1)}° Roll(X): ${r.toFixed(1)}°`;
    // Mirror onto sliders so user sees live values (sliders are read-only while BLE drives).
    if (yawSlider) {
      yawSlider.value = Math.round(y);
      pitchSlider.value = Math.round(p);
      rollSlider.value = Math.round(r);
      document.getElementById("yaw-value").textContent = Math.round(y);
      document.getElementById("pitch-value").textContent = Math.round(p);
      document.getElementById("roll-value").textContent = Math.round(r);
    }
    sendProbeUpdate();
  };

  // Probe position on chest diagram. Source is "mat" (absolute, drift-free)
  // or "imu" (double-integrated linear accel — used only when mat is absent).
  // For "imu" source, `confidence` is the current accel magnitude (m/s²),
  // which we surface as a live readout so the user can confirm the
  // accelerometer is actually driving translation.
  bleManager.onPosition = (nx, ny, confidence, source) => {
    probeOverlay.setProbePosition(nx, ny);
    if (window.probe3d) window.probe3d.setPosition(nx, ny);
    const t = document.getElementById("ble-translation");
    if (t) {
      const tag = source === "mat" ? "mat" : "imu";
      const mag = source === "imu" ? `${confidence.toFixed(2)} m/s²` : "—";
      t.textContent = `Pos (${tag}): ${nx.toFixed(2)}, ${ny.toFixed(2)} | |a|: ${mag}`;
    }
    sendProbeUpdate();
  };
  document.getElementById("ble-connect-btn").addEventListener("click", () => { if (bleManager.connected) bleManager.disconnect(); else bleManager.connect(); });
  const calibrateOverlay = document.getElementById("calibrate-overlay");
  const showCalibrateOverlay = () => calibrateOverlay.classList.remove("hidden");
  const hideCalibrateOverlay = () => calibrateOverlay.classList.add("hidden");
  document.getElementById("ble-calibrate-btn").addEventListener("click", showCalibrateOverlay);
  document.getElementById("calibrate-cancel-btn").addEventListener("click", hideCalibrateOverlay);
  document.getElementById("calibrate-confirm-btn").addEventListener("click", () => {
    bleManager.calibrateYaw();
    bleManager.calibratePosition();  // anchor IMU integrator at chest center
    hideCalibrateOverlay();
  });
  calibrateOverlay.addEventListener("click", (e) => {
    if (e.target === calibrateOverlay) hideCalibrateOverlay();
  });
  document.getElementById("probe-reset-btn").addEventListener("click", () => {
    probeOverlay.resetOrientation();
    bmode.setProbeOrientation(0, 0, 0);
    if (window.probe3d) window.probe3d.resetOrientation();
    document.getElementById("yaw-slider").value = 0;
    document.getElementById("pitch-slider").value = 0;
    document.getElementById("roll-slider").value = 0;
    document.getElementById("yaw-value").textContent = "0";
    document.getElementById("pitch-value").textContent = "0";
    document.getElementById("roll-value").textContent = "0";
    sendProbeUpdate(true);
  });

  // Orientation sliders (manual control when BLE not connected)
  const yawSlider = document.getElementById("yaw-slider");
  const pitchSlider = document.getElementById("pitch-slider");
  const rollSlider = document.getElementById("roll-slider");

  function onSliderInput() {
    // Sliders are read-only mirrors while BLE drives; ignore input then.
    if (probeOverlay.imuConnected) return;
    const y = parseFloat(yawSlider.value);
    const p = parseFloat(pitchSlider.value);
    const r = parseFloat(rollSlider.value);
    document.getElementById("yaw-value").textContent = y;
    document.getElementById("pitch-value").textContent = p;
    document.getElementById("roll-value").textContent = r;
    probeOverlay.yaw = y;
    probeOverlay.pitch = p;
    probeOverlay.roll = r;
    probeOverlay._updateAngleDisplay();
    probeOverlay._dirty = true;
    bmode.setProbeOrientation(y, p, r);
    if (window.probe3d) window.probe3d.setOrientation(y, p, r);
    sendProbeUpdate();
  }

  yawSlider.addEventListener("input", onSliderInput);
  pitchSlider.addEventListener("input", onSliderInput);
  rollSlider.addEventListener("input", onSliderInput);

  // Sync sliders + 3D viewer when probe orientation changes via drag
  probeOverlay.onOrientationChange = (y, p, r) => {
    bmode.setProbeOrientation(y, p, r);
    if (window.probe3d) window.probe3d.setOrientation(y, p, r);
    yawSlider.value = Math.round(y);
    pitchSlider.value = Math.round(p);
    rollSlider.value = Math.round(r);
    document.getElementById("yaw-value").textContent = Math.round(y);
    document.getElementById("pitch-value").textContent = Math.round(p);
    document.getElementById("roll-value").textContent = Math.round(r);
    sendProbeUpdate();
  };

  // Disable sliders when BLE is active
  bleManager.onStatus = (msg, connected) => {
    document.getElementById("ble-status").textContent = msg;
    document.getElementById("ble-status").className = connected ? "green small" : "dim small";
    document.getElementById("ble-calibrate-btn").disabled = !connected;
    probeOverlay.setIMUConnected(connected);
    document.getElementById("ble-connect-btn").textContent = connected ? "Disconnect" : "Connect Probe";
    const sliderPanel = document.getElementById("orientation-sliders");
    if (connected) {
      sliderPanel.classList.add("ble-active");
      // Anchor the integrator at chest center so the probe is immediately
      // visible and accelerometer-driven translation has a meaningful origin.
      bleManager.calibratePosition(0.5, 0.5);
      probeOverlay.setProbePosition(0.5, 0.5);
      if (window.probe3d) window.probe3d.setPosition(0.5, 0.5);
      sendProbeUpdate(true);
    } else {
      sliderPanel.classList.remove("ble-active");
      const t = document.getElementById("ble-translation");
      if (t) t.textContent = "Pos: -- | |a|: --";
    }
  };
});

// ---------------------------------------------------------------------------
// Training Studio HTML (injected only when GPU available)
// ---------------------------------------------------------------------------

function buildTrainingHTML() {
  return `<div class="training-layout">
    <div class="card training-card">
      <div class="card-title">Train Diffusion Model</div>
      <div class="form-grid">
        <label>Epochs</label><input type="number" id="train-epochs" class="input" value="100">
        <label>Batch Size</label><input type="number" id="train-batch" class="input" value="4">
        <label>Samples/Class</label><input type="number" id="train-npc" class="input" value="200">
        <label>Image Size</label><input type="number" id="train-imgsize" class="input" value="256">
        <label>From Scratch</label><input type="checkbox" id="train-scratch">
      </div>
      <button id="train-start-btn" class="btn btn-success">Start Training</button>
      <div id="train-progress" class="progress-container hidden">
        <div class="progress-bar"><div id="train-progress-fill" class="progress-fill"></div></div>
        <div id="train-progress-text" class="dim small">--</div>
      </div>
      <canvas id="train-loss-chart" class="loss-chart hidden" width="400" height="100"></canvas>
    </div>
    <div class="card training-card">
      <div class="card-title">Build HDF5 Dataset</div>
      <div class="form-grid">
        <label>Samples/Class</label><input type="number" id="ds-npc" class="input" value="500">
        <label>Output Path</label><input type="text" id="ds-output" class="input" value="data/lung_us_moculus.h5">
      </div>
      <button id="ds-start-btn" class="btn btn-success">Build Dataset</button>
      <div id="ds-progress" class="progress-container hidden">
        <div class="progress-bar"><div id="ds-progress-fill" class="progress-fill"></div></div>
        <div id="ds-progress-text" class="dim small">--</div>
      </div>
    </div>
    <div class="card training-card">
      <div class="card-title">Generate Previews</div>
      <div class="form-grid">
        <label>Type</label>
        <select id="preview-type" class="select">
          <option value="grid">Pathology Grid</option>
          <option value="bmode_mmode">B-mode / M-mode</option>
          <option value="scenario">Scenario</option>
        </select>
        <label>Scenario</label><select id="preview-scenario" class="select"></select>
      </div>
      <button id="preview-btn" class="btn btn-primary">Generate</button>
      <div id="preview-container" class="hidden"><img id="preview-img" alt="Preview"></div>
    </div>
    <div class="card training-card">
      <div class="card-title">Checkpoints</div>
      <div id="checkpoints-list" class="dim small">Loading...</div>
      <button id="refresh-ckpts-btn" class="btn btn-small">Refresh</button>
    </div>
  </div>`;
}
