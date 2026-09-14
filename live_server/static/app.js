let currentSession = null;
let pollTimer = null;
let replayTimer = null;
let datasetMatches = [];
let datasetAllMatches = [];
let datasetCatalogStats = {};
let demoBuilderCatalog = null;
let selectedDemoSeed = null;
let latestBevData = null;
let selectedDatasetInstruction = "";
let selectedDatasetItem = null;
let lastHealthData = null;
let lastHealthCheckedAt = 0;
let lastGpuInventory = null;
let healthCheckInFlight = false;
let healthRefreshTimer = null;
let pollInFlight = false;
let datasetLoadSequence = 0;
let lastStatusText = "";
let lastStatusPanelAt = 0;
let streamCanvasRaf = null;
let streamCanvasLastDraw = 0;
let streamCanvasDrawCount = 0;
let streamCanvasFps = 0;
let streamCanvasFpsWindowAt = 0;
let streamCanvasHasFrame = false;
let streamCanvasClearAt = 0;
let streamCanvasCtx = null;
let streamCanvasMetrics = null;
let streamCanvasLastResizeCheck = 0;
let lastBevDrawAt = 0;
let lastTraceFetchAt = 0;
let cachedTraceBevData = null;
let delayedFrameFetchTimer = null;
let delayedFramePlayTimer = null;
let delayedFrameSession = null;
let delayedFrameItems = [];
let delayedFrameSeen = new Set();
let delayedFrameCursor = 0;
let delayedFrameTerminal = false;
let delayedFrameRaf = null;
let delayedFrameCurrentImage = null;
let delayedFramePreviousImage = null;
let delayedFrameFrameStartedAt = 0;
let delayedFrameImageCache = new Map();
let delayedFrameImageLoading = new Set();
let delayedFrameReplayRestarted = false;
let delayedFrameReplayStreamStarted = false;
let terminalVideoReplayStarted = false;
let terminalVideoQualityTimer = null;
let terminalVideoRafMonitor = null;
let terminalVideoRafLastAt = 0;
let terminalVideoRafLongGapCount = 0;
let terminalVideoReplayProfile = "primary";
let terminalVideoLastTime = 0;
let terminalVideoStallChecks = 0;
let terminalEnhancedReplayStarted = false;
let terminalEnhancedReplayApplied = false;
let terminalVideoTakeoverStarted = false;
let terminalVideoTakeoverApplied = false;
let terminalVideoTakeoverPending = false;
let terminalVideoTakeoverPreload = null;
let terminalVideoTakeoverObjectUrl = null;
let terminalVideoObjectUrl = null;
let terminalAnimationObjectUrls = [];
let terminalSpriteBitmaps = [];
let terminalAnimationChunkTimer = null;
let terminalAnimationPlaybackToken = 0;
let terminalSpriteRaf = null;
let terminalSpriteFrameTimer = null;
let terminalSpriteRenderScale = 1.0;
let terminalReplayPrefetchTimers = [];
let demoVideoPrewarmSessions = new Set();
let demoReplayBlobCache = new Map();
let demoReplayBlobInflight = new Map();
let demoReplayNeighborPrefetchTimers = [];
let demoReplayActive = false;
let viewerMessageText = null;
let viewerMessageHasFrame = null;

const $ = (id) => document.getElementById(id);

const STREAM_CANVAS_FPS = 24;
// Showcase mode favors continuous motion over minimal latency: keep a short
// playback buffer and consume queued frames instead of always jumping to latest.
const STREAM_QUERY = "fps=24&turbo=0&cinematic=1&interpolate=1&buffer=3.50&hold=1&hold_age=5.00&blend=0&speed=1.15&latest=0&transition_cap=24";
const STREAM_TEMPORAL_ALPHA = 1.0;
const STREAM_RENDER_SCALE = 1.0;
const TERMINAL_SPRITE_RENDER_DPR_CAP = 1.0;
const TERMINAL_SPRITE_RENDER_SCALE_MIN = 0.75;
const SHOWCASE_NATIVE_STREAM = true;
const SHOWCASE_DELAYED_FRAME_PLAYER = true;
// Start showing frames after a short buffer instead of hiding the whole flight
// until terminal state. This keeps high-resolution capture without a blank UI.
const SHOWCASE_RECORD_ONLY_UNTIL_TERMINAL = false;
// Start playback once the initial buffer is ready, but slow down automatically
// when live capture cannot keep up. On terminal sessions, replay from the
// beginning so the final demo pass is smooth even if the source capture was
// sparse while the model was running.
const SHOWCASE_RECORD_THEN_REPLAY = false;
const SHOWCASE_REPLAY_FROM_START_ON_TERMINAL = true;
const SHOWCASE_TERMINAL_VIDEO_REPLAY = true;
const SHOWCASE_TERMINAL_ANIMATION_REPLAY = true;
const SHOWCASE_TERMINAL_MJPEG_REPLAY = true;
const SHOWCASE_TERMINAL_REPLAY_FPS = 24;
const SHOWCASE_TERMINAL_SMOOTH_MULTIPLIER = 1.5;
const SHOWCASE_TERMINAL_LIGHT_WIDTH = 420;
const SHOWCASE_TERMINAL_LIGHT_SMOOTH_MULTIPLIER = 1.15;
const SHOWCASE_TERMINAL_PURE_VIDEO_REPLAY = true;
const SHOWCASE_TERMINAL_PURE_VIDEO_FPS = 20;
const SHOWCASE_TERMINAL_PURE_VIDEO_STRIDE = 4;
const SHOWCASE_TERMINAL_PURE_VIDEO_WIDTH = 400;
// Verified showcase replays must preserve the captured AirSim frames.  Optical
// flow made the motion look denser but invented/warped building detail.
const SHOWCASE_DEMO_SMOOTH_VIDEO_FPS = 8;
const SHOWCASE_DEMO_SMOOTH_VIDEO_STRIDE = 1;
const SHOWCASE_DEMO_SMOOTH_VIDEO_WIDTH = 768;
const SHOWCASE_DEMO_SMOOTH_VIDEO_SHARPEN = 0.0;
const SHOWCASE_DEMO_LITE_VIDEO_WIDTH = 768;
const SHOWCASE_DEMO_SMOOTH_VIDEO_MULTIPLIER = 1.0;
const SHOWCASE_DEMO_LITE_VIDEO_FPS = 8;
const SHOWCASE_DEMO_LITE_VIDEO_MULTIPLIER = 1.0;
const SHOWCASE_TERMINAL_ULTRA_VIDEO_FPS = 12;
const SHOWCASE_TERMINAL_ULTRA_VIDEO_STRIDE = 6;
const SHOWCASE_TERMINAL_ULTRA_VIDEO_WIDTH = 320;
const SHOWCASE_TERMINAL_STABLE_ANIMATION_FPS = 10;
const SHOWCASE_TERMINAL_STABLE_ANIMATION_STRIDE = 8;
const SHOWCASE_TERMINAL_STABLE_ANIMATION_WIDTH = 320;
const SHOWCASE_TERMINAL_STABLE_ANIMATION_QUALITY = 35;
const SHOWCASE_TERMINAL_FAST_PREVIEW = true;
const SHOWCASE_TERMINAL_FAST_STRIDE = 3;
const SHOWCASE_TERMINAL_ENHANCED_REPLAY = false;
const SHOWCASE_TERMINAL_ENHANCED_WIDTH = 480;
const SHOWCASE_TERMINAL_ENHANCED_SMOOTH_MULTIPLIER = 1.2;
const SHOWCASE_TERMINAL_BLOB_PLAYBACK = true;
const SHOWCASE_TERMINAL_ANIMATION_QUALITY = 40;
const SHOWCASE_TERMINAL_CHUNKED_ANIMATION_REPLAY = true;
const SHOWCASE_TERMINAL_SPRITE_REPLAY = true;
const SHOWCASE_TERMINAL_VIDEO_TAKEOVER = true;
const SHOWCASE_TERMINAL_VIDEO_TAKEOVER_FPS = 20;
const SHOWCASE_TERMINAL_VIDEO_TAKEOVER_STRIDE = 4;
const SHOWCASE_TERMINAL_VIDEO_TAKEOVER_MIN_PROGRESS = 0.38;
const SHOWCASE_TERMINAL_VIDEO_TAKEOVER_PRELOAD_TIMEOUT_MS = 22000;
const SHOWCASE_TERMINAL_VIDEO_TAKEOVER_START_DELAY_MS = 8200;
const SHOWCASE_TERMINAL_VIDEO_TAKEOVER_WIDTH = 400;
const SHOWCASE_TERMINAL_ANIMATION_CHUNK_FRAMES = 30;
const SHOWCASE_TERMINAL_SPRITE_COLUMNS = 6;
const SHOWCASE_TERMINAL_REPLAY_PREFETCH_AHEAD = 6;
const SHOWCASE_TERMINAL_REPLAY_PREFETCH_STAGGER_MS = 90;
const SHOWCASE_TERMINAL_BACKGROUND_PREFETCH_AHEAD = 4;
const SHOWCASE_TERMINAL_BACKGROUND_PREFETCH_STAGGER_MS = 160;
const SHOWCASE_TERMINAL_SPRITE_START_BUFFER_CHUNKS = 4;
const SHOWCASE_TERMINAL_SPRITE_KEEP_BEHIND = 2;
const SHOWCASE_TERMINAL_SPRITE_KEEP_AHEAD = 8;
const SHOWCASE_TERMINAL_SPRITE_FRAME_BITMAPS = true;
const SHOWCASE_TERMINAL_SPRITE_FRAME_BITMAP_AHEAD = 1;
const SHOWCASE_TERMINAL_SPRITE_BITMAP_PRESSURE_DISABLES = 2;
const SHOWCASE_TERMINAL_SPRITE_MAX_READY_CHUNKS = 11;
const SHOWCASE_TERMINAL_SPRITE_LONG_GAP_MS = 110;
const DELAYED_FRAME_API_LIMIT = 8000;
const DELAYED_FRAME_FETCH_INTERVAL_MS = 650;
const DELAYED_FRAME_PLAY_INTERVAL_MS = 42;
const DELAYED_FRAME_LIVE_PLAY_INTERVAL_MS = 140;
const DELAYED_FRAME_MAX_LIVE_PLAY_INTERVAL_MS = 180;
const DELAYED_FRAME_CROSSFADE_MS = 80;
const DELAYED_FRAME_START_BUFFER = 45;
const DELAYED_FRAME_LOW_BUFFER = 15;
const DELAYED_FRAME_PRELOAD_AHEAD = 48;
const DELAYED_FRAME_MAX_IMAGE_LOADS = 12;
const STREAM_CLEAR_INTERVAL_MS = 4500;
const STATUS_PANEL_INTERVAL_MS = 7000;
const BEV_DRAW_INTERVAL_MS = 12000;
const TRACE_FETCH_INTERVAL_MS = 15000;
const LIVE_POLL_INTERVAL_MS = 3600;
const SHOWCASE_LIVE_DIAGNOSTICS = false;
const SHOWCASE_RUNNING_MINIMAL_UI = true;
const SHOWCASE_PREFER_VERIFIED_SOURCE_REPLAY = true;
const SHOWCASE_DEMO_BLOB_PREFETCH = true;
const SHOWCASE_DEMO_BLOB_CACHE_LIMIT = 32;
const SHOWCASE_DEMO_BLOB_CACHE_MAX_BYTES = 80 * 1024 * 1024;
const SHOWCASE_DEMO_NEIGHBOR_PREFETCH_COUNT = 1;
const SHOWCASE_DEMO_NEIGHBOR_PREFETCH_STAGGER_MS = 3200;
const SHOWCASE_DEMO_WEBM_RAF_LONG_GAP_LIMIT = 6;
const SHOWCASE_DEMO_SMOOTH_RAF_LONG_GAP_LIMIT = 4;
const HEALTH_REFRESH_INTERVAL_MS = 10000;
const HEALTH_STALE_MS = 30000;
const MAX_BEV_POINTS = 140;
const DEMO_FALLBACK_SPLIT = "train";
const DEMO_FALLBACK_SCENE = "16";
const SHOWCASE_FILTER_STORAGE_KEY = "airvln_showcase_filter";
const QUALITY_GATE_STORAGE_KEY = "airvln_quality_gate";
const CONFIRM_CONTROL_STORAGE_KEY = "airvln_confirm_control";
const SIMULATOR_PORT_STORAGE_KEY = "airvln_simulator_port";

function isTerminalSessionStatus(status) {
  return ["done", "error", "stopped"].includes(String(status || ""));
}

function liveRecordingMessage(data = null) {
  const step = data && data.step !== undefined && data.step !== null ? `step ${data.step}` : "running";
  const frames = data && data.stream_frame_count ? `${data.stream_frame_count} frames` : "recording frames";
  return `飞行运行中，正在录制平滑回放... ${step} | ${frames}`;
}

function setStatus(data) {
  let displayed = data;
  if (data && typeof data === "object" && data.session_id && data.status) {
    const success = Boolean(data.success_20m_with_surface ?? data.success_20m);
    const progressPercent = success
      ? 100
      : data.progress_to_goal === null || data.progress_to_goal === undefined
        ? null
        : Math.round(Number(data.progress_to_goal) * 100);
    displayed = {
      status: data.status,
      step: data.step,
      action: data.action,
      distance_to_goal: data.distance_to_goal,
      best_distance: data.best_distance,
      progress_percent: progressPercent,
      success_20m_with_surface: data.success_20m_with_surface,
      stream_fps: data.stream_fps,
      stream_recent_fps: data.stream_recent_fps,
      stream_pacing_label: data.stream_pacing_label,
      stream_recent_gap_max_sec: data.stream_recent_gap_max_sec,
      stream_recent_gaps_over_0_5s: data.stream_recent_gaps_over_0_5s,
      stream_average_fps: data.stream_average_fps,
      stream_frame_count: data.stream_frame_count,
      black_frame_drop_count: data.black_frame_drop_count,
      collision_rollback_count: data.collision_rollback_count,
      navigation_phase: data.navigation_phase,
      stop_reason: data.stop_reason,
      error: data.error,
    };
  }
  const now = performance.now();
  const isLiveRunning = displayed && typeof displayed === "object" && displayed.status === "running";
  const shouldUpdatePanel = !isLiveRunning || now - lastStatusPanelAt >= STATUS_PANEL_INTERVAL_MS;
  const text = isLiveRunning
    ? `flight running | step ${displayed.step ?? "-"} | ${String(displayed.stream_pacing_label || "waiting").toUpperCase()} | source ${formatNumber(displayed.stream_recent_fps ?? displayed.stream_average_fps)}fps | max gap ${formatNumber(displayed.stream_recent_gap_max_sec, 3)}s`
    : typeof displayed === "string" ? displayed : JSON.stringify(displayed, null, 2);
  if (shouldUpdatePanel && text !== lastStatusText) {
    $("status").textContent = text;
    lastStatusText = text;
    lastStatusPanelAt = now;
  }
  if (data && data.status) $("badge").textContent = data.status;
}

function updateRunningShowcaseSummary(data = {}) {
  if (!data || data.status !== "running") return false;
  const fps = data.stream_recent_fps ?? data.stream_average_fps ?? data.stream_fps;
  const gap = data.stream_recent_gap_max_sec;
  const distance = data.distance_to_goal;
  const summary = [
    `LIVE step ${data.step ?? "-"}`,
    Number.isFinite(Number(distance)) ? `${formatNumber(distance)}m` : null,
    Number.isFinite(Number(fps)) ? `source ${formatNumber(fps)}fps` : null,
    Number.isFinite(Number(gap)) ? `gap ${formatNumber(gap, 2)}s` : null,
  ].filter(Boolean).join(" | ");
  setText("liveFps", summary);
  return true;
}

function setViewerMessage(message, hasFrame = false) {
  if (viewerMessageText === message && viewerMessageHasFrame === Boolean(hasFrame)) return;
  viewerMessageText = message;
  viewerMessageHasFrame = Boolean(hasFrame);
  $("placeholder").textContent = message;
  $("placeholder").style.display = hasFrame ? "none" : "block";
  const canvas = $("streamCanvas");
  if (canvas) canvas.style.opacity = hasFrame ? "1" : "0.08";
  const stream = $("stream");
  if (stream) stream.style.opacity = hasFrame ? "1" : "0.08";
  const replayVideo = $("replayVideo");
  if (replayVideo) replayVideo.style.opacity = hasFrame ? "1" : "0.08";
  if (!hasFrame) {
    streamCanvasHasFrame = false;
    streamCanvasClearAt = 0;
  }
}

function drawImageCover(ctx, image, width, height) {
  const sourceWidth = image.naturalWidth || image.videoWidth || image.width;
  const sourceHeight = image.naturalHeight || image.videoHeight || image.height;
  if (!sourceWidth || !sourceHeight || !width || !height) return false;
  const scale = Math.max(width / sourceWidth, height / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  const drawX = (width - drawWidth) / 2;
  const drawY = (height - drawHeight) / 2;
  ctx.drawImage(image, drawX, drawY, drawWidth, drawHeight);
  return true;
}

function resizeStreamCanvas(canvas) {
  const now = performance.now();
  const terminalSpriteMode = document.body.classList.contains("terminalSpriteReplay");
  const modeKey = terminalSpriteMode ? "terminal-sprite" : "stream";
  if (
    streamCanvasMetrics
    && streamCanvasMetrics.modeKey === modeKey
    && now - streamCanvasLastResizeCheck < 1500
  ) {
    streamCanvasMetrics.resized = false;
    return streamCanvasMetrics;
  }
  streamCanvasLastResizeCheck = now;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const dprCap = terminalSpriteMode ? TERMINAL_SPRITE_RENDER_DPR_CAP : 1.5;
  const renderScale = terminalSpriteMode ? terminalSpriteRenderScale : STREAM_RENDER_SCALE;
  const minDpr = terminalSpriteMode ? TERMINAL_SPRITE_RENDER_SCALE_MIN : 1;
  const renderDpr = Math.max(minDpr, Math.min(dprCap, dpr * renderScale));
  const width = Math.max(1, Math.floor(rect.width * renderDpr));
  const height = Math.max(1, Math.floor(rect.height * renderDpr));
  let resized = false;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    resized = true;
    streamCanvasHasFrame = false;
  }
  streamCanvasMetrics = { width: rect.width, height: rect.height, dpr: renderDpr, resized, modeKey };
  return streamCanvasMetrics;
}

function renderStreamCanvas(timestamp) {
  const canvas = $("streamCanvas");
  const image = $("stream");
  if (!canvas || !image) return;
  if (document.hidden) {
    streamCanvasRaf = window.requestAnimationFrame(renderStreamCanvas);
    return;
  }
  const minInterval = 1000 / STREAM_CANVAS_FPS;
  if (!streamCanvasLastDraw || timestamp - streamCanvasLastDraw >= minInterval) {
    const { width, height, dpr, resized } = resizeStreamCanvas(canvas);
    const ctx = streamCanvasCtx || canvas.getContext("2d", { alpha: false, desynchronized: true });
    streamCanvasCtx = ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (image.complete && image.naturalWidth > 0) {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      const shouldRefreshBase = resized || !streamCanvasHasFrame || timestamp - streamCanvasClearAt >= STREAM_CLEAR_INTERVAL_MS;
      if (shouldRefreshBase) {
        ctx.fillStyle = "#02070d";
        ctx.fillRect(0, 0, width, height);
        streamCanvasClearAt = timestamp;
      }
      ctx.globalAlpha = STREAM_TEMPORAL_ALPHA;
      if (drawImageCover(ctx, image, width, height)) {
        ctx.globalAlpha = 1;
        streamCanvasHasFrame = true;
        streamCanvasDrawCount += 1;
        if (!streamCanvasFpsWindowAt) streamCanvasFpsWindowAt = timestamp;
        const elapsed = timestamp - streamCanvasFpsWindowAt;
        if (elapsed >= 1000) {
          streamCanvasFps = Math.round((streamCanvasDrawCount * 1000) / elapsed);
          streamCanvasDrawCount = 0;
          streamCanvasFpsWindowAt = timestamp;
        }
      } else {
        ctx.globalAlpha = 1;
      }
    }
    streamCanvasLastDraw = timestamp;
  }
  streamCanvasRaf = window.requestAnimationFrame(renderStreamCanvas);
}

function startStreamCanvasRenderer() {
  if (SHOWCASE_NATIVE_STREAM) return;
  if (streamCanvasRaf) return;
  streamCanvasRaf = window.requestAnimationFrame(renderStreamCanvas);
}

function releaseTerminalVideoObjectUrl() {
  if (!terminalVideoObjectUrl) return;
  for (const cached of demoReplayBlobCache.values()) {
    if (cached && cached.url === terminalVideoObjectUrl) {
      terminalVideoObjectUrl = null;
      return;
    }
  }
  URL.revokeObjectURL(terminalVideoObjectUrl);
  terminalVideoObjectUrl = null;
}

function stopTerminalVideoRafMonitor() {
  if (terminalVideoRafMonitor) window.cancelAnimationFrame(terminalVideoRafMonitor);
  terminalVideoRafMonitor = null;
  terminalVideoRafLastAt = 0;
  terminalVideoRafLongGapCount = 0;
}

function fallbackTerminalVideoReplay(sessionId, video, profile, reason) {
  const perf = window.__airvlnLastReplayPerf;
  if (perf && (perf.mode === "demo-smooth-video" || perf.mode === "demo-webm-video" || perf.mode === "pure-video" || perf.mode === "ultra-video")) {
    perf.video_fallback_reason = reason;
    perf.video_fallback_profile = profile;
  }
  const duration = video ? Number(video.duration || 0) : 0;
  const current = video ? Number(video.currentTime || 0) : 0;
  const progress = duration > 0 ? Math.max(0, Math.min(0.98, current / duration)) : 0;
  if (video) video.pause();
  document.body.classList.remove("videoReplay");
  document.body.classList.remove("videoTakeoverActive");
  terminalVideoReplayStarted = false;
  stopTerminalVideoRafMonitor();
  if (profile === "demo-smooth") {
    startTerminalVideoReplay(sessionId, "demo-smooth-lite", true, progress);
  } else if (profile === "demo-smooth-lite") {
    startTerminalVideoReplay(sessionId, "demo-webm", true, progress);
  } else if (profile === "demo-webm") {
    startTerminalVideoReplay(sessionId, "demo-webm-lite", true, progress);
  } else if (profile === "demo-webm-lite") {
    startTerminalVideoReplay(sessionId, "pure", true);
  } else if (profile === "pure") {
    startTerminalVideoReplay(sessionId, "ultra", true);
  } else if (profile === "video-takeover" || profile === "ultra") {
    startTerminalStableAnimationReplay(sessionId, true)
      .then((started) => {
        if (!started) startTerminalMjpegReplay(sessionId);
      })
      .catch(() => startTerminalMjpegReplay(sessionId));
  } else {
    startTerminalVideoReplay(sessionId, "light", true);
  }
}

function startTerminalVideoRafMonitor(sessionId, video) {
  stopTerminalVideoRafMonitor();
  const watchedProfiles = ["demo-smooth", "demo-smooth-lite", "demo-webm", "demo-webm-lite", "pure", "ultra"];
  const step = (timestamp) => {
    if (
      !video
      || video.paused
      || video.ended
      || !watchedProfiles.includes(terminalVideoReplayProfile)
    ) {
      stopTerminalVideoRafMonitor();
      return;
    }
    if (terminalVideoRafLastAt > 0) {
      const gap = timestamp - terminalVideoRafLastAt;
      const perf = window.__airvlnLastReplayPerf;
      if (perf) {
        perf.video_raf_max_gap_ms = Math.max(Number(perf.video_raf_max_gap_ms || 0), Math.round(gap));
      }
      if (gap > 170) {
        terminalVideoRafLongGapCount += 1;
        if (perf) {
          perf.video_raf_long_gap_count = terminalVideoRafLongGapCount;
        }
      } else if (terminalVideoRafLongGapCount > 0) {
        terminalVideoRafLongGapCount -= 1;
      }
      if (
        (terminalVideoReplayProfile === "demo-webm" && terminalVideoRafLongGapCount >= SHOWCASE_DEMO_WEBM_RAF_LONG_GAP_LIMIT)
        || (terminalVideoReplayProfile === "demo-webm-lite" && terminalVideoRafLongGapCount >= SHOWCASE_DEMO_WEBM_RAF_LONG_GAP_LIMIT)
        || (terminalVideoReplayProfile === "demo-smooth" && terminalVideoRafLongGapCount >= SHOWCASE_DEMO_SMOOTH_RAF_LONG_GAP_LIMIT)
        || (terminalVideoReplayProfile === "demo-smooth-lite" && terminalVideoRafLongGapCount >= SHOWCASE_DEMO_SMOOTH_RAF_LONG_GAP_LIMIT)
        || (!["demo-smooth", "demo-smooth-lite", "demo-webm", "demo-webm-lite"].includes(terminalVideoReplayProfile) && terminalVideoRafLongGapCount >= 3)
      ) {
        fallbackTerminalVideoReplay(sessionId, video, terminalVideoReplayProfile, "raf_long_gap");
        return;
      }
    }
    terminalVideoRafLastAt = timestamp;
    terminalVideoRafMonitor = window.requestAnimationFrame(step);
  };
  terminalVideoRafMonitor = window.requestAnimationFrame(step);
}

function rememberDemoReplayBlobUrl(url, objectUrl, metadata = {}) {
  if (!url || !objectUrl) return objectUrl;
  const existing = demoReplayBlobCache.get(url);
  if (existing && existing.url && existing.url !== objectUrl) {
    try {
      URL.revokeObjectURL(existing.url);
    } catch (err) {
      // Best-effort cleanup only.
    }
  }
  demoReplayBlobCache.delete(url);
  demoReplayBlobCache.set(url, {
    url: objectUrl,
    metadata,
    size: Number(metadata?.size || 0),
    touched_at_ms: Math.round(performance.now()),
  });
  const cachedBytes = () => Array.from(demoReplayBlobCache.values())
    .reduce((total, item) => total + Number(item?.size || item?.metadata?.size || 0), 0);
  while (
    demoReplayBlobCache.size > SHOWCASE_DEMO_BLOB_CACHE_LIMIT
    || cachedBytes() > SHOWCASE_DEMO_BLOB_CACHE_MAX_BYTES
  ) {
    const oldestKey = demoReplayBlobCache.keys().next().value;
    const oldest = demoReplayBlobCache.get(oldestKey);
    demoReplayBlobCache.delete(oldestKey);
    if (oldest && oldest.url && oldest.url !== terminalVideoObjectUrl) {
      try {
        URL.revokeObjectURL(oldest.url);
      } catch (err) {
        // Best-effort cleanup only.
      }
    }
  }
  return objectUrl;
}

function cachedDemoReplayBlobUrl(url) {
  const cached = demoReplayBlobCache.get(url);
  if (!cached || !cached.url) return "";
  demoReplayBlobCache.delete(url);
  cached.touched_at_ms = Math.round(performance.now());
  demoReplayBlobCache.set(url, cached);
  return cached.url;
}

function releaseTerminalVideoTakeoverObjectUrl() {
  if (!terminalVideoTakeoverObjectUrl) return;
  try {
    URL.revokeObjectURL(terminalVideoTakeoverObjectUrl);
  } catch (err) {
    // Best-effort cleanup only.
  }
  terminalVideoTakeoverObjectUrl = null;
}

function releaseTerminalVideoTakeoverPreload(force = false) {
  const url = terminalVideoTakeoverPreload;
  terminalVideoTakeoverPreload = null;
  const preservePlayingVideo = !force && document.body.classList.contains("videoReplay");
  if (!preservePlayingVideo) {
    releaseTerminalVideoTakeoverObjectUrl();
  }
  if (!url || preservePlayingVideo) return;
  const video = $("replayVideo");
  if (!video) return;
  try {
    video.pause();
    video.removeAttribute("src");
    video.load();
  } catch (err) {
    // Best-effort cleanup only.
  }
}

function releaseTerminalAnimationObjectUrls() {
  for (const url of terminalAnimationObjectUrls) {
    try {
      URL.revokeObjectURL(url);
    } catch (err) {
      // Best-effort cleanup only.
    }
  }
  terminalAnimationObjectUrls = [];
}

function forgetTerminalAnimationObjectUrl(url) {
  if (!url) return;
  const index = terminalAnimationObjectUrls.indexOf(url);
  if (index >= 0) terminalAnimationObjectUrls.splice(index, 1);
}

function releaseTerminalSpriteBitmaps() {
  for (const bitmap of terminalSpriteBitmaps) {
    try {
      if (bitmap && typeof bitmap.close === "function") {
        bitmap.close();
      }
    } catch (err) {
      // Best-effort cleanup only.
    }
  }
  terminalSpriteBitmaps = [];
}

function forgetTerminalSpriteBitmap(bitmap) {
  const index = terminalSpriteBitmaps.indexOf(bitmap);
  if (index >= 0) terminalSpriteBitmaps.splice(index, 1);
}

function releaseTerminalSpriteChunkFrameImages(chunk) {
  if (!chunk || !Array.isArray(chunk.frameImages)) return;
  for (const frameImage of chunk.frameImages) {
    try {
      if (frameImage && typeof frameImage.close === "function") {
        frameImage.close();
      }
    } catch (err) {
      // Best-effort cleanup only.
    }
  }
  chunk.frameImages = null;
  chunk.frameImagePromise = null;
  chunk.frameSliceFailed = true;
}

function releaseTerminalSpriteChunkImage(chunk) {
  if (!chunk || chunk.released) return;
  try {
    releaseTerminalSpriteChunkFrameImages(chunk);
    if (chunk.image && typeof chunk.image.close === "function") {
      chunk.image.close();
      forgetTerminalSpriteBitmap(chunk.image);
    } else if (chunk.image && typeof chunk.image.removeAttribute === "function") {
      chunk.image.removeAttribute("src");
    }
  } catch (err) {
    // Best-effort cleanup only.
  }
  if (chunk.objectUrl) {
    try {
      URL.revokeObjectURL(chunk.objectUrl);
    } catch (err) {
      // Best-effort cleanup only.
    }
    forgetTerminalAnimationObjectUrl(chunk.objectUrl);
  }
  chunk.image = null;
  chunk.released = true;
}

function clearTerminalReplayPrefetchTimers() {
  for (const timer of terminalReplayPrefetchTimers) {
    clearTimeout(timer);
  }
  terminalReplayPrefetchTimers = [];
}

function scheduleReplayPrefetch(loadChunk, chunkCount, fromIndex, token) {
  for (
    let index = fromIndex;
    index < Math.min(chunkCount, fromIndex + SHOWCASE_TERMINAL_REPLAY_PREFETCH_AHEAD);
    index += 1
  ) {
    const delay = Math.max(0, index - fromIndex) * SHOWCASE_TERMINAL_REPLAY_PREFETCH_STAGGER_MS;
    const timer = setTimeout(() => {
      if (token !== terminalAnimationPlaybackToken) return;
      const promise = loadChunk(index, true);
      if (promise && typeof promise.catch === "function") promise.catch(() => {});
    }, delay);
    terminalReplayPrefetchTimers.push(timer);
  }
}

function scheduleReplayBackgroundPrefetch(loadChunk, chunkCount, fromIndex, token) {
  const endIndex = Math.min(chunkCount, fromIndex + SHOWCASE_TERMINAL_BACKGROUND_PREFETCH_AHEAD);
  for (let index = fromIndex; index < endIndex; index += 1) {
    const delay = (index - fromIndex + 1) * SHOWCASE_TERMINAL_BACKGROUND_PREFETCH_STAGGER_MS;
    const timer = setTimeout(() => {
      if (token !== terminalAnimationPlaybackToken) return;
      const run = () => {
        if (token !== terminalAnimationPlaybackToken) return;
        const promise = loadChunk(index, true);
        if (promise && typeof promise.catch === "function") promise.catch(() => {});
      };
      if (typeof window.requestIdleCallback === "function") {
        window.requestIdleCallback(run, { timeout: SHOWCASE_TERMINAL_BACKGROUND_PREFETCH_STAGGER_MS });
      } else {
        run();
      }
    }, delay);
    terminalReplayPrefetchTimers.push(timer);
  }
}

function stopDelayedFramePlayer() {
  setDemoReplayActive(false);
  if (delayedFrameFetchTimer) clearInterval(delayedFrameFetchTimer);
  if (delayedFramePlayTimer) clearInterval(delayedFramePlayTimer);
  if (delayedFrameRaf) window.cancelAnimationFrame(delayedFrameRaf);
  if (terminalAnimationChunkTimer) clearTimeout(terminalAnimationChunkTimer);
  if (terminalSpriteRaf) window.cancelAnimationFrame(terminalSpriteRaf);
  if (terminalSpriteFrameTimer) clearTimeout(terminalSpriteFrameTimer);
  clearTerminalReplayPrefetchTimers();
  clearDemoNeighborPrefetchTimers();
  delayedFrameFetchTimer = null;
  delayedFramePlayTimer = null;
  delayedFrameRaf = null;
  terminalAnimationChunkTimer = null;
  terminalSpriteRaf = null;
  terminalSpriteFrameTimer = null;
  terminalAnimationPlaybackToken += 1;
  delayedFrameSession = null;
  delayedFrameItems = [];
  delayedFrameSeen = new Set();
  delayedFrameCursor = 0;
  delayedFrameTerminal = false;
  delayedFrameCurrentImage = null;
  delayedFramePreviousImage = null;
  delayedFrameFrameStartedAt = 0;
  delayedFrameImageCache = new Map();
  delayedFrameImageLoading = new Set();
  delayedFrameReplayRestarted = false;
  delayedFrameReplayStreamStarted = false;
  streamCanvasMetrics = null;
  streamCanvasLastResizeCheck = 0;
  terminalSpriteRenderScale = 1.0;
  terminalVideoReplayStarted = false;
  terminalVideoReplayProfile = "primary";
  terminalVideoLastTime = 0;
  terminalVideoStallChecks = 0;
  stopTerminalVideoRafMonitor();
  terminalEnhancedReplayStarted = false;
  terminalEnhancedReplayApplied = false;
  terminalVideoTakeoverStarted = false;
  terminalVideoTakeoverApplied = false;
  terminalVideoTakeoverPending = false;
  releaseTerminalVideoTakeoverPreload(true);
  if (terminalVideoQualityTimer) clearInterval(terminalVideoQualityTimer);
  terminalVideoQualityTimer = null;
  const stream = $("stream");
  if (stream && document.body.classList.contains("delayedFramePlayer")) {
    stream.removeAttribute("src");
  }
  if (stream && document.body.classList.contains("terminalImageReplay")) {
    stream.removeAttribute("src");
  }
  const replayVideo = $("replayVideo");
  if (replayVideo) {
    replayVideo.pause();
    replayVideo.removeAttribute("src");
    releaseTerminalVideoObjectUrl();
    releaseTerminalVideoTakeoverObjectUrl();
    releaseTerminalAnimationObjectUrls();
    releaseTerminalSpriteBitmaps();
    replayVideo.load();
  }
  document.body.classList.remove("delayedFramePlayer");
  document.body.classList.remove("videoReplay");
  document.body.classList.remove("videoTakeoverActive");
  document.body.classList.remove("terminalImageReplay");
  document.body.classList.remove("terminalSpriteReplay");
}

async function fetchDelayedFrameItems() {
  if (!delayedFrameSession) return;
  try {
    const data = await api(`/api/agent/sessions/${delayedFrameSession}/frames?limit=${DELAYED_FRAME_API_LIMIT}&t=${Date.now()}`);
    for (const item of data.frames || []) {
      if (!item || !item.name || delayedFrameSeen.has(item.name)) continue;
      delayedFrameSeen.add(item.name);
      delayedFrameItems.push(item);
    }
    if (!SHOWCASE_REPLAY_FROM_START_ON_TERMINAL && delayedFrameItems.length > 1600) {
      const drop = Math.max(0, delayedFrameCursor - 200);
      if (drop > 0) {
        delayedFrameItems = delayedFrameItems.slice(drop);
        delayedFrameCursor -= drop;
      }
    }
  } catch (err) {
    // The session may not have created frames yet; keep the player alive.
  }
}

function preloadDelayedFrames() {
  const end = Math.min(delayedFrameItems.length, delayedFrameCursor + DELAYED_FRAME_PRELOAD_AHEAD);
  for (let index = delayedFrameCursor; index < end; index += 1) {
    if (delayedFrameImageLoading.size >= DELAYED_FRAME_MAX_IMAGE_LOADS) break;
    const item = delayedFrameItems[index];
    if (!item || delayedFrameImageCache.has(item.name) || delayedFrameImageLoading.has(item.name)) continue;
    delayedFrameImageLoading.add(item.name);
    const image = new Image();
    image.decoding = "async";
    image.onload = () => {
      delayedFrameImageCache.set(item.name, image);
      delayedFrameImageLoading.delete(item.name);
    };
    image.onerror = () => delayedFrameImageLoading.delete(item.name);
    image.src = `${item.url}?v=${Math.round(item.mtime || 0)}`;
  }
}

function drawDelayedFrameCanvas(image, previousImage, alpha) {
  const canvas = $("streamCanvas");
  if (!canvas || !image) return false;
  const { width, height, dpr } = resizeStreamCanvas(canvas);
  const ctx = streamCanvasCtx || canvas.getContext("2d", { alpha: false, desynchronized: true });
  streamCanvasCtx = ctx;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#02070d";
  ctx.fillRect(0, 0, width, height);
  if (previousImage && alpha < 1) {
    ctx.globalAlpha = 1;
    drawImageCover(ctx, previousImage, width, height);
  }
  ctx.globalAlpha = alpha;
  drawImageCover(ctx, image, width, height);
  ctx.globalAlpha = 1;
  return true;
}

function delayedFramePlaybackInterval(buffered) {
  if (delayedFrameTerminal) return DELAYED_FRAME_PLAY_INTERVAL_MS;
  if (buffered <= DELAYED_FRAME_LOW_BUFFER) return DELAYED_FRAME_MAX_LIVE_PLAY_INTERVAL_MS;
  if (buffered >= DELAYED_FRAME_START_BUFFER) return DELAYED_FRAME_LIVE_PLAY_INTERVAL_MS;
  const ratio = (buffered - DELAYED_FRAME_LOW_BUFFER) / Math.max(1, DELAYED_FRAME_START_BUFFER - DELAYED_FRAME_LOW_BUFFER);
  return DELAYED_FRAME_MAX_LIVE_PLAY_INTERVAL_MS
    - ((DELAYED_FRAME_MAX_LIVE_PLAY_INTERVAL_MS - DELAYED_FRAME_LIVE_PLAY_INTERVAL_MS) * ratio);
}

function renderDelayedFramePlayer(timestamp) {
  if (!delayedFrameSession) return;
  preloadDelayedFrames();
  const buffered = delayedFrameItems.length - delayedFrameCursor;
  const startReady = delayedFrameCursor > 0
    || delayedFrameTerminal
    || (!SHOWCASE_RECORD_THEN_REPLAY && buffered >= DELAYED_FRAME_START_BUFFER);
  const bufferReady = buffered > 0 && (buffered >= DELAYED_FRAME_LOW_BUFFER || delayedFrameTerminal);
  if (!startReady || !bufferReady) {
    const label = SHOWCASE_RECORD_THEN_REPLAY && !delayedFrameTerminal
      ? "正在录制平滑回放"
      : "平滑缓冲中";
    setViewerMessage(label, false);
    delayedFrameRaf = window.requestAnimationFrame(renderDelayedFramePlayer);
    return;
  }
  let item = delayedFrameItems[delayedFrameCursor];
  let image = item ? delayedFrameImageCache.get(item.name) : null;
  if (!image) {
    // Keep the last decoded frame visible while the next high-resolution JPEG
    // is still loading. A blank/dim canvas makes normal source pacing look
    // like a rendering failure.
    if (delayedFrameCurrentImage) {
      drawDelayedFrameCanvas(delayedFrameCurrentImage, delayedFramePreviousImage, 1);
      setViewerMessage("", true);
    } else {
      setViewerMessage("平滑缓冲中", false);
    }
    delayedFrameRaf = window.requestAnimationFrame(renderDelayedFramePlayer);
    return;
  }
  if (!delayedFrameCurrentImage) {
    delayedFrameCurrentImage = image;
    delayedFrameFrameStartedAt = timestamp;
  }
  const playInterval = delayedFramePlaybackInterval(buffered);
  if (
    timestamp - delayedFrameFrameStartedAt >= playInterval
    && (buffered > DELAYED_FRAME_LOW_BUFFER || delayedFrameTerminal)
  ) {
    const nextCursor = delayedFrameCursor + 1;
    const nextItem = delayedFrameItems[nextCursor];
    const nextImage = nextItem ? delayedFrameImageCache.get(nextItem.name) : null;
    // Do not advance into an undecoded frame. Holding the current frame keeps
    // motion continuous and prevents cursor jumps when capture gaps are large.
    if (nextImage) {
      delayedFramePreviousImage = delayedFrameCurrentImage;
      delayedFrameCursor = nextCursor;
      delayedFrameCurrentImage = nextImage;
      delayedFrameFrameStartedAt = timestamp;
    }
  }
  const fadeMs = delayedFrameTerminal
    ? DELAYED_FRAME_CROSSFADE_MS
    : Math.min(playInterval, Math.max(DELAYED_FRAME_CROSSFADE_MS, playInterval * 0.8));
  const fadeProgress = Math.max(0, Math.min(1, (timestamp - delayedFrameFrameStartedAt) / fadeMs));
  drawDelayedFrameCanvas(delayedFrameCurrentImage, delayedFramePreviousImage, fadeProgress);
  setViewerMessage("", true);
  delayedFrameRaf = window.requestAnimationFrame(renderDelayedFramePlayer);
}

function startDelayedFramePlayer(sessionId) {
  if (SHOWCASE_RECORD_ONLY_UNTIL_TERMINAL || !SHOWCASE_DELAYED_FRAME_PLAYER || !sessionId) return false;
  stopDelayedFramePlayer();
  delayedFrameSession = sessionId;
  document.body.classList.add("delayedFramePlayer");
  delayedFrameFetchTimer = setInterval(fetchDelayedFrameItems, DELAYED_FRAME_FETCH_INTERVAL_MS);
  fetchDelayedFrameItems();
  delayedFrameRaf = window.requestAnimationFrame(renderDelayedFramePlayer);
   setViewerMessage("准备画面...", false);
  return true;
}

function terminalVideoReplayUrl(sessionId, profile, directUrl = "") {
  if (directUrl) return directUrl;
  const light = profile === "light";
  if (profile === "demo-smooth") {
    return `/api/agent/sessions/${sessionId}/replay-video?format=mp4&video_codec=h264&smooth=0&interp=hold&stride=${SHOWCASE_DEMO_SMOOTH_VIDEO_STRIDE}&fps=${SHOWCASE_DEMO_SMOOTH_VIDEO_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_DEMO_SMOOTH_VIDEO_WIDTH}&upscale=0&sharpen=${SHOWCASE_DEMO_SMOOTH_VIDEO_SHARPEN}&t=${Date.now()}`;
  }
  if (profile === "demo-smooth-lite") {
    return `/api/agent/sessions/${sessionId}/replay-video?format=mp4&video_codec=h264&smooth=0&interp=hold&stride=1&fps=${SHOWCASE_DEMO_LITE_VIDEO_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_DEMO_LITE_VIDEO_WIDTH}&upscale=0&sharpen=0&t=${Date.now()}`;
  }
  if (profile === "demo-webm") {
    return `/api/agent/sessions/${sessionId}/replay-video?format=webm&smooth=0&interp=hold&stride=${SHOWCASE_DEMO_SMOOTH_VIDEO_STRIDE}&fps=${SHOWCASE_DEMO_SMOOTH_VIDEO_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_DEMO_SMOOTH_VIDEO_WIDTH}&upscale=0&sharpen=0&t=${Date.now()}`;
  }
  if (profile === "demo-webm-lite") {
    return `/api/agent/sessions/${sessionId}/replay-video?format=webm&smooth=0&interp=hold&stride=1&fps=${SHOWCASE_DEMO_LITE_VIDEO_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_DEMO_LITE_VIDEO_WIDTH}&upscale=0&sharpen=0&t=${Date.now()}`;
  }
  if (profile === "pure") {
    return `/api/agent/sessions/${sessionId}/replay-video?format=mp4&smooth=0&stride=${SHOWCASE_TERMINAL_PURE_VIDEO_STRIDE}&fps=${SHOWCASE_TERMINAL_PURE_VIDEO_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_TERMINAL_PURE_VIDEO_WIDTH}&t=${Date.now()}`;
  }
  if (profile === "ultra") {
    return `/api/agent/sessions/${sessionId}/replay-video?format=mp4&smooth=0&stride=${SHOWCASE_TERMINAL_ULTRA_VIDEO_STRIDE}&fps=${SHOWCASE_TERMINAL_ULTRA_VIDEO_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_TERMINAL_ULTRA_VIDEO_WIDTH}&t=${Date.now()}`;
  }
  if (profile === "enhanced") {
    return `/api/agent/sessions/${sessionId}/replay-video?format=mp4&smooth=1&smooth_multiplier=${SHOWCASE_TERMINAL_ENHANCED_SMOOTH_MULTIPLIER}&fps=${SHOWCASE_TERMINAL_REPLAY_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_TERMINAL_ENHANCED_WIDTH}&t=${Date.now()}`;
  }
  if (light && SHOWCASE_TERMINAL_FAST_PREVIEW) {
    return `/api/agent/sessions/${sessionId}/replay-video?format=mp4&smooth=0&stride=${SHOWCASE_TERMINAL_FAST_STRIDE}&fps=${SHOWCASE_TERMINAL_REPLAY_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_TERMINAL_LIGHT_WIDTH}&t=${Date.now()}`;
  }
  const smoothMultiplier = light
    ? SHOWCASE_TERMINAL_LIGHT_SMOOTH_MULTIPLIER
    : SHOWCASE_TERMINAL_SMOOTH_MULTIPLIER;
  const maxWidth = light ? `&max_width=${SHOWCASE_TERMINAL_LIGHT_WIDTH}` : "";
  return `/api/agent/sessions/${sessionId}/replay-video?format=mp4&smooth=1&smooth_multiplier=${smoothMultiplier}&fps=${SHOWCASE_TERMINAL_REPLAY_FPS}&limit=${DELAYED_FRAME_API_LIMIT}${maxWidth}&t=${Date.now()}`;
}

function terminalVideoTakeoverUrl(sessionId) {
  return `/api/agent/sessions/${sessionId}/replay-video?format=mp4&smooth=0&stride=${SHOWCASE_TERMINAL_VIDEO_TAKEOVER_STRIDE}&fps=${SHOWCASE_TERMINAL_VIDEO_TAKEOVER_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_TERMINAL_VIDEO_TAKEOVER_WIDTH}&t=${Date.now()}`;
}

function terminalAnimationReplayUrl(sessionId) {
  return `/api/agent/sessions/${sessionId}/replay-animation?stride=${SHOWCASE_TERMINAL_FAST_STRIDE}&fps=${SHOWCASE_TERMINAL_REPLAY_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_TERMINAL_LIGHT_WIDTH}&quality=${SHOWCASE_TERMINAL_ANIMATION_QUALITY}&t=${Date.now()}`;
}

function terminalStableAnimationReplayUrl(sessionId) {
  return `/api/agent/sessions/${sessionId}/replay-animation?stride=${SHOWCASE_TERMINAL_STABLE_ANIMATION_STRIDE}&fps=${SHOWCASE_TERMINAL_STABLE_ANIMATION_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_TERMINAL_STABLE_ANIMATION_WIDTH}&quality=${SHOWCASE_TERMINAL_STABLE_ANIMATION_QUALITY}&t=${Date.now()}`;
}

function terminalAnimationManifestUrl(sessionId) {
  return `/api/agent/sessions/${sessionId}/replay-animation-manifest?stride=${SHOWCASE_TERMINAL_FAST_STRIDE}&fps=${SHOWCASE_TERMINAL_REPLAY_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_TERMINAL_LIGHT_WIDTH}&quality=${SHOWCASE_TERMINAL_ANIMATION_QUALITY}&chunk_frames=${SHOWCASE_TERMINAL_ANIMATION_CHUNK_FRAMES}&t=${Date.now()}`;
}

function terminalSpriteManifestUrl(sessionId) {
  return `/api/agent/sessions/${sessionId}/replay-sprite-manifest?stride=${SHOWCASE_TERMINAL_FAST_STRIDE}&fps=${SHOWCASE_TERMINAL_REPLAY_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_TERMINAL_LIGHT_WIDTH}&quality=${SHOWCASE_TERMINAL_ANIMATION_QUALITY}&chunk_frames=${SHOWCASE_TERMINAL_ANIMATION_CHUNK_FRAMES}&columns=${SHOWCASE_TERMINAL_SPRITE_COLUMNS}&t=${Date.now()}`;
}

function terminalReplayPrewarmUrl(sessionId, startChunk = 2) {
  return `/api/agent/sessions/${sessionId}/replay-prewarm?stride=${SHOWCASE_TERMINAL_FAST_STRIDE}&fps=${SHOWCASE_TERMINAL_REPLAY_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&max_width=${SHOWCASE_TERMINAL_LIGHT_WIDTH}&quality=${SHOWCASE_TERMINAL_ANIMATION_QUALITY}&chunk_frames=${SHOWCASE_TERMINAL_ANIMATION_CHUNK_FRAMES}&columns=${SHOWCASE_TERMINAL_SPRITE_COLUMNS}&start_chunk=${Math.max(0, startChunk)}&max_chunks=96&t=${Date.now()}`;
}

function startTerminalReplayPrewarm(sessionId, startChunk = 2) {
  if (!sessionId) return;
  api(terminalReplayPrewarmUrl(sessionId, startChunk), { method: "POST" }).catch(() => {});
}

function verifiedHighresDemoVideo(item) {
  const url = String(item?.demo_video_url || "").trim();
  const metadata = item?.demo_video_metadata;
  if (
    !url
    || !metadata
    || String(metadata.codec || "").toLowerCase() !== "h264"
    || String(metadata.interp || "").toLowerCase() !== "hold"
    || Number(metadata.width || 0) < 960
    || Number(metadata.height || 0) < 540
  ) return null;
  return { url, metadata };
}

function prefetchDemoReplayBlob(item, profile = "pure") {
  if (!SHOWCASE_DEMO_BLOB_PREFETCH || !SHOWCASE_TERMINAL_BLOB_PLAYBACK) return Promise.resolve(false);
  const sessionId = String(item?.demo_source_session_id || item?.source_session_id || "").trim();
  if (!sessionId) return Promise.resolve(false);
  const key = `${sessionId}:${profile}`;
  if (demoReplayBlobInflight.has(key)) return demoReplayBlobInflight.get(key);
  const task = (async () => {
    const highres = verifiedHighresDemoVideo(item);
    const data = highres || await api(terminalVideoReplayUrl(sessionId, profile));
    if (!data || !data.url) return false;
    if (cachedDemoReplayBlobUrl(data.url)) return true;
    const response = await fetch(data.url, { cache: "default" });
    if (!response.ok) throw new Error(`demo replay prefetch failed: ${response.status}`);
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    rememberDemoReplayBlobUrl(data.url, objectUrl, data.metadata || {});
    const current = $("liveFps")?.textContent || "";
    if (!current || current === "0 frames" || current.startsWith("demo video") || current.startsWith("demo replay")) {
      const size = Number(data.metadata?.size || blob.size || 0);
      const sizeLabel = size > 0 ? ` ${(size / 1024 / 1024).toFixed(1)}MB` : "";
      setText("liveFps", `demo replay ready${sizeLabel}`);
    }
    return true;
  })()
    .catch(() => false)
    .finally(() => {
      demoReplayBlobInflight.delete(key);
    });
  demoReplayBlobInflight.set(key, task);
  return task;
}

async function waitForDemoReplayBlob(item, profile = "pure", timeoutMs = 1500) {
  const sessionId = String(item?.demo_source_session_id || item?.source_session_id || "").trim();
  if (!sessionId) return false;
  prefetchDemoReplayBlob(item, profile);
  const key = `${sessionId}:${profile}`;
  const task = demoReplayBlobInflight.get(key);
  if (!task) return true;
  let timeoutHandle = null;
  try {
    await Promise.race([
      task,
      new Promise((resolve) => {
        timeoutHandle = window.setTimeout(resolve, timeoutMs);
      }),
    ]);
  } catch (err) {
    // Prefetch failures should not block the regular replay path.
  } finally {
    if (timeoutHandle) window.clearTimeout(timeoutHandle);
  }
  return true;
}

function startDemoVideoPrewarm(item) {
  const sessionId = String(item?.demo_source_session_id || item?.source_session_id || "").trim();
  if (!sessionId) return;
  if (demoVideoPrewarmSessions.has(sessionId)) {
    prefetchDemoReplayBlob(item, "demo-smooth");
    return;
  }
  demoVideoPrewarmSessions.add(sessionId);
  prefetchDemoReplayBlob(item, "demo-smooth")
    .then((prefetched) => {
      if (prefetched) return null;
      return api(`/api/agent/sessions/${encodeURIComponent(sessionId)}/demo-video-prewarm`, { method: "POST" });
    })
    .then((data) => {
      if (!data) return;
      if (!data || data.error) return;
      const current = $("liveFps")?.textContent || "";
      if (!current || current === "0 frames" || current.startsWith("demo video")) {
          setText("liveFps", `demo video cache ${data.status || "queued"}`);
      }
    })
    .catch(() => {
      demoVideoPrewarmSessions.delete(sessionId);
    });
}

function isVerifiedDemoReplayItem(item) {
  if (!item) return false;
  const sessionId = String(item.demo_source_session_id || item.source_session_id || "").trim();
  if (!sessionId) return false;
  return Boolean(
    item.verified_success
    || item.variant_verified_success
    || item.showcase_ready
    || item.verification_tier === "current_smooth"
    || item.display_showcase_tier === "frontline"
    || item.display_showcase_tier === "stable"
  );
}

function clearDemoNeighborPrefetchTimers() {
  for (const timer of demoReplayNeighborPrefetchTimers) {
    window.clearTimeout(timer);
  }
  demoReplayNeighborPrefetchTimers = [];
}

function setDemoReplayActive(active) {
  demoReplayActive = Boolean(active);
  document.body.classList.toggle("demoReplayActive", demoReplayActive);
  if (demoReplayActive) {
    clearDemoNeighborPrefetchTimers();
  } else {
    document.documentElement.style.removeProperty("--demo-video-display-width");
    document.documentElement.style.removeProperty("--demo-video-display-height");
  }
}

function applyDemoReplayDisplaySize(metadata = {}) {
  const width = Number(metadata.width || 0);
  const height = Number(metadata.height || 0);
  if (!width || !height) return;
  const scale = width <= 256 ? 2.75 : width <= 360 ? 2.05 : 1.12;
  const displayWidth = Math.round(Math.min(780, Math.max(620, width * scale)));
  const displayHeight = Math.round(Math.min(780, Math.max(620, height * scale)));
  document.documentElement.style.setProperty("--demo-video-display-width", `${displayWidth}px`);
  document.documentElement.style.setProperty("--demo-video-display-height", `${displayHeight}px`);
}

function demoNeighborPrefetchIndices(centerIndex) {
  if (!Array.isArray(datasetMatches) || !datasetMatches.length) return [];
  const indices = [];
  const total = datasetMatches.length;
  for (let offset = 1; offset <= Math.min(SHOWCASE_DEMO_NEIGHBOR_PREFETCH_COUNT, total - 1); offset += 1) {
    indices.push((centerIndex + offset) % total);
  }
  return indices;
}

function scheduleDemoNeighborPrewarm(centerIndex) {
  if (!SHOWCASE_DEMO_BLOB_PREFETCH || currentSession || demoReplayActive) return;
  clearDemoNeighborPrefetchTimers();
  const indices = demoNeighborPrefetchIndices(Number(centerIndex || 0));
  indices.forEach((index, order) => {
    const item = datasetMatches[index];
    if (!item) return;
    const timer = window.setTimeout(() => {
      if (currentSession || demoReplayActive || document.hidden) return;
      const run = () => {
        if (currentSession || demoReplayActive || document.hidden) return;
        startDemoVideoPrewarm(item);
      };
      if (typeof window.requestIdleCallback === "function") {
        window.requestIdleCallback(run, { timeout: SHOWCASE_DEMO_NEIGHBOR_PREFETCH_STAGGER_MS });
      } else {
        run();
      }
    }, (order + 1) * SHOWCASE_DEMO_NEIGHBOR_PREFETCH_STAGGER_MS);
    demoReplayNeighborPrefetchTimers.push(timer);
  });
}

async function startVerifiedSourceReplay(item) {
  if (!SHOWCASE_PREFER_VERIFIED_SOURCE_REPLAY || !item) return false;
  const sessionId = String(item.demo_source_session_id || item.source_session_id || "").trim();
  if (!sessionId) return false;
  stopDelayedFramePlayer();
  setDemoReplayActive(true);
  clearPolling();
  currentSession = null;
  latestBevData = null;
  cachedTraceBevData = null;
  lastTraceFetchAt = 0;
  setFlightControlsRunning(true);
  $("sessionId").textContent = `demo replay: ${sessionId}`;
  setStatus({
    replay_demo: true,
    episode_id: item.episode_id || "-",
    source_session_id: sessionId,
    mode: "verified_source_replay",
    note: "播放已验证成功 session 的缓存回放，避免在线推理和实时采集造成展示卡顿。",
  });
  setViewerMessage("正在加载已验证原始帧演示回放...", false);
  try {
    await waitForDemoReplayBlob(item, "demo-smooth", 1800);
    const highres = verifiedHighresDemoVideo(item);
    const smoothVideoStarted = await startTerminalVideoReplay(
      sessionId,
      "demo-smooth",
      true,
      0,
      highres?.url || "",
      highres?.metadata || {},
    );
    const smoothLiteVideoStarted = smoothVideoStarted ? false : await startTerminalVideoReplay(sessionId, "demo-smooth-lite", true);
    const webmVideoStarted = (smoothVideoStarted || smoothLiteVideoStarted) ? false : await startTerminalVideoReplay(sessionId, "demo-webm", true);
    const liteVideoStarted = (smoothVideoStarted || smoothLiteVideoStarted || webmVideoStarted) ? false : await startTerminalVideoReplay(sessionId, "demo-webm-lite", true);
    const pureVideoStarted = (smoothVideoStarted || smoothLiteVideoStarted || liteVideoStarted || webmVideoStarted) ? false : await startTerminalVideoReplay(sessionId, "pure", true);
    const ultraVideoStarted = (smoothVideoStarted || smoothLiteVideoStarted || liteVideoStarted || webmVideoStarted || pureVideoStarted) ? false : await startTerminalVideoReplay(sessionId, "ultra", true);
    const stableAnimationStarted = (smoothVideoStarted || smoothLiteVideoStarted || liteVideoStarted || webmVideoStarted || pureVideoStarted || ultraVideoStarted) ? false : await startTerminalStableAnimationReplay(sessionId, true);
    if (smoothVideoStarted || smoothLiteVideoStarted || liteVideoStarted || webmVideoStarted || pureVideoStarted || ultraVideoStarted || stableAnimationStarted) {
      setFlightControlsRunning(false);
      return true;
    }
  } catch (err) {
    // Fall through to the online agent path; this keeps Start usable even when
    // an older catalog entry has no local replay frames.
  }
  setDemoReplayActive(false);
  setFlightControlsRunning(false);
  setViewerMessage("缓存回放不可用，切换在线飞行...", false);
  return false;
}

function monitorTerminalVideoQuality(sessionId, video) {
  if (terminalVideoQualityTimer) clearInterval(terminalVideoQualityTimer);
  terminalVideoLastTime = 0;
  terminalVideoStallChecks = 0;
  terminalVideoQualityTimer = setInterval(() => {
    if (
      !video
      || video.paused
      || video.ended
      || !["primary", "video-takeover", "demo-smooth", "demo-smooth-lite", "demo-webm", "demo-webm-lite", "pure", "ultra", "light"].includes(terminalVideoReplayProfile)
    ) {
      return;
    }
    const current = Number(video.currentTime || 0);
    if (current > 2 && current - terminalVideoLastTime < 0.08) {
      terminalVideoStallChecks += 1;
    } else {
      terminalVideoStallChecks = 0;
    }
    terminalVideoLastTime = current;
    let dropRatio = 0;
    if (typeof video.getVideoPlaybackQuality === "function") {
      const quality = video.getVideoPlaybackQuality();
      const total = Number(quality.totalVideoFrames || 0);
      const dropped = Number(quality.droppedVideoFrames || 0);
      dropRatio = total > 30 ? dropped / total : 0;
    }
    if (terminalVideoStallChecks >= 3 || dropRatio > 0.12) {
      const stalledProfile = terminalVideoReplayProfile;
      const perf = window.__airvlnLastReplayPerf;
      if (perf && (perf.mode === "demo-smooth-video" || perf.mode === "demo-webm-video" || perf.mode === "pure-video" || perf.mode === "ultra-video")) {
        perf.video_stall_profile = stalledProfile;
        perf.video_stall_checks = terminalVideoStallChecks;
        perf.video_drop_ratio = Math.round(dropRatio * 1000) / 1000;
      }
      clearInterval(terminalVideoQualityTimer);
      terminalVideoQualityTimer = null;
      fallbackTerminalVideoReplay(sessionId, video, stalledProfile, "video_quality");
    }
  }, 1200);
}

function prepareReplayVideoElement(video) {
  if (!video) return;
  video.preload = "auto";
  video.muted = true;
  video.playsInline = true;
  video.disablePictureInPicture = true;
  video.setAttribute("playsinline", "");
  video.setAttribute("preload", "auto");
  video.setAttribute("disableRemotePlayback", "");
  video.style.transform = "translate3d(0, 0, 0)";
  video.style.backfaceVisibility = "hidden";
}

function waitForReplayVideoBuffer(video, timeoutMs = 1200) {
  if (!video) return Promise.resolve(false);
  if (video.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) return Promise.resolve(true);
  return new Promise((resolve) => {
    let resolved = false;
    let timeoutHandle = null;
    const finish = (ready) => {
      if (resolved) return;
      resolved = true;
      video.removeEventListener("canplay", onReady);
      video.removeEventListener("canplaythrough", onReady);
      video.removeEventListener("loadeddata", onReady);
      if (timeoutHandle) window.clearTimeout(timeoutHandle);
      resolve(Boolean(ready));
    };
    const onReady = () => finish(true);
    video.addEventListener("canplay", onReady, { once: true });
    video.addEventListener("canplaythrough", onReady, { once: true });
    video.addEventListener("loadeddata", onReady, { once: true });
    timeoutHandle = window.setTimeout(() => finish(video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA), timeoutMs);
  });
}

async function prepareEnhancedTerminalReplay(sessionId, currentVideo) {
  if (!SHOWCASE_TERMINAL_ENHANCED_REPLAY || terminalEnhancedReplayStarted || terminalEnhancedReplayApplied) return;
  terminalEnhancedReplayStarted = true;
  try {
    const data = await api(terminalVideoReplayUrl(sessionId, "enhanced"));
    const video = $("replayVideo");
    if (!data || !data.url || !video || video !== currentVideo || video.ended) return;
    const duration = Number(video.duration || 0);
    const current = Number(video.currentTime || 0);
    const progress = duration > 0 ? Math.max(0, Math.min(0.98, current / duration)) : 0;
    terminalEnhancedReplayApplied = true;
    terminalVideoReplayProfile = "enhanced";
    const nextUrl = `${data.url}?t=${Date.now()}`;
    video.pause();
    video.src = nextUrl;
    video.onloadedmetadata = () => {
      const nextDuration = Number(video.duration || 0);
      if (nextDuration > 0) {
        video.currentTime = Math.min(nextDuration - 0.1, Math.max(0, progress * nextDuration));
      }
      const playPromise = video.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch(() => {});
      }
    };
    setViewerMessage("", true);
  } catch (err) {
    // Keep the fast preview playing; enhanced replay is opportunistic.
  }
}

async function preloadTerminalVideoTakeoverUrl(url, token) {
  if (!url || typeof document === "undefined") return Promise.resolve(null);
  releaseTerminalVideoTakeoverPreload();
  let playbackUrl = url;
  try {
    const response = await fetch(url, { cache: "default" });
    if (response.ok && token === terminalAnimationPlaybackToken) {
      const blob = await response.blob();
      if (token !== terminalAnimationPlaybackToken) return null;
      terminalVideoTakeoverObjectUrl = URL.createObjectURL(blob);
      playbackUrl = terminalVideoTakeoverObjectUrl;
    }
  } catch (err) {
    // Fall back to streaming from the server URL if blob preloading fails.
  }
  return new Promise((resolve) => {
    const video = $("replayVideo");
    if (!video) {
      resolve(null);
      return;
    }
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      video.oncanplay = null;
      video.onloadeddata = null;
      video.onerror = null;
      if (!value && terminalVideoTakeoverPreload !== playbackUrl) {
        releaseTerminalVideoTakeoverObjectUrl();
        try {
          video.pause();
          video.removeAttribute("src");
          video.load();
        } catch (err) {
          // Best-effort cleanup only.
        }
      }
      resolve(value);
    };
    const timeout = setTimeout(() => {
      if (token !== terminalAnimationPlaybackToken) {
        finish(null);
        return;
      }
      finish(null);
    }, SHOWCASE_TERMINAL_VIDEO_TAKEOVER_PRELOAD_TIMEOUT_MS);
    const ready = () => {
      if (token !== terminalAnimationPlaybackToken) {
        finish(null);
        return;
      }
      terminalVideoTakeoverPreload = playbackUrl;
      finish(playbackUrl);
    };
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.oncanplay = ready;
    video.onloadeddata = ready;
    video.onerror = () => {
      finish(null);
    };
    video.src = playbackUrl;
    video.load();
  });
}

async function prepareTerminalVideoTakeover(sessionId, token) {
  if (!SHOWCASE_TERMINAL_VIDEO_TAKEOVER || terminalVideoTakeoverStarted || terminalVideoTakeoverApplied) return null;
  terminalVideoTakeoverStarted = true;
  try {
    const data = await api(terminalVideoTakeoverUrl(sessionId));
    if (token !== terminalAnimationPlaybackToken || !data || !data.url) return null;
    return await preloadTerminalVideoTakeoverUrl(data.url, token);
  } catch (err) {
    return null;
  }
}

function applyTerminalVideoTakeover(sessionId, playbackUrl, progress, token) {
  if (
    !SHOWCASE_TERMINAL_VIDEO_TAKEOVER
    || terminalVideoTakeoverApplied
    || terminalVideoTakeoverPending
    || token !== terminalAnimationPlaybackToken
    || !playbackUrl
  ) {
    return false;
  }
  const video = $("replayVideo");
  if (!video) return false;
  terminalVideoTakeoverPending = true;
  const stream = $("stream");
  video.onerror = () => {
    terminalVideoTakeoverPending = false;
    const perf = window.__airvlnLastReplayPerf;
    if (perf && perf.mode === "sprite-chunks") {
      perf.video_takeover_pending = false;
      perf.video_takeover_applied = false;
    }
    video.controls = true;
    document.body.classList.remove("videoReplay");
    document.body.classList.remove("videoTakeoverActive");
    terminalVideoReplayStarted = false;
    startTerminalMjpegReplay(sessionId);
  };
  video.onplaying = () => {
    if (token !== terminalAnimationPlaybackToken) return;
    terminalVideoTakeoverPending = false;
    terminalVideoTakeoverApplied = true;
    terminalVideoReplayProfile = "video-takeover";
    const perf = window.__airvlnLastReplayPerf;
    if (perf && perf.mode === "sprite-chunks") {
      perf.video_takeover_pending = false;
      perf.video_takeover_applied = true;
      perf.video_takeover_started_at_ms = Math.round(performance.now());
    }
    if (terminalSpriteRaf) window.cancelAnimationFrame(terminalSpriteRaf);
    if (terminalSpriteFrameTimer) clearTimeout(terminalSpriteFrameTimer);
    terminalSpriteRaf = null;
    terminalSpriteFrameTimer = null;
    clearTerminalReplayPrefetchTimers();
    releaseTerminalAnimationObjectUrls();
    releaseTerminalSpriteBitmaps();
    document.body.classList.remove("terminalSpriteReplay");
    document.body.classList.remove("terminalImageReplay");
    document.body.classList.add("videoReplay");
    document.body.classList.add("videoTakeoverActive");
    if (stream) stream.removeAttribute("src");
    video.controls = false;
    terminalVideoTakeoverPreload = null;
    setText(
      "liveFps",
      `video takeover | progress ${Math.round(progress * 100)}% | ${SHOWCASE_TERMINAL_VIDEO_TAKEOVER_FPS}fps ${SHOWCASE_TERMINAL_VIDEO_TAKEOVER_WIDTH}w`
    );
    monitorTerminalVideoQuality(sessionId, video);
  };
  video.onended = () => {
    if (terminalVideoQualityTimer) clearInterval(terminalVideoQualityTimer);
    terminalVideoQualityTimer = null;
    video.controls = true;
    document.body.classList.remove("videoTakeoverActive");
  };
  video.playbackRate = 1;
  video.loop = false;
  const playFromProgress = () => {
    const duration = Number(video.duration || 0);
    if (duration > 0) {
      video.currentTime = Math.max(0, Math.min(duration - 0.1, progress * duration));
    }
    setViewerMessage("", true);
    const playPromise = video.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {
        terminalVideoTakeoverPending = false;
        const perf = window.__airvlnLastReplayPerf;
        if (perf && perf.mode === "sprite-chunks") {
          perf.video_takeover_pending = false;
          perf.video_takeover_applied = false;
        }
      });
    }
  };
  if (video.currentSrc !== playbackUrl && video.getAttribute("src") !== playbackUrl) {
    video.src = playbackUrl;
    video.onloadedmetadata = playFromProgress;
    video.load();
  } else if (video.readyState >= 1) {
    playFromProgress();
  } else {
    video.onloadedmetadata = playFromProgress;
    video.load();
  }
  return false;
}

async function terminalVideoPlaybackUrl(data, profile, releasePrevious = true, silent = false) {
  if (!SHOWCASE_TERMINAL_BLOB_PLAYBACK || !data || !data.url) return data ? data.url : "";
  const cachedUrl = cachedDemoReplayBlobUrl(data.url);
  if (cachedUrl) {
    if (data.metadata && typeof data.metadata === "object") {
      data.metadata.client_blob_cached = true;
    }
    if (releasePrevious) {
      releaseTerminalVideoObjectUrl();
      terminalVideoObjectUrl = cachedUrl;
    } else if (!terminalAnimationObjectUrls.includes(cachedUrl)) {
      terminalAnimationObjectUrls.push(cachedUrl);
    }
    return cachedUrl;
  }
  const size = data.metadata && Number(data.metadata.size || 0);
  const sizeMb = size > 0 ? ` ${(size / 1024 / 1024).toFixed(1)}MB` : "";
  if (!silent) {
    setViewerMessage(`飞行完成，正在加载本地播放缓存${sizeMb}...`, false);
  }
  try {
    const response = await fetch(data.url, { cache: "default" });
    if (!response.ok) throw new Error(`video fetch failed: ${response.status}`);
    const blob = await response.blob();
    if (releasePrevious) {
      releaseTerminalVideoObjectUrl();
      terminalVideoObjectUrl = URL.createObjectURL(blob);
      return terminalVideoObjectUrl;
    }
    const objectUrl = URL.createObjectURL(blob);
    terminalAnimationObjectUrls.push(objectUrl);
    return objectUrl;
  } catch (err) {
    // If the browser cannot create a Blob URL, fall back to direct video loading.
    return data.url;
  }
}

async function fetchTerminalAnimationChunk(chunk, silent = false) {
  const data = await api(chunk.url);
  if (!data || !data.url) throw new Error("missing replay chunk url");
  const playbackUrl = await terminalVideoPlaybackUrl(data, "animation chunk", false, silent);
  const metadata = data.metadata || {};
  return {
    url: playbackUrl,
    durationMs: Number(metadata.duration_ms || chunk.duration_ms || 1000),
    index: Number(metadata.chunk_index ?? chunk.index ?? 0),
    frameCount: Number(metadata.frame_count || chunk.frame_count || 0),
    size: Number(metadata.size || 0),
  };
}

function loadReplayImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => {
      if (typeof image.decode !== "function") {
        resolve(image);
        return;
      }
      image.decode().then(() => resolve(image)).catch(() => resolve(image));
    };
    image.onerror = reject;
    image.src = url;
  });
}

async function replayBitmapFromResponseData(data, silent = false) {
  if (!data || !data.url || typeof createImageBitmap !== "function") return null;
  const size = data.metadata && Number(data.metadata.size || 0);
  const sizeMb = size > 0 ? ` ${(size / 1024 / 1024).toFixed(1)}MB` : "";
  if (!silent) {
    setViewerMessage(`飞行完成，正在解码本地帧缓存${sizeMb}...`, false);
  }
  try {
    const response = await fetch(data.url, { cache: "default" });
    if (!response.ok) throw new Error(`sprite fetch failed: ${response.status}`);
    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob);
    terminalSpriteBitmaps.push(bitmap);
    return bitmap;
  } catch (err) {
    return null;
  }
}

async function fetchTerminalSpriteChunk(chunk, silent = false) {
  const data = await api(chunk.url);
  if (!data || !data.url) throw new Error("missing sprite replay chunk url");
  let image = await replayBitmapFromResponseData(data, silent);
  let objectUrl = null;
  if (!image) {
    const playbackUrl = await terminalVideoPlaybackUrl(data, "sprite chunk", false, silent);
    objectUrl = playbackUrl && playbackUrl.startsWith("blob:") ? playbackUrl : null;
    image = await loadReplayImage(playbackUrl);
  }
  const metadata = data.metadata || {};
  return {
    image,
    objectUrl,
    released: false,
    index: Number(metadata.chunk_index ?? chunk.index ?? 0),
    frameCount: Number(metadata.frame_count || chunk.frame_count || 0),
    frameWidth: Number(metadata.frame_width || 1),
    frameHeight: Number(metadata.frame_height || 1),
    columns: Number(metadata.columns || chunk.columns || SHOWCASE_TERMINAL_SPRITE_COLUMNS),
    fps: Number(metadata.fps || SHOWCASE_TERMINAL_REPLAY_FPS),
    size: Number(metadata.size || 0),
  };
}

function scheduleTerminalSpriteIdleTask(task) {
  if (typeof window.requestIdleCallback === "function") {
    window.requestIdleCallback(task, { timeout: 90 });
    return;
  }
  window.setTimeout(() => task({ didTimeout: true, timeRemaining: () => 0 }), 16);
}

function prepareTerminalSpriteFrameBitmaps(chunk) {
  if (
    !SHOWCASE_TERMINAL_SPRITE_FRAME_BITMAPS
    || !chunk
    || !chunk.image
    || chunk.frameImages
    || chunk.frameImagePromise
    || chunk.frameSliceFailed
    || typeof createImageBitmap !== "function"
  ) {
    return chunk ? chunk.frameImagePromise : null;
  }
  chunk.frameImages = new Array(chunk.frameCount);
  chunk.frameImagePromise = new Promise((resolve) => {
    let frame = 0;
    const step = async () => {
      if (chunk.released || chunk.frameSliceFailed) {
        chunk.frameImagePromise = null;
        resolve(chunk);
        return;
      }
      try {
        const sourceX = (frame % chunk.columns) * chunk.frameWidth;
        const sourceY = Math.floor(frame / chunk.columns) * chunk.frameHeight;
        const bitmap = await createImageBitmap(
          chunk.image,
          sourceX,
          sourceY,
          chunk.frameWidth,
          chunk.frameHeight,
        );
        if (chunk.released) {
          if (bitmap && typeof bitmap.close === "function") bitmap.close();
        } else {
          chunk.frameImages[frame] = bitmap;
        }
        frame += 1;
      } catch (err) {
        if (Array.isArray(chunk.frameImages)) {
          for (const bitmap of chunk.frameImages) {
            try {
              if (bitmap && typeof bitmap.close === "function") bitmap.close();
            } catch (closeErr) {
              // Best-effort cleanup only.
            }
          }
          chunk.frameImages = null;
        }
        chunk.frameSliceFailed = true;
      }
      if (chunk.frameSliceFailed || chunk.released || frame >= chunk.frameCount) {
        chunk.frameImagePromise = null;
        resolve(chunk);
        return;
      }
      scheduleTerminalSpriteIdleTask(step);
    };
    scheduleTerminalSpriteIdleTask(step);
  });
  return chunk.frameImagePromise;
}

function drawSpriteReplayFrame(chunk, frameIndex) {
  const canvas = $("streamCanvas");
  if (!canvas || !chunk || !chunk.image) return false;
  const ctx = streamCanvasCtx || canvas.getContext("2d", { alpha: false, desynchronized: true });
  if (!ctx) return false;
  streamCanvasCtx = ctx;
  return drawSpriteReplayFrameOnCanvas(canvas, ctx, chunk, frameIndex);
}

function drawSpriteReplayFrameOnCanvas(canvas, ctx, chunk, frameIndex) {
  if (!canvas || !ctx || !chunk || !chunk.image) return false;
  const { width, height, dpr } = resizeStreamCanvas(canvas);
  const frame = Math.max(0, Math.min(chunk.frameCount - 1, frameIndex));
  const drawStateKey = `${width}x${height}@${dpr}`;
  if (ctx.__airvlnSpriteDrawStateKey !== drawStateKey) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "medium";
    ctx.__airvlnSpriteDrawStateKey = drawStateKey;
  }
  const frameImage = Array.isArray(chunk.frameImages) ? chunk.frameImages[frame] : null;
  if (frameImage) {
    ctx.drawImage(
      frameImage,
      0,
      0,
      chunk.frameWidth,
      chunk.frameHeight,
      0,
      0,
      width,
      height,
    );
    return true;
  }
  const sourceX = (frame % chunk.columns) * chunk.frameWidth;
  const sourceY = Math.floor(frame / chunk.columns) * chunk.frameHeight;
  ctx.drawImage(
    chunk.image,
    sourceX,
    sourceY,
    chunk.frameWidth,
    chunk.frameHeight,
    0,
    0,
    width,
    height,
  );
  return true;
}

async function startTerminalSpriteReplay(sessionId, force = false) {
  if (!SHOWCASE_TERMINAL_SPRITE_REPLAY || !sessionId) return false;
  if (terminalVideoReplayStarted && !force) return false;
  terminalVideoReplayStarted = true;
  terminalVideoReplayProfile = "sprite-chunks";
  terminalAnimationPlaybackToken += 1;
  const token = terminalAnimationPlaybackToken;
  setViewerMessage("飞行完成，正在生成首段帧级回放...", false);
  try {
    const manifest = await api(terminalSpriteManifestUrl(sessionId));
    const chunks = Array.isArray(manifest.chunks) ? manifest.chunks : [];
    const canvas = $("streamCanvas");
    if (!chunks.length || !canvas) return false;
    const chunkFrameCounts = chunks.map((chunk) => Number(chunk.frame_count || 0));
    const cumulativeChunkFrames = [];
    let totalSpriteFrames = 0;
    for (const count of chunkFrameCounts) {
      cumulativeChunkFrames.push(totalSpriteFrames);
      totalSpriteFrames += count;
    }
    const ctx = streamCanvasCtx || canvas.getContext("2d", { alpha: false, desynchronized: true });
    if (!ctx) return false;
    streamCanvasCtx = ctx;
    ctx.__airvlnSpriteDrawStateKey = "";

    if (terminalAnimationChunkTimer) clearTimeout(terminalAnimationChunkTimer);
    if (terminalSpriteRaf) window.cancelAnimationFrame(terminalSpriteRaf);
    releaseTerminalAnimationObjectUrls();
    releaseTerminalSpriteBitmaps();
    terminalAnimationChunkTimer = null;
    terminalSpriteRaf = null;
    if (delayedFrameFetchTimer) clearInterval(delayedFrameFetchTimer);
    if (delayedFrameRaf) window.cancelAnimationFrame(delayedFrameRaf);
    if (terminalVideoQualityTimer) clearInterval(terminalVideoQualityTimer);
    delayedFrameFetchTimer = null;
    delayedFrameRaf = null;
    terminalVideoQualityTimer = null;
  document.body.classList.remove("delayedFramePlayer");
  document.body.classList.remove("videoReplay");
  document.body.classList.remove("videoTakeoverActive");
  document.body.classList.remove("terminalImageReplay");
  document.body.classList.add("terminalSpriteReplay");
    streamCanvasMetrics = null;
    streamCanvasLastResizeCheck = 0;
    terminalSpriteRenderScale = 1.0;
    const stream = $("stream");
    if (stream) stream.removeAttribute("src");
    const video = $("replayVideo");
    if (video) {
      video.pause();
      video.removeAttribute("src");
      video.load();
    }
    let videoTakeoverUrl = null;
    let videoTakeoverPromise = null;
    const scheduleVideoTakeoverPrewarm = () => {
      if (!SHOWCASE_TERMINAL_VIDEO_TAKEOVER) return;
      const timer = setTimeout(() => {
        if (token !== terminalAnimationPlaybackToken) return;
        videoTakeoverPromise = prepareTerminalVideoTakeover(sessionId, token).then((url) => {
          videoTakeoverUrl = url;
          if (spritePerf) {
            spritePerf.video_takeover_ready = Boolean(url);
            spritePerf.video_takeover_blob = typeof url === "string" && url.startsWith("blob:");
          }
          return url;
        });
        videoTakeoverPromise.catch(() => {});
      }, SHOWCASE_TERMINAL_VIDEO_TAKEOVER_START_DELAY_MS);
      terminalReplayPrefetchTimers.push(timer);
    };

    const chunkPromises = new Map();
    const chunkReady = new Map();
    const chunkErrors = new Map();
    const loadChunk = (index, silent = true) => {
      if (index < 0 || index >= chunks.length) return null;
      if (!chunkPromises.has(index)) {
        const promise = fetchTerminalSpriteChunk(chunks[index], silent)
          .then((chunk) => {
            chunkReady.set(index, chunk);
            return chunk;
          })
          .catch((err) => {
            chunkErrors.set(index, err);
            throw err;
          });
        chunkPromises.set(index, promise);
      }
      return chunkPromises.get(index);
    };
    const prefetchChunks = (fromIndex) => scheduleReplayPrefetch(loadChunk, chunks.length, fromIndex, token);
    const prefetchFutureChunks = (fromIndex) => {
      prefetchChunks(fromIndex);
      scheduleReplayBackgroundPrefetch(
        loadChunk,
        chunks.length,
        fromIndex + SHOWCASE_TERMINAL_REPLAY_PREFETCH_AHEAD,
        token
      );
    };
    let spriteFrameBitmapsEnabled = SHOWCASE_TERMINAL_SPRITE_FRAME_BITMAPS;
    let spriteFrameBitmapsDisabledReason = "";
    let spritePerf = null;
    const disableSpriteFrameBitmaps = (reason) => {
      if (!spriteFrameBitmapsEnabled) return;
      spriteFrameBitmapsEnabled = false;
      spriteFrameBitmapsDisabledReason = reason;
      if (spritePerf) {
        spritePerf.bitmap_disabled = true;
        spritePerf.bitmap_disabled_reason = reason;
      }
      for (const chunk of chunkReady.values()) {
        releaseTerminalSpriteChunkFrameImages(chunk);
      }
    };
    const prepareFrameBitmapsAround = (fromIndex) => {
      if (!SHOWCASE_TERMINAL_SPRITE_FRAME_BITMAPS || !spriteFrameBitmapsEnabled) return;
      const endIndex = Math.min(chunks.length, fromIndex + SHOWCASE_TERMINAL_SPRITE_FRAME_BITMAP_AHEAD + 1);
      for (let index = Math.max(0, fromIndex); index < endIndex; index += 1) {
        const readyChunk = chunkReady.get(index);
        if (readyChunk) {
          const framePromise = prepareTerminalSpriteFrameBitmaps(readyChunk);
          if (framePromise && typeof framePromise.catch === "function") framePromise.catch(() => {});
          continue;
        }
        const promise = loadChunk(index, true);
        if (promise && typeof promise.then === "function") {
          promise
            .then((loadedChunk) => prepareTerminalSpriteFrameBitmaps(loadedChunk))
            .catch(() => {});
        }
      }
    };
    const releaseChunksOutsideWindow = (activeIndex) => {
      for (const [index, promise] of Array.from(chunkPromises.entries())) {
        if (
          index >= activeIndex - SHOWCASE_TERMINAL_SPRITE_KEEP_BEHIND
          && index <= activeIndex + SHOWCASE_TERMINAL_SPRITE_KEEP_AHEAD
        ) {
          continue;
        }
        promise.then(releaseTerminalSpriteChunkImage).catch(() => {});
        chunkPromises.delete(index);
        chunkReady.delete(index);
        chunkErrors.delete(index);
      }
    };
    const trimReadyChunkPressure = (activeIndex) => {
      if (chunkReady.size <= SHOWCASE_TERMINAL_SPRITE_MAX_READY_CHUNKS) return;
      const candidates = Array.from(chunkPromises.keys())
        .filter((index) => index !== activeIndex && index !== activeIndex + 1)
        .sort((left, right) => Math.abs(right - activeIndex) - Math.abs(left - activeIndex));
      for (const index of candidates) {
        if (chunkReady.size <= SHOWCASE_TERMINAL_SPRITE_MAX_READY_CHUNKS) break;
        const promise = chunkPromises.get(index);
        if (!promise) continue;
        promise.then(releaseTerminalSpriteChunkImage).catch(() => {});
        chunkPromises.delete(index);
        chunkReady.delete(index);
        chunkErrors.delete(index);
      }
    };

    const startBufferCount = Math.min(chunks.length, SHOWCASE_TERMINAL_SPRITE_START_BUFFER_CHUNKS);
    const startBuffer = await Promise.all(
      Array.from({ length: startBufferCount }, (_, index) => loadChunk(index, index !== 0))
    );
    const firstChunk = startBuffer[0];
    if (token !== terminalAnimationPlaybackToken || !firstChunk) return false;
    prefetchFutureChunks(startBufferCount);
    prepareFrameBitmapsAround(0);
    setViewerMessage("", true);
    startTerminalReplayPrewarm(sessionId, startBufferCount);
    scheduleVideoTakeoverPrewarm();

    let currentChunk = firstChunk;
    let currentIndex = 0;
    let chunkStartedAt = 0;
    let lastDrawnChunk = -1;
    let lastDrawnFrame = -1;
    let spriteDrawsSinceStats = 0;
    let spriteWaitsSinceStats = 0;
    let spriteLongGapsSinceStats = 0;
    let spritePlaybackFps = SHOWCASE_TERMINAL_REPLAY_FPS;
    let spriteStableStatsWindows = 0;
    let spritePressureWindows = 0;
    let spriteLastStatsAt = performance.now();
    let spriteLastRenderAt = 0;
    let spriteFinished = false;
    spritePerf = {
      session_id: sessionId,
      mode: "sprite-chunks",
      version: "20260825-crisp-slow-v44",
      target_fps_initial: SHOWCASE_TERMINAL_REPLAY_FPS,
      target_fps_final: SHOWCASE_TERMINAL_REPLAY_FPS,
      min_target_fps: SHOWCASE_TERMINAL_REPLAY_FPS,
      render_scale_final: 1.0,
      min_render_scale: 1.0,
      total_frames: totalSpriteFrames,
      chunk_count: chunks.length,
      draws: 0,
      wait_count: 0,
      long_gap_count: 0,
      max_gap_ms: 0,
      pressure_windows: 0,
      fps_downshift_count: 0,
      scale_downshift_count: 0,
      bitmap_disabled: false,
      bitmap_disabled_reason: "",
      video_takeover_requested: SHOWCASE_TERMINAL_VIDEO_TAKEOVER,
      video_takeover_fps: SHOWCASE_TERMINAL_VIDEO_TAKEOVER_FPS,
      video_takeover_stride: SHOWCASE_TERMINAL_VIDEO_TAKEOVER_STRIDE,
      video_takeover_width: SHOWCASE_TERMINAL_VIDEO_TAKEOVER_WIDTH,
      video_takeover_ready: false,
      video_takeover_blob: false,
      video_takeover_pending: false,
      video_takeover_applied: false,
      video_takeover_progress: null,
      max_ready_chunks: 0,
      started_at_ms: Math.round(performance.now()),
      ended_at_ms: null,
      duration_ms: null,
    };
    const updateSpriteReplayStats = (timestamp) => {
      if (spriteLastRenderAt) {
        const gapMs = timestamp - spriteLastRenderAt;
        if (gapMs > SHOWCASE_TERMINAL_SPRITE_LONG_GAP_MS) {
          spriteLongGapsSinceStats += 1;
          spritePerf.long_gap_count += 1;
          spritePerf.max_gap_ms = Math.max(spritePerf.max_gap_ms, Math.round(gapMs));
        }
      }
      spritePerf.max_ready_chunks = Math.max(spritePerf.max_ready_chunks, chunkReady.size);
      spriteLastRenderAt = timestamp;
      const elapsed = performance.now() - spriteLastStatsAt;
      if (elapsed < 1000) return;
      const fps = Math.round((spriteDrawsSinceStats * 1000) / Math.max(1, elapsed));
      const severePressure = spriteWaitsSinceStats >= 6 || spriteLongGapsSinceStats >= 3;
      const moderatePressure = spriteWaitsSinceStats >= 4 || spriteLongGapsSinceStats >= 2;
      if (moderatePressure) {
        spritePressureWindows += 1;
        spritePerf.pressure_windows += 1;
        if (spritePressureWindows >= SHOWCASE_TERMINAL_SPRITE_BITMAP_PRESSURE_DISABLES) {
          disableSpriteFrameBitmaps(severePressure ? "severe pressure" : "pressure");
        }
      } else if (!spriteWaitsSinceStats && !spriteLongGapsSinceStats) {
        spritePressureWindows = 0;
      }
      if (moderatePressure && spritePlaybackFps > 18) {
        spritePlaybackFps = Math.max(18, spritePlaybackFps - 2);
        spritePerf.fps_downshift_count += 1;
        spritePerf.min_target_fps = Math.min(spritePerf.min_target_fps, spritePlaybackFps);
        spriteStableStatsWindows = 0;
        if (chunkStartedAt && lastDrawnFrame >= 0) {
          chunkStartedAt = timestamp - (lastDrawnFrame * (1000 / spritePlaybackFps));
        }
      } else if (severePressure && terminalSpriteRenderScale > TERMINAL_SPRITE_RENDER_SCALE_MIN) {
        terminalSpriteRenderScale = Math.max(TERMINAL_SPRITE_RENDER_SCALE_MIN, terminalSpriteRenderScale - 0.125);
        spritePerf.scale_downshift_count += 1;
        spritePerf.min_render_scale = Math.min(spritePerf.min_render_scale, terminalSpriteRenderScale);
        streamCanvasMetrics = null;
        streamCanvasLastResizeCheck = 0;
        ctx.__airvlnSpriteDrawStateKey = "";
        spriteStableStatsWindows = 0;
      } else if (!spriteWaitsSinceStats && !spriteLongGapsSinceStats && spritePlaybackFps < SHOWCASE_TERMINAL_REPLAY_FPS) {
        spriteStableStatsWindows += 1;
        if (spriteStableStatsWindows >= 3) {
          spritePlaybackFps = Math.min(SHOWCASE_TERMINAL_REPLAY_FPS, spritePlaybackFps + 1);
          spriteStableStatsWindows = 0;
          if (chunkStartedAt && lastDrawnFrame >= 0) {
            chunkStartedAt = timestamp - (lastDrawnFrame * (1000 / spritePlaybackFps));
          }
        }
      } else if (!spriteWaitsSinceStats && !spriteLongGapsSinceStats && terminalSpriteRenderScale < 1.0) {
        spriteStableStatsWindows += 1;
        if (spriteStableStatsWindows >= 4) {
          terminalSpriteRenderScale = Math.min(1.0, terminalSpriteRenderScale + 0.125);
          streamCanvasMetrics = null;
          streamCanvasLastResizeCheck = 0;
          ctx.__airvlnSpriteDrawStateKey = "";
          spriteStableStatsWindows = 0;
        }
      } else {
        spriteStableStatsWindows = 0;
      }
      const suffix = spriteWaitsSinceStats || spriteLongGapsSinceStats
        ? ` | wait ${spriteWaitsSinceStats} gap ${spriteLongGapsSinceStats}`
        : "";
      const target = spritePlaybackFps !== SHOWCASE_TERMINAL_REPLAY_FPS ? ` target ${spritePlaybackFps}` : "";
      const scaleLabel = terminalSpriteRenderScale < 1.0
        ? ` scale ${Math.round(terminalSpriteRenderScale * 100)}%`
        : "";
      const bitmapLabel = spriteFrameBitmapsDisabledReason ? " bitmap off" : "";
      setText("liveFps", `sprite ${fps} fps${target}${scaleLabel}${bitmapLabel}${suffix}`);
      spritePerf.target_fps_final = spritePlaybackFps;
      spritePerf.render_scale_final = terminalSpriteRenderScale;
      spriteDrawsSinceStats = 0;
      spriteWaitsSinceStats = 0;
      spriteLongGapsSinceStats = 0;
      spriteLastStatsAt = performance.now();
    };
    const drawSpriteFrameOnce = (chunk, chunkIndex, frameIndex) => {
      if (lastDrawnChunk === chunkIndex && lastDrawnFrame === frameIndex) return;
      if (drawSpriteReplayFrameOnCanvas(canvas, ctx, chunk, frameIndex)) {
        lastDrawnChunk = chunkIndex;
        lastDrawnFrame = frameIndex;
        spriteDrawsSinceStats += 1;
        spritePerf.draws += 1;
      }
    };
    const finishSpriteReplay = () => {
      if (spriteFinished) return;
      spriteFinished = true;
      spritePerf.ended_at_ms = Math.round(performance.now());
      spritePerf.duration_ms = Math.max(0, spritePerf.ended_at_ms - spritePerf.started_at_ms);
      spritePerf.target_fps_final = spritePlaybackFps;
      spritePerf.render_scale_final = terminalSpriteRenderScale;
      window.__airvlnLastReplayPerf = spritePerf;
      const averageFps = Math.round((spritePerf.draws * 1000) / Math.max(1, spritePerf.duration_ms));
      const summary = `replay done | avg ${averageFps}fps | gap ${spritePerf.long_gap_count} max ${spritePerf.max_gap_ms}ms | wait ${spritePerf.wait_count} | chunks ${spritePerf.max_ready_chunks}`;
      setText("liveFps", summary);
      const status = $("status");
      if (status) {
        status.textContent = JSON.stringify({ replay_perf: spritePerf }, null, 2);
        lastStatusText = status.textContent;
        lastStatusPanelAt = performance.now();
      }
    };
    const scheduleSpriteFrame = (delayMs = 0) => {
      if (terminalSpriteFrameTimer) clearTimeout(terminalSpriteFrameTimer);
      terminalSpriteFrameTimer = null;
      if (terminalSpriteRaf) window.cancelAnimationFrame(terminalSpriteRaf);
      const schedule = () => {
        if (token !== terminalAnimationPlaybackToken) return;
        terminalSpriteRaf = window.requestAnimationFrame(render);
      };
      if (delayMs > 0 && delayMs > 80) {
        terminalSpriteFrameTimer = setTimeout(schedule, Math.max(0, delayMs - 50));
      } else {
        schedule();
      }
    };
    const render = (timestamp) => {
      if (token !== terminalAnimationPlaybackToken || !currentChunk) return;
      updateSpriteReplayStats(timestamp);
      if (!chunkStartedAt) chunkStartedAt = timestamp;
      const frameMs = 1000 / Math.max(1, spritePlaybackFps || currentChunk.fps);
      const frameIndex = Math.floor((timestamp - chunkStartedAt) / frameMs);
      const completedFrames = cumulativeChunkFrames[currentIndex] || 0;
      const progress = totalSpriteFrames > 0
        ? Math.max(0, Math.min(0.98, (completedFrames + Math.min(frameIndex, currentChunk.frameCount)) / totalSpriteFrames))
        : 0;
      if (
        videoTakeoverUrl
        && progress >= SHOWCASE_TERMINAL_VIDEO_TAKEOVER_MIN_PROGRESS
        && !terminalVideoTakeoverPending
        && !terminalVideoTakeoverApplied
      ) {
        spritePerf.video_takeover_pending = true;
        spritePerf.video_takeover_progress = Math.round(progress * 1000) / 1000;
        spritePerf.ended_at_ms = Math.round(performance.now());
        spritePerf.duration_ms = Math.max(0, spritePerf.ended_at_ms - spritePerf.started_at_ms);
        spritePerf.target_fps_final = spritePlaybackFps;
        spritePerf.render_scale_final = terminalSpriteRenderScale;
        window.__airvlnLastReplayPerf = spritePerf;
        applyTerminalVideoTakeover(sessionId, videoTakeoverUrl, progress, token);
      }
      if (frameIndex >= currentChunk.frameCount) {
        const nextIndex = currentIndex + 1;
        if (nextIndex >= chunks.length) {
          drawSpriteFrameOnce(currentChunk, currentIndex, currentChunk.frameCount - 1);
          finishSpriteReplay();
          return;
        }
        if (chunkErrors.has(nextIndex)) {
          startTerminalChunkedAnimationReplay(sessionId, true);
          return;
        }
        loadChunk(nextIndex, true);
        const nextChunk = chunkReady.get(nextIndex);
        if (!nextChunk) {
          spriteWaitsSinceStats += 1;
          spritePerf.wait_count += 1;
          drawSpriteFrameOnce(currentChunk, currentIndex, currentChunk.frameCount - 1);
          scheduleSpriteFrame(0);
          return;
        }
        currentChunk = nextChunk;
        currentIndex = nextIndex;
        chunkStartedAt = timestamp;
        releaseChunksOutsideWindow(currentIndex);
        trimReadyChunkPressure(currentIndex);
        prefetchFutureChunks(nextIndex + 1);
        prepareFrameBitmapsAround(currentIndex);
        drawSpriteFrameOnce(currentChunk, currentIndex, 0);
      } else {
        if (frameIndex >= Math.max(0, currentChunk.frameCount - 12)) {
          loadChunk(currentIndex + 1, true);
          prepareFrameBitmapsAround(currentIndex + 1);
        }
        drawSpriteFrameOnce(currentChunk, currentIndex, frameIndex);
      }
      scheduleSpriteFrame(0);
    };
    scheduleSpriteFrame(0);
    return true;
  } catch (err) {
    terminalVideoReplayStarted = false;
    return false;
  }
}

async function startTerminalChunkedAnimationReplay(sessionId, force = false) {
  if (!SHOWCASE_TERMINAL_CHUNKED_ANIMATION_REPLAY || !sessionId) return false;
  if (terminalVideoReplayStarted && !force) return false;
  terminalVideoReplayStarted = true;
  terminalVideoReplayProfile = "animation-chunks";
  terminalAnimationPlaybackToken += 1;
  const token = terminalAnimationPlaybackToken;
  setViewerMessage("飞行完成，正在生成首段流畅动画...", false);
  try {
    const manifest = await api(terminalAnimationManifestUrl(sessionId));
    const chunks = Array.isArray(manifest.chunks) ? manifest.chunks : [];
    const stream = $("stream");
    if (!chunks.length || !stream) return false;

    if (terminalAnimationChunkTimer) clearTimeout(terminalAnimationChunkTimer);
    releaseTerminalAnimationObjectUrls();
    const chunkPromises = new Map();
    const loadChunk = (index, silent = true) => {
      if (index < 0 || index >= chunks.length) return null;
      if (!chunkPromises.has(index)) {
        chunkPromises.set(index, fetchTerminalAnimationChunk(chunks[index], silent));
      }
      return chunkPromises.get(index);
    };
    const prefetchChunks = (fromIndex) => scheduleReplayPrefetch(loadChunk, chunks.length, fromIndex, token);

    const firstChunk = await loadChunk(0, false);
    if (token !== terminalAnimationPlaybackToken) return false;
    prefetchChunks(1);

    if (delayedFrameFetchTimer) clearInterval(delayedFrameFetchTimer);
    if (delayedFrameRaf) window.cancelAnimationFrame(delayedFrameRaf);
    if (terminalVideoQualityTimer) clearInterval(terminalVideoQualityTimer);
    delayedFrameFetchTimer = null;
    delayedFrameRaf = null;
    terminalVideoQualityTimer = null;
    document.body.classList.remove("delayedFramePlayer");
    document.body.classList.remove("videoReplay");
    document.body.classList.add("terminalImageReplay");
    const video = $("replayVideo");
    if (video) {
      video.pause();
      video.removeAttribute("src");
      video.load();
    }

    let playingIndex = 0;
    const playChunk = async (index, loadedChunk = null) => {
      if (token !== terminalAnimationPlaybackToken || index >= chunks.length) return;
      let chunk = loadedChunk;
      if (!chunk) chunk = await loadChunk(index);
      if (token !== terminalAnimationPlaybackToken || !chunk) return;
      playingIndex = index;
      stream.onerror = () => {
        document.body.classList.remove("terminalImageReplay");
        terminalVideoReplayStarted = false;
        startTerminalAnimationReplay(sessionId, true);
      };
      stream.onload = () => setViewerMessage("", true);
      stream.src = chunk.url;
      setViewerMessage("", true);
      prefetchChunks(index + 1);
      const switchDelay = Math.max(250, Number(chunk.durationMs || 1000) - 80);
      terminalAnimationChunkTimer = setTimeout(async () => {
        if (token !== terminalAnimationPlaybackToken) return;
        const nextIndex = playingIndex + 1;
        if (nextIndex >= chunks.length) return;
        try {
          const nextChunk = await loadChunk(nextIndex);
          playChunk(nextIndex, nextChunk);
        } catch (err) {
          startTerminalAnimationReplay(sessionId, true);
        }
      }, switchDelay);
    };

    await playChunk(0, firstChunk);
    return true;
  } catch (err) {
    terminalVideoReplayStarted = false;
    return false;
  }
}

async function startTerminalAnimationReplay(sessionId, force = false) {
  if (!SHOWCASE_TERMINAL_ANIMATION_REPLAY || !sessionId) return false;
  if (terminalVideoReplayStarted && !force) return false;
  terminalVideoReplayStarted = true;
  terminalVideoReplayProfile = "animation";
  setViewerMessage("飞行完成，正在生成轻量清晰动画...", false);
  try {
    const data = await api(terminalAnimationReplayUrl(sessionId));
    const stream = $("stream");
    if (!data || !data.url || !stream) {
      terminalVideoReplayStarted = false;
      return false;
    }
    const playbackUrl = await terminalVideoPlaybackUrl(data, "animation");
    if (delayedFrameFetchTimer) clearInterval(delayedFrameFetchTimer);
    if (delayedFrameRaf) window.cancelAnimationFrame(delayedFrameRaf);
    if (terminalVideoQualityTimer) clearInterval(terminalVideoQualityTimer);
    delayedFrameFetchTimer = null;
    delayedFrameRaf = null;
    terminalVideoQualityTimer = null;
    document.body.classList.remove("delayedFramePlayer");
    document.body.classList.remove("videoReplay");
    document.body.classList.add("terminalImageReplay");
    const video = $("replayVideo");
    if (video) {
      video.pause();
      video.removeAttribute("src");
      video.load();
    }
    stream.onerror = () => {
      document.body.classList.remove("terminalImageReplay");
      terminalVideoReplayStarted = false;
      startTerminalVideoReplay(sessionId, "light", true);
    };
    stream.onload = () => setViewerMessage("", true);
    stream.src = playbackUrl;
    setViewerMessage("", true);
    return true;
  } catch (err) {
    terminalVideoReplayStarted = false;
    return false;
  }
}

async function startTerminalStableAnimationReplay(sessionId, force = false) {
  if (!SHOWCASE_TERMINAL_ANIMATION_REPLAY || !sessionId) return false;
  if (terminalVideoReplayStarted && !force) return false;
  terminalVideoReplayStarted = true;
  terminalVideoReplayProfile = "stable-animation";
  setViewerMessage("飞行完成，正在切换稳定低帧动画...", false);
  try {
    const data = await api(terminalStableAnimationReplayUrl(sessionId));
    const stream = $("stream");
    if (!data || !data.url || !stream) {
      terminalVideoReplayStarted = false;
      return false;
    }
    const playbackUrl = await terminalVideoPlaybackUrl(data, "stable animation");
    const metadata = data.metadata || {};
    if (delayedFrameFetchTimer) clearInterval(delayedFrameFetchTimer);
    if (delayedFrameRaf) window.cancelAnimationFrame(delayedFrameRaf);
    if (terminalVideoQualityTimer) clearInterval(terminalVideoQualityTimer);
    delayedFrameFetchTimer = null;
    delayedFrameRaf = null;
    terminalVideoQualityTimer = null;
    document.body.classList.remove("delayedFramePlayer");
    document.body.classList.remove("videoReplay");
    document.body.classList.add("terminalImageReplay");
    document.body.classList.add("videoTakeoverActive");
    const video = $("replayVideo");
    if (video) {
      video.pause();
      video.controls = true;
      video.removeAttribute("src");
      video.load();
    }
    stream.onerror = () => {
      document.body.classList.remove("terminalImageReplay");
      document.body.classList.remove("videoTakeoverActive");
      terminalVideoReplayStarted = false;
      startTerminalMjpegReplay(sessionId);
    };
    stream.onload = () => {
      const cacheLabel = metadata.cached ? "cached " : "";
      setText(
        "liveFps",
        `${cacheLabel}stable webp | ${SHOWCASE_TERMINAL_STABLE_ANIMATION_FPS}fps ${SHOWCASE_TERMINAL_STABLE_ANIMATION_WIDTH}w`,
      );
      setViewerMessage("", true);
    };
    window.__airvlnLastReplayPerf = {
      session_id: sessionId,
      mode: "stable-webp",
      version: "20260825-crisp-slow-v44",
      animation_fps: SHOWCASE_TERMINAL_STABLE_ANIMATION_FPS,
      animation_stride: SHOWCASE_TERMINAL_STABLE_ANIMATION_STRIDE,
      animation_width: SHOWCASE_TERMINAL_STABLE_ANIMATION_WIDTH,
      animation_quality: SHOWCASE_TERMINAL_STABLE_ANIMATION_QUALITY,
      animation_blob: typeof playbackUrl === "string" && playbackUrl.startsWith("blob:"),
      animation_cached: Boolean(metadata.cached),
      animation_size: Number(metadata.size || 0),
      metadata,
      started_at_ms: Math.round(performance.now()),
    };
    stream.src = playbackUrl;
    setViewerMessage("", true);
    return true;
  } catch (err) {
    terminalVideoReplayStarted = false;
    return false;
  }
}

async function startTerminalVideoReplay(sessionId, profile = "primary", force = false, startProgress = 0, directUrl = "", directMetadata = {}) {
  if (!SHOWCASE_TERMINAL_VIDEO_REPLAY || !sessionId) return false;
  if (terminalVideoReplayStarted && !force) return false;
  terminalVideoReplayStarted = true;
  terminalVideoReplayProfile = profile;
  const label = profile === "demo-smooth"
    ? "正在播放已验证原始帧演示视频..."
    : profile === "demo-smooth-lite"
    ? "正在切换轻量 H.264 平滑演示..."
    : profile === "demo-webm"
    ? "正在播放 WebM 平滑演示兜底..."
    : profile === "demo-webm-lite"
    ? "正在切换轻量 WebM 平滑演示..."
    : profile === "pure"
    ? "飞行完成，正在生成极简流畅视频..."
    : profile === "ultra"
    ? "飞行完成，正在切换稳定兜底视频..."
    : profile === "light"
    ? "飞行完成，正在生成轻量平滑视频..."
    : "飞行完成，正在生成平滑视频...";
  setViewerMessage(label, false);
  try {
    const data = directUrl
      ? { ok: true, url: directUrl, metadata: directMetadata }
      : await api(terminalVideoReplayUrl(sessionId, profile));
    const video = $("replayVideo");
    if (!data || !data.url || !video) {
      terminalVideoReplayStarted = false;
      return false;
    }
    prepareReplayVideoElement(video);
    const playbackUrl = await terminalVideoPlaybackUrl(data, profile);
    const videoMetadata = data.metadata || {};
    const videoCached = Boolean(videoMetadata.cached);
    if (profile === "demo-smooth" || profile === "demo-smooth-lite" || profile === "demo-webm" || profile === "demo-webm-lite") {
      applyDemoReplayDisplaySize(videoMetadata);
    }
    if (profile === "demo-smooth" || profile === "demo-smooth-lite" || profile === "demo-webm" || profile === "demo-webm-lite" || profile === "pure" || profile === "ultra") {
      const isDemoSmooth = profile === "demo-smooth";
      const isDemoSmoothLite = profile === "demo-smooth-lite";
      const isDemoWebm = profile === "demo-webm";
      const isDemoWebmLite = profile === "demo-webm-lite";
      const isUltra = profile === "ultra";
      window.__airvlnLastReplayPerf = {
        session_id: sessionId,
        mode: (isDemoSmooth || isDemoSmoothLite) ? "demo-smooth-video" : (isDemoWebm || isDemoWebmLite) ? "demo-webm-video" : isUltra ? "ultra-video" : "pure-video",
        version: "20260825-stable82-v55",
        video_fps: Number(videoMetadata.fps || 0) || ((isDemoSmoothLite || isDemoWebmLite) ? SHOWCASE_DEMO_LITE_VIDEO_FPS : (isDemoSmooth || isDemoWebm) ? SHOWCASE_DEMO_SMOOTH_VIDEO_FPS : isUltra ? SHOWCASE_TERMINAL_ULTRA_VIDEO_FPS : SHOWCASE_TERMINAL_PURE_VIDEO_FPS),
        video_stride: Number(videoMetadata.stride || 0) || ((isDemoSmooth || isDemoSmoothLite || isDemoWebm || isDemoWebmLite) ? SHOWCASE_DEMO_SMOOTH_VIDEO_STRIDE : isUltra ? SHOWCASE_TERMINAL_ULTRA_VIDEO_STRIDE : SHOWCASE_TERMINAL_PURE_VIDEO_STRIDE),
        video_width: Number(videoMetadata.width || 0) || ((isDemoSmoothLite || isDemoWebmLite) ? SHOWCASE_DEMO_LITE_VIDEO_WIDTH : (isDemoSmooth || isDemoWebm) ? SHOWCASE_DEMO_SMOOTH_VIDEO_WIDTH : isUltra ? SHOWCASE_TERMINAL_ULTRA_VIDEO_WIDTH : SHOWCASE_TERMINAL_PURE_VIDEO_WIDTH),
        video_smooth_multiplier: Number(videoMetadata.smooth_multiplier || 0) || ((isDemoSmoothLite || isDemoWebmLite) ? SHOWCASE_DEMO_LITE_VIDEO_MULTIPLIER : (isDemoSmooth || isDemoWebm) ? SHOWCASE_DEMO_SMOOTH_VIDEO_MULTIPLIER : null),
        video_interp: videoMetadata.interp || ((isDemoSmooth || isDemoWebm) ? "flow" : (isDemoSmoothLite || isDemoWebmLite) ? "hold" : null),
        video_blob: typeof playbackUrl === "string" && playbackUrl.startsWith("blob:"),
        video_cached: videoCached,
        client_blob_cached: Boolean(videoMetadata.client_blob_cached),
        video_size: Number(videoMetadata.size || 0),
        metadata: videoMetadata,
        started_at_ms: Math.round(performance.now()),
      };
    }
    if (delayedFrameFetchTimer) clearInterval(delayedFrameFetchTimer);
    if (delayedFrameRaf) window.cancelAnimationFrame(delayedFrameRaf);
    delayedFrameFetchTimer = null;
    delayedFrameRaf = null;
    document.body.classList.remove("delayedFramePlayer");
    document.body.classList.add("videoReplay");
    const stream = $("stream");
    if (stream) stream.removeAttribute("src");
    video.onerror = () => {
      document.body.classList.remove("videoReplay");
      document.body.classList.remove("videoTakeoverActive");
      video.controls = true;
      fallbackTerminalVideoReplay(sessionId, video, profile, "video_error");
    };
    video.onplaying = () => {
      if (profile === "demo-smooth" || profile === "demo-smooth-lite" || profile === "demo-webm" || profile === "demo-webm-lite" || profile === "pure" || profile === "ultra") {
        const smoothProfile = profile === "demo-smooth" || profile === "demo-smooth-lite" || profile === "demo-webm" || profile === "demo-webm-lite";
        const fps = Number(videoMetadata.fps || 0) || ((profile === "demo-smooth-lite" || profile === "demo-webm-lite") ? SHOWCASE_DEMO_LITE_VIDEO_FPS : smoothProfile ? SHOWCASE_DEMO_SMOOTH_VIDEO_FPS : profile === "ultra" ? SHOWCASE_TERMINAL_ULTRA_VIDEO_FPS : SHOWCASE_TERMINAL_PURE_VIDEO_FPS);
        const width = Number(videoMetadata.width || 0) || ((profile === "demo-smooth-lite" || profile === "demo-webm-lite") ? SHOWCASE_DEMO_LITE_VIDEO_WIDTH : smoothProfile ? SHOWCASE_DEMO_SMOOTH_VIDEO_WIDTH : profile === "ultra" ? SHOWCASE_TERMINAL_ULTRA_VIDEO_WIDTH : SHOWCASE_TERMINAL_PURE_VIDEO_WIDTH);
        const cacheLabel = videoCached ? "cached " : "";
        const profileLabel = profile === "demo-smooth" ? "smooth demo" : profile === "demo-smooth-lite" ? "lite h264" : profile === "demo-webm" ? "smooth webm" : profile === "demo-webm-lite" ? "lite webm" : `${profile} video`;
        document.body.classList.add("videoTakeoverActive");
        video.controls = false;
        const interpLabel = videoMetadata.interp ? ` ${videoMetadata.interp}` : "";
        setText("liveFps", `${cacheLabel}${profileLabel} | ${fps}fps ${width}w${interpLabel}`);
      }
      monitorTerminalVideoQuality(sessionId, video);
      startTerminalVideoRafMonitor(sessionId, video);
    };
    video.oncanplay = () => {
      if (profile === "light" && SHOWCASE_TERMINAL_FAST_PREVIEW) {
        prepareEnhancedTerminalReplay(sessionId, video);
      }
    };
    video.onended = () => {
      if (terminalVideoQualityTimer) clearInterval(terminalVideoQualityTimer);
      terminalVideoQualityTimer = null;
      stopTerminalVideoRafMonitor();
      if (profile === "demo-smooth" || profile === "demo-smooth-lite" || profile === "demo-webm" || profile === "demo-webm-lite" || profile === "pure" || profile === "ultra") {
        setDemoReplayActive(false);
        document.body.classList.remove("videoTakeoverActive");
        video.controls = true;
      }
    };
    video.src = playbackUrl;
    video.currentTime = 0;
    video.playbackRate = 1;
    video.loop = false;
    video.load();
    if (profile === "demo-smooth" || profile === "demo-smooth-lite" || profile === "demo-webm" || profile === "demo-webm-lite" || profile === "pure" || profile === "ultra") {
      await waitForReplayVideoBuffer(video, (profile === "demo-webm" || profile === "demo-webm-lite") ? 1600 : 1200);
    }
    const progress = Math.max(0, Math.min(0.98, Number(startProgress || 0)));
    if (progress > 0) {
      const duration = Number(video.duration || 0);
      if (duration > 0) {
        video.currentTime = Math.max(0, Math.min(duration - 0.1, progress * duration));
      }
    }
    setViewerMessage("", true);
    const playPromise = video.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {
        // Browser autoplay policy may require a click; controls stay visible.
        if (profile === "demo-smooth" || profile === "demo-smooth-lite" || profile === "demo-webm" || profile === "demo-webm-lite" || profile === "pure" || profile === "ultra") {
          document.body.classList.remove("videoTakeoverActive");
          video.controls = true;
        }
      });
    }
    return true;
  } catch (err) {
    return false;
  }
}

function startTerminalMjpegReplay(sessionId) {
  if (!SHOWCASE_TERMINAL_MJPEG_REPLAY || !sessionId || delayedFrameReplayStreamStarted) return false;
  delayedFrameReplayStreamStarted = true;
  if (delayedFrameFetchTimer) clearInterval(delayedFrameFetchTimer);
  if (delayedFrameRaf) window.cancelAnimationFrame(delayedFrameRaf);
  delayedFrameFetchTimer = null;
  delayedFrameRaf = null;
  document.body.classList.remove("delayedFramePlayer");
  const stream = $("stream");
  if (!stream) return false;
  stream.src = `/api/agent/sessions/${sessionId}/replay.mjpg?fps=${SHOWCASE_TERMINAL_REPLAY_FPS}&limit=${DELAYED_FRAME_API_LIMIT}&t=${Date.now()}`;
  setViewerMessage("", true);
  stream.onload = () => setViewerMessage("", true);
  return true;
}

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function setText(id, value) {
  const node = $(id);
  if (node) node.textContent = value;
}

function setFlightControlsRunning(isRunning) {
  $("startBtn").disabled = Boolean(isRunning);
  $("stopBtn").disabled = !Boolean(isRunning);
  document.body.classList.toggle("flightRunning", Boolean(isRunning));
  syncStartButtonReadiness();
}

function setStartButtonText(value) {
  const button = $("startBtn");
  if (!button) return;
  if (!button.dataset.baseLabel) {
    button.dataset.baseLabel = button.textContent || "开始";
  }
  button.textContent = value || button.dataset.baseLabel;
}

function clearPolling() {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function schedulePolling(delay = LIVE_POLL_INTERVAL_MS) {
  clearPolling();
  if (!currentSession) return;
  pollTimer = window.setTimeout(pollSession, delay);
}

function resetFlightUi(reason = "") {
  clearPolling();
  stopDelayedFramePlayer();
  setFlightControlsRunning(false);
  setViewerMessage("等待画面", false);
  currentSession = null;
  setText("badge", "idle");
  setText("sessionId", "session: -");
  if (reason) {
    setStatus({ idle: true, reason });
  }
}

function normalizeText(value) {
  return String(value || "").trim().replace(/\s+/g, " ").toLowerCase();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function latestSafetyOverride(data = {}) {
  const value = data.safety_override;
  if (Array.isArray(value)) return value.length ? value[value.length - 1] : {};
  if (value && typeof value === "object") return value;
  return {};
}

function actionMatchesSegment(actionName, segment = {}) {
  const action = String(actionName || "").toUpperCase();
  const actions = new Set(segment.segment_actions || []);
  const text = normalizeText(segment.segment_text || "");
  if (!actions.size) return null;
  const hasAnyText = (phrases) => phrases.some((phrase) => text.includes(phrase));
  const sweepTurn = actions.has("fly_over") && hasAnyText(["cover", "round", "around", "entire", "all over"]);
  if (action === "TURN_LEFT") return actions.has("turn_left") || actions.has("look_left") || actions.has("turn") || sweepTurn;
  if (action === "TURN_RIGHT") return actions.has("turn_right") || actions.has("look_right") || actions.has("turn") || sweepTurn;
  if (action === "MOVE_FORWARD") {
    const landingApproach = actions.has("descend") && hasAnyText(["land", "next to", "on ground", "on top", "onto"]);
    return actions.has("forward") || actions.has("fly_over") || landingApproach;
  }
  if (action === "GO_UP") {
    return actions.has("take_off")
      || actions.has("look_up")
      || actions.has("fly_over")
      || hasAnyText([
        "level",
        "levelled",
        "leveled",
        "height",
        "altitude",
        "raise yourself",
        "raise up",
        "raising up",
        "higher",
        "lower",
        "go over",
        "fly over",
        "fly above",
        "cross top",
      ]);
  }
  if (action === "GO_DOWN") {
    return actions.has("descend")
      || actions.has("look_down")
      || actions.has("stop")
      || actions.has("land")
      || hasAnyText([
        "level",
        "levelled",
        "leveled",
        "height",
        "altitude",
        "raise yourself",
        "higher",
        "lower",
        "get down",
        "drop down",
        "dropping down",
        "fly down",
        "land",
      ]);
  }
  if (action === "STOP") return actions.has("stop") || actions.has("land") || text.includes("stay there");
  return false;
}

function isTerminalLandingControl(actionName, data = {}, segment = {}) {
  const action = String(actionName || "").toUpperCase();
  const phase = data.navigation_phase || data.status || "";
  const index = segment.segment_index;
  const count = segment.segment_count;
  if (!["final_approach", "landing_to_surface"].includes(phase)) return false;
  if (!["GO_UP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "GO_DOWN"].includes(action)) return false;
  return Number.isInteger(Number(index)) && Number.isInteger(Number(count)) && Number(index) >= Number(count) - 1;
}

function compactSegmentToState(segment = {}) {
  return {
    segment_actions: segment.actions || [],
    segment_text: segment.text || "",
    segment_kind: segment.segment_kind,
  };
}

function updateSemanticPanel(data = {}) {
  const panel = $("semanticPanel");
  if (!panel) return;
  const override = latestSafetyOverride(data);
  const segment = override.segment_state || data.segment_state || {};
  const actionName = override.override_action || data.action || "-";
  const match = isTerminalLandingControl(actionName, data, segment) ? true : actionMatchesSegment(actionName, segment);
  const nearbyMatch = match === false && (
    actionMatchesSegment(actionName, compactSegmentToState(segment.previous_segment)) === true
    || actionMatchesSegment(actionName, compactSegmentToState(segment.next_segment)) === true
  );
  const index = segment.segment_index;
  const count = segment.segment_count;
  const segmentPrefix = index === null || index === undefined
    ? "segment -"
    : `segment ${Number(index) + 1}${count ? `/${count}` : ""}`;
  const segmentText = segment.segment_text || "-";
  const expected = (segment.segment_actions || []).join(", ") || "-";
  const target = [
    segment.segment_target_landmark,
    segment.segment_target_zone,
    segment.instruction_profile,
    segment.semantic_quality,
  ].filter(Boolean).join(" / ") || "-";

  setText("semanticSegment", `${segmentPrefix}: ${segmentText}`);
  setText("semanticExpected", expected);
  setText("semanticAction", actionName || "-");
  setText("semanticTarget", target);
  setText("semanticMatch", match === null ? "no label" : match ? "matched" : nearbyMatch ? "nearby matched" : "watching");
  panel.classList.toggle("matched", match === true || nearbyMatch);
  panel.classList.toggle("watching", match === false && !nearbyMatch);
}

function ensureLongInstructionSplitForLearned() {
  if (!$("learnedSegmentHoming")?.checked) return false;
  const split = $("split")?.value || "";
  if (!split.startsWith("train_scene16_segments")) return false;
  $("split").value = "train";
  selectedDatasetInstruction = null;
  datasetMatches = [];
  if ($("datasetEpisode")) $("datasetEpisode").innerHTML = "";
  return true;
}

function updateOracleModeNotice() {
  const notice = $("oracleModeNotice");
  if (!notice) return;
  const usingGoal = $("oracleGoalHoming")?.checked;
  const usingPath = $("oraclePathHoming")?.checked;
  const usingLearned = $("learnedSegmentHoming")?.checked;
  const instructionChanged = selectedDatasetInstruction
    && normalizeText($("instruction").value) !== normalizeText(selectedDatasetInstruction);
  if (!usingGoal && !usingPath && !usingLearned) {
    notice.textContent = "Oracle 诊断已关闭：模型会按当前文本指令推理，不使用数据集目标/参考路径上界。";
    return;
  }
  if (usingLearned && usingPath) {
    notice.textContent = instructionChanged
      ? "语义切段 + Oracle path 已启用：开始飞行时会恢复为当前下拉框选中的长指令；参考路径负责稳定落地，分段状态用于语义对齐观察。"
      : "语义切段 + Oracle path 已启用：参考路径负责稳定落地，Learned segment/segment state 用于分段语义对齐观察。";
    return;
  }
  if (usingLearned) {
    notice.textContent = instructionChanged
      ? "Learned segment 已启用：开始飞行时会自动恢复为当前下拉框选中的长指令；导航不读取 reference_path。"
      : "Learned segment 已启用：使用 local20 分段 grounding 模型预测局部目标，不读取 reference_path。";
    return;
  }
  notice.textContent = instructionChanged
    ? "当前文本只是检索词或被手动修改过；开始飞行时会自动恢复为当前下拉框选中的长指令。"
    : "当前飞行指令来自已选 Dataset Episode 或安全同义变体；Oracle 诊断使用同一个 episode 的 goal/reference_path。";
}

function showcaseFilterValue() {
  return $("showcaseFilter")?.value || "live";
}

function qualityGateValue() {
  const raw = Number($("qualityGate")?.value ?? 80);
  return Number.isFinite(raw) ? raw : 80;
}

function qualityGateLabel() {
  const value = qualityGateValue();
  if (value >= 90) return "Q90+";
  if (value >= 80) return "Q80+";
  if (value >= 70) return "Q70+";
  return "all-Q";
}

function qualityGateFallbackValues(requestedValue) {
  const gates = [90, 80, 70, 0];
  const requested = Number.isFinite(Number(requestedValue)) ? Number(requestedValue) : 80;
  const normalized = gates.includes(requested) ? requested : 80;
  return gates.filter((value) => value <= normalized);
}

function setQualityGateValue(value, { persist = false } = {}) {
  const node = $("qualityGate");
  if (!node) return;
  node.value = String(value);
  if (persist) saveQualityGatePreference();
}

function restoreShowcaseFilterPreference() {
  const node = $("showcaseFilter");
  if (!node) return;
  try {
    const stored = localStorage.getItem(SHOWCASE_FILTER_STORAGE_KEY);
    if (["frontline", "stable", "all", "live"].includes(stored)) {
      node.value = stored;
    }
  } catch (err) {
    // Storage can be unavailable in strict browser privacy modes.
  }
}

function restoreQualityGatePreference() {
  const node = $("qualityGate");
  if (!node) return;
  try {
    const stored = localStorage.getItem(QUALITY_GATE_STORAGE_KEY);
    if (["90", "80", "70", "0"].includes(stored)) {
      node.value = stored;
    }
  } catch (err) {
    // Storage can be unavailable in strict browser privacy modes.
  }
}

function saveShowcaseFilterPreference() {
  try {
    localStorage.setItem(SHOWCASE_FILTER_STORAGE_KEY, showcaseFilterValue());
  } catch (err) {
    // Keep the page usable even when localStorage is blocked.
  }
}

function saveQualityGatePreference() {
  try {
    localStorage.setItem(QUALITY_GATE_STORAGE_KEY, String(qualityGateValue()));
  } catch (err) {
    // Keep the page usable even when localStorage is blocked.
  }
}

function restoreRuntimePreferences() {
  try {
    const storedPort = Number(localStorage.getItem(SIMULATOR_PORT_STORAGE_KEY));
    if (Number.isInteger(storedPort) && storedPort >= 1 && storedPort <= 65535 && $("simulatorPort")) {
      $("simulatorPort").value = String(storedPort);
    }
    const storedConfirm = localStorage.getItem(CONFIRM_CONTROL_STORAGE_KEY);
    if (storedConfirm !== null && $("confirmControl")) {
      $("confirmControl").checked = storedConfirm === "1";
    }
  } catch (err) {
    // Keep defaults when localStorage is blocked.
  }
}

function saveSimulatorPortPreference() {
  const port = Number($("simulatorPort")?.value || 30014);
  if (!Number.isInteger(port) || port < 1 || port > 65535) return;
  try {
    localStorage.setItem(SIMULATOR_PORT_STORAGE_KEY, String(port));
  } catch (err) {
    // Keep the page usable even when localStorage is blocked.
  }
}

function saveConfirmControlPreference() {
  try {
    localStorage.setItem(CONFIRM_CONTROL_STORAGE_KEY, $("confirmControl")?.checked ? "1" : "0");
  } catch (err) {
    // Keep the page usable even when localStorage is blocked.
  }
}

function filterShowcaseItems(items) {
  const mode = showcaseFilterValue();
  if (mode === "all") return items.slice();
  return items.filter((item) => {
    if (mode === "live") return item.verification_tier === "current_smooth";
    const tier = item.display_showcase_tier || "";
    if (mode === "stable") return tier === "frontline" || tier === "stable";
    return tier === "frontline";
  });
}

function renderDatasetOptions(items) {
  $("datasetEpisode").innerHTML = items
    .map((item, index) => {
      const text = (item.instruction || "").replace(/\s+/g, " ");
      const primaryDistanceValue = item.variant_verified_success
        ? Number(item.variant_eval_distance)
        : Number(item.final_distance);
      const distance = Number.isFinite(primaryDistanceValue)
        ? `${primaryDistanceValue.toFixed(1)}m`
        : "verified";
      const evalDistance = Number.isFinite(Number(item.eval_best_distance))
        ? `best ${Number(item.eval_best_distance).toFixed(1)}m`
        : "";
      const genLabel = item.gen_eval_label || "GEN-RAW";
      const variantStability = item.variant_verified_success && Number.isFinite(primaryDistanceValue)
        ? primaryDistanceValue <= 18
          ? "stable"
          : primaryDistanceValue <= 19.5
            ? "ok"
            : "edge"
        : "";
      const variantTag = item.variant_verified_success
        ? ["VAR-OK", variantStability].filter(Boolean).join("/")
        : item.instruction_variant ? "VAR" : "";
      const verifiedTag = verificationTag(item);
      const prefix = item.verified_success ? `OK ${distance}` : [genLabel, evalDistance].filter(Boolean).join(" ");
      const showcaseTier = item.display_showcase_tier ? String(item.display_showcase_tier).toUpperCase() : "";
      const qualityScore = Number.isFinite(Number(item.display_quality_score))
        ? `Q${Number(item.display_quality_score).toFixed(0)}`
        : "";
      const qualityLabel = item.display_quality_label ? String(item.display_quality_label).toUpperCase() : "";
      const displayPrefix = [showcaseTier, qualityScore, qualityLabel, verifiedTag, variantTag, prefix].filter(Boolean).join(" ");
      const label = `${displayPrefix} | env_${item.scene_id} | ${item.episode_id || "-"} | ${text}`;
      return `<option value="${index}">${escapeHtml(label)}</option>`;
    })
    .join("");
}

function demoBuilderSeedLabel(item = {}) {
  const text = (item.instruction || "").replace(/\s+/g, " ");
  const playback = item.demo_playback_mode === "regenerated_preview"
    ? "高清成功回放"
    : item.demo_replay_available
      ? "历史成功回放"
      : "待重生成";
  const distance = Number.isFinite(Number(item.final_distance))
    ? `${Number(item.final_distance).toFixed(1)}m`
    : "success";
  const surface = Number.isFinite(Number(item.surface_clearance))
    ? `surface ${Number(item.surface_clearance).toFixed(1)}m`
    : "surface ok";
  const segments = Number.isFinite(Number(item.segment_count)) ? `${Number(item.segment_count)} seg` : "route";
  return `${playback} | ${distance} | ${surface} | ${segments} | ${item.episode_id || "-"} | ${text}`;
}

function demoBuilderVariantToInstructionItem(seed = {}, variant = {}) {
  const item = { ...seed };
  item.instruction = variant.instruction || seed.instruction || "";
  item.original_instruction = seed.instruction || "";
  item.instruction_variant = variant.id !== "original";
  item.variant_index = variant.id || "original";
  item.variant_source_episode_id = seed.episode_id;
  item.variant_note = variant.risk_note || "";
  item.demo_builder_variant_label = variant.label || "";
  item.demo_builder_variant_kind = variant.variant_kind || "";
  if (item.instruction_variant) {
    item.variant_verified_success = false;
    item.display_quality_score = Math.max(70, Number(item.display_quality_score || 80) - 6);
    item.display_quality_label = item.display_quality_score >= 90 ? "demo-ready" : item.display_quality_score >= 80 ? "strong" : "usable";
  }
  return item;
}

function renderDemoBuilderFamilies() {
  const families = demoBuilderCatalog?.families || [];
  const familySelect = $("demoFamily");
  if (!familySelect) return;
  if (!families.length) {
    familySelect.innerHTML = `<option value="">暂无稳定展示指令族</option>`;
    return;
  }
  familySelect.innerHTML = families
    .map((family, index) => `<option value="${family.id}">${escapeHtml(`${family.label} (${family.count})`)}</option>`)
    .join("");
  if (!familySelect.value) familySelect.value = families[0].id;
}

function renderDemoBuilderSeeds({ preserveEpisode = true } = {}) {
  const families = demoBuilderCatalog?.families || [];
  const familyId = $("demoFamily")?.value || families[0]?.id || "";
  const family = families.find((item) => item.id === familyId) || families[0];
  const seedSelect = $("demoSeed");
  if (!seedSelect) return null;
  const seeds = family?.seeds || [];
  if (!seeds.length) {
    seedSelect.innerHTML = `<option value="">该指令族暂无稳定种子</option>`;
    selectedDemoSeed = null;
    return null;
  }
  const currentEpisodeId = preserveEpisode ? $("episodeId")?.value?.trim() : "";
  seedSelect.innerHTML = seeds
    .map((seed, index) => `<option value="${index}">${escapeHtml(demoBuilderSeedLabel(seed))}</option>`)
    .join("");
  const preservedIndex = currentEpisodeId
    ? seeds.findIndex((seed) => String(seed.episode_id || "") === currentEpisodeId)
    : -1;
  seedSelect.value = String(preservedIndex >= 0 ? preservedIndex : 0);
  selectedDemoSeed = seeds[Number(seedSelect.value)] || seeds[0];
  renderDemoBuilderVariants();
  return selectedDemoSeed;
}

function renderDemoBuilderVariants() {
  const variantSelect = $("demoVariant");
  if (!variantSelect) return;
  const variants = selectedDemoSeed?.demo_variants || [];
  if (!variants.length) {
    variantSelect.innerHTML = `<option value="">暂无可用表达</option>`;
    return;
  }
  variantSelect.innerHTML = variants
    .map((variant, index) => `<option value="${index}">${escapeHtml(`${variant.label}${variant.verified ? " | strict seed" : " | paraphrase"}`)}</option>`)
    .join("");
  variantSelect.value = "0";
  applyDemoBuilderSelection();
}

function updateDemoBuilderSummary() {
  const node = $("demoBuilderSummary");
  if (!node) return;
  if (!demoBuilderCatalog) {
    node.textContent = "Demo Builder 未加载";
    return;
  }
  const replayReady = Number(demoBuilderCatalog.replay_ready_seed_count || 0);
  const stableCount = Number(demoBuilderCatalog.stable_count || 0);
  const seedCount = Number(demoBuilderCatalog.seed_count || 0);
  node.textContent = `stable records ${stableCount} | shown ${seedCount} | cached replay ${replayReady} | families ${demoBuilderCatalog.family_count || 0}`;
}

async function loadDemoBuilder({ preserveEpisode = true } = {}) {
  const summary = $("demoBuilderSummary");
  if (summary) summary.textContent = "正在读取稳定展示指令族...";
  try {
    demoBuilderCatalog = await api("/api/scene16/demo-builder?max_seeds_per_family=120");
    renderDemoBuilderFamilies();
    renderDemoBuilderSeeds({ preserveEpisode });
    updateDemoBuilderSummary();
  } catch (err) {
    demoBuilderCatalog = null;
    if ($("demoFamily")) $("demoFamily").innerHTML = `<option value="">Demo Builder 加载失败</option>`;
    if ($("demoSeed")) $("demoSeed").innerHTML = `<option value="">请检查 stable pool</option>`;
    if ($("demoVariant")) $("demoVariant").innerHTML = `<option value="">不可用</option>`;
    if (summary) summary.textContent = "稳定展示指令族加载失败";
    setStatus(err);
  }
}

function applyDemoBuilderSelection() {
  const variants = selectedDemoSeed?.demo_variants || [];
  const variant = variants[Number($("demoVariant")?.value || 0)] || variants[0];
  if (!selectedDemoSeed || !variant) return;
  const item = demoBuilderVariantToInstructionItem(selectedDemoSeed, variant);
  applyInstructionItem(item, {
    sourceLabel: "Scene16 Demo Builder",
    variantLabel: variant.label,
  });
  const note = $("demoVariantNote");
  if (note) {
    note.textContent = variant.verified
      ? "当前为严格成功原始长指令，适合作为展示首选。"
      : "当前为保守同路线改写：保持同一 episode、同一路线地标顺序，不代表任意新指令泛化。";
  }
}

function verificationTag(item = {}) {
  if (item.verification_tier === "current_smooth") return "LIVE-OK";
  if (item.variant_verified_success) return "VAR-OK";
  if (item.verified_success) return "VERIFIED";
  if (item.gen_eval_label === "GEN-SAFE") return "GEN-SAFE";
  return "";
}

function isDemoReadyInstruction(item = {}) {
  const qualityScore = Number(item.display_quality_score);
  const qualityLabel = String(item.display_quality_label || "").toLowerCase();
  return qualityLabel === "demo-ready" || (Number.isFinite(qualityScore) && qualityScore >= 90);
}

function demoReadyIndices() {
  const seenEpisodes = new Set();
  return datasetMatches
    .map((item, index) => {
      if (!isDemoReadyInstruction(item)) return -1;
      const episodeId = String(item.episode_id || `index-${index}`);
      if (seenEpisodes.has(episodeId)) return -1;
      seenEpisodes.add(episodeId);
      return index;
    })
    .filter((index) => index >= 0);
}

function qualityGateReadyIndices() {
  const threshold = qualityGateValue();
  const seenEpisodes = new Set();
  return datasetMatches
    .map((item, index) => {
      const qualityScore = Number(item.display_quality_score);
      if (!Number.isFinite(qualityScore) || qualityScore < threshold) return -1;
      const episodeId = String(item.episode_id || `index-${index}`);
      if (seenEpisodes.has(episodeId)) return -1;
      seenEpisodes.add(episodeId);
      return index;
    })
    .filter((index) => index >= 0);
}

function updateDatasetInstructionSummary(data = {}) {
  const verifiedCount = datasetAllMatches.filter((item) => item.verified_success && !item.instruction_variant).length;
  const genSafeCount = datasetAllMatches.filter((item) => item.gen_eval_label === "GEN-SAFE" && !item.instruction_variant).length;
  const allCount = Number.isFinite(Number(data.all_count)) ? Number(data.all_count) : datasetAllMatches.length;
  const safeDisplayCount = Number.isFinite(Number(data.safe_display_count)) ? Number(data.safe_display_count) : datasetAllMatches.length;
  const originalCount = Number.isFinite(Number(data.original_count)) ? Number(data.original_count) : datasetAllMatches.length;
  const variantCount = Number.isFinite(Number(data.variant_count)) ? Number(data.variant_count) : datasetAllMatches.filter((item) => item.instruction_variant).length;
  const variantOkCount = Number.isFinite(Number(data.variant_ok_count)) ? Number(data.variant_ok_count) : datasetAllMatches.filter((item) => item.variant_verified_success).length;
  const riskCount = Number.isFinite(Number(data.gen_risk_count)) ? Number(data.gen_risk_count) : 0;
  const rawCount = Number.isFinite(Number(data.gen_raw_count)) ? Number(data.gen_raw_count) : 0;
  const displayPoolCount = Number.isFinite(Number(data.display_pool_count)) ? Number(data.display_pool_count) : datasetAllMatches.length;
  const displayPoolUnfilteredCount = Number.isFinite(Number(data.display_pool_unfiltered_count))
    ? Number(data.display_pool_unfiltered_count)
    : displayPoolCount;
  const displayPoolOriginalCount = Number.isFinite(Number(data.display_pool_original_count)) ? Number(data.display_pool_original_count) : safeDisplayCount;
  const displayPoolVariantCount = Number.isFinite(Number(data.display_pool_variant_count)) ? Number(data.display_pool_variant_count) : variantCount;
  const displayPoolVariantOkCount = Number.isFinite(Number(data.display_pool_variant_ok_count)) ? Number(data.display_pool_variant_ok_count) : variantOkCount;
  const tierCounts = data.display_pool_tier_counts || data.showcase_tier_counts || {};
  const activeTierCounts = data.showcase_tier_counts || {};
  const qualityCounts = data.display_pool_quality_counts || {};
  const activeQualityCounts = data.active_quality_counts || {};
  const activeUnique = Number.isFinite(Number(data.unique_episode_count)) ? Number(data.unique_episode_count) : datasetMatches.length;
  const demoUnique = Number.isFinite(Number(data.demo_ready_unique_episode_count)) ? Number(data.demo_ready_unique_episode_count) : 0;
  const demoCurrentVerified = Number.isFinite(Number(data.demo_ready_current_smooth_unique_episode_count))
    ? Number(data.demo_ready_current_smooth_unique_episode_count)
    : 0;
  updateDemoCoverageSummary({
    activeUnique,
    demoUnique,
    demoCurrentVerified,
    displayPoolCount,
  });
  setText(
    "datasetInstructionCount",
    `shown ${datasetMatches.length}/${displayPoolCount} (${activeUnique} unique, ${showcaseFilterValue()}, ${qualityGateLabel()}) | demo coverage ${demoCurrentVerified}/${demoUnique} current-version LIVE-OK | gated pool ${displayPoolCount}/${displayPoolUnfilteredCount} | quality DEMO ${qualityCounts.demo_ready || 0} / STRONG ${qualityCounts.strong || 0} / USABLE ${qualityCounts.usable || 0} / CAUTION ${qualityCounts.caution || 0} / RISKY ${qualityCounts.risky || 0} | current DEMO ${activeQualityCounts.demo_ready || 0} / STRONG ${activeQualityCounts.strong || 0} | pool FRONT ${tierCounts.frontline || 0} / STABLE ${tierCounts.stable || 0} / CAUTION ${tierCounts.caution || 0} | current FRONT ${activeTierCounts.frontline || 0} / STABLE ${activeTierCounts.stable || 0} / CAUTION ${activeTierCounts.caution || 0} | original ${originalCount}/${displayPoolOriginalCount}, variants ${variantCount}/${displayPoolVariantCount} (VAR-OK ${variantOkCount}/${displayPoolVariantOkCount}), OK ${verifiedCount}, GEN-SAFE ${genSafeCount}, hidden risk/raw ${riskCount + rawCount}, all ${allCount}`
  );
}

function updateDemoCoverageSummary({ activeUnique = 0, demoUnique = 0, demoCurrentVerified = 0, displayPoolCount = 0 } = {}) {
  const node = $("demoCoverageSummary");
  if (!node) return;
  const complete = demoUnique > 0 && demoCurrentVerified >= demoUnique;
  const mode = showcaseFilterValue() === "live" ? "LIVE-OK demo pool" : "candidate pool";
  node.textContent = complete
    ? `Demo coverage: ${demoCurrentVerified}/${demoUnique} LIVE-OK unique episodes | ${mode} | ${datasetMatches.length}/${displayPoolCount} shown, ${activeUnique} unique`
    : `Demo coverage: ${demoCurrentVerified}/${demoUnique || "-"} LIVE-OK unique episodes | ${mode} | ${datasetMatches.length}/${displayPoolCount} shown, ${activeUnique} unique`;
  node.classList.toggle("complete", complete);
  node.classList.toggle("partial", !complete);
}

function selectedQualityText(item = {}) {
  const tier = String(item.display_showcase_tier || "unknown").toUpperCase();
  const quality = Number.isFinite(Number(item.display_quality_score))
    ? `Q${Number(item.display_quality_score).toFixed(1)} ${item.display_quality_label || ""}`.trim()
    : "Q-";
  const backendEvidence = Array.isArray(item.display_evidence) && item.display_evidence.length
    ? item.display_evidence.join(", ")
    : "";
  const best = Number.isFinite(Number(item.eval_best_distance))
    ? `${Number(item.eval_best_distance).toFixed(1)}m`
    : Number.isFinite(Number(item.final_distance))
      ? `${Number(item.final_distance).toFixed(1)}m`
      : "-";
  const surface = Number.isFinite(Number(item.eval_surface_clearance))
    ? `${Number(item.eval_surface_clearance).toFixed(1)}m`
    : Number.isFinite(Number(item.surface_clearance))
      ? `${Number(item.surface_clearance).toFixed(1)}m`
      : "-";
  const collision = Number.isFinite(Number(item.eval_collision_rollback_count))
    ? Number(item.eval_collision_rollback_count)
    : Number.isFinite(Number(item.collision_rollback_count))
      ? Number(item.collision_rollback_count)
      : 0;
  const streamGap = Number.isFinite(Number(item.eval_stream_sample_max_gap))
    ? `${Number(item.eval_stream_sample_max_gap).toFixed(3)}s`
    : Number.isFinite(Number(item.stream_sample_max_gap))
      ? `${Number(item.stream_sample_max_gap).toFixed(3)}s`
    : "-";
  const route = Number.isFinite(Number(item.candidate_reference_path_length))
    ? `${Number(item.candidate_reference_path_length).toFixed(0)}m`
    : Number.isFinite(Number(item.reference_path_length))
      ? `${Number(item.reference_path_length).toFixed(0)}m`
      : "-";
  const water = item.display_water_landing_risk ? "water-risk" : "no-water-stop";
  const variant = item.instruction_variant ? "variant" : "original";
  const verified = verificationTag(item) || "unverified";
  const fallbackEvidence = `best ${best}, surface ${surface}, collision ${collision}, stream gap ${streamGap}, route ${route}, ${water}`;
  return `${tier} | ${quality} | ${verified} | ${variant} | ${backendEvidence || fallbackEvidence}`;
}

function updateSelectedQualitySummary(item = {}) {
  const node = $("selectedQualitySummary");
  if (!node) return;
  const tier = String(item.display_showcase_tier || "").toLowerCase();
  const label = String(item.display_quality_label || "").toLowerCase();
  node.textContent = `Selected quality: ${selectedQualityText(item)}`;
  node.classList.toggle("demo", label === "demo-ready");
  node.classList.toggle("stable", tier === "stable");
  node.classList.toggle("caution", tier === "caution" || label === "caution" || label === "risky" || item.display_water_landing_risk);
}

function selectedGpuInfo() {
  const selectedId = Number($("gpuId")?.value);
  const gpus = lastGpuInventory?.gpus || [];
  return gpus.find((gpu) => Number(gpu.id) === selectedId) || null;
}

function updateReadinessSummary() {
  const snapshot = readinessSnapshot();
  syncStartButtonReadiness(snapshot);
  const node = $("readinessSummary");
  if (!node) return;
  const issueTail = snapshot.issues.length
    ? ` | issue ${snapshot.issues.slice(0, 2).join("; ")}${snapshot.issues.length > 2 ? "..." : ""}`
    : "";
  const displayLabel = snapshot.state === "ready"
    ? (snapshot.demoReady ? "DEMO READY" : "CANDIDATE READY")
    : snapshot.label;
  node.textContent = `${displayLabel} | ${snapshot.pieces.join(" | ")}${issueTail}`;
  node.classList.toggle("ready", snapshot.state === "ready" && snapshot.demoReady);
  node.classList.toggle("check", snapshot.state === "check" || (snapshot.state === "ready" && !snapshot.demoReady));
  node.classList.toggle("blocked", snapshot.state === "blocked");
}

function syncStartButtonReadiness(snapshot = null) {
  const button = $("startBtn");
  if (!button) return;
  if (!button.dataset.baseLabel) {
    button.dataset.baseLabel = button.textContent || "开始";
  }
  button.classList.remove("readinessReady", "readinessCheck", "readinessBlocked");
  if (button.disabled) {
    setStartButtonText("飞行中...");
    button.title = "Flight is running or controls are temporarily locked.";
    return;
  }
  if ($("mode")?.value !== "agent") {
    setStartButtonText(button.dataset.baseLabel);
    button.title = "Start flight.";
    return;
  }
  const state = snapshot || readinessSnapshot();
  if (state.ready) {
    button.classList.add(state.demoReady ? "readinessReady" : "readinessCheck");
    setStartButtonText(`${button.dataset.baseLabel} ${state.demoReady ? "DEMO READY" : "CANDIDATE"}`);
    button.title = state.demoReady
      ? "DEMO READY: preflight checks are satisfied for a demo-ready instruction."
      : `CANDIDATE: ${state.issues.join("; ") || "launch is allowed, but this is not a demo-ready instruction"}`;
  } else if (state.state === "blocked") {
    button.classList.add("readinessBlocked");
    setStartButtonText(`${button.dataset.baseLabel} BLOCKED`);
    button.title = `BLOCKED: ${state.issues.join("; ") || "readiness checks failed"}`;
  } else {
    button.classList.add("readinessCheck");
    setStartButtonText(`${button.dataset.baseLabel} CHECK`);
    button.title = `CHECK: ${state.issues.join("; ") || "launch will re-check readiness"}`;
  }
}

function readinessSnapshot() {
  const sim = lastHealthData?.airsim || null;
  const healthAgeMs = lastHealthCheckedAt ? Date.now() - lastHealthCheckedAt : Number.POSITIVE_INFINITY;
  const healthFresh = healthAgeMs <= HEALTH_STALE_MS;
  const healthAgeText = Number.isFinite(healthAgeMs) ? `${Math.max(0, Math.round(healthAgeMs / 1000))}s` : "never";
  const simReachable = Boolean(sim?.ok);
  const simReady = simReachable && healthFresh;
  const gpu = selectedGpuInfo();
  const gpuReady = gpu ? Boolean(gpu.compatible) : Boolean($("gpuId")?.value);
  const qualityScore = Number(selectedDatasetItem?.display_quality_score);
  const qualityReady = Number.isFinite(qualityScore) ? qualityScore >= qualityGateValue() : false;
  const demoReady = Number.isFinite(qualityScore) && qualityScore >= 90;
  const candidateReady = qualityReady && !demoReady;
  const qualityText = Number.isFinite(qualityScore)
    ? `Q${qualityScore.toFixed(0)} ${selectedDatasetItem?.display_quality_label || ""}`.trim()
    : "no selected instruction";
  const controlReady = Boolean($("confirmControl")?.checked);
  const pieces = [
    `sim ${simReady ? "ok" : "check"}:${sim?.port || $("simulatorPort")?.value || "-"} health ${healthAgeText}`,
    `gpu ${gpuReady ? "ok" : "check"}:${gpu ? `GPU${gpu.id}` : $("gpuId")?.value || "-"}`,
    `instruction ${qualityReady ? "ok" : "check"}:${qualityText}`,
    `control ${controlReady ? "confirmed" : "not confirmed"}`,
  ];
  const issues = [];
  if (!simReachable) issues.push(`simulator ${sim?.port || $("simulatorPort")?.value || "-"} is not reachable`);
  else if (!healthFresh) issues.push(`simulator health is stale (${healthAgeText})`);
  if (!gpuReady) issues.push("selected GPU is not compatible or unavailable");
  if (!selectedDatasetItem) issues.push("no dataset instruction is selected");
  else if (!qualityReady) issues.push(`selected instruction ${qualityText} is below ${qualityGateLabel()}`);
  if (!controlReady) issues.push("control confirmation is not checked");
  if (candidateReady) issues.push(`candidate-level instruction ${qualityText}; use 下一条演示 for demo-ready flight`);
  let state = "check";
  if (simReady && gpuReady && qualityReady && controlReady) {
    state = "ready";
  } else if (!simReady || !gpuReady || !selectedDatasetItem) {
    state = "blocked";
  }
  return {
    state,
    ready: state === "ready",
    label: state === "ready" ? "READY" : state === "blocked" ? "BLOCKED" : "CHECK",
    demoReady,
    candidateReady,
    pieces,
    issues,
  };
}

function refreshShowcaseFilter({ preserveEpisode = true } = {}) {
  const currentEpisodeId = preserveEpisode ? $("episodeId")?.value?.trim() : "";
  datasetMatches = filterShowcaseItems(datasetAllMatches);
  if (!datasetMatches.length) {
    $("datasetEpisode").innerHTML = `<option value="">No instructions in this showcase filter</option>`;
    updateDatasetInstructionSummary(datasetCatalogStats);
    return null;
  }
  renderDatasetOptions(datasetMatches);
  updateDatasetInstructionSummary(datasetCatalogStats);
  const preservedIndex = currentEpisodeId
    ? datasetMatches.findIndex((item) => String(item.episode_id || "") === currentEpisodeId)
    : -1;
  const selectedIndex = preservedIndex >= 0 ? preservedIndex : 0;
  $("datasetEpisode").value = String(selectedIndex);
  applyDatasetEpisode(selectedIndex);
  return datasetMatches[selectedIndex];
}

function ensureLongDatasetStepBudget() {
  const maxSteps = $("maxSteps");
  if (!maxSteps) return false;
  const instruction = $("instruction")?.value || "";
  const wordCount = instruction.split(/\s+/).filter(Boolean).length;
  const sentenceCount = instruction.split(".").filter((part) => part.trim()).length;
  const selectedDataset = Boolean(selectedDatasetInstruction);
  const longInstruction = wordCount >= 25 || sentenceCount >= 4;
  if ($("mode")?.value === "agent" && (selectedDataset || longInstruction) && Number(maxSteps.value || 0) < 900) {
    maxSteps.value = "900";
    return true;
  }
  return false;
}

function updateMetricPanel(data = {}) {
  const distance = data.distance_to_goal;
  const best = data.best_distance;
  const successDistance = Number(data.success_distance || 20);
  const verticalError = data.vertical_error_to_goal;
  const altitudeAligned = Boolean(data.altitude_aligned);
  const surfaceClearance = data.surface_clearance;
  const surfaceAligned = Boolean(data.surface_aligned);
  const progress = data.progress_to_goal === null || data.progress_to_goal === undefined
    ? null
    : Math.max(0, Math.min(1, Number(data.progress_to_goal)));
  const horizontalSuccess = Boolean(data.success_20m);
  const success = data.success_20m_with_surface === undefined
    ? horizontalSuccess
    : Boolean(data.success_20m_with_surface);
  const oracleSuccess = Boolean(data.oracle_success_20m);
  const displayedProgress = success ? 1 : progress;

  setText(
    "liveFps",
    SHOWCASE_DELAYED_FRAME_PLAYER ? "LIVE" : SHOWCASE_NATIVE_STREAM ? "LIVE" : "LIVE"
  );

  setText("metricDistance", `${formatNumber(distance)} m`);
  setText("metricBest", `${formatNumber(best)} m`);
  setText(
    "metricSuccess",
    success
      ? `SUCCESS <= ${formatNumber(successDistance, 0)}m + surface`
      : horizontalSuccess
        ? "XY hit, landing"
        : `not within ${formatNumber(successDistance, 0)}m`
  );
  setText(
    "metricAltitude",
    surfaceAligned
      ? `surface aligned${verticalError === null || verticalError === undefined ? "" : ` / dataset z Δ ${formatNumber(Math.abs(Number(verticalError)))} m`}`
      : verticalError === null || verticalError === undefined
      ? "- m"
      : `${formatNumber(Math.abs(Number(verticalError)))} m ${altitudeAligned ? "aligned" : "to go"}`
  );
  setText(
    "metricSurface",
    surfaceClearance === null || surfaceClearance === undefined
      ? (data.surface_probe_error ? "surface unknown" : "- m")
      : `${formatNumber(Number(surfaceClearance))} m ${surfaceAligned ? "on surface" : "to land"}`
  );
  setText("metricOracle", oracleSuccess ? "oracle hit" : "oracle miss");
  setText("metricPhase", data.navigation_phase || data.status || "-");
  const collisionRollbacks = Number(data.collision_rollback_count || 0);
  setText(
    "metricCollision",
    collisionRollbacks > 0 ? `${collisionRollbacks} blocked + rolled back` : "collision-free"
  );

  const progressBar = $("goalProgressBar");
  if (progressBar) {
    progressBar.style.width = `${Math.round((displayedProgress || 0) * 100)}%`;
  }
  setText("goalProgressText", displayedProgress === null ? "progress -" : `progress ${Math.round(displayedProgress * 100)}%`);

  const panel = $("metricsPanel");
  if (panel) {
    panel.classList.toggle("success", success);
    panel.classList.toggle("oracle", !success && oracleSuccess);
  }
  updateSemanticPanel(data);
}

function extractPathPoints(data) {
  const path = Array.isArray(data?.path) ? data.path : [];
  const sourcePath = path.length > MAX_BEV_POINTS
    ? path.filter((_, index) => index === 0 || index === path.length - 1 || index % Math.ceil(path.length / MAX_BEV_POINTS) === 0)
    : path;
  const points = sourcePath
    .map((item) => {
      const pos = item.position || item;
      if (!Array.isArray(pos) || pos.length < 2) return null;
      return {
        x: Number(pos[0]),
        y: Number(pos[1]),
        z: Number(pos[2] || 0),
        step: item.step,
        action: item.action,
        distance: item.distance_to_goal,
        distance3d: item.distance_3d_to_goal,
        verticalError: item.vertical_error_to_goal,
        altitudeAligned: item.altitude_aligned,
        surfaceClearance: item.surface_clearance,
        surfaceAligned: item.surface_aligned,
        bestDistance: item.best_distance,
        success: item.success_20m_with_surface ?? item.success_20m_with_altitude ?? item.success_20m,
        phase: item.navigation_phase,
        goal: item.goal_position,
      };
    })
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  if (!points.length && Array.isArray(data?.position) && data.position.length >= 2) {
    points.push({
      x: Number(data.position[0]),
      y: Number(data.position[1]),
      z: Number(data.position[2] || 0),
      step: data.step,
      action: data.action,
      distance: data.distance_to_goal,
      distance3d: data.distance_3d_to_goal,
      verticalError: data.vertical_error_to_goal,
      altitudeAligned: data.altitude_aligned,
      surfaceClearance: data.surface_clearance,
      surfaceAligned: data.surface_aligned,
      bestDistance: data.best_distance,
      success: data.success_20m_with_surface ?? data.success_20m_with_altitude ?? data.success_20m,
      phase: data.navigation_phase,
      goal: data.goal_position,
    });
  }
  return points;
}

function extractGoalPoint(data, points) {
  const raw = data?.goal_position || [...points].reverse().find((point) => Array.isArray(point.goal))?.goal;
  if (!Array.isArray(raw) || raw.length < 2) return null;
  const goal = {
    x: Number(raw[0]),
    y: Number(raw[1]),
    z: Number(raw[2] || 0),
  };
  return Number.isFinite(goal.x) && Number.isFinite(goal.y) ? goal : null;
}

function drawBev(data) {
  latestBevData = data || latestBevData;
  const canvas = $("bevCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width;
  const h = rect.height;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#090c10";
  ctx.fillRect(0, 0, w, h);

  const points = extractPathPoints(latestBevData);
  if (!points.length) {
    $("bevStats").textContent = "no trajectory";
    ctx.fillStyle = "#687282";
    ctx.font = "13px Arial";
    ctx.fillText("Waiting for trajectory...", 16, 28);
    return;
  }
  const goal = extractGoalPoint(latestBevData, points);
  const boundsPoints = goal ? points.concat([goal]) : points;

  let minX = Math.min(...boundsPoints.map((p) => p.x));
  let maxX = Math.max(...boundsPoints.map((p) => p.x));
  let minY = Math.min(...boundsPoints.map((p) => p.y));
  let maxY = Math.max(...boundsPoints.map((p) => p.y));
  const spanX = Math.max(20, maxX - minX);
  const spanY = Math.max(20, maxY - minY);
  const pad = 26;
  const scale = Math.min((w - pad * 2) / spanX, (h - pad * 2) / spanY);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  const map = (p) => ({
    x: w / 2 + (p.x - centerX) * scale,
    y: h / 2 - (p.y - centerY) * scale,
  });

  ctx.strokeStyle = "#1a222e";
  ctx.lineWidth = 1;
  ctx.beginPath();
  const grid = 40;
  for (let x = pad; x <= w - pad; x += grid) {
    ctx.moveTo(x, pad);
    ctx.lineTo(x, h - pad);
  }
  for (let y = pad; y <= h - pad; y += grid) {
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
  }
  ctx.stroke();

  const start = map(points[0]);
  const current = map(points[points.length - 1]);
  const goalCanvas = goal ? map(goal) : null;
  const best = points.reduce((acc, point) => {
    if (point.distance === null || point.distance === undefined || Number.isNaN(Number(point.distance))) return acc;
    if (!acc || Number(point.distance) < Number(acc.distance)) return point;
    return acc;
  }, null);
  const bestCanvas = best ? map(best) : null;

  if (goalCanvas) {
    const successRadius = Math.max(5, 20 * scale);
    ctx.fillStyle = "rgba(87, 201, 154, 0.12)";
    ctx.strokeStyle = "rgba(87, 201, 154, 0.72)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(goalCanvas.x, goalCanvas.y, successRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#57c99a";
    ctx.beginPath();
    ctx.arc(goalCanvas.x, goalCanvas.y, 7, 0, Math.PI * 2);
    ctx.fill();
  }

  if (points.length > 1) {
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#4fa3ff";
    ctx.beginPath();
    points.forEach((point, index) => {
      const p = map(point);
      if (index === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
  }

  ctx.fillStyle = "#55d17a";
  ctx.beginPath();
  ctx.arc(start.x, start.y, 5, 0, Math.PI * 2);
  ctx.fill();

  if (bestCanvas) {
    ctx.fillStyle = "#f3c969";
    ctx.beginPath();
    ctx.arc(bestCanvas.x, bestCanvas.y, 6, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#ff6b6b";
  ctx.beginPath();
  ctx.arc(current.x, current.y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = "#c9d1dc";
  ctx.font = "12px Arial";
  ctx.fillText("start", start.x + 8, start.y - 8);
  if (goalCanvas) ctx.fillText("goal / 20m", goalCanvas.x + 8, goalCanvas.y - 8);
  if (bestCanvas) ctx.fillText("best", bestCanvas.x + 8, bestCanvas.y - 8);
  ctx.fillText("current", current.x + 8, current.y - 8);

  const last = points[points.length - 1];
  const bestText = best ? ` best ${formatNumber(best.distance)}m@${best.step ?? "-"}` : "";
  const successText = latestBevData?.success_20m_with_surface ? " | SURFACE SUCCESS" : latestBevData?.success_20m ? " | XY HIT" : "";
  const zError = latestBevData?.vertical_error_to_goal;
  const zText = zError === null || zError === undefined ? "" : ` | zerr ${formatNumber(Math.abs(Number(zError)))}m`;
  const surface = latestBevData?.surface_clearance;
  const surfaceText = surface === null || surface === undefined ? "" : ` | surface ${formatNumber(Number(surface))}m`;
  $("bevStats").textContent =
    `step ${last.step ?? latestBevData?.step ?? "-"} | x ${formatNumber(last.x)} y ${formatNumber(last.y)} z ${formatNumber(last.z)} | dist ${formatNumber(latestBevData?.distance_to_goal)}m${zText}${surfaceText}${bestText}${successText}`;
}

function drawBevThrottled(data, force = false) {
  const now = performance.now();
  if (!force && now - lastBevDrawAt < BEV_DRAW_INTERVAL_MS) return;
  lastBevDrawAt = now;
  drawBev(data);
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json();
  if (!res.ok) {
    data.status = res.status;
    throw data;
  }
  return data;
}

async function loadScenes() {
  try {
    const data = await api("/api/scenes");
    $("scene").innerHTML = data.scenes.map((id) => `<option value="${id}">env_${id}</option>`).join("");
    $("scene").value = $("mode").value === "agent" ? "16" : "4";
  } catch (err) {
    setStatus(err);
  }
}

async function loadGpus() {
  try {
    const data = await api("/api/gpus");
    lastGpuInventory = data;
    if (!data.gpus || !data.gpus.length) {
      updateReadinessSummary();
      return;
    }
    $("gpuId").innerHTML = data.gpus
      .map((gpu) => {
        const freeGb = Math.max(0, gpu.memory_free_mb || 0) / 1024;
        const label = `GPU ${gpu.id} - ${gpu.name} (${freeGb.toFixed(1)} GB free)${gpu.compatible ? "" : " - unsupported"}`;
        return `<option value="${gpu.id}" ${gpu.compatible ? "" : "disabled"}>${label}</option>`;
      })
      .join("");
    if (data.recommended_gpu_id !== null && data.recommended_gpu_id !== undefined) {
      $("gpuId").value = String(data.recommended_gpu_id);
    }
    updateReadinessSummary();
  } catch (err) {
    setStatus(err);
    updateReadinessSummary();
  }
}

async function checkHealth({ silent = false } = {}) {
  if (healthCheckInFlight) return lastHealthData;
  healthCheckInFlight = true;
  try {
    const params = new URLSearchParams({
      airsim_port: String(Number($("simulatorPort")?.value || 30014)),
    });
    const data = await api(`/api/health?${params.toString()}`);
    lastHealthData = data;
    lastHealthCheckedAt = Date.now();
    if (!silent) setStatus(data);
    updateReadinessSummary();
    return data;
  } catch (err) {
    lastHealthData = { ok: false, airsim: { ok: false, message: err?.message || "health check failed" } };
    lastHealthCheckedAt = Date.now();
    if (!silent) setStatus(err);
    updateReadinessSummary();
    return lastHealthData;
  } finally {
    healthCheckInFlight = false;
  }
}

function startHealthRefresh() {
  if (healthRefreshTimer) return;
  healthRefreshTimer = window.setInterval(() => {
    updateReadinessSummary();
    if (document.hidden || currentSession || demoReplayActive) return;
    checkHealth({ silent: true });
  }, HEALTH_REFRESH_INTERVAL_MS);
}

function applyInstructionItem(item, { sourceLabel = "数据集 episode", variantLabel = "" } = {}) {
  if (!item) return;
  selectedDatasetItem = item;
  selectedDatasetInstruction = item.instruction || "";
  const instructionBox = $("instruction");
  instructionBox.value = item.instruction || instructionBox.value;
  instructionBox.scrollTop = 0;
  instructionBox.classList.remove("replaced");
  void instructionBox.offsetWidth;
  instructionBox.classList.add("replaced");
  window.setTimeout(() => instructionBox.classList.remove("replaced"), 2400);
  $("episodeId").value = item.episode_id || "";
  $("scene").value = String(item.scene_id || $("scene").value);
  $("split").value = item.split || $("split").value;
  updateSelectedQualitySummary(item);
  updateReadinessSummary();
  ensureLongDatasetStepBudget();
  updateOracleModeNotice();
  startDemoVideoPrewarm(item);
  const wordCount = (item.instruction || "").trim().split(/\s+/).filter(Boolean).length;
  const proofParts = [];
  if (item.verified_success) proofParts.push("已验证成功");
  if (item.variant_verified_success) proofParts.push("变体已实飞成功");
  else if (item.instruction_variant) proofParts.push("同 episode 保守改写");
  if (!item.verified_success && item.gen_eval_label) proofParts.push(item.gen_eval_label);
  if (Number.isFinite(Number(item.final_distance))) proofParts.push(`最终距离 ${Number(item.final_distance).toFixed(1)}m`);
  if (Number.isFinite(Number(item.eval_best_distance))) proofParts.push(`评估最近 ${Number(item.eval_best_distance).toFixed(1)}m`);
  if (Number.isFinite(Number(item.surface_clearance))) proofParts.push(`实体距离 ${Number(item.surface_clearance).toFixed(1)}m`);
  if (item.demo_family_label) proofParts.push(`指令族 ${item.demo_family_label}`);
  if (variantLabel) proofParts.push(variantLabel);
  if (!proofParts.length && item.score !== undefined) proofParts.push(`score ${item.score}`);
  const proofText = proofParts.length ? `，${proofParts.join("，")}` : "";
  setStatus(
    `已选中${sourceLabel} ${item.episode_id || "-"}：飞行指令已替换为${item.instruction_variant ? "安全同义变体" : "原始长 instruction"}（${wordCount} 词${proofText}）。`
  );
}

function applyDatasetEpisode(index) {
  const item = datasetMatches[index];
  applyInstructionItem(item);
  scheduleDemoNeighborPrewarm(index);
}

async function loadDatasetInstructions({ preserveEpisode = true, allowCatalogFallback = true } = {}) {
  const loadSequence = ++datasetLoadSequence;
  const switchedToTrain = ensureLongInstructionSplitForLearned();
  const requestedQualityGate = qualityGateValue();
  const liveOkOnly = showcaseFilterValue() === "live";
  const buildParams = (minQualityScore) => new URLSearchParams({
    split: $("split").value,
    scene_id: $("scene").value || "all",
    dataset: "aerialvln",
    verified_only: "0",
    safe_only: "1",
    include_variants: liveOkOnly ? "0" : "1",
    variants_per_safe: liveOkOnly ? "0" : "5",
    showcase_filter: showcaseFilterValue(),
    min_quality_score: String(minQualityScore),
    dedupe_episode: "1",
  });
  const currentEpisodeId = preserveEpisode ? $("episodeId").value.trim() : "";
  $("datasetEpisode").innerHTML = `<option value="">正在加载单场景稳定展示指令...</option>`;
  setText("datasetInstructionCount", "正在读取单场景稳定展示指令目录");
  try {
    if (switchedToTrain) {
      setStatus({ info: "Learned segment 模式已自动切换到原始 train 数据集，用于加载已验证成功的长指令目录。" });
    }
    let data = null;
    let appliedQualityGate = requestedQualityGate;
    for (const candidateQualityGate of qualityGateFallbackValues(requestedQualityGate)) {
      const params = buildParams(candidateQualityGate);
      data = await api(`/api/dataset/instructions?${params.toString()}`);
      if (loadSequence !== datasetLoadSequence) return null;
      const candidateMatches = filterShowcaseItems(data.instructions || []);
      if (candidateMatches.length || candidateQualityGate === 0) {
        appliedQualityGate = candidateQualityGate;
        break;
      }
    }
    if (loadSequence !== datasetLoadSequence) return null;
    if (appliedQualityGate !== requestedQualityGate) {
      setQualityGateValue(appliedQualityGate, { persist: true });
      setStatus({
        info: "quality gate fallback applied",
        from: `Q${requestedQualityGate}+`,
        to: qualityGateLabel(),
        reason: "current gate returned no dataset instructions",
      });
    }
    datasetCatalogStats = data;
    datasetAllMatches = data.instructions || [];
    datasetMatches = filterShowcaseItems(datasetAllMatches);
    if (!datasetMatches.length) {
      const shouldUseDemoFallback = allowCatalogFallback
        && $("mode")?.value === "agent"
        && ($("split")?.value !== DEMO_FALLBACK_SPLIT || $("scene")?.value !== DEMO_FALLBACK_SCENE);
      if (shouldUseDemoFallback) {
        const fromSplit = $("split")?.value || "-";
        const fromScene = $("scene")?.value || "-";
        $("split").value = DEMO_FALLBACK_SPLIT;
        $("scene").value = DEMO_FALLBACK_SCENE;
        setQualityGateValue(80, { persist: true });
        setStatus({
          info: "dataset catalog fallback applied",
          from: `${fromSplit}/env_${fromScene}`,
          to: `${DEMO_FALLBACK_SPLIT}/env_${DEMO_FALLBACK_SCENE}`,
          quality_gate: "Q80+",
          reason: "current catalog has no safe showcase instructions; restored demo-ready quality gate",
        });
        return loadDatasetInstructions({ preserveEpisode: false, allowCatalogFallback: false });
      }
      $("datasetEpisode").innerHTML = `<option value="">当前范围内没有可用指令</option>`;
      setText("datasetInstructionCount", "共 0 条可用指令");
      selectedDatasetItem = null;
      selectedDatasetInstruction = "";
      updateReadinessSummary();
      setStatus({ error: "no dataset instructions are available for this scene and split" });
      return;
    }
    renderDatasetOptions(datasetMatches);
    const verifiedCount = datasetMatches.filter((item) => item.verified_success && !item.instruction_variant).length;
    const genSafeCount = datasetMatches.filter((item) => item.gen_eval_label === "GEN-SAFE" && !item.instruction_variant).length;
    const allCount = Number.isFinite(Number(data.all_count)) ? Number(data.all_count) : datasetMatches.length;
    const safeDisplayCount = Number.isFinite(Number(data.safe_display_count)) ? Number(data.safe_display_count) : datasetMatches.length;
    const originalCount = Number.isFinite(Number(data.original_count)) ? Number(data.original_count) : datasetMatches.length;
    const variantCount = Number.isFinite(Number(data.variant_count)) ? Number(data.variant_count) : datasetMatches.filter((item) => item.instruction_variant).length;
    const variantOkCount = Number.isFinite(Number(data.variant_ok_count)) ? Number(data.variant_ok_count) : datasetMatches.filter((item) => item.variant_verified_success).length;
    const riskCount = Number.isFinite(Number(data.gen_risk_count)) ? Number(data.gen_risk_count) : 0;
    const rawCount = Number.isFinite(Number(data.gen_raw_count)) ? Number(data.gen_raw_count) : 0;
    const tierCounts = data.showcase_tier_counts || {};
    setText(
      "datasetInstructionCount",
      `稳定展示池 ${originalCount}/${safeDisplayCount} 条，保守改写 ${variantCount} 条（VAR-OK ${variantOkCount}）：OK ${verifiedCount}，GEN-SAFE ${genSafeCount}；全量候选 ${allCount} 条，风险/未验证 ${riskCount + rawCount} 条已隐藏；选择后自动填入完整长指令`
    );
    updateDatasetInstructionSummary(data);
    const preservedIndex = currentEpisodeId
      ? datasetMatches.findIndex((item) => String(item.episode_id || "") === currentEpisodeId)
      : -1;
    const selectedIndex = preservedIndex >= 0 ? preservedIndex : 0;
    $("datasetEpisode").value = String(selectedIndex);
    applyDatasetEpisode(selectedIndex);
    return datasetMatches[selectedIndex];
  } catch (err) {
    if (loadSequence !== datasetLoadSequence) return null;
    $("datasetEpisode").innerHTML = `<option value="">指令目录加载失败</option>`;
    setText("datasetInstructionCount", "加载失败，请检查服务状态");
    selectedDatasetItem = null;
    updateReadinessSummary();
    setStatus(err);
    return null;
  }
}

async function pollSession() {
  if (!currentSession || pollInFlight) return;
  pollInFlight = true;
  let keepPolling = false;
  try {
    const mode = $("mode").value;
    const data = await api(mode === "agent" ? `/api/agent/sessions/${currentSession}` : `/api/sessions/${currentSession}`);
    setStatus(data);
    const terminalStatus = isTerminalSessionStatus(data.status);
    const minimalLiveUi = SHOWCASE_RUNNING_MINIMAL_UI && data.status === "running" && !terminalStatus;
    const bevData = SHOWCASE_LIVE_DIAGNOSTICS || terminalStatus
      ? await loadBevData(mode, currentSession, data)
      : data;
    if (minimalLiveUi) {
      updateRunningShowcaseSummary(data);
    } else {
      updateMetricPanel(bevData);
    }
    if (!minimalLiveUi && (SHOWCASE_LIVE_DIAGNOSTICS || terminalStatus)) {
      drawBevThrottled(bevData, terminalStatus);
    }
    if (SHOWCASE_RECORD_ONLY_UNTIL_TERMINAL && !terminalStatus) {
      setViewerMessage(liveRecordingMessage(data), false);
    } else if (data.last_frame_at && data.last_frame_at > 0) {
      setViewerMessage("", true);
    } else if (["error", "stopped", "done"].includes(data.status)) {
      setViewerMessage(data.error || data.stop_reason || "No frame was produced.");
    } else {
      setViewerMessage("Waiting for the first frame...");
    }
    if (terminalStatus) {
      delayedFrameTerminal = true;
      const pureVideoStarted = SHOWCASE_TERMINAL_PURE_VIDEO_REPLAY
        ? await startTerminalVideoReplay(currentSession, "pure")
        : false;
      const ultraVideoStarted = pureVideoStarted ? false : await startTerminalVideoReplay(currentSession, "ultra", true);
      const stableAnimationStarted = (pureVideoStarted || ultraVideoStarted) ? false : await startTerminalStableAnimationReplay(currentSession, true);
      const spriteStarted = (pureVideoStarted || ultraVideoStarted || stableAnimationStarted) ? false : await startTerminalSpriteReplay(currentSession);
      const chunkedAnimationStarted = (pureVideoStarted || ultraVideoStarted || stableAnimationStarted || spriteStarted) ? false : await startTerminalChunkedAnimationReplay(currentSession);
      const animationStarted = (pureVideoStarted || ultraVideoStarted || stableAnimationStarted || spriteStarted || chunkedAnimationStarted) ? false : await startTerminalAnimationReplay(currentSession);
      const videoStarted = (pureVideoStarted || ultraVideoStarted || stableAnimationStarted || spriteStarted || chunkedAnimationStarted || animationStarted) ? false : await startTerminalVideoReplay(currentSession, "light");
      if (pureVideoStarted || ultraVideoStarted || stableAnimationStarted || spriteStarted || chunkedAnimationStarted || animationStarted || videoStarted) {
        // Prefer browser-decoded replay assets over thousands of image requests.
      } else if (startTerminalMjpegReplay(currentSession)) {
        // The saved-frame MJPEG stream is smoother for long routes than issuing
        // thousands of separate image requests from the browser.
      } else if (
        SHOWCASE_REPLAY_FROM_START_ON_TERMINAL
        && !delayedFrameReplayRestarted
        && delayedFrameItems.length > 1
      ) {
        delayedFrameReplayRestarted = true;
        delayedFrameCursor = 0;
        delayedFrameCurrentImage = null;
        delayedFramePreviousImage = null;
        delayedFrameFrameStartedAt = 0;
        setViewerMessage("飞行完成，正在平滑回放...", false);
      }
      const endedSession = currentSession;
      clearPolling();
      currentSession = null;
      setFlightControlsRunning(false);
      if (endedSession) $("sessionId").textContent = `last session: ${endedSession}`;
    } else {
      keepPolling = true;
    }
  } catch (err) {
    if (err && (err.status === 404 || String(err.message || "").includes("not found"))) {
      resetFlightUi("session is no longer active");
      return;
    }
    setStatus(err);
    keepPolling = Boolean(currentSession);
  } finally {
    pollInFlight = false;
    if (keepPolling) schedulePolling();
  }
}

async function loadBevData(mode, sessionId, fallbackData) {
  if (mode !== "agent" || !sessionId) return fallbackData;
  const now = performance.now();
  const terminal = ["done", "error", "stopped"].includes(fallbackData?.status);
  if (!terminal && !cachedTraceBevData && Number(fallbackData?.step || 0) < 20) {
    return fallbackData;
  }
  if (!terminal && cachedTraceBevData && now - lastTraceFetchAt < TRACE_FETCH_INTERVAL_MS) {
    return {
      ...fallbackData,
      path: cachedTraceBevData.path,
      goal_position: cachedTraceBevData.goal_position ?? fallbackData.goal_position,
    };
  }
  try {
    const trace = await api(`/api/agent/sessions/${sessionId}/trace?t=${Date.now()}`);
    if (Array.isArray(trace.path) && trace.path.length) {
      const merged = {
        ...fallbackData,
        path: trace.path,
        ...(trace.latest || {}),
      };
      cachedTraceBevData = merged;
      lastTraceFetchAt = now;
      return merged;
    }
  } catch (err) {
    return cachedTraceBevData ? { ...fallbackData, path: cachedTraceBevData.path } : fallbackData;
  }
  return fallbackData;
}

async function start() {
  if (replayTimer) clearInterval(replayTimer);
  stopDelayedFramePlayer();
  clearPolling();
  currentSession = null;
  const mode = $("mode").value;
  // Stable-pool items without a pre-rendered replay are intentionally kept out
  // of the smooth demo path. Starting live inference for them would reintroduce
  // the 5-6 fps capture jitter the demo is meant to avoid.
  if (
    mode === "agent"
    && selectedDatasetItem
    && selectedDatasetItem.demo_playback_mode
    && !selectedDatasetItem.demo_replay_available
  ) {
    setFlightControlsRunning(false);
    setStatus({
      pending: true,
      reason: "smooth demo video is still being generated",
      episode_id: selectedDatasetItem.episode_id || "-",
      demo_playback_mode: selectedDatasetItem.demo_playback_mode,
      suggestion: "请先选择标记为高分辨率重生成或旧缓存回放的成功指令。",
    });
    setViewerMessage("该指令的高清流畅视频尚未生成", false);
    return;
  }
  if (mode === "agent" && selectedDatasetItem) {
    const replayStarted = await startVerifiedSourceReplay(selectedDatasetItem);
    if (replayStarted) return;
    if (isVerifiedDemoReplayItem(selectedDatasetItem)) {
      setDemoReplayActive(false);
      setFlightControlsRunning(false);
      setStatus({
        error: "已验证演示回放未能启动，已停止在本地演示模式，未切换在线飞行。",
        reason: "verified_demo_replay_failed",
        source_session_id: selectedDatasetItem.demo_source_session_id || selectedDatasetItem.source_session_id || "",
        suggestion: "请重试开始，或切换下一条演示指令；系统不会再误触发在线 agent start。",
      });
      return;
    }
  }
  if (mode === "agent") {
    await checkHealth();
  }
  if (false && mode === "agent" && $("oracleGoalHoming").checked && $("oraclePathHoming").checked) {
    $("oracleGoalHoming").checked = false;
    updateOracleModeNotice();
    setStatus({
      warning: "已自动关闭 Oracle goal homing。请只保留 Oracle path homing，否则 goal/path 两种诊断目标会让验证含义不清。",
    });
  }
  if (mode === "agent") {
    ensureLongInstructionSplitForLearned();
  }
  const usingDatasetBackedGuidance = mode === "agent"
    && ($("oracleGoalHoming").checked || $("oraclePathHoming").checked || $("learnedSegmentHoming").checked);
  if (usingDatasetBackedGuidance && !selectedDatasetInstruction) {
    const matched = await loadDatasetInstructions();
    if (!matched || !selectedDatasetInstruction) {
      setStatus({
        error: "请选择一条可用飞行指令。该模式必须使用数据集 episode 的原始长 instruction。",
      });
      return;
    }
  }
  const instructionChanged = selectedDatasetInstruction
    && normalizeText($("instruction").value) !== normalizeText(selectedDatasetInstruction);
  if (usingDatasetBackedGuidance && instructionChanged) {
    $("instruction").value = selectedDatasetInstruction;
    updateOracleModeNotice();
    setStatus({
      info: "已自动把飞行指令恢复为当前 Dataset Episode 的原始 instruction。",
      dataset_instruction: selectedDatasetInstruction,
    });
  }
  if (mode === "agent") {
    const readiness = readinessSnapshot();
    updateReadinessSummary();
    if (!readiness.ready) {
      setFlightControlsRunning(false);
      setStatus({
        blocked: true,
        reason: "agent launch readiness check failed",
        issues: readiness.issues,
        readiness: readiness.label,
      });
      return;
    }
  }
  const stepBudgetRaised = ensureLongDatasetStepBudget();
  const payload = {
    instruction: $("instruction").value.trim(),
    scene_id: Number($("scene").value || 4),
    max_steps: Number($("maxSteps").value || 900),
    mode,
    confirm_control: $("confirmControl").checked,
  };
  if (mode === "agent") {
    payload.episode_id = $("episodeId").value.trim();
    payload.split = $("split").value;
    payload.checkpoint = $("checkpoint").value.trim();
    payload.simulator_tool_port = Number($("simulatorPort").value || 30000);
    payload.gpu_id = Number($("gpuId").value || 0);
    payload.save_frames = true;
    payload.enable_oracle_goal_homing = $("oracleGoalHoming").checked;
    payload.enable_oracle_path_homing = $("oraclePathHoming").checked;
    payload.enable_learned_segment_homing = $("learnedSegmentHoming").checked;
  }
  try {
    setFlightControlsRunning(true);
    setStatus({
      launching: true,
      instruction: payload.instruction,
      episode_id: payload.episode_id,
      split: payload.split,
      max_steps: payload.max_steps,
      simulator_tool_port: payload.simulator_tool_port,
      enable_oracle_goal_homing: payload.enable_oracle_goal_homing,
      enable_oracle_path_homing: payload.enable_oracle_path_homing,
      enable_learned_segment_homing: payload.enable_learned_segment_homing,
      step_budget_raised: stepBudgetRaised,
    });
    const data = await api(mode === "agent" ? "/api/agent/start" : "/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    currentSession = data.session_id;
    latestBevData = null;
    cachedTraceBevData = null;
    lastTraceFetchAt = 0;
    drawBevThrottled(null, true);
    $("sessionId").textContent = `session: ${currentSession}`;
    setViewerMessage(liveRecordingMessage(), false);
    if (!SHOWCASE_RECORD_ONLY_UNTIL_TERMINAL && !startDelayedFramePlayer(currentSession)) {
      $("stream").src = `${data.stream}?${STREAM_QUERY}&t=${Date.now()}`;
    }
    setFlightControlsRunning(true);
    await pollSession();
  } catch (err) {
    setFlightControlsRunning(false);
    setStatus(err);
  }
}

async function previewPlan() {
  const payload = {
    instruction: $("instruction").value.trim(),
    max_steps: Number($("maxSteps").value || 120),
  };
  try {
    const data = await api("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setStatus(data);
  } catch (err) {
    setStatus(err);
  }
}

async function loadModelReplay() {
  try {
    stopDelayedFramePlayer();
    const data = await api("/api/model-runs/latest");
    if (!data.frames || !data.frames.length) {
      setStatus({ error: "latest model run has no frames" });
      return;
    }
    if (replayTimer) clearInterval(replayTimer);
    $("stream").style.display = "block";
    $("stream").style.opacity = "1";
    $("placeholder").style.display = "none";
    $("sessionId").textContent = `model run: ${data.run_id}`;
    $("badge").textContent = "replay";

    let index = 0;
    const show = () => {
      $("stream").src = `${data.frames[index]}?t=${Date.now()}`;
      const last = data.last || {};
      const trace = data.trace || [];
      const currentTrace = trace[index] || last;
      drawBevThrottled({
        status: "replay",
        step: currentTrace.step,
        action: currentTrace.action,
        position: currentTrace.position,
        distance_to_goal: currentTrace.distance_to_goal,
        distance_3d_to_goal: currentTrace.distance_3d_to_goal,
        vertical_error_to_goal: currentTrace.vertical_error_to_goal,
        altitude_aligned: currentTrace.altitude_aligned,
        surface_clearance: currentTrace.surface_clearance,
        surface_aligned: currentTrace.surface_aligned,
        surface_probe_ok: currentTrace.surface_probe_ok,
        surface_probe_error: currentTrace.surface_probe_error,
        best_distance: currentTrace.best_distance,
        success_20m: currentTrace.success_20m,
        success_20m_with_altitude: currentTrace.success_20m_with_altitude,
        success_20m_with_surface: currentTrace.success_20m_with_surface,
        oracle_success_20m: currentTrace.oracle_success_20m,
        progress_to_goal: currentTrace.progress_to_goal,
        navigation_phase: currentTrace.navigation_phase,
        goal_position: currentTrace.goal_position,
        path: trace.slice(0, Math.max(1, index + 1)),
      });
      updateMetricPanel({
        status: "replay",
        distance_to_goal: currentTrace.distance_to_goal,
        distance_3d_to_goal: currentTrace.distance_3d_to_goal,
        vertical_error_to_goal: currentTrace.vertical_error_to_goal,
        altitude_aligned: currentTrace.altitude_aligned,
        surface_clearance: currentTrace.surface_clearance,
        surface_aligned: currentTrace.surface_aligned,
        surface_probe_ok: currentTrace.surface_probe_ok,
        surface_probe_error: currentTrace.surface_probe_error,
        best_distance: currentTrace.best_distance,
        success_20m: currentTrace.success_20m,
        success_20m_with_altitude: currentTrace.success_20m_with_altitude,
        success_20m_with_surface: currentTrace.success_20m_with_surface,
        oracle_success_20m: currentTrace.oracle_success_20m,
        progress_to_goal: currentTrace.progress_to_goal,
        navigation_phase: currentTrace.navigation_phase,
        goal_position: currentTrace.goal_position,
      });
      setStatus({
        mode: "model replay",
        run_id: data.run_id,
        episode_id: data.episode_id,
        scene_id: data.scene_id,
        last_action: last.action,
        last_distance_to_goal: last.distance_to_goal,
        instruction: data.instruction,
      });
      index = (index + 1) % data.frames.length;
    };
    show();
    replayTimer = setInterval(show, 160);
  } catch (err) {
    setStatus(err);
  }
}

async function stop() {
  $("stopBtn").disabled = true;
  try {
    const mode = $("mode").value;
    const path = mode === "agent"
      ? (currentSession ? `/api/agent/stop/${currentSession}` : "/api/agent/stop-active")
      : `/api/stop/${currentSession}`;
    const data = await api(path, { method: "POST" });
    setStatus(data);
    resetFlightUi("user_stop");
  } catch (err) {
    resetFlightUi("stop request finished locally");
    setStatus(err);
  }
}

async function applyDemoDefaults() {
  if (currentSession) {
    setStatus({ warning: "demo defaults are not applied while a flight session is running" });
    return;
  }
  $("mode").value = "agent";
  $("scene").value = DEMO_FALLBACK_SCENE;
  $("split").value = DEMO_FALLBACK_SPLIT;
  $("maxSteps").value = "900";
  $("simulatorPort").value = "30014";
  $("confirmControl").checked = true;
  $("showcaseFilter").value = "live";
  setQualityGateValue(80, { persist: true });
  saveShowcaseFilterPreference();
  saveSimulatorPortPreference();
  saveConfirmControlPreference();
  setStatus({
    info: "demo defaults applied",
    mode: "agent",
    split: DEMO_FALLBACK_SPLIT,
    scene: `env_${DEMO_FALLBACK_SCENE}`,
    quality_gate: "Q80+ LIVE-OK",
    showcase_filter: "LIVE-OK only",
    demo_pool: "current-version verified demo/strong flights",
    simulator_port: 30014,
  });
  await checkHealth({ silent: true });
  await loadDatasetInstructions({ preserveEpisode: false, allowCatalogFallback: false });
  await loadDemoBuilder({ preserveEpisode: true });
  updateOracleModeNotice();
  updateReadinessSummary();
}

async function applyGeneralizationPoolDefaults() {
  if (currentSession) {
    setStatus({ warning: "generalization pool is not changed while a flight session is running" });
    return;
  }
  $("mode").value = "agent";
  $("scene").value = DEMO_FALLBACK_SCENE;
  $("split").value = DEMO_FALLBACK_SPLIT;
  $("maxSteps").value = "900";
  $("simulatorPort").value = "30014";
  $("confirmControl").checked = true;
  $("showcaseFilter").value = "frontline";
  setQualityGateValue(70, { persist: true });
  saveShowcaseFilterPreference();
  saveSimulatorPortPreference();
  saveConfirmControlPreference();
  setStatus({
    info: "generalization candidate pool applied",
    mode: "agent",
    split: DEMO_FALLBACK_SPLIT,
    scene: `env_${DEMO_FALLBACK_SCENE}`,
    quality_gate: "Q70+ candidate pool",
    showcase_filter: "FRONT only",
    note: "use this for manual generalization tests; 演示配置 returns to LIVE-OK demo flights",
    simulator_port: 30014,
  });
  await checkHealth({ silent: true });
  await loadDatasetInstructions({ preserveEpisode: false, allowCatalogFallback: false });
  updateOracleModeNotice();
  updateReadinessSummary();
}

async function selectNextDemoInstruction() {
  if (demoReplayActive) {
    stopDelayedFramePlayer();
    setFlightControlsRunning(false);
  }
  if (currentSession) {
    try {
      const data = await api(`/api/agent/sessions/${currentSession}?t=${Date.now()}`);
      if (isTerminalSessionStatus(data.status)) {
        currentSession = null;
        setFlightControlsRunning(false);
        $("sessionId").textContent = `last session: ${data.session_id || "-"}`;
      } else {
        setStatus({ warning: "demo instruction cannot be changed while a flight session is running" });
        return;
      }
    } catch (err) {
      currentSession = null;
      setFlightControlsRunning(false);
    }
  }
  if (!datasetMatches.length) {
    await applyDemoDefaults();
    if (!datasetMatches.length) return;
  }
  const demoIndices = demoReadyIndices();
  const qualityIndices = qualityGateReadyIndices();
  const cycleIndices = qualityIndices.length ? qualityIndices : datasetMatches.map((_, index) => index);
  const currentIndex = Number($("datasetEpisode")?.value || 0);
  const currentDemoPosition = cycleIndices.indexOf(currentIndex);
  const nextCyclePosition = currentDemoPosition >= 0 ? (currentDemoPosition + 1) % cycleIndices.length : 0;
  const nextIndex = cycleIndices[nextCyclePosition];
  $("datasetEpisode").value = String(nextIndex);
  applyDatasetEpisode(nextIndex);
  setStatus({
    info: "next demo instruction selected",
    demo_index: nextCyclePosition + 1,
    demo_total: cycleIndices.length,
    pool_index: nextIndex + 1,
    pool_total: datasetMatches.length,
    demo_ready_count: demoIndices.length,
    quality_gate_count: qualityIndices.length,
    episode_id: datasetMatches[nextIndex]?.episode_id || "-",
    quality: datasetMatches[nextIndex]?.display_quality_score,
  });
}

async function restoreActiveAgentSession() {
  if ($("mode").value !== "agent") return;
  try {
    const data = await api(`/api/agent/sessions/active?t=${Date.now()}`);
    if (!data || !data.session_id) return;
    currentSession = data.session_id;
    cachedTraceBevData = null;
    lastTraceFetchAt = 0;
    $("sessionId").textContent = `session: ${currentSession}`;
    setFlightControlsRunning(true);
    if (SHOWCASE_RECORD_ONLY_UNTIL_TERMINAL) {
      setViewerMessage(liveRecordingMessage(data), false);
    } else if (!startDelayedFramePlayer(currentSession)) {
      $("stream").src = `/stream/${currentSession}?${STREAM_QUERY}&t=${Date.now()}`;
      setViewerMessage(data.last_frame_at ? "" : "Restored active session; waiting for frame...", Boolean(data.last_frame_at));
    }
    updateMetricPanel(data);
    drawBevThrottled(data, true);
    clearPolling();
    await pollSession();
    return true;
  } catch (err) {
    // No active session is a normal page-load state.
  }
  return false;
}

async function recoverStaleControls() {
  if (demoReplayActive) return;
  if (currentSession && !$("stopBtn")?.disabled) return;
  const badge = $("badge")?.textContent || "";
  if (["done", "error", "stopped"].includes(badge)) {
    setFlightControlsRunning(false);
    return;
  }
  if (!$("startBtn")?.disabled) return;
  try {
    await api(`/api/agent/sessions/active?t=${Date.now()}`);
  } catch (err) {
    if (currentSession) {
      try {
        const completed = await api(`/api/agent/sessions/${currentSession}?t=${Date.now()}`);
        if (["done", "error", "stopped"].includes(completed.status)) {
          setStatus(completed);
          updateMetricPanel(completed);
          drawBevThrottled(completed, true);
          setFlightControlsRunning(false);
          return;
        }
      } catch (sessionError) {
        // The session may have been removed during a server restart.
      }
    }
    resetFlightUi("no active backend session");
  }
}

$("startBtn").addEventListener("click", start);
$("datasetEpisode").addEventListener("change", () => applyDatasetEpisode(Number($("datasetEpisode").value)));
$("showcaseFilter")?.addEventListener("change", () => {
  saveShowcaseFilterPreference();
  loadDatasetInstructions({ preserveEpisode: false });
});
$("qualityGate")?.addEventListener("change", () => {
  saveQualityGatePreference();
  loadDatasetInstructions({ preserveEpisode: false });
});
$("gpuId")?.addEventListener("change", updateReadinessSummary);
$("simulatorPort")?.addEventListener("change", () => {
  saveSimulatorPortPreference();
  checkHealth();
});
$("confirmControl")?.addEventListener("change", () => {
  saveConfirmControlPreference();
  updateReadinessSummary();
});
$("scene").addEventListener("change", () => loadDatasetInstructions({ preserveEpisode: false }));
$("split").addEventListener("change", () => loadDatasetInstructions({ preserveEpisode: false }));
$("instruction").addEventListener("input", updateOracleModeNotice);
$("oraclePathHoming").addEventListener("change", () => {
  updateOracleModeNotice();
});
$("oracleGoalHoming").addEventListener("change", () => {
  updateOracleModeNotice();
});
$("learnedSegmentHoming").addEventListener("change", () => {
  if ($("learnedSegmentHoming").checked) {
    ensureLongInstructionSplitForLearned();
  }
  loadDatasetInstructions();
  updateOracleModeNotice();
});
$("planBtn").addEventListener("click", previewPlan);
$("replayBtn").addEventListener("click", loadModelReplay);
$("stopBtn").addEventListener("click", stop);
$("demoDefaultsBtn")?.addEventListener("click", applyDemoDefaults);
$("nextDemoBtn")?.addEventListener("click", selectNextDemoInstruction);
$("generalizationPoolBtn")?.addEventListener("click", applyGeneralizationPoolDefaults);
$("demoFamily")?.addEventListener("change", () => renderDemoBuilderSeeds({ preserveEpisode: false }));
$("demoSeed")?.addEventListener("change", () => {
  const families = demoBuilderCatalog?.families || [];
  const family = families.find((item) => item.id === $("demoFamily")?.value) || families[0];
  const seeds = family?.seeds || [];
  selectedDemoSeed = seeds[Number($("demoSeed")?.value || 0)] || seeds[0] || null;
  renderDemoBuilderVariants();
});
$("demoVariant")?.addEventListener("change", applyDemoBuilderSelection);
$("mode").addEventListener("change", () => {
  if ($("mode").value === "agent" && $("scene").value !== "16") {
    $("scene").value = "16";
    $("maxSteps").value = "900";
    loadDatasetInstructions({ preserveEpisode: false });
  }
  updateReadinessSummary();
});

async function initializePage() {
  restoreShowcaseFilterPreference();
  restoreQualityGatePreference();
  restoreRuntimePreferences();
  startStreamCanvasRenderer();
  startHealthRefresh();
  await loadScenes();
  await loadGpus();
  await checkHealth();
  await loadDatasetInstructions();
  await loadDemoBuilder({ preserveEpisode: true });
  const restored = await restoreActiveAgentSession();
  if (!restored) drawBevThrottled(null, true);
  updateOracleModeNotice();
  updateReadinessSummary();
}

initializePage();
window.addEventListener("resize", () => drawBevThrottled(latestBevData, true));
window.addEventListener("focus", () => {
  recoverStaleControls();
  if (!demoReplayActive) checkHealth({ silent: true });
});
setInterval(recoverStaleControls, 3000);
