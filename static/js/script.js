/* Dashboard logic: polls the Flask API and updates the page. Plain JavaScript + Chart.js. */

const EMOTIONS = ["Happy", "Neutral", "Surprise", "Sad", "Angry", "Fear", "Disgust"];
const COLORS = {
  Happy: "#22c55e", Neutral: "#94a3b8", Surprise: "#f59e0b", Sad: "#3b82f6",
  Angry: "#ef4444", Fear: "#8b5cf6", Disgust: "#a16207", "No face": "#dbe0ef", "N/A": "#dbe0ef",
};
const STATUS_POLL_MS = 600;
const ANALYTICS_POLL_MS = 2000;
const GAUGE_CIRCUMFERENCE = 326.7;

const $ = (id) => document.getElementById(id);
let sessionActive = false;
let sessionStartEpoch = null;
let failedPolls = 0;
let lastStatus = null;
let viewSessionId = null;   // null = follow the running / latest session
let emotionChart = null;
let engagementChart = null;

/* ---------------- helpers ---------------- */
function setPill(id, kind, text) {
  const pill = $(id);
  pill.className = "pill " + kind;
  pill.lastChild.textContent = text;
}
function formatDuration(totalSeconds) {
  const s = Math.max(0, Math.floor(totalSeconds));
  const m = Math.floor(s / 60);
  return String(m).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
}
function engagementLabel(score, faceDetected) {
  if (!faceDetected) return "No face visible";
  if (score >= 70) return "Higher estimated engagement";
  if (score >= 40) return "Moderate estimated engagement";
  return "Lower estimated engagement";
}

/* ---------------- one-time UI setup ---------------- */
function buildProbabilityRows() {
  $("prob-list").innerHTML = EMOTIONS.map((name) => `
    <div class="prob-row" data-name="${name}">
      <span class="prob-name">${name}</span>
      <div class="prob-track"><div class="prob-fill" style="background:${COLORS[name]}"></div></div>
      <span class="prob-val">0%</span>
    </div>`).join("");
  $("legend").innerHTML = EMOTIONS.concat(["No face"]).map(
    (name) => `<span><i style="background:${COLORS[name]}"></i>${name}</span>`).join("");
  $("timeline").innerHTML = '<span class="timeline-empty">Timeline appears once a session is running.</span>';
}

function whenChartReady(callback, triesLeft = 40) {
  if (typeof Chart !== "undefined") return callback();
  if (triesLeft <= 0) {   // CDN blocked / offline and no local copy
    ["emotion-empty", "engagement-empty"].forEach((id) => {
      $(id).textContent = "Chart.js could not be loaded (see README: offline setup).";
      $(id).classList.remove("hidden");
    });
    return;
  }
  setTimeout(() => whenChartReady(callback, triesLeft - 1), 150);
}

function buildCharts() {
  Chart.defaults.font.family = '"Inter", system-ui, sans-serif';
  Chart.defaults.color = "#6b7491";

  emotionChart = new Chart($("emotion-chart"), {
    type: "doughnut",
    data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderWidth: 2, borderColor: "#fff" }] },
    options: {
      maintainAspectRatio: false, cutout: "64%",
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 10, usePointStyle: true } },
        tooltip: { callbacks: { label: (c) => ` ${c.label}: ${c.parsed}%` } },
      },
    },
  });

  engagementChart = new Chart($("engagement-chart"), {
    type: "line",
    data: { labels: [], datasets: [{
      label: "Estimated engagement", data: [], borderColor: "#4f5bd5", backgroundColor: "rgba(79,91,213,.12)",
      fill: true, tension: 0.35, pointRadius: 0, borderWidth: 2.5,
    }] },
    options: {
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 100, grid: { color: "#eef0f8" }, ticks: { stepSize: 25 } },
        x: { grid: { display: false }, ticks: { maxTicksLimit: 6 } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

/* ---------------- live status ---------------- */
function applyStatus(s) {
  failedPolls = 0;
  lastStatus = s;
  $("server-banner").classList.add("hidden");

  // header pills
  setPill("pill-camera", s.camera_available ? "ok" : "bad", s.camera_available ? "Camera online" : "No camera");
  const modelText = { loading: "Loading model…", ready: "Model ready", error: "Model error" }[s.model_status];
  setPill("pill-model", { loading: "warn", ready: "ok", error: "bad" }[s.model_status], modelText);
  setPill("pill-session", s.session_active ? "ok" : "", s.session_active ? "Session running" : "Session idle");
  $("camera-warning").classList.toggle("hidden", s.camera_available);
  $("live-tag").classList.toggle("on", s.camera_available);

  // expression + confidence
  const hasEmotion = !["N/A", "No face"].includes(s.emotion);
  $("status-emotion").textContent = s.model_status === "loading" ? "Loading…" : (s.emotion === "N/A" ? "—" : s.emotion);
  $("emotion-dot").style.background = COLORS[s.emotion] || "#cbd2e6";
  $("status-confidence").textContent = hasEmotion ? s.confidence.toFixed(0) + "%" : "—";

  // face
  $("status-face").textContent = !s.camera_available ? "—" : (s.face_detected ? "Detected" : "Not detected");
  $("face-dot").style.background = s.face_detected ? "#22c55e" : "#cbd2e6";
  $("face-sub").textContent = !s.camera_available ? "Waiting for camera…"
    : s.face_count > 1 ? `${s.face_count} faces — analysing the largest`
    : s.face_detected ? "One face in view" : "Look toward the camera";

  // engagement gauge
  const score = s.engagement_score;
  $("gauge-value").textContent = score;
  $("gauge-arc").style.strokeDashoffset = GAUGE_CIRCUMFERENCE * (1 - score / 100);
  $("gauge-arc").style.stroke = score >= 70 ? "#22c55e" : score >= 40 ? "#4f5bd5" : "#f59e0b";
  $("engagement-label").textContent = engagementLabel(score, s.face_detected);

  // probability bars
  const topName = hasEmotion ? s.emotion : null;
  document.querySelectorAll("#prob-list .prob-row").forEach((row) => {
    const value = (s.scores && s.scores[row.dataset.name]) || 0;
    row.querySelector(".prob-fill").style.width = value + "%";
    row.querySelector(".prob-val").textContent = value.toFixed(0) + "%";
    row.classList.toggle("top", row.dataset.name === topName);
  });

  syncSession(s);
}

function syncSession(s) {
  sessionActive = s.session_active;
  sessionStartEpoch = s.session_start_epoch;
  $("start-btn").disabled = sessionActive || !$("consent").checked;
  $("end-btn").disabled = !sessionActive;
  $("consent").disabled = sessionActive;
  if (sessionActive) $("summary-card").classList.add("hidden");
  document.querySelectorAll(".view-btn").forEach((b) => { b.disabled = sessionActive; });
}

function tickTimer() {
  if (sessionActive && sessionStartEpoch) $("timer").textContent = formatDuration(Date.now() / 1000 - sessionStartEpoch);
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status");
    applyStatus(await response.json());
  } catch (error) {
    if (++failedPolls >= 3) $("server-banner").classList.remove("hidden");
  }
}

/* ---------------- analytics ---------------- */
function applyAnalytics(data) {
  const { timeline, summary } = data;

  // which session are we looking at?
  const running = lastStatus && lastStatus.session_active && lastStatus.session_id === data.session_id;
  $("viewing-label").textContent = data.session_id === null ? "no session yet"
    : `Session #${data.session_id}` + (running ? " (running now)" : "");
  $("show-latest-btn").classList.toggle("hidden", viewSessionId === null);

  // timeline strip
  $("timeline-count").textContent = `${timeline.length} observation${timeline.length === 1 ? "" : "s"}`;
  $("timeline").innerHTML = timeline.length === 0
    ? '<span class="timeline-empty">Timeline appears once a session is running.</span>'
    : timeline.slice(-150).map((t) =>
        `<span class="tick" style="background:${COLORS[t.emotion] || "#dbe0ef"}" title="${t.timestamp.slice(11)} — ${t.emotion} (engagement ${t.engagement_score})"></span>`).join("");

  // doughnut
  if (emotionChart) {
    const names = Object.keys(summary.emotion_distribution);
    emotionChart.data.labels = names;
    emotionChart.data.datasets[0].data = names.map((n) => summary.emotion_distribution[n]);
    emotionChart.data.datasets[0].backgroundColor = names.map((n) => COLORS[n] || "#cbd5e1");
    emotionChart.update("none");
    $("emotion-empty").classList.toggle("hidden", names.length > 0);
  }
  // line
  if (engagementChart) {
    engagementChart.data.labels = timeline.map((t) => t.timestamp.slice(11));
    engagementChart.data.datasets[0].data = timeline.map((t) => t.engagement_score);
    engagementChart.update("none");
    $("engagement-empty").classList.toggle("hidden", timeline.length > 0);
  }
  return data;
}

async function fetchAnalytics() {
  const url = "/api/analytics" + (viewSessionId !== null ? "?session_id=" + viewSessionId : "");
  const response = await fetch(url);
  return applyAnalytics(await response.json());
}

async function pollAnalytics() {
  try { await fetchAnalytics(); } catch (error) { /* status poll shows the banner */ }
}

/* ---------------- session history ---------------- */
function formatStart(iso) {
  return iso.replace("T", " ").slice(0, 16);
}

async function pollSessions() {
  try {
    const response = await fetch("/api/sessions");
    renderSessions(await response.json());
  } catch (error) { /* ignore */ }
}

function renderSessions(sessions) {
  const body = $("history-body");
  if (sessions.length === 0) {
    body.innerHTML = '<tr><td colspan="6" class="muted">No sessions yet. Start a session to create one.</td></tr>';
    return;
  }
  body.innerHTML = sessions.map((s) => `
    <tr class="${s.id === viewSessionId ? "selected" : ""}">
      <td>${s.id}</td>
      <td>${formatStart(s.start_time)}${s.active ? ' <span class="tag-live">running</span>' : ""}</td>
      <td>${formatDuration(s.duration_seconds)}</td>
      <td>${s.observation_count}</td>
      <td>${s.observation_count ? s.average_engagement + "%" : "—"}</td>
      <td class="actions">
        <button class="mini view-btn" data-id="${s.id}" ${sessionActive ? "disabled" : ""}>View</button>
        <a class="mini" href="/api/export_csv?session_id=${s.id}" download>CSV</a>
      </td>
    </tr>`).join("");
  body.querySelectorAll(".view-btn").forEach((b) =>
    b.addEventListener("click", () => viewSession(Number(b.dataset.id))));
}

async function viewSession(id) {
  viewSessionId = id;
  try {
    const data = await fetchAnalytics();
    showSummary(data.summary, false);
    pollSessions();
  } catch (error) {
    alert("Could not load that session.");
  }
}

async function showLatest() {
  viewSessionId = null;
  $("summary-card").classList.add("hidden");
  await pollAnalytics();
  pollSessions();
}

async function clearHistory() {
  if (sessionActive) { alert("End the running session first."); return; }
  if (!confirm("Delete ALL saved sessions? This cannot be undone.")) return;
  try {
    await fetch("/api/clear_history", { method: "POST" });
    viewSessionId = null;
    $("summary-card").classList.add("hidden");
    await pollAnalytics();
    await pollSessions();
  } catch (error) {
    alert("Could not clear the history.");
  }
}

/* ---------------- session buttons ---------------- */
async function startSession() {
  $("start-btn").disabled = true;
  try {
    const response = await fetch("/api/start_session", { method: "POST" });
    if (!response.ok) throw new Error((await response.json()).error || "Could not start");
    viewSessionId = null;
    $("summary-card").classList.add("hidden");
    $("timer").textContent = "00:00";
    await pollStatus();
    await pollAnalytics();
    pollSessions();
  } catch (error) {
    alert("Could not start the session: " + error.message);
    $("start-btn").disabled = !$("consent").checked;
  }
}

async function endSession() {
  $("end-btn").disabled = true;
  try {
    const response = await fetch("/api/end_session", { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not end");
    showSummary(data.summary, true);
    await pollStatus();
    await pollAnalytics();
    pollSessions();
  } catch (error) {
    alert("Could not end the session: " + error.message);
  }
}

function showSummary(summary, scroll) {
  $("sum-duration").textContent = formatDuration(summary.duration_seconds) + " min";
  $("sum-count").textContent = summary.observation_count;
  $("sum-avg").textContent = summary.average_engagement + "%";
  $("sum-visible").textContent = summary.face_visible_percent + "%";

  const entries = Object.entries(summary.emotion_distribution).sort((a, b) => b[1] - a[1]);
  $("sum-distribution").innerHTML = entries.length === 0
    ? '<span class="muted">No observations were recorded in this session.</span>'
    : entries.map(([name, pct]) => `
        <div class="prob-row">
          <span class="prob-name">${name}</span>
          <div class="prob-track"><div class="prob-fill" style="width:${pct}%;background:${COLORS[name] || "#94a3b8"}"></div></div>
          <span class="prob-val">${pct}%</span>
        </div>`).join("");
  $("summary-card").classList.remove("hidden");
  if (scroll) $("summary-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ---------------- start up ---------------- */
$("consent").addEventListener("change", () => { $("start-btn").disabled = sessionActive || !$("consent").checked; });
$("start-btn").addEventListener("click", startSession);
$("end-btn").addEventListener("click", endSession);
$("show-latest-btn").addEventListener("click", showLatest);
$("clear-history-btn").addEventListener("click", clearHistory);

buildProbabilityRows();
whenChartReady(buildCharts);
pollStatus();
pollAnalytics();
pollSessions();
setInterval(pollStatus, STATUS_POLL_MS);
setInterval(pollAnalytics, ANALYTICS_POLL_MS);
setInterval(pollSessions, 4000);
setInterval(tickTimer, 500);
