# web_server.py
from flask import Flask, request, jsonify
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py_frame import SlideshowController  # adjust import


def create_app(controller: "SlideshowController") -> Flask:
    app = Flask(__name__)

    @app.route("/api/state")
    def api_state():
        with controller.lock:
            slides = [
                {
                    "index": i,
                    "path": s.path,
                    "marked": i in controller.current_marks,
                    "pattern_type": controller.current_pattern_type,
                }
                for i, s in enumerate(controller.current_slides)
            ]
            paused = controller.paused
        return jsonify({"slides": slides, "paused": paused})

    @app.route("/api/mark", methods=["POST"])
    def api_mark():
        data = request.json or {}
        slot = int(data.get("slot", -1))
        if not (0 <= slot < 5):
            return jsonify({"ok": False, "error": "invalid slot"}), 400

        with controller.lock:
            if slot in controller.current_marks:
                controller.current_marks.remove(slot)
            else:
                controller.current_marks.add(slot)
        return jsonify({"ok": True})

    @app.route("/api/command", methods=["POST"])
    def api_command():
        data = request.json or {}
        cmd = data.get("cmd")
        steps = int(data.get("steps", 1))

        if cmd not in ("next", "prev", "pause", "play"):
            return jsonify({"ok": False, "error": "bad cmd"}), 400

        with controller.lock:
            if cmd in ("pause", "play"):
                controller.pending_command = {"type": cmd}
            else:
                controller.pending_command = {"type": cmd, "steps": steps}

        return jsonify({"ok": True})

    @app.route("/")
    def index():
        return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Frame Control</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    button { font-size: 1.5rem; margin: 0.3rem; }
    .slot { border: 1px solid #ccc; margin: 0.5rem; padding: 0.5rem; }
    .marked { background: #fdd; }
  </style>
</head>
<body>
  <div>
    <button onclick="sendCommand('prev', 1)">&laquo; Prev</button>
    <button onclick="sendCommand('next', 1)">Next &raquo;</button>
    <button onclick="sendCommand('pause')">Pause</button>
    <button onclick="sendCommand('play')">Play</button>
  </div>
  <div id="status"></div>
  <div id="slots"></div>

<script>
async function refreshState() {
  const res = await fetch('/api/state');
  const data = await res.json();
  const slotsDiv = document.getElementById('slots');
  const statusDiv = document.getElementById('status');

  statusDiv.textContent = data.paused ? "Status: PAUSED" : "Status: PLAYING";

  slotsDiv.innerHTML = '';
  data.slides.forEach(slide => {
    const div = document.createElement('div');
    div.className = 'slot' + (slide.marked ? ' marked' : '');
    div.innerHTML = `
      <div><b>Slot ${slide.index + 1}</b></div>
      <div>${slide.path}</div>
      <button onclick="toggleMark(${slide.index})">
        ${slide.marked ? 'Unmark' : 'Mark'}
      </button>
    `;
    slotsDiv.appendChild(div);
  });
}

async function toggleMark(slot) {
  await fetch('/api/mark', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({slot})
  });
  refreshState();
}

async function sendCommand(cmd, steps) {
  const body = { cmd };
  if (steps !== undefined) {
    body.steps = steps;
  }
  await fetch('/api/command', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  setTimeout(refreshState, 800);
}

setInterval(refreshState, 3000);
refreshState();
</script>
</body>
</html>
"""

    return app


def run_web(controller: "SlideshowController"):
    app = create_app(controller)
    # host=0.0.0.0 so phones on LAN can reach it
    app.run(host="0.0.0.0", port=7654, threaded=True)
