# web_server.py
from flask import Flask, request, jsonify
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py_frame import SlideshowController  # adjust import


def _parse_int_field(data: dict, key: str, default: int):
    """
    Parse an integer field from a request's JSON body. Returns
    (value, None) on success, or (None, error_response) on failure, where
    error_response is a ready-to-return (jsonify(...), 400) tuple.
    """
    try:
        return int(data.get(key, default)), None
    except (TypeError, ValueError):
        return None, (jsonify({"ok": False, "error": f"invalid {key}"}), 400)


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
            black = controller.black_screen
            video_playing = controller.pending_video is not None
        return jsonify({"slides": slides, "paused": paused, "black": black, "video_playing": video_playing})

    @app.route("/api/mark", methods=["POST"])
    def api_mark():
        data = request.json or {}
        slot, error = _parse_int_field(data, "slot", -1)
        if error:
            return error

        with controller.lock:
            if not (0 <= slot < len(controller.current_slides)):
                return jsonify({"ok": False, "error": "invalid slot"}), 400

            if slot in controller.current_marks:
                controller.current_marks.remove(slot)
            else:
                controller.current_marks.add(slot)
        return jsonify({"ok": True})

    @app.route("/api/settings", methods=["GET", "POST"])
    def api_settings():
        if request.method == "POST":
            data = request.json or {}
            if "shuffle_enabled" in data:
                shuffle_enabled = bool(data["shuffle_enabled"])
                with controller.lock:
                    controller.shuffle_enabled = shuffle_enabled

                import json
                with open(controller.settings_file, "w") as f:
                    json.dump({"shuffle_enabled": shuffle_enabled}, f)

        with controller.lock:
            shuffle_enabled = controller.shuffle_enabled
        return jsonify({"ok": True, "shuffle_enabled": shuffle_enabled})

    @app.route("/api/command", methods=["POST"])
    def api_command():
        data = request.json or {}
        cmd = data.get("cmd")

        if cmd not in ("next", "prev", "pause", "play", "screen_off", "screen_on"):
            return jsonify({"ok": False, "error": "bad cmd"}), 400

        steps, error = _parse_int_field(data, "steps", 1)
        if error:
            return error

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
  body {
    font-family: sans-serif;
    margin: 10px;
  }

  .controls {
    display: grid;
    grid-template-columns: 1fr 1fr;   /* 2 buttons per row */
    gap: 12px;
    margin-bottom: 16px;
  }

  .controls button {
    font-size: 20px;
    padding: 18px 10px;              /* tall buttons */
    border-radius: 10px;
    border: none;
    background: #2c7be5;
    color: white;
    cursor: pointer;
  }

  .controls button:active {
    background: #1a5dc9;
  }

  .controls button:focus {
    outline: none;
  }

  .controls button.active {
    background: #1a5dc9;
    box-shadow: inset 0 0 0 3px #ffd166;
  }

  #status {
    margin: 10px 0;
    font-weight: bold;
  }

  #settings-note {
    margin: 0 0 16px 0;
    color: #666;
    font-size: 13px;
  }

  .slot {
    border: 1px solid #ccc;
    padding: 8px;
    margin-bottom: 8px;
    border-radius: 6px;
  }

  .slot.marked {
    border-color: red;
    background: #ffecec;
  }
</style>
</head>
<body>
    <div class="controls">
      <button onclick="sendCommand('pause')">Pause</button>
      <button onclick="sendCommand('play')">Play</button>
    
      <button onclick="sendCommand('prev', 1)"><<&nbsp;Prev</button>
      <button onclick="sendCommand('next', 1)">Next&nbsp;>></button>
    
      <button onclick="sendCommand('screen_off')">Screen Off</button>
      <button onclick="sendCommand('screen_on')">Screen On</button>

      <button id="btn-shuffle" onclick="setShuffleMode(true)">Shuffle</button>
      <button id="btn-random-start" onclick="setShuffleMode(false)">Random Start</button>
    </div>
  <div id="settings-note">Order takes effect next time the frame restarts.</div>
  <div id="status"></div>
  <div id="slots"></div>

<script>
async function refreshState() {
  const res = await fetch('/api/state');
  const data = await res.json();
  const slotsDiv = document.getElementById('slots');
  const statusDiv = document.getElementById('status');

  let status = data.paused ? "PAUSED" : "PLAYING";
  if (data.video_playing) status = "VIDEO PLAYING";
  if (data.black) status += " (SCREEN OFF)";
  statusDiv.textContent = "Status: " + status;

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

async function refreshSettings() {
  const res = await fetch('/api/settings');
  const data = await res.json();
  document.getElementById('btn-shuffle').classList.toggle('active', data.shuffle_enabled);
  document.getElementById('btn-random-start').classList.toggle('active', !data.shuffle_enabled);
}

async function setShuffleMode(shuffleEnabled) {
  await fetch('/api/settings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({shuffle_enabled: shuffleEnabled})
  });
  refreshSettings();
}

setInterval(refreshState, 3000);
refreshState();
refreshSettings();
</script>
</body>
</html>
"""

    return app


def run_web(controller: "SlideshowController"):
    import threading
    print("Web server thread native_id:", threading.get_native_id())

    app = create_app(controller)
    # host=0.0.0.0 so phones on LAN can reach it
    app.run(host="0.0.0.0", port=7654, threaded=True)
