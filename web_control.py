#!/usr/bin/env python3
"""
web_control.py  (corre en la Raspberry Pi)

Panel web de control. Es el único punto de entrada para controlar la
GoPro: tanto quien lo use desde el navegador como quien lo use por SSH
(vía `curl` contra estos mismos endpoints) pasan por acá, así el
`state_lock` evita condiciones de carrera entre los dos canales.

Ya NO sube nada a ningún lado: al terminar una grabación deja el path
del archivo disponible en /api/last_file para que gopro_file_transfer.py
(que corre en la computadora) lo detecte y lo baje directo de la GoPro.
El borrado de la SD se expone como endpoint separado, para llamarlo solo
después de confirmar la descarga.

Run con:
    source ~/gopro-venv/bin/activate
    python3 web_control.py

Visitar, desde cualquier dispositivo en la misma red que la Raspi:
    http://<ip-de-la-raspi>:5000

Requires:
    pip install flask
(más lo que ya pide gopro_controller.py)

NOTA DE SEGURIDAD: no tiene login. Está bien para una red doméstica/privada;
no exponer este puerto a internet sin agregar autenticación.
"""

import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template_string, request

import gopro_controller as gc

# Werkzeug (el servidor HTTP de Flask) loguea una línea por cada request.
# Como /api/status se pollea cada 2s desde el panel web y cada 5s desde
# gopro_file_transfer.py, eso solo es ruido en uso normal. Lo subimos a
# WARNING para que solo aparezcan errores reales del servidor.
logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Estado compartido
# --------------------------------------------------------------------------- #

state = {
    "phase": "idle",  # idle | connecting | recording | capturing_photo | stopping | ready_to_transfer | error
    "message": "Listo.",
    "gopro_connected": False,       # reflejado en vivo por el ConnectionMonitor
    "battery": None,
    "sd_free": None,
    "recording_started_at": None,
    "last_recorded_file": None,     # path "FOLDER/FILE.MP4" o ".JPG" pendiente de transferir
    "last_transfer_status": None,   # None | "pending" | "transferred"
    "last_photo_file": None,
    "error": None,
}
state_lock = threading.Lock()
session_thread: Optional[threading.Thread] = None

_camera_lock = threading.Lock()  # evita comandos concurrentes a la cámara (ej. foto durante grabación)


def set_state(**kwargs):
    with state_lock:
        state.update(kwargs)


# --------------------------------------------------------------------------- #
# Hooks de sesión externa
# --------------------------------------------------------------------------- #
# Por default, este módulo maneja la grabación él mismo (run_recording_session).
# Pero cuando lo levanta main_final.py, el dueño de la cámara es main_final:
# el panel NO debe mandarle comandos a la GoPro por su cuenta, porque serían
# comandos duplicados sobre el mismo hardware. Registrando estos hooks, los
# botones Iniciar/Detener delegan en main_final y este módulo queda solo como
# interfaz + estado.

_session_hooks = {"start": None, "stop": None}


def register_session_hooks(start, stop) -> None:
    """Llamado por main_final.py para tomar el control de la sesión."""
    _session_hooks["start"] = start
    _session_hooks["stop"] = stop


def publish_recorded_file(camera_file_path: str) -> None:
    """
    main_final llama a esto cuando la GoPro terminó de grabar y ya sabe el
    path del archivo en la SD. Es lo que hace que gopro_file_transfer.py
    (en la computadora) lo vea por /api/last_file y lo baje.
    """
    if not camera_file_path:
        set_state(phase="error", message="La grabación terminó pero no se encontró el archivo.",
                  error="no_file_found")
        return
    set_state(
        phase="ready_to_transfer",
        message=f"Grabación lista para transferir: {camera_file_path}",
        last_recorded_file=camera_file_path,
        last_transfer_status="pending",
    )


def _run_external_start():
    """Corre el hook de arranque de main_final en su propio thread."""
    try:
        set_state(phase="connecting", message="Arrancando el sistema...", error=None)
        if _session_hooks["start"]():
            set_state(phase="recording", message="Grabando...",
                      recording_started_at=datetime.now().isoformat())
        else:
            set_state(phase="error", message="El arranque del sistema falló.",
                      error="start_failed")
    except Exception as e:
        set_state(phase="error", message=f"Error al arrancar: {e}", error=str(e))


def _run_external_stop():
    """Corre el hook de parada de main_final en su propio thread."""
    try:
        set_state(phase="stopping", message="Deteniendo el sistema...")
        _session_hooks["stop"]()
        # main_final publica el archivo con publish_recorded_file(); si no
        # hubo archivo (ej. corrida --sin-gopro), volvemos a idle.
        with state_lock:
            if state["phase"] == "stopping":
                state["phase"] = "idle"
                state["message"] = "Listo."
    except Exception as e:
        set_state(phase="error", message=f"Error al detener: {e}", error=str(e))


def _on_connection_change(connected: bool):
    set_state(gopro_connected=connected)
    if not connected:
        with state_lock:
            if state["phase"] == "recording":
                # No tocamos la grabación (puede seguir en la SD), solo avisamos.
                state["message"] = "Se perdió la conexión con la GoPro durante la grabación."


connection_monitor = gc.ConnectionMonitor(on_change=_on_connection_change)


# --------------------------------------------------------------------------- #
# Sesión de grabación
# --------------------------------------------------------------------------- #

def run_recording_session():
    try:
        set_state(phase="connecting", message="Esperando Wi-Fi de la GoPro...", error=None)
        with _camera_lock:
            gopro = gc.wait_for_gopro()
            set_state(gopro_connected=True, message="Conectado. Sincronizando reloj y estado...")

            gc.sync_camera_clock(gopro)
            battery = gc.get_battery_level(gopro)
            sd_free = gc.get_remaining_sd_capacity(gopro)
            set_state(battery=battery, sd_free=sd_free)

            set_state(phase="recording", message="Grabando...",
                      recording_started_at=datetime.now().isoformat())
            gc.start_recording(gopro)

        # Fuera del lock mientras graba: así /api/photo o /api/stop pueden
        # llegar; el lock se vuelve a tomar para cada comando puntual.
        while True:
            with state_lock:
                phase = state["phase"]
            if phase != "recording":
                break
            gc.time.sleep(1)  # type: ignore[attr-defined]
            if gc.stop_requested():
                break

        with _camera_lock:
            set_state(phase="stopping", message="Deteniendo grabación...")
            camera_file = gc.stop_recording(gopro)

        if not camera_file:
            set_state(phase="error", message="La grabación terminó pero no se encontró el archivo en la cámara.",
                      error="no_file_found")
            return

        set_state(
            phase="ready_to_transfer",
            message=f"Grabación lista para transferir: {camera_file}",
            last_recorded_file=camera_file,
            last_transfer_status="pending",
        )

    except Exception as e:
        set_state(phase="error", message=f"Error inesperado: {e}", error=str(e))


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #

@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(dict(state))


@app.route("/api/start", methods=["POST"])
def api_start():
    global session_thread
    with state_lock:
        if state["phase"] in ("connecting", "recording", "stopping", "capturing_photo"):
            return jsonify({"ok": False, "message": "Ya hay una sesión en curso."}), 409

    gc.clear_stop_flag()

    if _session_hooks["start"] is not None:
        # main_final es el dueño de la cámara: delegamos en él.
        session_thread = threading.Thread(target=_run_external_start, daemon=True)
    else:
        session_thread = threading.Thread(target=run_recording_session, daemon=True)

    session_thread.start()
    return jsonify({"ok": True, "message": "Sesión iniciada."})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state_lock:
        if state["phase"] != "recording":
            return jsonify({"ok": False, "message": "No hay ninguna grabación activa."}), 409

    if _session_hooks["stop"] is not None:
        threading.Thread(target=_run_external_stop, daemon=True).start()
        return jsonify({"ok": True, "message": "Stop solicitado."})

    gc.STOP_FLAG_FILE.touch()
    return jsonify({"ok": True, "message": "Stop solicitado."})


@app.route("/api/photo", methods=["POST"])
def api_photo():
    """Saca una foto. Se bloquea si hay una grabación en curso, para no
    mandarle dos comandos de modo distintos a la cámara a la vez."""
    with state_lock:
        if state["phase"] == "recording":
            return jsonify({"ok": False, "message": "No se puede sacar una foto mientras se graba."}), 409
        if state["phase"] in ("connecting", "stopping", "capturing_photo"):
            return jsonify({"ok": False, "message": "Ya hay una operación en curso."}), 409

    def _take_photo_job():
        try:
            set_state(phase="capturing_photo", message="Sacando foto...", error=None)
            with _camera_lock:
                gopro = gc.wait_for_gopro()
                photo_file = gc.take_photo(gopro)
            if photo_file:
                set_state(
                    phase="ready_to_transfer",
                    message=f"Foto lista para transferir: {photo_file}",
                    last_recorded_file=photo_file,
                    last_photo_file=photo_file,
                    last_transfer_status="pending",
                )
            else:
                set_state(phase="error", message="No se pudo confirmar la foto en la cámara.",
                          error="photo_not_found")
        except Exception as e:
            set_state(phase="error", message=f"Error sacando foto: {e}", error=str(e))

    threading.Thread(target=_take_photo_job, daemon=True).start()
    return jsonify({"ok": True, "message": "Foto solicitada."})


@app.route("/api/last_file")
def api_last_file():
    """Endpoint que consulta gopro_file_transfer.py (desde la compu) para
    saber si hay un archivo nuevo listo para bajar."""
    with state_lock:
        return jsonify({
            "camera_file_path": state["last_recorded_file"],
            "transfer_status": state["last_transfer_status"],
        })


@app.route("/api/delete_file", methods=["POST"])
def api_delete_file():
    """
    Llamado por la computadora SOLO después de confirmar que la descarga
    terminó bien (tamaño verificado). Body JSON: {"camera_file_path": "..."}.
    """
    payload = request.get_json(silent=True) or {}
    camera_file_path = payload.get("camera_file_path")
    if not camera_file_path:
        return jsonify({"ok": False, "message": "Falta camera_file_path."}), 400

    with _camera_lock:
        gopro = gc.wait_for_gopro()
        deleted = gc.delete_file_from_camera(gopro, camera_file_path)

    if deleted:
        with state_lock:
            if state["last_recorded_file"] == camera_file_path:
                state["last_transfer_status"] = "transferred"
                state["message"] = f"{camera_file_path} transferido y borrado de la SD."
        return jsonify({"ok": True})
    return jsonify({"ok": False, "message": "El borrado en la cámara falló."}), 500


@app.route("/api/reset", methods=["POST"])
def api_reset():
    set_state(phase="idle", message="Listo.", error=None, last_recorded_file=None,
              last_transfer_status=None, recording_started_at=None)
    return jsonify({"ok": True})


@app.route("/api/debug/connection")
def api_debug_connection():
    connected = gc.is_connected_to_gopro_wifi()
    visible = gc.gopro_ssid_visible() if not connected else None
    reachable = gc.gopro_api_reachable() if connected else False
    return jsonify({
        "connected_to_gopro_wifi": connected,
        "gopro_ssid_visible": visible,
        "gopro_api_reachable": reachable,
        "monitor_last_known_state": connection_monitor.connected,
    })


@app.route("/api/debug/battery")
def api_debug_battery():
    if not gc.is_connected_to_gopro_wifi():
        return jsonify({"ok": False, "message": "No conectado a la Wi-Fi de la GoPro."}), 409
    try:
        gopro = gc.create_camera()
        return jsonify({
            "ok": True,
            "battery_percent": gc.get_battery_level(gopro),
            "sd_free_bytes": gc.get_remaining_sd_capacity(gopro),
        })
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/api/debug/media")
def api_debug_media():
    if not gc.is_connected_to_gopro_wifi():
        return jsonify({"ok": False, "message": "No conectado a la Wi-Fi de la GoPro."}), 409
    try:
        gopro = gc.create_camera()
        media = gc.list_all_media(gopro)
        return jsonify({"ok": True, "media": media})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/api/logs")
def api_logs():
    try:
        lines = Path(gc.LOG_FILE).read_text().splitlines()
        return jsonify({"ok": True, "lines": lines[-100:]})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


# --------------------------------------------------------------------------- #
# Front-end: HTML/JS de una sola página
# --------------------------------------------------------------------------- #

PAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GoPro Control</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 480px; margin: 0 auto; padding: 20px; background: #111; color: #eee; }
  h1 { font-size: 20px; }
  .card { background: #1c1c1c; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
  .phase { font-size: 22px; font-weight: bold; text-transform: capitalize; }
  .phase.idle { color: #888; }
  .phase.connecting, .phase.recording, .phase.capturing_photo, .phase.stopping { color: #f5a623; }
  .phase.ready_to_transfer { color: #4caf50; }
  .phase.error { color: #e5484d; }
  .message { color: #ccc; margin-top: 6px; font-size: 14px; }
  button { width: 100%; padding: 14px; font-size: 16px; border: none; border-radius: 10px; margin-top: 8px; cursor: pointer; }
  .start { background: #4caf50; color: white; }
  .stop { background: #e5484d; color: white; }
  .photo { background: #2f80ed; color: white; }
  .secondary { background: #333; color: #eee; }
  .row { display: flex; gap: 10px; }
  .row button { flex: 1; }
  .stat { display: flex; justify-content: space-between; font-size: 14px; padding: 4px 0; border-bottom: 1px solid #2a2a2a; }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
  .dot.on { background: #4caf50; } .dot.off { background: #e5484d; }
  #logbox { background: #000; color: #0f0; font-family: monospace; font-size: 11px; padding: 10px; border-radius: 8px; height: 200px; overflow-y: auto; white-space: pre-wrap; }
</style>
</head>
<body>
  <h1>GoPro Control Panel</h1>

  <div class="card">
    <div class="stat"><span><span class="dot" id="connDot"></span>Conexión con la GoPro</span><span id="connText">-</span></div>
  </div>

  <div class="card">
    <div class="phase idle" id="phase">idle</div>
    <div class="message" id="message">Cargando...</div>
  </div>

  <div class="card">
    <button class="start" id="startBtn" onclick="startSession()">Iniciar grabación</button>
    <button class="stop" id="stopBtn" onclick="stopSession()">Detener grabación</button>
    <button class="photo" id="photoBtn" onclick="takePhoto()">Sacar foto</button>
    <button class="secondary" id="resetBtn" onclick="resetSession()">Reset</button>
  </div>

  <div class="card">
    <div class="stat"><span>Batería</span><span id="battery">-</span></div>
    <div class="stat"><span>Espacio libre SD</span><span id="sd">-</span></div>
    <div class="stat"><span>Último archivo</span><span id="lastFile">-</span></div>
    <div class="stat"><span>Estado transferencia</span><span id="transferStatus">-</span></div>
  </div>

  <div class="card">
    <div class="row">
      <button class="secondary" onclick="checkConnection()">Chequear Wi-Fi</button>
      <button class="secondary" onclick="checkBattery()">Chequear batería</button>
    </div>
    <div class="row" style="margin-top:8px">
      <button class="secondary" onclick="checkMedia()">Listar archivos cámara</button>
      <button class="secondary" onclick="loadLogs()">Refrescar logs</button>
    </div>
  </div>

  <div class="card">
    <div id="logbox">Los logs aparecen acá...</div>
  </div>

<script>
async function refreshStatus() {
  const r = await fetch('/api/status');
  const s = await r.json();
  document.getElementById('phase').textContent = s.phase;
  document.getElementById('phase').className = 'phase ' + s.phase;
  document.getElementById('message').textContent = s.message;

  document.getElementById('connDot').className = 'dot ' + (s.gopro_connected ? 'on' : 'off');
  document.getElementById('connText').textContent = s.gopro_connected ? 'Conectada' : 'Desconectada';

  document.getElementById('battery').textContent = s.battery !== null ? s.battery + '%' : '-';
  document.getElementById('sd').textContent = s.sd_free !== null ? s.sd_free : '-';
  document.getElementById('lastFile').textContent = s.last_recorded_file || '-';
  document.getElementById('transferStatus').textContent = s.last_transfer_status || '-';

  const busy = ['connecting', 'recording', 'stopping', 'capturing_photo'].includes(s.phase);
  document.getElementById('startBtn').disabled = busy;
  document.getElementById('stopBtn').disabled = s.phase !== 'recording';
  document.getElementById('photoBtn').disabled = busy;
  [ 'startBtn', 'photoBtn' ].forEach(id => document.getElementById(id).style.opacity = busy ? 0.5 : 1);
  document.getElementById('stopBtn').style.opacity = s.phase !== 'recording' ? 0.5 : 1;
}

async function startSession() { await fetch('/api/start', { method: 'POST' }); refreshStatus(); }
async function stopSession() { await fetch('/api/stop', { method: 'POST' }); refreshStatus(); }
async function takePhoto() { await fetch('/api/photo', { method: 'POST' }); refreshStatus(); }
async function resetSession() { await fetch('/api/reset', { method: 'POST' }); refreshStatus(); }

async function checkConnection() {
  const r = await fetch('/api/debug/connection');
  log('Chequeo de conexión: ' + JSON.stringify(await r.json()));
}
async function checkBattery() {
  const r = await fetch('/api/debug/battery');
  log('Chequeo de batería: ' + JSON.stringify(await r.json()));
}
async function checkMedia() {
  const r = await fetch('/api/debug/media');
  log('Media en la cámara: ' + JSON.stringify(await r.json()));
}
async function loadLogs() {
  const r = await fetch('/api/logs');
  const d = await r.json();
  if (d.ok) {
    document.getElementById('logbox').textContent = d.lines.join('\\n');
    document.getElementById('logbox').scrollTop = document.getElementById('logbox').scrollHeight;
  }
}
function log(msg) {
  const box = document.getElementById('logbox');
  box.textContent += '\\n' + msg;
  box.scrollTop = box.scrollHeight;
}

refreshStatus();
setInterval(refreshStatus, 2000);
</script>
</body>
</html>
"""

# --------------------------------------------------------------------------- #
# Inicialización / main
# --------------------------------------------------------------------------- #
#
# Acá NO hay un loop manual tipo `while True: ...`. El "loop" del programa
# son dos cosas que ya corren solas una vez arrancadas:
#   1. connection_monitor: tiene su propio thread con su propio while
#      interno (ver ConnectionMonitor._loop en gopro_controller.py), que
#      chequea la conexión cada CONNECTION_CHECK_INTERVAL segundos.
#   2. app.run(): el loop de eventos de Flask, que atiende cada request
#      HTTP (botones del panel, polls de gopro_file_transfer.py) a medida
#      que llegan.
#
# Por eso todo lo que es "chequeo único al arrancar" va ANTES de esas dos
# líneas (inicialización), y todo lo que es "chequeo continuo" se delega
# a esos dos loops ya existentes, en vez de escribir un tercer loop nuevo.

def startup_checks() -> None:
    """
    Chequeos de arranque, una sola vez, antes de aceptar requests:
    - Confirma que la GoPro está conectada y respondiendo (bloquea acá,
      no en medio de un request del usuario).
    - Sincroniza el reloj de la cámara.
    - Deja batería y espacio de SD ya cargados en el estado, para que el
      panel muestre datos reales desde el primer refresh en vez de "-".
    """
    print("Buscando la GoPro...", flush=True)
    gopro = gc.wait_for_gopro()  # bloquea hasta que la Wi-Fi + API respondan
    print("GoPro conectada.", flush=True)

    gc.sync_camera_clock(gopro)
    battery = gc.get_battery_level(gopro)
    sd_free = gc.get_remaining_sd_capacity(gopro)
    set_state(gopro_connected=True, battery=battery, sd_free=sd_free)

    print(f"Batería: {battery}% | SD libre: {sd_free}", flush=True)


def main():
    print("=== web_control: iniciando ===", flush=True)

    # 1) Chequeos de arranque, bloqueantes, antes de levantar el server.
    #    Si la GoPro todavía no está prendida, esto espera acá (con sus
    #    propios reintentos, ver wait_for_gopro) en vez de levantar el
    #    panel mostrando datos falsos.
    startup_checks()

    # 2) A partir de acá arrancan los dos loops continuos:
    connection_monitor.start()  # loop propio en su thread: vigila la conexión
    print("Panel disponible en http://0.0.0.0:5000", flush=True)
    app.run(host="0.0.0.0", port=5000, debug=False)  # loop de eventos de Flask


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.", flush=True)
        sys.exit(0)
