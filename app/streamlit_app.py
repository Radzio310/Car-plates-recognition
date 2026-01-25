from __future__ import annotations

import io
import re
import sys
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from contextlib import contextmanager
from textwrap import dedent
from ultralytics import YOLO

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anpr.pipeline import ANPRPipeline
from anpr.utils import normalize_plate, validate_plate
from anpr.db import PlateDB

EMERGENCY_MODEL_PATH = Path("runs/train/emergency_classification_v2/weights/best.pt")
EMERGENCY_THRESHOLD = 0.60  # możesz zmienić

# =========================================================
# CONFIG / PAGE
# =========================================================

st.set_page_config(page_title="ANPR – kontrola bramy", layout="wide")

APP_CSS = dedent(
    """
<style>
:root{
  --bg0:#070707;
  --bg1:#0d0b08;

  --card0: rgba(12,11,10,0.92);
  --card1: rgba(18,16,13,0.72);

  --border: rgba(216,199,163,0.18);
  --border2: rgba(216,199,163,0.28);

  --text:#f3efe7;
  --muted:#b9b0a3;

  --beige:#d8c7a3;
  --beige2:#cbb78c;

  --ok:#2ecc71;
  --bad:#ff4d4d;
  --warn:#f4b740;
  --info:#8bb6ff;

  --shadow: 0 18px 55px rgba(0,0,0,0.55);
}

.stApp {
  background:
    radial-gradient(1100px 520px at 50% -12%, rgba(216,199,163,0.12), transparent 60%),
    radial-gradient(900px 360px at 10% 12%, rgba(203,183,140,0.10), transparent 60%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
}

.block-container { padding-top: 1.25rem; padding-bottom: 2.2rem; }
section[data-testid="stSidebar"] > div{
  background:
    radial-gradient(900px 300px at 50% 0%, rgba(216,199,163,0.12), transparent 60%),
    linear-gradient(180deg, rgba(0,0,0,0.92), rgba(13,11,8,0.97));
  border-right: 1px solid rgba(216,199,163,0.16);
}

h1,h2,h3,h4,p,span,div,label { color: var(--text); }
small, .stCaption { color: var(--muted) !important; }

.card{
  background: linear-gradient(180deg, var(--card0), var(--card1));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: var(--shadow);
}

.card.soft{
  background: linear-gradient(180deg, rgba(12,11,10,0.72), rgba(12,11,10,0.55));
}

.hrSoft{
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(216,199,163,0.22), transparent);
  margin: 12px 0;
  border-radius: 999px;
}

.badge{
  display:inline-flex; align-items:center; gap:8px;
  padding:7px 11px;
  border-radius: 999px;
  border: 1px solid rgba(216,199,163,0.22);
  background: rgba(0,0,0,0.25);
  font-weight: 900;
  font-size: 12px;
  color: var(--text);
}
.badge.ok{ border-color: rgba(46,204,113,0.45); background: rgba(46,204,113,0.10); }
.badge.bad{ border-color: rgba(255,77,77,0.45); background: rgba(255,77,77,0.10); }
.badge.warn{ border-color: rgba(244,183,64,0.45); background: rgba(244,183,64,0.10); }
.badge.info{ border-color: rgba(139,182,255,0.45); background: rgba(139,182,255,0.10); }

.sectionTitle{
  font-size: 18px;
  font-weight: 1000;
  letter-spacing: .2px;
  margin: 0 0 10px 0;
}
.subtle{
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
}

/* Bigger uploader */
div[data-testid="stFileUploader"] > section{
  padding: 18px !important;
  border: 1px dashed rgba(216,199,163,0.42) !important;
  border-radius: 18px !important;
  background: rgba(0,0,0,0.20) !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]{
  min-height: 170px !important;
}
div[data-testid="stFileUploader"] section:hover{
  border-color: rgba(216,199,163,0.70) !important;
  background: rgba(0,0,0,0.26) !important;
}

/* Preview boxes */
.previewBox{
  border: 1px solid rgba(216,199,163,0.18);
  background: rgba(0,0,0,0.20);
  border-radius: 18px;
  padding: 12px;
}
.previewLabel{
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 8px;
}

/* Gate animation */
.gateWrap{
  border-radius: 18px;
  border: 1px solid rgba(216,199,163,0.22);
  background:
    radial-gradient(1000px 280px at 50% -20%, rgba(216,199,163,0.16), transparent 60%),
    linear-gradient(180deg, rgba(0,0,0,0.25), rgba(12,11,10,0.55));
  padding: 14px;
  overflow: hidden;
}
.gateHeader{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap: 10px;
  margin-bottom: 10px;
}
.gate{
  position: relative;
  height: 180px;
  border-radius: 16px;
  border: 1px solid rgba(216,199,163,0.22);
  background: linear-gradient(180deg, rgba(0,0,0,0.55), rgba(0,0,0,0.18));
  overflow:hidden;
}
.gateFloor{
  position:absolute; left:0; right:0; bottom:0; height:48px;
  background: linear-gradient(90deg, rgba(216,199,163,0.06), rgba(216,199,163,0.02));
  border-top: 1px solid rgba(216,199,163,0.18);
}
.gateLeaf{
  position:absolute; top:0; bottom:48px;
  width: 50%;
  background:
    linear-gradient(180deg, rgba(216,199,163,0.10), rgba(216,199,163,0.04)),
    repeating-linear-gradient(90deg, rgba(216,199,163,0.16) 0px, rgba(216,199,163,0.16) 6px,
                                  rgba(216,199,163,0.05) 6px, rgba(216,199,163,0.05) 14px);
  border: 1px solid rgba(216,199,163,0.20);
  box-shadow: inset 0 0 0 1px rgba(0,0,0,0.40);
  transform-origin: center left;
  transform: perspective(900px) rotateY(0deg);
}
.gateLeaf.right { left:50%; transform-origin: center right; }
.gateLeaf.left  { left:0; }

.gateStateLabel{
  position:absolute; left: 14px; top: 14px;
  padding: 7px 11px;
  border-radius: 999px;
  border: 1px solid rgba(216,199,163,0.22);
  background: rgba(0,0,0,0.28);
  font-weight: 1000;
  font-size: 12px;
}

@keyframes gateOpenLeft {
  0% { transform: perspective(900px) rotateY(0deg); }
  100% { transform: perspective(900px) rotateY(-78deg); }
}
@keyframes gateOpenRight {
  0% { transform: perspective(900px) rotateY(0deg); }
  100% { transform: perspective(900px) rotateY(78deg); }
}
@keyframes gateCloseLeft {
  0% { transform: perspective(900px) rotateY(-12deg); }
  100% { transform: perspective(900px) rotateY(0deg); }
}
@keyframes gateCloseRight {
  0% { transform: perspective(900px) rotateY(12deg); }
  100% { transform: perspective(900px) rotateY(0deg); }
}

.gate.open  .gateLeaf.left  { animation: gateOpenLeft 720ms cubic-bezier(.2,.9,.2,1) forwards; }
.gate.open  .gateLeaf.right { animation: gateOpenRight 720ms cubic-bezier(.2,.9,.2,1) forwards; }

.gate.closed .gateLeaf.left  { animation: gateCloseLeft 520ms ease-in-out forwards; }
.gate.closed .gateLeaf.right { animation: gateCloseRight 520ms ease-in-out forwards; }

.gate.idle .gateLeaf.left,
.gate.idle .gateLeaf.right { opacity: .88; }

.gateGlow{
  position:absolute; inset:-130px -130px auto -130px; height: 260px;
  filter: blur(2px);
  opacity: 0;
  transition: opacity 360ms ease-in-out;
}
.gate.open .gateGlow{ opacity: 1; background: radial-gradient(closest-side, rgba(46,204,113,0.22), transparent 70%); }
.gate.closed .gateGlow{ opacity: 1; background: radial-gradient(closest-side, rgba(255,77,77,0.18), transparent 70%); }

/* Fullscreen loader */
.loaderOverlay{
  position: fixed;
  inset: 0;
  z-index: 999999;
  display:flex;
  align-items:center;
  justify-content:center;
  background:
    radial-gradient(1100px 520px at 50% -10%, rgba(216,199,163,0.14), transparent 60%),
    linear-gradient(180deg, rgba(0,0,0,0.88), rgba(13,11,8,0.96));
  backdrop-filter: blur(10px);
}
.loaderCard{
  width: min(760px, 92vw);
  border-radius: 20px;
  border: 1px solid rgba(216,199,163,0.22);
  background: linear-gradient(180deg, rgba(12,11,10,0.88), rgba(0,0,0,0.40));
  box-shadow: 0 26px 78px rgba(0,0,0,0.62);
  padding: 16px;
}
.loaderTop{
  display:flex; align-items:flex-start; justify-content:space-between; gap: 12px;
  margin-bottom: 12px;
}
.loaderTitle{ font-weight: 1000; font-size: 18px; letter-spacing: .25px; }
.loaderSub{ color: var(--muted); font-size: 13px; margin-top: 4px; line-height: 1.45; }

.road{
  position: relative;
  height: 160px;
  border-radius: 16px;
  border: 1px solid rgba(216,199,163,0.18);
  background: linear-gradient(180deg, rgba(0,0,0,0.32), rgba(0,0,0,0.14));
  overflow:hidden;
}
.road:before{
  content:"";
  position:absolute; left:-30%; right:-30%;
  top: 56%;
  height: 3px;
  background: repeating-linear-gradient(90deg,
      rgba(216,199,163,0.0) 0px,
      rgba(216,199,163,0.0) 24px,
      rgba(216,199,163,0.26) 24px,
      rgba(216,199,163,0.26) 44px
  );
  animation: dashMove 0.7s linear infinite;
  opacity: .9;
}
@keyframes dashMove{
  0% { transform: translateX(0); }
  100% { transform: translateX(120px); }
}

.car{
  position:absolute;
  left: -200px;
  bottom: 44px;
  width: 190px;
  height: 68px;
  animation: carMove 1.35s ease-in-out infinite;
}
@keyframes carMove{
  0%   { transform: translateX(0); opacity: 0; }
  10%  { opacity: 1; }
  50%  { transform: translateX(calc(46vw - 90px)); }
  90%  { opacity: 1; }
  100% { transform: translateX(calc(100vw + 220px)); opacity: 0; }
}
.carBody{
  position:absolute; left: 16px; top: 22px;
  width: 150px; height: 34px;
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(216,199,163,0.95), rgba(203,183,140,0.70));
  border: 1px solid rgba(216,199,163,0.22);
  box-shadow: inset 0 0 0 1px rgba(0,0,0,0.25);
}
.carCabin{
  position:absolute; left: 58px; top: 10px;
  width: 72px; height: 26px;
  border-radius: 12px;
  background: linear-gradient(180deg, rgba(0,0,0,0.35), rgba(0,0,0,0.20));
  border: 1px solid rgba(216,199,163,0.20);
}
.headLight{
  position:absolute; right: 6px; top: 26px;
  width: 10px; height: 8px;
  border-radius: 6px;
  background: rgba(255,255,255,0.76);
  box-shadow: 0 0 14px rgba(255,255,255,0.35);
}
.wheel{
  position:absolute; bottom: 8px;
  width: 22px; height: 22px;
  border-radius: 999px;
  background: linear-gradient(180deg, rgba(0,0,0,0.70), rgba(0,0,0,0.45));
  border: 1px solid rgba(216,199,163,0.18);
  box-shadow: inset 0 0 0 3px rgba(216,199,163,0.18);
  animation: spin 0.25s linear infinite;
}
@keyframes spin{ to { transform: rotate(360deg); } }
.wheel.left { left: 36px; }
.wheel.right{ left: 134px; }

.loaderDots{ display:inline-flex; gap:6px; align-items:center; margin-top: 10px; }
.dot{
  width: 7px; height: 7px; border-radius: 999px;
  background: rgba(216,199,163,0.35);
  animation: dotPulse 0.9s ease-in-out infinite;
}
.dot:nth-child(2){ animation-delay: .15s; }
.dot:nth-child(3){ animation-delay: .3s; }
@keyframes dotPulse{
  0%,100%{ transform: scale(1); opacity: .6; }
  50%{ transform: scale(1.45); opacity: 1; }
}

/* Floating "See results" button */
.fabResults{
  position: fixed;
  left: 50%;
  transform: translateX(-50%);
  bottom: 26px;
  z-index: 99999;
  text-decoration: none;
}
.fabBtn{
  display:inline-flex;
  align-items:center;
  gap:10px;
  padding: 12px 16px;
  border-radius: 999px;
  border: 1px solid rgba(216,199,163,0.28);
  background: rgba(0,0,0,0.55);
  box-shadow: 0 18px 45px rgba(0,0,0,0.55);
  color: var(--text);
  font-weight: 1000;
  letter-spacing: .2px;
}
.fabBtn:hover{
  border-color: rgba(216,199,163,0.55);
  background: rgba(0,0,0,0.62);
}
.fabArrow{
  width: 26px; height: 26px;
  border-radius: 999px;
  display:flex; align-items:center; justify-content:center;
  border: 1px solid rgba(216,199,163,0.22);
  background: rgba(216,199,163,0.10);
}

/* Admin tiles */
.tile{
  border: 1px solid rgba(216,199,163,0.18);
  background: rgba(0,0,0,0.22);
  border-radius: 16px;
  padding: 10px 12px;
}
.tilePlate{ font-weight: 1000; letter-spacing: .65px; font-size: 14px; }
.tileMeta{ color: var(--muted); font-size: 12px; }

@keyframes pulseNew{
  0%{ box-shadow: 0 0 0 rgba(216,199,163,0.0); transform: translateY(0); }
  50%{ box-shadow: 0 0 28px rgba(216,199,163,0.22); transform: translateY(-1px); }
  100%{ box-shadow: 0 0 0 rgba(216,199,163,0.0); transform: translateY(0); }
}
.tileNew{
  border-color: rgba(216,199,163,0.70);
  background: rgba(216,199,163,0.08);
  animation: pulseNew 900ms ease-in-out 2;
}
</style>
"""
)
st.markdown(APP_CSS, unsafe_allow_html=True)


# =========================================================
# HELPERS: PIPELINE + DB (zgodne z Twoją konsolą)
# =========================================================

@st.cache_resource
def load_emergency_model_cached(model_path: str) -> YOLO:
    return YOLO(model_path)

def predict_emergency_on_rgb(img_rgb: np.ndarray) -> tuple[bool, float, str]:
    """
    Returns:
      is_emergency (bool),
      confidence (float),
      predicted_class_name (str)
    """
    if not EMERGENCY_MODEL_PATH.exists():
        return False, 0.0, "model_missing"

    model = load_emergency_model_cached(str(EMERGENCY_MODEL_PATH))

    # Ultralytics działa na RGB ok, ale przyjmie też np. ndarray
    r = model.predict(source=img_rgb, verbose=False)[0]

    pred_idx = int(r.probs.top1)
    conf = float(r.probs.top1conf)
    pred_name = r.names[pred_idx]  # np. "emergency" / "non_emergency"

    is_emergency = (pred_name == "emergency" and conf >= EMERGENCY_THRESHOLD)
    return is_emergency, conf, pred_name


@st.cache_resource
def load_pipeline_cached(config_path: str) -> ANPRPipeline:
    return ANPRPipeline(config_path=config_path)

def get_db_config(cfg_path: str) -> dict:
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("access_control", {})

def get_db_instance(cfg_path: str) -> PlateDB:
    conf = get_db_config(cfg_path)
    return PlateDB(
        host=conf.get("host", "localhost"),
        database=conf.get("database", "anpr_db"),
        user=conf.get("user", "user"),
        password=conf.get("password", "password123"),
        port=int(conf.get("port", 5432))
    )

def _run_db_manage(args: List[str], db_path: Optional[str] = None) -> subprocess.CompletedProcess:
    """
    Dokładnie jak w konsoli:
      python -m scripts.db_manage add --plate "SK12345"
    + próbujemy dopiąć --db <path> jeśli skrypt wspiera.
    """
    base = [sys.executable, "-m", "scripts.db_manage"]

    if db_path:
        try_cmd = base + args + ["--db", db_path]
        p = subprocess.run(try_cmd, cwd=str(ROOT), capture_output=True, text=True)
        if "unrecognized arguments: --db" in (p.stderr or "") or "unrecognized arguments: --db" in (p.stdout or ""):
            p = subprocess.run(base + args, cwd=str(ROOT), capture_output=True, text=True)
        return p

    return subprocess.run(base + args, cwd=str(ROOT), capture_output=True, text=True)

def db_add_plate(plate_norm: str) -> None:
    p = subprocess.run([sys.executable, "-m", "scripts.db_manage", "add", "--plate", plate_norm], cwd=str(ROOT))
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "").strip() or "db_manage add failed")

def _looks_like_plate(s: str) -> bool:
    s = (s or "").strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return 4 <= len(s) <= 10

def db_list_plates(cfg_path: str) -> List[str]:
    try:
        db = get_db_instance(cfg_path)
        return db.list()
    except Exception as e:
        st.error(f"Błąd listowania bazy: {e}")
        return []

def db_remove_plate(cfg_path: str, plate_norm: str) -> None:
    db = get_db_instance(cfg_path)
    try:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM trusted_plates WHERE plate = %s", (plate_norm,))
            conn.commit()
    except Exception as e:
        raise RuntimeError(f"Błąd usuwania: {e}")


# =========================================================
# UI HELPERS
# =========================================================

def friendly_confidence(conf: float) -> str:
    if conf >= 0.85:
        return "Wysoka"
    if conf >= 0.60:
        return "Średnia"
    if conf > 0.0:
        return "Niska"
    return "Brak"

def conf_badge_class(conf: float) -> str:
    if conf >= 0.85:
        return "ok"
    if conf >= 0.60:
        return "warn"
    return "bad"

def gate_model(access_granted: Optional[bool], plate: str, valid: bool, detected: bool) -> Tuple[str, str, str]:
    """
    return: (state_class, label, subtitle)
    state_class: idle/open/closed
    """

    if not detected:
        return (
            "idle",
            "Brama: BRAK DANYCH",
            "Nie wykryto pojazdu – system nie podejmuje decyzji."
        )

    # 🚑 POJAZD UPRZYWILEJOWANY
    if plate == "EMERGENCY" and access_granted is True:
        return (
            "open",
            "Brama: OTWARTA",
            "🚑 Pojazd uprzywilejowany – brama otwarta automatycznie."
        )

    if not valid or not plate:
        return (
            "idle",
            "Brama: BRAK DECYZJI",
            "Tablica wykryta, ale odczyt jest niepewny / niepełny – nie otwieramy."
        )

    if access_granted is True:
        return (
            "open",
            "Brama: OTWARTA",
            "Numer jest na liście zaufanych – wjazd dozwolony."
        )

    return (
        "closed",
        "Brama: ZAMKNIĘTA",
        "Numer nie jest na liście zaufanych – wjazd zablokowany."
    )

def render_gate(access_granted: Optional[bool], plate: str, valid: bool, detected: bool, animate: bool) -> None:
    state, label, subtitle = gate_model(access_granted, plate, valid, detected)

    # animate=True -> użyj open/closed (z keyframes)
    # animate=False -> pokaż idle (spokojnie)
    gate_class = state if animate else "idle"

    badge = "info"
    if state == "open":
        badge = "ok"
    elif state == "closed":
        badge = "bad"
    elif state == "idle" and detected:
        badge = "warn"

    st.markdown(
        dedent(
            f"""
            <div class="gateWrap">
              <div class="gateHeader">
                <div>
                  <div class="badge {badge}">{label}</div>
                  <div class="subtle" style="margin-top:8px;">{subtitle}</div>
                </div>
                <div class="badge info">Tryb demonstracyjny</div>
              </div>

              <div class="gate {gate_class}">
                <div class="gateGlow"></div>
                <div class="gateStateLabel">{label}</div>
                <div class="gateLeaf left"></div>
                <div class="gateLeaf right"></div>
                <div class="gateFloor"></div>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

@contextmanager
def fullscreen_loader(title: str, subtitle: str):
    slot = st.empty()
    slot.markdown(
        dedent(
            f"""
            <div class="loaderOverlay">
              <div class="loaderCard">
                <div class="loaderTop">
                  <div>
                    <div class="loaderTitle">{title}</div>
                    <div class="loaderSub">{subtitle}</div>
                    <div class="loaderDots" aria-hidden="true">
                      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
                    </div>
                  </div>
                  <div class="badge info">Pracuję…</div>
                </div>

                <div class="road">
                  <div class="car" aria-hidden="true">
                    <div class="carBody"></div>
                    <div class="carCabin"></div>
                    <div class="headLight"></div>
                    <div class="wheel left"></div>
                    <div class="wheel right"></div>
                  </div>
                </div>

                <div class="subtle" style="margin-top:10px;">
                  Pro tip: najlepszy odczyt jest, gdy tablica jest ostra i zajmuje większą część zdjęcia.
                </div>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        slot.empty()


# =========================================================
# SESSION STATE
# =========================================================

if "mode" not in st.session_state:
    st.session_state.mode = "Użytkownik"

if "cfg_path" not in st.session_state:
    st.session_state.cfg_path = "configs/app_config.yaml"

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 1

if "upload_image_rgb" not in st.session_state:
    st.session_state.upload_image_rgb = None  # np.ndarray

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_image_rgb" not in st.session_state:
    st.session_state.last_image_rgb = None

if "last_crop_rgb" not in st.session_state:
    st.session_state.last_crop_rgb = None

if "analysis_just_finished" not in st.session_state:
    st.session_state.analysis_just_finished = False

if "admin_recent_add" not in st.session_state:
    st.session_state.admin_recent_add = ""


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("## ANPR – kontrola bramy")
    st.caption("Aplikacja rozpoznaje tablicę na zdjęciu i decyduje, czy brama ma się otworzyć.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.session_state.mode = st.radio(
        "Tryb",
        ["Użytkownik", "Administrator"],
        index=0 if st.session_state.mode == "Użytkownik" else 1,
    )

    st.markdown("<div class='card soft'>", unsafe_allow_html=True)
    st.markdown("### Ustawienia systemu")

    if st.session_state.mode == "Użytkownik":
        st.text_input(
            "Ścieżka do configu (YAML)",
            value=st.session_state.cfg_path,
            disabled=True,
            help="W trybie użytkownika konfiguracja jest zablokowana.",
        )
        st.caption("Konfigurację może zmieniać administrator.")
    else:
        new_cfg = st.text_input("Ścieżka do configu (YAML)", value=st.session_state.cfg_path)
        if new_cfg.strip() and new_cfg.strip() != st.session_state.cfg_path:
            st.session_state.cfg_path = new_cfg.strip()

        if st.button("Przeładuj pipeline", use_container_width=True):
            load_pipeline_cached.clear()  # type: ignore[attr-defined]
            st.session_state.last_result = None
            st.session_state.last_image_rgb = None
            st.session_state.last_crop_rgb = None
            st.session_state.analysis_just_finished = False
            st.success("Pipeline przeładowany.")

    st.markdown("</div>", unsafe_allow_html=True)

cfg_path = st.session_state.cfg_path

# =========================================================
# LOAD PIPELINE
# =========================================================

try:
    pipeline = load_pipeline_cached(cfg_path)
except Exception as e:
    st.error(f"Nie udało się załadować pipeline: {e}")
    st.stop()

allowed_chars = getattr(pipeline, "allowed_chars", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
uppercase = bool(getattr(pipeline, "uppercase", True))
strip_spaces = bool(getattr(pipeline, "strip_spaces", True))
plate_regex = getattr(pipeline, "plate_regex", "^[A-Z]{1,3}[A-Z0-9]{4,5}$")


# =========================================================
# PIPELINE RUN
# =========================================================

def run_pipeline_on_rgb(img_rgb: np.ndarray) -> None:
    # 0) najpierw emergency classifier
    with fullscreen_loader("Analizuję zdjęcie…", "Sprawdzam czy to pojazd uprzywilejowany (emergency)."):
        is_emg, emg_conf, emg_name = predict_emergency_on_rgb(img_rgb)

    # Jeśli EMERGENCY -> otwieramy bramę i NIE sprawdzamy tablicy
    if is_emg:
        class DummyResult:
            detected = True
            bbox = None
            plate_text_raw = ""
            plate_text_norm = "EMERGENCY"
            plate_valid_format = True
            access_granted = True
            ocr_conf = emg_conf
            error = None
            timing_ms = {
                "emergency_cls_conf": emg_conf,
                "emergency_cls_name": emg_name,
            }

            is_emergency_vehicle = True

        out = DummyResult()
        st.session_state.last_result = out
        st.session_state.last_image_rgb = img_rgb
        st.session_state.last_crop_rgb = None
        st.session_state.analysis_just_finished = True
        return

    # 1) jeśli nie emergency -> normalny ANPR pipeline
    img_bgr = img_rgb[:, :, ::-1].copy()
    with fullscreen_loader("Analizuję zdjęcie…", "Wykrywam tablicę i odczytuję numer rejestracyjny."):
        out = pipeline.run(img_bgr)

    # (opcjonalnie) dopisz info o emergency-predykcji do wyniku
    try:
        out.timing_ms = dict(getattr(out, "timing_ms", {}) or {})
        out.timing_ms.update({"emergency_cls_conf": emg_conf, "emergency_cls_name": emg_name})
    except Exception:
        pass

    st.session_state.last_result = out
    st.session_state.last_image_rgb = img_rgb
    st.session_state.analysis_just_finished = True

    if getattr(out, "detected", False) and getattr(out, "bbox", None):
        x1, y1, x2, y2 = out.bbox
        crop_rgb = img_bgr[y1:y2, x1:x2][:, :, ::-1]
        st.session_state.last_crop_rgb = crop_rgb
    else:
        st.session_state.last_crop_rgb = None


# =========================================================
# MAIN UI
# =========================================================

st.markdown("# ANPR – rozpoznawanie tablic i kontrola bramy")
st.markdown(
    "<div class='card'>"
    "<div class='subtle'>"
    "Wgraj zdjęcie lub wybierz test. System znajdzie tablicę, odczyta numer i podejmie decyzję o bramie "
    "na podstawie listy zaufanych."
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# =========================================================
# USER MODE
# =========================================================

if st.session_state.mode == "Użytkownik":
    colL, colR = st.columns([1.15, 0.85], gap="large")

    with colL:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sectionTitle'>Wejście</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtle'>Najpierw wybierz obraz, potem uruchom analizę.</div>", unsafe_allow_html=True)
        st.markdown("<div class='hrSoft'></div>", unsafe_allow_html=True)

        tab_up, tab_test = st.tabs(["Wgraj zdjęcie", "Wybierz testowe"])

        # -------------------------
        # Upload TAB
        # -------------------------
        with tab_up:
            # jeśli mamy już wybrane zdjęcie -> pokaż podgląd + akcje
            if st.session_state.upload_image_rgb is not None:
                st.markdown("<div class='previewBox'>", unsafe_allow_html=True)
                st.markdown("<div class='previewLabel'>Podgląd wybranego zdjęcia</div>", unsafe_allow_html=True)
                st.image(st.session_state.upload_image_rgb, width=420)
                st.markdown("</div>", unsafe_allow_html=True)

                btns = st.columns([1, 1], gap="small")
                with btns[0]:
                    if st.button("Uruchom analizę", use_container_width=True):
                        run_pipeline_on_rgb(st.session_state.upload_image_rgb)
                        st.rerun()
                with btns[1]:
                    if st.button("Wybierz inne zdjęcie", use_container_width=True):
                        st.session_state.upload_image_rgb = None
                        st.session_state.uploader_key += 1
                        st.session_state.analysis_just_finished = False
                        st.rerun()

            else:
                uploaded = st.file_uploader(
                    "Wgraj zdjęcie (JPG/PNG)",
                    type=["jpg", "jpeg", "png"],
                    key=f"uploader_{st.session_state.uploader_key}",
                )
                st.caption("Wskazówka: najlepsze wyniki są, gdy tablica jest ostra i zajmuje większą część kadru.")

                if uploaded is not None:
                    img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
                    st.session_state.upload_image_rgb = np.array(img)
                    st.rerun()

        # -------------------------
        # Test TAB
        # -------------------------
        with tab_test:
            test_dir = Path("data/test_images")
            if not test_dir.exists():
                st.info("Brak folderu data/test_images.")
            else:
                imgs = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
                if not imgs:
                    st.info("Brak obrazów w data/test_images.")
                else:
                    pick = st.selectbox("Wybierz testowy obraz", [p.name for p in imgs], index=0)
                    p = test_dir / pick

                    try:
                        img_prev = Image.open(p).convert("RGB")
                        prev_rgb = np.array(img_prev)
                        st.markdown("<div class='previewBox'>", unsafe_allow_html=True)
                        st.markdown("<div class='previewLabel'>Podgląd testu</div>", unsafe_allow_html=True)
                        st.image(prev_rgb, width=420)
                        st.markdown("</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Nie udało się wczytać podglądu: {e}")
                        prev_rgb = None

                    if st.button("Uruchom analizę testu", use_container_width=True):
                        if prev_rgb is None:
                            st.error("Brak obrazu testowego do analizy.")
                        else:
                            run_pipeline_on_rgb(prev_rgb)
                            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with colR:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sectionTitle'>Decyzja bramy</div>", unsafe_allow_html=True)

        res = st.session_state.last_result
        if res is None:
            render_gate(None, "", False, False, animate=False)
            st.markdown("<div class='subtle' style='margin-top:10px;'>Uruchom analizę, aby zobaczyć decyzję.</div>", unsafe_allow_html=True)
        else:
            detected = bool(getattr(res, "detected", False))
            plate_norm = (getattr(res, "plate_text_norm", "") or "").strip()
            valid = bool(getattr(res, "plate_valid_format", False))
            access = getattr(res, "access_granted", None)

            # Po świeżej analizie -> animuj otwieranie/zamykanie
            render_gate(access, plate_norm, valid, detected, animate=bool(st.session_state.analysis_just_finished))

            # FAB "Zobacz wyniki"
            st.markdown(
                dedent(
                    """
                    <a class="fabResults" href="#results_anchor">
                      <div class="fabBtn">
                        Zobacz wyniki
                        <div class="fabArrow">↓</div>
                      </div>
                    </a>
                    """
                ),
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # anchor for scrolling
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div id='results_anchor'></div>", unsafe_allow_html=True)

    # RESULTS SECTION
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sectionTitle'>Co odczytał system?</div>", unsafe_allow_html=True)

    res = st.session_state.last_result
    if res is None:
        st.info("Brak wyniku. Wybierz obraz i uruchom analizę.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        if not getattr(res, "detected", False):
            st.markdown("<div class='badge bad'>Nie wykryto tablicy</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='subtle' style='margin-top:8px;'>{res.error or 'System nie znalazł tablicy na zdjęciu.'}</div>", unsafe_allow_html=True)
        else:
            conf = float(getattr(res, "ocr_conf", 0.0) or 0.0)
            plate_raw = (getattr(res, "plate_text_raw", "") or "").strip()
            plate_norm = (getattr(res, "plate_text_norm", "") or "").strip()
            valid = bool(getattr(res, "plate_valid_format", False))
            access = bool(getattr(res, "access_granted", False)) if valid else False

            cA, cB = st.columns([1, 1], gap="large")
            with cA:
                st.markdown("<div class='previewBox'>", unsafe_allow_html=True)
                st.markdown("<div class='previewLabel'>Odczytany numer (po oczyszczeniu)</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:26px; font-weight:1000; letter-spacing:.4px;'>{plate_norm or '—'}</div>", unsafe_allow_html=True)
                st.markdown("<div class='subtle' style='margin-top:6px;'>To numer bez spacji i znaków spoza A–Z / 0–9.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with cB:
                st.markdown("<div class='previewBox'>", unsafe_allow_html=True)
                st.markdown("<div class='previewLabel'>Pewność odczytu</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:26px; font-weight:1000;'>{friendly_confidence(conf)}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:8px;'><span class='badge {conf_badge_class(conf)}'>Wartość: {conf:.3f}</span></div>", unsafe_allow_html=True)
                st.markdown("<div class='subtle' style='margin-top:6px;'>Im wyżej, tym stabilniejszy odczyt.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='hrSoft'></div>", unsafe_allow_html=True)

            if not plate_norm or not valid:
                st.markdown("<div class='badge warn'>Brak bezpiecznej decyzji</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='subtle' style='margin-top:8px;'>"
                    "Tablica została znaleziona, ale numer jest niepełny albo ma nietypowy format – "
                    "dla bezpieczeństwa brama nie otwiera się automatycznie."
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                if access:
                    st.markdown("<div class='badge ok'>Wjazd dozwolony</div>", unsafe_allow_html=True)
                    st.markdown("<div class='subtle' style='margin-top:8px;'>Numer jest na liście zaufanych – brama powinna się otworzyć.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='badge bad'>Wjazd zablokowany</div>", unsafe_allow_html=True)
                    st.markdown("<div class='subtle' style='margin-top:8px;'>Numer nie jest na liście zaufanych – brama pozostaje zamknięta.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sectionTitle'>Podgląd wyników</div>", unsafe_allow_html=True)

    if res is None:
        st.info("Najpierw uruchom analizę.")
    else:
        img_rgb = st.session_state.last_image_rgb
        crop_rgb = st.session_state.last_crop_rgb

        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("<div class='previewBox'>", unsafe_allow_html=True)
            st.markdown("<div class='previewLabel'>Oryginalne zdjęcie</div>", unsafe_allow_html=True)
            if img_rgb is not None:
                st.image(img_rgb, use_container_width=True)
            else:
                st.info("Brak obrazu w pamięci sesji.")
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='previewBox'>", unsafe_allow_html=True)
            st.markdown("<div class='previewLabel'>Wykrycie tablicy (ramka)</div>", unsafe_allow_html=True)
            if img_rgb is not None and getattr(res, "detected", False) and getattr(res, "bbox", None):
                img_bgr = img_rgb[:, :, ::-1]
                vis_bgr = pipeline.draw_bbox(img_bgr, res.bbox)
                st.image(vis_bgr[:, :, ::-1], use_container_width=True)
            else:
                st.info("Brak ramki (bbox).")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='previewBox'>", unsafe_allow_html=True)
        st.markdown("<div class='previewLabel'>Wycięta tablica (crop)</div>", unsafe_allow_html=True)
        if crop_rgb is not None:
            st.image(crop_rgb, use_container_width=True)
        else:
            st.info("Crop nie jest dostępny.")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Detale techniczne (opcjonalnie)"):
            st.write(f"Raw OCR: `{getattr(res, 'plate_text_raw', '')}`")
            st.write(f"Normalized: `{getattr(res, 'plate_text_norm', '')}`")
            st.write(f"Format (regex): `{getattr(res, 'plate_valid_format', False)}`")
            st.json(getattr(res, "timing_ms", {}) or {})

    st.markdown("</div>", unsafe_allow_html=True)

    # Po pokazaniu animacji bramy już nie traktujemy wyniku jako "świeżo zakończony"
    if st.session_state.analysis_just_finished:
        st.session_state.analysis_just_finished = False


# =========================================================
# ADMIN MODE
# =========================================================

else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sectionTitle'>Panel administratora</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>Dodawaj i usuwaj zaufane numery. Operacje wykonują się tak jak w Twojej konsoli (scripts.db_manage).</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    colL, colR = st.columns([1.0, 1.2], gap="large")

    with colL:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sectionTitle'>Dodaj zaufaną tablicę</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtle'>Wpisz numer, a zapisze się do bazy.</div>", unsafe_allow_html=True)
        st.markdown("<div class='hrSoft'></div>", unsafe_allow_html=True)

        raw = st.text_input("Numer rejestracyjny", value="", placeholder="np. SK12345 / KOL27308")
        norm = normalize_plate(raw, allowed_chars=allowed_chars, uppercase=uppercase, strip_spaces=strip_spaces)
        is_valid = validate_plate(norm, plate_regex) if norm else False

        st.markdown("<div class='previewBox'>", unsafe_allow_html=True)
        st.markdown("<div class='previewLabel'>Podgląd wpisu</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:22px; font-weight:1000; letter-spacing:.45px;'>{norm or '—'}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='margin-top:10px;'><span class='badge {'ok' if is_valid else 'warn'}'>{'Format OK' if is_valid else 'Sprawdź wpis'}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Dodaj do zaufanych", use_container_width=True):
            if not norm:
                st.error("Najpierw wpisz numer.")
            elif not is_valid:
                st.error("Numer ma niepoprawny format (wg reguł w configu). Popraw i spróbuj ponownie.")
            else:
                try:
                    with fullscreen_loader("Zapisuję do bazy…", "Dodaję numer do listy zaufanych."):
                        db_add_plate(norm)
                    st.session_state.admin_recent_add = norm
                    st.success(f"Dodano: {norm}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Nie udało się dodać do bazy: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    with colR:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sectionTitle'>Zaufane tablice</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtle'>Kliknij „Usuń”, aby natychmiast zdjąć numer z listy.</div>", unsafe_allow_html=True)
        st.markdown("<div class='hrSoft'></div>", unsafe_allow_html=True)

        plates = db_list_plates(cfg_path)

        if not plates:
            st.info("Lista jest pusta albo nie udało się wykryć tabeli z tablicami w tej bazie.")
        else:
            st.caption(f"Liczba zaufanych tablic: {len(plates)}")
            per_row = 3
            rows = [plates[i:i + per_row] for i in range(0, len(plates), per_row)]

            for r_i, row in enumerate(rows):
                cols = st.columns(per_row, gap="small")
                for c_i in range(per_row):
                    with cols[c_i]:
                        if c_i >= len(row):
                            st.write("")
                            continue

                        p = row[c_i]
                        is_new = (p == (st.session_state.admin_recent_add or "").strip().upper())

                        st.markdown(
                            dedent(
                                f"""
                                <div class="tile {'tileNew' if is_new else ''}">
                                  <div class="tilePlate">{p}</div>
                                  <div class="tileMeta">zaufana tablica</div>
                                </div>
                                """
                            ),
                            unsafe_allow_html=True,
                        )

                        if st.button("Usuń", key=f"del_{p}_{r_i}_{c_i}", use_container_width=True):
                            try:
                                with fullscreen_loader("Aktualizuję bazę…", "Usuwam numer z listy zaufanych."):
                                    db_remove_plate(cfg_path, p)
                                if st.session_state.admin_recent_add == p:
                                    st.session_state.admin_recent_add = ""
                                st.success(f"Usunięto: {p}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Nie udało się usunąć `{p}`: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='card soft'>"
        "<div class='subtle'><b>Info:</b> Operacje dodawania/usuwnia są wykonywane tak jak w Twojej konsoli (scripts.db_manage), "
        "a listowanie działa dla różnych schematów bazy.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
