from __future__ import annotations

import io
import re
import sys
import sqlite3
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from contextlib import contextmanager
from textwrap import dedent

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from ultralytics import YOLO

# =========================================================
# PATHS / IMPORTS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anpr.pipeline import ANPRPipeline
from anpr.utils import normalize_plate, validate_plate


# =========================================================
# EMERGENCY CLS CONFIG
# =========================================================

EMERGENCY_THRESHOLD = 0.80  # ustaw próg dla "emergency"


def _find_emergency_model_path() -> Optional[Path]:
    """
    Szuka najlepszego kandydatu na model emergency w typowych lokalizacjach.
    Priorytet:
      1) Dokładna ścieżka jak u Ciebie: runs/classify/runs/train/.../best.pt
      2) runs/**/emergency_classification*/weights/best.pt (rekurencyjnie)
    """
    candidates: List[Path] = []

    p1 = ROOT / "runs" / "classify" / "runs" / "train" / "emergency_classification_v2" / "weights" / "best.pt"
    if p1.exists():
        return p1

    for pat in [
        "runs/**/emergency_classification*/weights/best.pt",
        "runs/**/emergency*/weights/best.pt",
    ]:
        candidates.extend(ROOT.glob(pat))

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


EMERGENCY_MODEL_PATH = _find_emergency_model_path()


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(page_title="ANPR – kontrola bramy", layout="wide")


# =========================================================
# SMALL UTILS
# =========================================================

def _html_noindent(html: str) -> str:
    """
    Streamlit markdown potrafi potraktować wcięty HTML jako blok kodu.
    Usuwamy wiodące spacje z każdej linii, żeby HTML zawsze był renderowany.
    """
    lines = html.splitlines()
    return "\n".join([ln.lstrip() for ln in lines])


# =========================================================
# STYLE + UX (CSS + JS smooth scroll + auto-hide FAB)
# =========================================================

APP_CSS = dedent(
    """
<style>
:root{
  --bg0:#070707;
  --bg1:#0d0b08;

  --card0: rgba(14,13,12,0.92);
  --card1: rgba(18,16,13,0.70);

  --border: rgba(216,199,163,0.20);
  --border2: rgba(216,199,163,0.32);

  --text:#f3efe7;
  --muted:#b9b0a3;

  --beige:#d8c7a3;
  --beige2:#cbb78c;

  --ok:#2ecc71;
  --bad:#ff4d4d;
  --warn:#f4b740;
  --info:#8bb6ff;

  --shadow: 0 18px 55px rgba(0,0,0,0.55);
  --shadow2: 0 24px 78px rgba(0,0,0,0.60);

  --radius: 18px;
  --radius2: 20px;

  --font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}

html, body, [class*="css"]{ font-family: var(--font) !important; }
.stApp {
  background:
    radial-gradient(1100px 520px at 50% -12%, rgba(216,199,163,0.12), transparent 60%),
    radial-gradient(900px 360px at 10% 12%, rgba(203,183,140,0.10), transparent 60%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
}

.block-container { padding-top: 1.1rem; padding-bottom: 2.0rem; max-width: 1200px; }
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
  border-radius: var(--radius2);
  padding: 16px 16px;
  box-shadow: var(--shadow);
}
.card.soft{
  background: linear-gradient(180deg, rgba(14,13,12,0.68), rgba(14,13,12,0.50));
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
  background: rgba(0,0,0,0.26);
  font-weight: 900;
  font-size: 12px;
  color: var(--text);
  letter-spacing: .15px;
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
  line-height: 1.55;
}

/* Better buttons */
.stButton > button{
  border-radius: 14px !important;
  border: 1px solid rgba(216,199,163,0.28) !important;
  background: rgba(0,0,0,0.30) !important;
  color: var(--text) !important;
  font-weight: 900 !important;
  letter-spacing: .2px !important;
  padding: 0.65rem 0.9rem !important;
  box-shadow: 0 14px 40px rgba(0,0,0,0.35) !important;
  transition: transform .12s ease, border-color .12s ease, background .12s ease;
}
.stButton > button:hover{
  border-color: rgba(216,199,163,0.55) !important;
  background: rgba(0,0,0,0.40) !important;
  transform: translateY(-1px);
}
.stButton > button:active{
  transform: translateY(0px);
}

/* Bigger uploader */
div[data-testid="stFileUploader"] > section{
  padding: 18px !important;
  border: 1px dashed rgba(216,199,163,0.42) !important;
  border-radius: 18px !important;
  background: rgba(0,0,0,0.20) !important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"]{
  min-height: 160px !important;
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

/* Gate */
.gateWrap{
  border-radius: 18px;
  border: 1px solid rgba(216,199,163,0.22);
  background:
    radial-gradient(1000px 280px at 50% -20%, rgba(216,199,163,0.16), transparent 60%),
    linear-gradient(180deg, rgba(0,0,0,0.25), rgba(12,11,10,0.55));
  padding: 14px;
  overflow: hidden;
  box-shadow: var(--shadow2);
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
  height: 178px;
  border-radius: 16px;
  border: 1px solid rgba(216,199,163,0.22);
  background: linear-gradient(180deg, rgba(0,0,0,0.58), rgba(0,0,0,0.16));
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

@keyframes gateOpenLeft { 0% { transform: perspective(900px) rotateY(0deg); } 100% { transform: perspective(900px) rotateY(-78deg); } }
@keyframes gateOpenRight{ 0% { transform: perspective(900px) rotateY(0deg); } 100% { transform: perspective(900px) rotateY(78deg); } }
@keyframes gateCloseLeft{ 0% { transform: perspective(900px) rotateY(-12deg);} 100% { transform: perspective(900px) rotateY(0deg);} }
@keyframes gateCloseRight{0% { transform: perspective(900px) rotateY(12deg);} 100% { transform: perspective(900px) rotateY(0deg);} }

.gate.open  .gateLeaf.left  { animation: gateOpenLeft 720ms cubic-bezier(.2,.9,.2,1) forwards; }
.gate.open  .gateLeaf.right { animation: gateOpenRight 720ms cubic-bezier(.2,.9,.2,1) forwards; }
.gate.closed .gateLeaf.left  { animation: gateCloseLeft 520ms ease-in-out forwards; }
.gate.closed .gateLeaf.right { animation: gateCloseRight 520ms ease-in-out forwards; }
.gate.idle .gateLeaf.left, .gate.idle .gateLeaf.right { opacity: .88; }

.gateGlow{
  position:absolute; inset:-130px -130px auto -130px; height: 260px;
  filter: blur(2px);
  opacity: 0;
  transition: opacity 360ms ease-in-out;
}
.gate.open .gateGlow{ opacity: 1; background: radial-gradient(closest-side, rgba(46,204,113,0.22), transparent 70%); }
.gate.closed .gateGlow{ opacity: 1; background: radial-gradient(closest-side, rgba(255,77,77,0.18), transparent 70%); }

/* Loader */
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
.loaderTop{ display:flex; align-items:flex-start; justify-content:space-between; gap: 12px; margin-bottom: 12px; }
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
@keyframes dashMove{ 0% { transform: translateX(0); } 100% { transform: translateX(120px); } }

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
.carBody{ position:absolute; left: 16px; top: 22px; width: 150px; height: 34px; border-radius: 14px;
  background: linear-gradient(180deg, rgba(216,199,163,0.95), rgba(203,183,140,0.70));
  border: 1px solid rgba(216,199,163,0.22);
  box-shadow: inset 0 0 0 1px rgba(0,0,0,0.25);
}
.carCabin{ position:absolute; left: 58px; top: 10px; width: 72px; height: 26px; border-radius: 12px;
  background: linear-gradient(180deg, rgba(0,0,0,0.35), rgba(0,0,0,0.20));
  border: 1px solid rgba(216,199,163,0.20);
}
.headLight{ position:absolute; right: 6px; top: 26px; width: 10px; height: 8px; border-radius: 6px;
  background: rgba(255,255,255,0.76); box-shadow: 0 0 14px rgba(255,255,255,0.35);
}
.wheel{
  position:absolute; bottom: 8px; width: 22px; height: 22px; border-radius: 999px;
  background: linear-gradient(180deg, rgba(0,0,0,0.70), rgba(0,0,0,0.45));
  border: 1px solid rgba(216,199,163,0.18);
  box-shadow: inset 0 0 0 3px rgba(216,199,163,0.18);
  animation: spin 0.25s linear infinite;
}
@keyframes spin{ to { transform: rotate(360deg); } }
.wheel.left { left: 36px; } .wheel.right{ left: 134px; }

.loaderDots{ display:inline-flex; gap:6px; align-items:center; margin-top: 10px; }
.dot{ width: 7px; height: 7px; border-radius: 999px; background: rgba(216,199,163,0.35);
  animation: dotPulse 0.9s ease-in-out infinite;
}
.dot:nth-child(2){ animation-delay: .15s; } .dot:nth-child(3){ animation-delay: .3s; }
@keyframes dotPulse{ 0%,100%{ transform: scale(1); opacity: .6; } 50%{ transform: scale(1.45); opacity: 1; } }

/* Floating "See results" */
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
  cursor: pointer;
  user-select: none;
  transition: transform .12s ease, border-color .12s ease, background .12s ease, opacity .18s ease;
}
.fabBtn:hover{
  border-color: rgba(216,199,163,0.55);
  background: rgba(0,0,0,0.62);
  transform: translateY(-1px);
}
.fabBtn:active{ transform: translateY(0px); }
.fabArrow{
  width: 26px; height: 26px;
  border-radius: 999px;
  display:flex; align-items:center; justify-content:center;
  border: 1px solid rgba(216,199,163,0.22);
  background: rgba(216,199,163,0.10);
}
.fabHidden{ opacity: 0 !important; pointer-events: none !important; transform: translateX(-50%) translateY(8px) !important; }
</style>
"""
)

st.markdown(APP_CSS, unsafe_allow_html=True)

# JS: wstrzykujemy jako komponent i sterujemy parent DOM (Streamlit często ignoruje JS z markdown).
APP_JS_COMPONENT = dedent(
    """
<!doctype html>
<html>
  <head><meta charset="utf-8" /></head>
  <body>
    <script>
      (function(){
        const P = window.parent;
        if(!P) return;

        function getScrollRoot(){
          // Streamlit zwykle scrolluje tu:
          const candidates = [
            P.document.querySelector('[data-testid="stAppViewContainer"]'),
            P.document.querySelector('[data-testid="stMain"]'),
            P.document.querySelector('section.main'),
            P.document.scrollingElement,
            P.document.documentElement,
            P.document.body
          ].filter(Boolean);

          // wybierz pierwszy, który realnie jest scrollowalny
          for (const el of candidates){
            try{
              const cs = P.getComputedStyle(el);
              const oy = cs.overflowY;
              const scrollable = (oy === 'auto' || oy === 'scroll') && (el.scrollHeight > el.clientHeight + 2);
              if(scrollable) return el;
            } catch(_) {}
          }

          // fallback
          return P.document.scrollingElement || P.document.documentElement;
        }

        function scrollToAnchor(anchorId){
          const el = P.document.getElementById(anchorId);
          if(!el) return;
          // działa i na window-scroll i na kontenerze streamlit
          try{
            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
          } catch(_) {
            // bardzo stary fallback
            const root = getScrollRoot();
            const r = el.getBoundingClientRect();
            const top = (root.scrollTop || 0) + r.top - 10;
            root.scrollTop = top;
          }
        }

        function wireFab(){
          const fab = P.document.getElementById('fabResultsBtn');
          if(!fab) return false;
          if(fab.__wired) return true;
          fab.__wired = true;

          const onClick = (e) => {
            try{ e.preventDefault(); } catch(_) {}
            scrollToAnchor('results_anchor');
          };

          fab.addEventListener('click', onClick);
          fab.addEventListener('keydown', (e)=>{
            const k = e.key || '';
            if(k === 'Enter' || k === ' '){ onClick(e); }
          });

          return true;
        }

        function observeResults(){
          const anchor = P.document.getElementById('results_anchor');
          const fabWrap = P.document.getElementById('fabResultsWrap');
          if(!anchor || !fabWrap) return false;

          if(P.__anprObsAttached) return true;
          P.__anprObsAttached = true;

          const root = getScrollRoot();
          const Obs = P.IntersectionObserver;
          if(!Obs) return false;

          const obs = new Obs((entries)=>{
          const e = entries && entries[0];
          if(!e) return;
        
          // rect.top:
          //  > 0  => anchor jest poniżej górnej krawędzi widoku (czyli jesteś WYŻEJ niż wyniki)
          // <= 0  => anchor jest na górze lub nad widokiem (czyli jesteś NA/PONIŻEJ wyniku)
          const rect = anchor.getBoundingClientRect();
          const reachedOrPassed = rect.top <= 10; // margines 10px
        
          if(e.isIntersecting || reachedOrPassed){
            fabWrap.classList.add('fabHidden');
          } else {
            fabWrap.classList.remove('fabHidden');
          }
        }, {
          root: root === P.document.documentElement ? null : root,
          threshold: 0.15
        });


          obs.observe(anchor);
          return true;
        }

        function ensure(){
          wireFab();
          observeResults();
        }

        // 1) od razu
        ensure();

        // 2) i przy każdej przebudowie DOM Streamlit
        const mo = new P.MutationObserver(()=>ensure());
        mo.observe(P.document.body, { childList:true, subtree:true });

      })();
    </script>
  </body>
</html>
"""
)

# Wstrzyknięcie JS (bez UI)
components.html(APP_JS_COMPONENT, height=1, width=1)


# =========================================================
# HELPERS: CACHE + MODELS
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
    global EMERGENCY_MODEL_PATH
    if EMERGENCY_MODEL_PATH is None or not EMERGENCY_MODEL_PATH.exists():
        EMERGENCY_MODEL_PATH = _find_emergency_model_path()

    if EMERGENCY_MODEL_PATH is None or not EMERGENCY_MODEL_PATH.exists():
        return False, 0.0, "model_missing"

    model = load_emergency_model_cached(str(EMERGENCY_MODEL_PATH))
    r = model.predict(source=img_rgb, verbose=False)[0]

    pred_idx = int(r.probs.top1)
    conf = float(r.probs.top1conf)
    pred_name = r.names[pred_idx]  # "emergency" / "non_emergency"

    is_emergency = (pred_name == "emergency" and conf >= EMERGENCY_THRESHOLD)
    return is_emergency, conf, pred_name


@st.cache_resource
def load_pipeline_cached(config_path: str) -> ANPRPipeline:
    return ANPRPipeline(config_path=config_path)


def get_db_path_from_config(cfg_path: str) -> str:
    import yaml
    p = Path(cfg_path)
    if not p.exists():
        return "data/plates.db"
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return str(cfg.get("access_control", {}).get("sqlite_path", "data/plates.db"))


# =========================================================
# DB HELPERS
# =========================================================

def _ensure_db_dir(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def _sqlite_conn(db_path: str) -> sqlite3.Connection:
    _ensure_db_dir(db_path)
    return sqlite3.connect(db_path, check_same_thread=False)


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


def db_add_plate(db_path: str, plate_norm: str) -> None:
    p = _run_db_manage(["add", "--plate", plate_norm], db_path=db_path)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "").strip() or "db_manage add failed")


def _looks_like_plate(s: str) -> bool:
    s = (s or "").strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return 4 <= len(s) <= 10


def _pick_best_table_and_col(conn: sqlite3.Connection, plate_regex: str) -> Optional[Tuple[str, str]]:
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
    if not tables:
        return None

    rx = re.compile(plate_regex)
    best: Optional[Tuple[str, str]] = None
    best_score = -1

    for t in tables:
        try:
            cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
        except Exception:
            continue

        for (_, colname, coltype, *_rest) in cols:
            ctype = (coltype or "").upper()
            if not any(x in ctype for x in ["CHAR", "TEXT", "CLOB", "VARCHAR"]):
                continue

            try:
                rows = cur.execute(f"SELECT {colname} FROM {t} WHERE {colname} IS NOT NULL LIMIT 500").fetchall()
            except Exception:
                continue

            if not rows:
                continue

            looks = 0
            matches = 0
            for (v,) in rows:
                s = str(v).strip().upper()
                s2 = re.sub(r"[^A-Z0-9]", "", s)
                if _looks_like_plate(s2):
                    looks += 1
                if s2 and rx.match(s2):
                    matches += 1

            score = matches * 10 + looks
            if score > best_score:
                best_score = score
                best = (t, colname)

    return best


def db_list_plates(
    db_path: str,
    plate_regex: str,
    allowed_chars: str,
    uppercase: bool,
    strip_spaces: bool,
) -> List[str]:
    conn = _sqlite_conn(db_path)
    pick = _pick_best_table_and_col(conn, plate_regex)
    if not pick:
        conn.close()
        return []

    table, col = pick
    cur = conn.cursor()
    try:
        rows = cur.execute(f"SELECT {col} FROM {table}").fetchall()
    except Exception:
        conn.close()
        return []

    conn.close()

    out: List[str] = []
    for (v,) in rows:
        if v is None:
            continue
        norm = normalize_plate(str(v), allowed_chars=allowed_chars, uppercase=uppercase, strip_spaces=strip_spaces)
        if norm:
            out.append(norm)

    return sorted(set(out))


def db_remove_plate(db_path: str, plate_norm: str, plate_regex: str) -> None:
    for cmd in (["remove"], ["del"], ["rm"]):
        p = _run_db_manage(cmd + ["--plate", plate_norm], db_path=db_path)
        if p.returncode == 0:
            return

    conn = _sqlite_conn(db_path)
    pick = _pick_best_table_and_col(conn, plate_regex)
    if not pick:
        conn.close()
        raise RuntimeError("Nie potrafię wykryć tabeli z tablicami w tej bazie.")

    table, col = pick
    cur = conn.cursor()
    cur.execute(f"DELETE FROM {table} WHERE UPPER({col}) = ?", (plate_norm.upper(),))
    conn.commit()
    conn.close()


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


def gate_model(
    access_granted: Optional[bool],
    plate: str,
    valid: bool,
    detected: bool,
    is_emergency: bool,
) -> Tuple[str, str, str]:
    if not detected:
        return ("idle", "Brama: BRAK DANYCH", "Nie wykryto pojazdu – system nie podejmuje decyzji.")

    if is_emergency:
        return ("open", "Brama: OTWARTA", "🚑 Pojazd uprzywilejowany – brama otwarta automatycznie (bez OCR).")

    if not valid or not plate:
        return ("idle", "Brama: BRAK DECYZJI", "Tablica wykryta, ale odczyt jest niepewny / niepełny – nie otwieramy.")

    if access_granted is True:
        return ("open", "Brama: OTWARTA", "Numer jest na liście zaufanych – wjazd dozwolony.")

    return ("closed", "Brama: ZAMKNIĘTA", "Numer nie jest na liście zaufanych – wjazd zablokowany.")


def render_gate(
    access_granted: Optional[bool],
    plate: str,
    valid: bool,
    detected: bool,
    is_emergency: bool,
    animate: bool,
) -> None:
    state, label, subtitle = gate_model(access_granted, plate, valid, detected, is_emergency)
    gate_class = state if animate else "idle"

    badge = "info"
    if state == "open":
        badge = "ok"
    elif state == "closed":
        badge = "bad"
    elif state == "idle" and detected:
        badge = "warn"

    html = dedent(
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
    )
    st.markdown(_html_noindent(html), unsafe_allow_html=True)


@contextmanager
def fullscreen_loader(title: str, subtitle: str):
    slot = st.empty()
    html = dedent(
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
    )
    slot.markdown(_html_noindent(html), unsafe_allow_html=True)
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
    st.caption("Najpierw rozpoznajemy pojazd uprzywilejowany, a dopiero potem (jeśli trzeba) tablicę.")
    if EMERGENCY_MODEL_PATH and EMERGENCY_MODEL_PATH.exists():
        st.markdown("<div class='subtle'>Model emergency: <span class='badge ok'>OK</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='subtle'>Model emergency: <span class='badge warn'>BRAK</span></div>", unsafe_allow_html=True)
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
db_path = get_db_path_from_config(cfg_path)


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
# PIPELINE RUN (EMERGENCY FIRST, THEN ANPR)
# =========================================================

def run_pipeline_on_rgb(img_rgb: np.ndarray) -> None:
    with fullscreen_loader("Analizuję zdjęcie…", "Krok 1/2: sprawdzam, czy to pojazd uprzywilejowany."):
        is_emg, emg_conf, emg_name = predict_emergency_on_rgb(img_rgb)

    if is_emg:
        class DummyResult:
            detected = True
            bbox = None

            plate_text_raw = ""
            plate_text_norm = ""
            plate_valid_format = False

            access_granted = True

            ocr_conf = 0.0
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

    img_bgr = img_rgb[:, :, ::-1].copy()
    with fullscreen_loader("Analizuję zdjęcie…", "Krok 2/2: wykrywam tablicę i odczytuję numer rejestracyjny."):
        out = pipeline.run(img_bgr)

    try:
        out.is_emergency_vehicle = False
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
    _html_noindent(
        "<div class='card'>"
        "<div class='subtle'>"
        "<b>Flow:</b> Najpierw klasyfikacja pojazdu uprzywilejowanego. "
        "Jeśli nie uprzywilejowany, uruchamiamy ANPR (detekcja tablicy + OCR) i sprawdzamy czy jest na liście zaufanych"
        "</div>"
        "</div>"
    ),
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
        st.markdown("<div class='subtle'>Wybierz obraz, potem uruchom analizę.</div>", unsafe_allow_html=True)
        st.markdown("<div class='hrSoft'></div>", unsafe_allow_html=True)

        tab_up, tab_test = st.tabs(["Wgraj zdjęcie", "Wybierz testowe"])

        with tab_up:
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
                st.caption("Najlepsze wyniki, gdy tablica jest ostra i zajmuje większą część kadru.")

                if uploaded is not None:
                    img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
                    st.session_state.upload_image_rgb = np.array(img)
                    st.rerun()

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
            render_gate(None, "", False, False, False, animate=False)
            st.markdown("<div class='subtle' style='margin-top:10px;'>Uruchom analizę, aby zobaczyć decyzję.</div>", unsafe_allow_html=True)
        else:
            detected = bool(getattr(res, "detected", False))
            plate_norm = (getattr(res, "plate_text_norm", "") or "").strip()
            valid = bool(getattr(res, "plate_valid_format", False))
            access = getattr(res, "access_granted", None)
            is_emergency = bool(getattr(res, "is_emergency_vehicle", False))

            render_gate(
                access_granted=access,
                plate=plate_norm,
                valid=valid,
                detected=detected,
                is_emergency=is_emergency,
                animate=bool(st.session_state.analysis_just_finished),
            )

            # FAB (JS jest wstrzyknięty przez components.html i działa na parent DOM)
            st.markdown(
                _html_noindent(
                    """
                    <div id="fabResultsWrap" class="fabResults">
                      <div id="fabResultsBtn"
                           class="fabBtn"
                           role="button"
                           tabindex="0"
                           aria-label="Zobacz wyniki">
                        Zobacz wyniki
                        <div class="fabArrow">↓</div>
                      </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # anchor for scrolling
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div id='results_anchor' style='height:2px;'></div>", unsafe_allow_html=True)

    # RESULTS SECTION
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sectionTitle'>Wynik analizy</div>", unsafe_allow_html=True)

    res = st.session_state.last_result
    if res is None:
        st.info("Brak wyniku. Wybierz obraz i uruchom analizę.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        is_emergency = bool(getattr(res, "is_emergency_vehicle", False))
        detected = bool(getattr(res, "detected", False))

        if is_emergency:
            emg_conf = float((getattr(res, "timing_ms", {}) or {}).get("emergency_cls_conf", 0.0) or 0.0)
            st.markdown("<div class='badge ok'>🚑 Pojazd uprzywilejowany</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='subtle' style='margin-top:8px;'>"
                "Wykryto pojazd uprzywilejowany. System <b>nie uruchamia OCR</b> i otwiera bramę automatycznie."
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='margin-top:10px;'><span class='badge info'>Pewność klasyfikacji: {emg_conf:.3f}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            if not detected:
                st.markdown("<div class='badge bad'>Nie wykryto tablicy</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='subtle' style='margin-top:8px;'>{getattr(res, 'error', None) or 'System nie znalazł tablicy na zdjęciu.'}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                conf = float(getattr(res, "ocr_conf", 0.0) or 0.0)
                plate_norm = (getattr(res, "plate_text_norm", "") or "").strip()
                valid = bool(getattr(res, "plate_valid_format", False))
                access = bool(getattr(res, "access_granted", False)) if valid else False

                cA, cB = st.columns([1, 1], gap="large")
                with cA:
                    st.markdown("<div class='previewBox'>", unsafe_allow_html=True)
                    st.markdown("<div class='previewLabel'>Odczytany numer (po normalizacji)</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='font-size:26px; font-weight:1000; letter-spacing:.4px;'>{plate_norm or '—'}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("<div class='subtle' style='margin-top:6px;'>Bez spacji i znaków spoza A–Z / 0–9.</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with cB:
                    st.markdown("<div class='previewBox'>", unsafe_allow_html=True)
                    st.markdown("<div class='previewLabel'>Pewność odczytu OCR</div>", unsafe_allow_html=True)
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
    st.markdown("<div class='sectionTitle'>Podgląd</div>", unsafe_allow_html=True)

    res = st.session_state.last_result
    if res is None:
        st.info("Najpierw uruchom analizę.")
    else:
        img_rgb = st.session_state.last_image_rgb
        crop_rgb = st.session_state.last_crop_rgb
        is_emergency = bool(getattr(res, "is_emergency_vehicle", False))

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
            if (not is_emergency) and img_rgb is not None and getattr(res, "detected", False) and getattr(res, "bbox", None):
                img_bgr = img_rgb[:, :, ::-1]
                vis_bgr = pipeline.draw_bbox(img_bgr, res.bbox)
                st.image(vis_bgr[:, :, ::-1], use_container_width=True)
            else:
                st.info("Brak ramki (bbox) — albo tryb emergency.")
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
            st.write(f"Emergency: `{bool(getattr(res, 'is_emergency_vehicle', False))}`")
            st.write(f"Raw OCR: `{getattr(res, 'plate_text_raw', '')}`")
            st.write(f"Normalized: `{getattr(res, 'plate_text_norm', '')}`")
            st.write(f"Format (regex): `{getattr(res, 'plate_valid_format', False)}`")
            st.json(getattr(res, "timing_ms", {}) or {})

    st.markdown("</div>", unsafe_allow_html=True)

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
                        db_add_plate(db_path, norm)
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

        plates = db_list_plates(
            db_path=db_path,
            plate_regex=plate_regex,
            allowed_chars=allowed_chars,
            uppercase=uppercase,
            strip_spaces=strip_spaces,
        )

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
                            _html_noindent(
                                f"""
                                <div class="previewBox" style="padding:10px 12px;">
                                  <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
                                    <div>
                                      <div style="font-weight:1000; letter-spacing:.65px; font-size:14px;">{p}</div>
                                      <div class="subtle" style="font-size:12px; margin-top:4px;">zaufana tablica</div>
                                    </div>
                                    <div>{'<span class="badge ok">NOWA</span>' if is_new else ''}</div>
                                  </div>
                                </div>
                                """
                            ),
                            unsafe_allow_html=True,
                        )

                        if st.button("Usuń", key=f"del_{p}_{r_i}_{c_i}", use_container_width=True):
                            try:
                                with fullscreen_loader("Aktualizuję bazę…", "Usuwam numer z listy zaufanych."):
                                    db_remove_plate(db_path=db_path, plate_norm=p, plate_regex=plate_regex)
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
        "<div class='subtle'><b>Info:</b> Operacje dodawania/usuwania są wykonywane przez scripts.db_manage, "
        "a listowanie działa dla różnych schematów bazy.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
