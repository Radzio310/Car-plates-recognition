from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re

import numpy as np
import cv2


@dataclass
class OcrResult:
    text: str
    confidence: float


ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
BAD_TOKENS = {"PL", "EU"}


def _clean(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def _has_letters(s: str) -> bool:
    return any("A" <= c <= "Z" for c in s)


def _has_digits(s: str) -> bool:
    return any("0" <= c <= "9" for c in s)


def _bbox_x_center(bbox) -> float:
    xs = [p[0] for p in bbox]
    return float(sum(xs)) / max(1, len(xs))


# -----------------------------------------
# BLUE STRIP DETECTION (UE) - conservative
# -----------------------------------------

def _detect_blue_strip_cut_x(bgr: np.ndarray) -> int:
    """
    Bardzo konserwatywne wykrycie paska UE.
    Zwraca x cięcia (0 = nie tnij).
    """
    h, w = bgr.shape[:2]
    if w < 140:
        return 0

    left_w = int(0.22 * w)
    roi = bgr[:, :left_w].copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 60, 60], dtype=np.uint8)
    upper = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if ratio < 0.16:
        return 0

    col_hits = (mask.max(axis=0) > 0).astype(np.uint8)
    if col_hits.sum() < int(0.10 * left_w):
        return 0

    cols = np.where(col_hits > 0)[0]
    if cols.size == 0:
        return 0

    raw_cut = int(cols.max()) + 1

    safe_margin = 18
    cut = max(0, raw_cut - safe_margin)

    cut = min(cut, int(0.16 * w))
    return max(0, cut)


# -----------------------------------------
# ORIGINAL PREPROCESS (EasyOCR/Tesseract)
# -----------------------------------------

def _preprocess(img_bgr: np.ndarray, cut_blue: bool) -> List[np.ndarray]:
    if img_bgr is None or img_bgr.size == 0:
        return []

    img = img_bgr.copy()
    h, w = img.shape[:2]

    if cut_blue:
        cut = _detect_blue_strip_cut_x(img)
        if 0 < cut < w - 10:
            img = img[:, cut:].copy()

    img = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2 = clahe.apply(gray)

    thr = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return [img, g2, thr]


# -----------------------------------------
# Scoring (wybór najlepszego kandydata)
# -----------------------------------------

def _score(text: str, conf: float) -> float:
    t = _clean(text)
    if not t:
        return -1e9
    if t in BAD_TOKENS:
        return -1e9

    n = len(t)
    score = 0.0

    # długość
    if 5 <= n <= 8:
        score += 40
    elif 4 <= n <= 10:
        score += 15
    else:
        score -= 40

    # literki/cyfry
    if _has_letters(t):
        score += 15
    else:
        score -= 25

    if _has_digits(t):
        score += 10
    else:
        score -= 15

    if _has_letters(t) and _has_digits(t):
        score += 20

    # start literą
    if t[0].isalpha():
        score += 10
    else:
        score -= 10

    score += float(conf) * 25.0
    return score


def _join_easyocr_tokens(results) -> List[Tuple[str, float]]:
    items: List[Tuple[float, str, float]] = []
    for bbox, text, conf in results:
        tok = _clean(str(text))
        if not tok:
            continue
        if tok in BAD_TOKENS:
            continue
        if len(tok) <= 1:
            continue
        xc = _bbox_x_center(bbox)
        items.append((xc, tok, float(conf)))

    if not items:
        return []

    items.sort(key=lambda x: x[0])
    joined = "".join(t for _, t, _ in items)
    joined_conf = float(sum(c for _, _, c in items) / max(1, len(items)))

    cands: List[Tuple[str, float]] = [(joined, joined_conf)]
    for _, tok, c in items:
        cands.append((tok, c))
    return cands


# -----------------------------------------
# FAST TESSERACT PREPROCESS (ważne!)
# -----------------------------------------

def _fast_candidates_for_tesseract(
    img_bgr: np.ndarray,
    cut_blue_auto: bool,
    resize_fx: float,
    border: int = 10,
) -> List[np.ndarray]:
    """
    Buduje listę obrazów do Tesseract (w kolejności od najtańszego/najczęściej skutecznego):
      1) gray+clahe
      2) otsu (binary)
      3) otsu inverted
      4) adaptive
      5) adaptive inverted
    Każdy kandydat dostaje biały border, bo tesseract lubi margines.
    """
    if img_bgr is None or img_bgr.size == 0:
        return []

    img = img_bgr.copy()
    h, w = img.shape[:2]

    if cut_blue_auto:
        cut = _detect_blue_strip_cut_x(img)
        if 0 < cut < w - 10:
            img = img[:, cut:].copy()

    if resize_fx and abs(resize_fx - 1.0) > 1e-6:
        img = cv2.resize(img, None, fx=resize_fx, fy=resize_fx, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # delikatne odszumienie + „czytelność” bez przesady
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # OTSU
    thr_otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr_otsu_inv = cv2.bitwise_not(thr_otsu)

    # Adaptive (czasem ratuje nierówne oświetlenie)
    thr_ad = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 6
    )
    thr_ad_inv = cv2.bitwise_not(thr_ad)

    def add_border(x: np.ndarray) -> np.ndarray:
        if border <= 0:
            return x
        return cv2.copyMakeBorder(x, border, border, border, border, cv2.BORDER_CONSTANT, value=255)

    cands: List[np.ndarray] = [
        add_border(g),
        add_border(thr_otsu),
        add_border(thr_otsu_inv),
        add_border(thr_ad),
        add_border(thr_ad_inv),
    ]
    return cands


# -----------------------------------------
# Engines
# -----------------------------------------

class EasyOcrEngine:
    def __init__(self, languages: List[str] = ["en"]):
        import easyocr
        self.reader = easyocr.Reader(languages, gpu=False)

        try:
            import pytesseract  # noqa: F401
            self._has_tesseract = True
        except Exception:
            self._has_tesseract = False

    def _easyocr(self, img) -> List[Tuple[str, float]]:
        if img.ndim == 3:
            rgb = img[:, :, ::-1]
        else:
            rgb = img
        out = self.reader.readtext(rgb, detail=1, paragraph=False, allowlist=ALLOWED)
        if not out:
            return []
        return _join_easyocr_tokens(out)

    def _tesseract(self, img) -> Tuple[str, float]:
        import pytesseract
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        txt = pytesseract.image_to_string(thr, lang="eng", config=config)
        txt = _clean(txt)
        return txt, (0.55 if txt else 0.0)

    def read(self, plate_bgr: np.ndarray) -> OcrResult:
        best_text = ""
        best_conf = 0.0
        best_score = -1e9

        for cut_blue in (False, True):
            variants = _preprocess(plate_bgr, cut_blue=cut_blue)
            for v in variants:
                cands = self._easyocr(v)
                for txt, conf in cands:
                    sc = _score(txt, conf)
                    if sc > best_score:
                        best_score = sc
                        best_text = _clean(txt)
                        best_conf = float(conf)

            if self._has_tesseract:
                if not best_text or best_text in BAD_TOKENS or len(best_text) < 4:
                    for v in variants:
                        txt, conf = self._tesseract(v)
                        sc = _score(txt, conf)
                        if sc > best_score:
                            best_score = sc
                            best_text = _clean(txt)
                            best_conf = float(conf)

        if best_text in BAD_TOKENS:
            return OcrResult(text="", confidence=0.0)
        return OcrResult(text=best_text, confidence=float(best_conf))


class TesseractEngine:
    def __init__(self, lang: str = "eng"):
        import pytesseract
        self.pytesseract = pytesseract
        self.lang = lang

    def read(self, plate_bgr: np.ndarray) -> OcrResult:
        variants = _preprocess(plate_bgr, cut_blue=False)
        best_text = ""
        best_conf = 0.0
        best_score = -1e9

        for v in variants:
            if v.ndim == 3:
                gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
            else:
                gray = v
            thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            text = self.pytesseract.image_to_string(thr, lang=self.lang, config=config)
            text = _clean(text)
            conf = 0.55 if text else 0.0
            sc = _score(text, conf)
            if sc > best_score:
                best_score = sc
                best_text = text
                best_conf = float(conf)

        if best_text in BAD_TOKENS:
            return OcrResult(text="", confidence=0.0)
        return OcrResult(text=best_text, confidence=best_conf)


class FastTesseractEngine:
    """
    Tesseract-fast, ale "niegłupi":
      - kilka szybkich kandydatów (gray/otsu/otsu_inv/ad/ad_inv)
      - border (bardzo ważny dla Tesseract)
      - próby PSM: 7 -> 8 -> 6 (dopiero gdy wynik słaby)
      - early-exit gdy scoring OK, żeby trzymać czas
    """
    def __init__(
        self,
        lang: str = "eng",
        cut_blue_auto: bool = False,
        resize_fx: float = 2.1,
        oem: int = 3,
        score_ok_threshold: float = 70.0,
        score_hard_retry_threshold: float = 45.0,
    ):
        import pytesseract
        self.pytesseract = pytesseract
        self.lang = lang
        self.cut_blue_auto = bool(cut_blue_auto)
        self.resize_fx = float(resize_fx)
        self.oem = int(oem)
        self.score_ok_threshold = float(score_ok_threshold)
        self.score_hard_retry_threshold = float(score_hard_retry_threshold)

        # bez słowników
        self._base = (
            f"--oem {self.oem} "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
            "-c load_system_dawg=0 -c load_freq_dawg=0"
        )

    def _ocr_psm(self, img: np.ndarray, psm: int) -> str:
        cfg = f"{self._base} --psm {int(psm)}"
        txt = self.pytesseract.image_to_string(img, lang=self.lang, config=cfg)
        return _clean(txt)

    def read(self, plate_bgr: np.ndarray) -> OcrResult:
        cands = _fast_candidates_for_tesseract(
            plate_bgr,
            cut_blue_auto=self.cut_blue_auto,
            resize_fx=self.resize_fx,
            border=10,
        )
        if not cands:
            return OcrResult(text="", confidence=0.0)

        best_t = ""
        best_c = 0.0
        best_s = -1e9

        # kolejność psm: najpierw 7 (linia), potem 8 (słowo), potem 6 (blok)
        psm_order = (7, 8, 6)

        # 1) spróbuj szybko na pierwszych 2 kandydatach tylko PSM7
        for i in range(min(2, len(cands))):
            t = self._ocr_psm(cands[i], psm=7)
            c = 0.62 if t else 0.0
            s = _score(t, c)
            if s > best_s:
                best_s, best_t, best_c = s, t, c
            if best_s >= self.score_ok_threshold:
                if best_t in BAD_TOKENS:
                    return OcrResult(text="", confidence=0.0)
                return OcrResult(text=best_t, confidence=float(best_c))

        # 2) jeśli nadal słabo, przeleć po pozostałych kandydatach PSM7
        for i in range(2, len(cands)):
            t = self._ocr_psm(cands[i], psm=7)
            c = 0.62 if t else 0.0
            s = _score(t, c)
            if s > best_s:
                best_s, best_t, best_c = s, t, c
            if best_s >= self.score_ok_threshold:
                if best_t in BAD_TOKENS:
                    return OcrResult(text="", confidence=0.0)
                return OcrResult(text=best_t, confidence=float(best_c))

        # 3) „hard retry”: jeśli wynik jest dramatyczny, dopiero wtedy odpal PSM8/6 na TOP2 kandydatach
        if best_s < self.score_hard_retry_threshold:
            for psm in (8, 6):
                for i in range(min(2, len(cands))):
                    t = self._ocr_psm(cands[i], psm=psm)
                    c = 0.62 if t else 0.0
                    s = _score(t, c)
                    if s > best_s:
                        best_s, best_t, best_c = s, t, c
                    if best_s >= self.score_ok_threshold:
                        if best_t in BAD_TOKENS:
                            return OcrResult(text="", confidence=0.0)
                        return OcrResult(text=best_t, confidence=float(best_c))

        if best_t in BAD_TOKENS:
            return OcrResult(text="", confidence=0.0)
        return OcrResult(text=best_t, confidence=float(best_c))


def make_ocr_engine(
    engine: str,
    languages: List[str],
    tesseract_lang: str,
    fast_try_psm8: bool = True,          # zachowane dla kompatybilności
    fast_cut_blue_auto: bool = False,
    fast_resize_fx: float = 2.1,
    fast_oem: int = 3,
    **_ignored: object,
):
    engine = (engine or "").lower().strip()

    if engine == "tesseract":
        return TesseractEngine(lang=tesseract_lang)

    if engine == "easyocr":
        return EasyOcrEngine(languages=languages)

    if engine in ("tesseract_fast", "fast_tesseract", "tesseract-fast"):
        _ = fast_try_psm8  # kompatybilność
        return FastTesseractEngine(
            lang=tesseract_lang,
            cut_blue_auto=bool(fast_cut_blue_auto),
            resize_fx=float(fast_resize_fx),
            oem=int(fast_oem),
        )

    raise ValueError(f"Unsupported OCR engine: {engine}. Use easyocr, tesseract, or tesseract_fast.")
