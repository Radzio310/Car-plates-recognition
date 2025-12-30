from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
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


def _detect_blue_strip_cut_x(bgr: np.ndarray) -> int:
    """
    Wykrywa niebieski pasek UE i wylicza x cięcia.
    FIX: cięcie jest "safe" (zostawiamy margines po lewej), żeby nie zjadać 1. litery.
    """
    h, w = bgr.shape[:2]
    if w < 120:  # przy małych cropach nie tnij w ogóle
        return 0

    left_w = int(0.25 * w)
    roi = bgr[:, :left_w].copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([90, 50, 50], dtype=np.uint8)
    upper = np.array([135, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if ratio < 0.10:  # podnieśliśmy próg -> mniej fałszywych cięć
        return 0

    cols = np.where(mask.max(axis=0) > 0)[0]
    if cols.size == 0:
        return 0

    raw_cut = int(cols.max()) + 1

    # SAFE: zostawiamy 10px (po resize będzie więcej), żeby nie "podgryzać" pierwszej litery
    safe_margin = 10
    cut = max(0, raw_cut - safe_margin)

    # nigdy nie tnij więcej niż 18% szerokości
    cut = min(cut, int(0.18 * w))
    return max(0, cut)


def _preprocess(img_bgr: np.ndarray, cut_blue: bool) -> List[np.ndarray]:
    """
    Zwraca warianty preprocessu:
    - color (dla EasyOCR),
    - gray+CLAHE,
    - threshold.
    cut_blue=True -> próbujemy uciąć pasek UE (safe).
    """
    if img_bgr is None or img_bgr.size == 0:
        return []

    img = img_bgr.copy()
    h, w = img.shape[:2]

    if cut_blue:
        cut = _detect_blue_strip_cut_x(img)
        if 0 < cut < w - 10:
            img = img[:, cut:].copy()

    # umiarkowany resize
    img = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2 = clahe.apply(gray)

    thr = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return [img, g2, thr]


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

    # mieszanka liter/cyfr
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

    # preferuj start literą
    if t[0].isalpha():
        score += 10
    else:
        score -= 10

    # confidence
    score += float(conf) * 25.0
    return score


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
        out = self.reader.readtext(
            rgb, detail=1, paragraph=False, allowlist=ALLOWED
        )
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
        # KLUCZ: liczymy OCR dla dwóch ścieżek:
        #  - bez cięcia paska
        #  - z safe-cięciem paska
        # i wybieramy wynik lepszym scoringiem.
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

            # fallback tesseract tylko gdy mamy fatalny wynik
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


def make_ocr_engine(engine: str, languages: List[str], tesseract_lang: str):
    engine = (engine or "").lower().strip()
    if engine == "tesseract":
        return TesseractEngine(lang=tesseract_lang)
    if engine == "easyocr":
        return EasyOcrEngine(languages=languages)
    raise ValueError(f"Unsupported OCR engine: {engine}. Use easyocr or tesseract.")
