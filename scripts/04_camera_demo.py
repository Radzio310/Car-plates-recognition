from __future__ import annotations
import argparse
import time

import cv2

from anpr.pipeline import ANPRPipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/app_config.yaml")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--every", type=int, default=5, help="Analizuj co N klatek (dla wydajności)")
    args = ap.parse_args()

    pipeline = ANPRPipeline(config_path=args.config)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Nie mogę otworzyć kamery.")

    frame_i = 0
    last_text = ""
    last_granted = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        if frame_i % args.every == 0:
            out = pipeline.run(frame)
            if out.detected and out.bbox:
                x1, y1, x2, y2 = out.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = out.plate_text_norm or out.plate_text_raw or "?"
                granted = out.access_granted if out.plate_valid_format else None
                last_text, last_granted = text, granted

                if out.plate_valid_format:
                    print(f"Odczyt: {text} | access: {'GRANTED' if granted else 'DENIED'}")
                else:
                    print(f"Odczyt: {text} | (format niepoprawny)")

        # overlay status
        overlay = f"PLATE: {last_text}"
        if last_granted is True:
            overlay += " | GRANTED"
        elif last_granted is False:
            overlay += " | DENIED"
        cv2.putText(frame, overlay, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        cv2.imshow("ANPR camera demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
