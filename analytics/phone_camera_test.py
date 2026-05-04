import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import logging
import time
from pathlib import Path

import cv2

from interfaces.phone_camera import PhoneCameraServer, _local_ip


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser(description="Proof-of-life test for ManeNavSystem PhoneCam.")
    p.add_argument("--port", type=int, default=8443)
    p.add_argument("--out-dir", type=str, default="phone_recordings")
    p.add_argument("--labels", type=str, default="xiaomi,iphone")
    p.add_argument("--fps", type=float, default=20.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [s.strip().lower() for s in args.labels.split(",") if s.strip()]
    if not labels:
        labels = ["phone"]

    server = PhoneCameraServer(port=args.port)
    server.start()

    print()
    print("=" * 78)
    print(f"  PhoneCameraServer running on https://{_local_ip()}:{args.port}")
    print("-" * 78)
    for lbl in labels:
        print(f"  Open on {lbl.upper():>8s}  ->  {server.url_for(lbl)}")
    print("-" * 78)
    print("  1) Phone and laptop must be on the SAME Wi-Fi.")
    print("  2) Browser warns about a self-signed cert -> Advanced -> Proceed.")
    print("  3) Tap 'Start camera' on the phone, allow camera permission.")
    print("  4) A window appears on the laptop with that label streaming live.")
    print("  5) Recording starts automatically as soon as frames arrive.")
    print("  6) Press 'q' (in any video window) to stop and save MP4 files.")
    print("=" * 78)
    print()

    writers = {}
    last_seen = {}
    frame_counts = {lbl: 0 for lbl in labels}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    quit_flag = False
    last_summary = time.time()

    try:
        while not quit_flag:
            for lbl in labels:
                got = server.get_frame(lbl)
                if got is None:
                    continue
                frame, ts = got
                if last_seen.get(lbl) == ts:
                    continue
                last_seen[lbl] = ts
                frame_counts[lbl] += 1

                if lbl not in writers:
                    h, w = frame.shape[:2]
                    out_path = str(out_dir / f"proof_{lbl}.mp4")
                    writers[lbl] = cv2.VideoWriter(out_path, fourcc, float(args.fps), (w, h))
                    print(f"[{lbl}] connected: {w}x{h} -> recording to {out_path}")

                writers[lbl].write(frame)

                disp = frame.copy()
                txt = f"{lbl.upper()}  frames={frame_counts[lbl]}  ts={time.strftime('%H:%M:%S')}"
                cv2.rectangle(disp, (0, 0), (disp.shape[1], 40), (0, 0, 0), -1)
                cv2.putText(disp, txt, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(f"PhoneCam: {lbl}", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                quit_flag = True

            now = time.time()
            if now - last_summary > 5.0:
                connected = [lbl for lbl in labels if server.has_frames(lbl)]
                pending = [lbl for lbl in labels if lbl not in connected]
                print(
                    f"[status] connected={connected} pending={pending} "
                    f"counts={ {l:frame_counts[l] for l in labels} }"
                )
                last_summary = now

            time.sleep(0.003)
    finally:
        print()
        print("-" * 78)
        for lbl, w in writers.items():
            try:
                w.release()
                out_path = out_dir / f"proof_{lbl}.mp4"
                size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0.0
                print(f"  saved: {out_path}  ({frame_counts[lbl]} frames, {size_mb:.1f} MB)")
            except Exception as e:
                print(f"  failed to save {lbl}: {e}")
        cv2.destroyAllWindows()
        server.stop()
        print("  server stopped.")
        print("-" * 78)


if __name__ == "__main__":
    main()
