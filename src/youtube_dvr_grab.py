#!/usr/bin/env python3
"""
YouTube live DVR grabber: records a window of a live stream from its
HLS DVR buffer instead of chasing the live edge.

Why this exists: YouTube live streams carry a DVR window (about
900 s) in their HLS media playlists. In this sandbox, ffmpeg reading
a live playlist through the egress proxy stalls (every segment URL
302 redirects to a chunk host, and each redirected fetch is slow),
and streamlink cannot reach YouTube through the proxy at all. But
individual segment fetches with curl are fast (about 1.7 s per 2 s
segment, faster in parallel), so this tool:

  1. asks yt-dlp for the media playlist URLs of one video and one
     audio rendition,
  2. downloads both playlists and picks an end aligned window of the
     requested length a few segments behind the live edge,
  3. downloads every segment in parallel with curl (redirects
     followed, retries on), in sequence order,
  4. concatenates each rendition and muxes them with ffmpeg.

The result is a normal .ts recording of real stream content, a few
minutes behind the live edge, ready for pipeline.py. Video and audio
windows are end aligned at the same playlist snapshot, so sync is
within about one segment (2 s worst case).

Usage:
  python3 src/youtube_dvr_grab.py --video <id> --seconds 620 \
      --out work/youtube-test/capture.ts
"""
import argparse, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YTDLP = os.path.join(ROOT, ".venv", "bin", "yt-dlp")


def playlist_url(video, fmt):
    r = subprocess.run([YTDLP, "--no-check-certificates",
                        "--socket-timeout", "45", "-g", "-f", fmt,
                        f"https://www.youtube.com/watch?v={video}"],
                       capture_output=True, text=True, timeout=180)
    url = (r.stdout or "").strip().splitlines()
    if r.returncode != 0 or not url:
        raise SystemExit(f"yt-dlp -g failed for format {fmt}: "
                         f"{(r.stderr or '')[-300:]}")
    return url[0].strip()


def fetch_playlist(url):
    r = subprocess.run(["curl", "-skL", "--max-time", "60", url],
                       capture_output=True, text=True, timeout=90)
    if r.returncode != 0 or "#EXTM3U" not in r.stdout:
        raise SystemExit("playlist fetch failed")
    segs = []  # (sq, duration, url)
    dur = None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("#EXTINF:"):
            dur = float(line.split(":", 1)[1].split(",")[0])
        elif line.startswith("http"):
            m = re.search(r"/sq/(\d+)/", line)
            segs.append((int(m.group(1)) if m else len(segs),
                         dur or 2.0, line))
            dur = None
    return segs


def pick_window(segs, seconds, edge_margin=6):
    """End aligned window: walk back from a few segments behind the
    live edge until the requested duration is covered."""
    end = max(0, len(segs) - edge_margin)
    total, start = 0.0, end
    while start > 0 and total < seconds:
        start -= 1
        total += segs[start][1]
    return segs[start:end]


def _grab(job):
    url, path = job
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    r = subprocess.run(["curl", "-skL", "--retry", "3", "--retry-delay", "2",
                        "--max-time", "90", url, "-o", path],
                       capture_output=True, timeout=180)
    if r.returncode != 0 or not os.path.exists(path) \
            or os.path.getsize(path) == 0:
        raise RuntimeError(f"segment download failed: {url[:80]}...")
    return path


def download_window(segs, folder, prefix, parallel=8):
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(folder, exist_ok=True)
    jobs = [(url, os.path.join(folder, f"{prefix}_{i:05d}.ts"))
            for i, (_sq, _d, url) in enumerate(segs)]
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        list(ex.map(_grab, jobs))
    out = os.path.join(folder, f"{prefix}_joined.ts")
    with open(out, "wb") as w:
        for _u, p in jobs:
            with open(p, "rb") as rf:
                w.write(rf.read())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seconds", type=float, default=620)
    ap.add_argument("--format-v", default="231", help="video rendition (231 = 480p)")
    ap.add_argument("--format-a", default="234", help="audio rendition (234 = high)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if True:
        td = os.path.join(ROOT, "work", "youtube-test", "dvr-parts")
        os.makedirs(td, exist_ok=True)
        print("[dvr] fetching playlists")
        vsegs = pick_window(fetch_playlist(playlist_url(args.video, args.format_v)),
                            args.seconds)
        asegs = pick_window(fetch_playlist(playlist_url(args.video, args.format_a)),
                            args.seconds)
        vdur = sum(d for _s, d, _u in vsegs)
        print(f"[dvr] window: video {len(vsegs)} segs ({vdur:.0f} s), "
              f"audio {len(asegs)} segs")
        vjoined = download_window(vsegs, td, "v")
        ajoined = download_window(asegs, td, "a")
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", vjoined,
                            "-i", ajoined, "-map", "0:v:0", "-map", "1:a:0",
                            "-c", "copy", args.out],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"mux failed: {r.stderr[-500:]}")
    print(f"[dvr] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
