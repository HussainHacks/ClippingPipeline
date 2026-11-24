#!/usr/bin/env python3
"""
End to end video clipping pipeline.

Stages:
  1. ingest     : local file copy, direct URL download, or yt-dlp for page URLs
  2. transcribe : faster-whisper (tiny model, CPU) -> transcript.json + transcript.srt
  3. highlights : heuristic ranking of candidate moments (no LLM key needed).
                  Scoring mirrors the offline modes of the open source clippers:
                  hook words, questions, numbers, imperatives, speech density,
                  transcription confidence, self contained sentence boundaries.
  4. render     : cut + reframe to vertical 1080x1920 + burn in captions (ASS/libass)
                  Reframe layouts: "fit" (blurred background fill, whole frame kept,
                  the Reelsaw fit layout) or "crop" (center crop to 9:16).

Usage:
  python3 src/pipeline.py run --input <file-or-url> --clips 3
  python3 src/pipeline.py run --input video.mp4 --layout crop
Outputs land in output/ and clips go to clips/.
"""
import argparse, json, math, os, re, shutil, subprocess, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work")
OUT = os.path.join(ROOT, "output")
CLIPS = os.path.join(ROOT, "clips")

# Watermark burned into every rendered clip :
# the clips page handle, semi transparent (45 percent opacity), present
# for the whole clip, bottom right corner. That corner is clear of the
# caption band (bottom center, ASS MarginV 420) and the hook line
# (top center). Toggle off with WATERMARK=0 or --no-watermark;
# change the text with WATERMARK_TEXT.
WATERMARK_TEXT = os.environ.get("WATERMARK_TEXT", "@yourchannel")
WATERMARK_ON = os.environ.get("WATERMARK", "1") != "0"
WATERMARK_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Clickbait title burned into every rendered clip (house rule,
# 2026-11-17): a funny, relatively absurd title naming the streamer,
# generated rule based from the clip's own transcript (zero LLM
# tokens, see make_clickbait_title), shown for the WHOLE clip in the
# upper center, bold white on a translucent black box. It sits below
# the 3.5 s hook line and well clear of the bottom caption band.
# Toggle off with --no-clickbait or CLICKBAIT=0; set the streamer
# name with --streamer or STREAMER_NAME.
CLICKBAIT_ON = os.environ.get("CLICKBAIT", "1") != "0"
CLICKBAIT_FONT = WATERMARK_FONT
CLICKBAIT_Y = 300          # px from the top of the 1080x1920 frame
CLICKBAIT_FONTSIZE = 58
CLICKBAIT_WRAP = 24        # max characters per line before wrapping

HOOK_WORDS = {
    "remember", "important", "never", "always", "must", "first", "watch",
    "listen", "look", "stop", "duck", "cover", "danger", "warning", "flash",
    "suddenly", "now", "ready", "protect", "safe", "safety", "help", "know",
    "learn", "secret", "why", "how", "everyone", "children", "family",
}
IMPERATIVES = {"duck", "cover", "stop", "watch", "listen", "look", "remember", "get", "stay", "keep"}

def sh(cmd):
    print("+", " ".join(cmd[:6]), "..." if len(cmd) > 6 else "")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-3000:])
        raise SystemExit(f"command failed: {cmd[0]}")
    return r

def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-show_entries", "stream=codec_type,width,height",
                        "-of", "json", path], capture_output=True, text=True)
    return json.loads(r.stdout)

# ---------- stage 1: ingest ----------
def ingest(src):
    os.makedirs(WORK, exist_ok=True)
    dst = os.path.join(WORK, "source.mp4")
    if os.path.exists(src):
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
        print(f"[ingest] local file -> {dst}")
    elif src.startswith("http"):
        if "youtube.com" in src or "youtu.be" in src or "twitch.tv" in src:
            ytdlp = os.path.join(ROOT, ".venv", "bin", "yt-dlp")
            tool = ytdlp if os.path.exists(ytdlp) else "yt-dlp"
            sh([tool, "-f", "mp4/best", "-o", dst, src])
        else:
            req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dst, "wb") as f:
                shutil.copyfileobj(resp, f)
        print(f"[ingest] downloaded -> {dst}")
    else:
        raise SystemExit(f"cannot ingest: {src}")
    info = probe(dst)
    dur = float(info["format"]["duration"])
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    print(f"[ingest] duration {dur:.1f}s, video {v['width']}x{v['height']}")
    return dst, dur

# ---------- stage 1.5: cheap signal gate (zero tokens, zero ASR) ----------
def cheap_signals(wav):
    """Per second audio energy from the raw PCM. Pure numpy, no model.

    Gate rule: a second passes when its RMS level is within 6 dB of the
    median level (active audio, not silence or near silence). Passing
    seconds are merged into windows with 4 s of padding. On a gameplay
    stream with long quiet stretches this rejects most of the timeline
    before any transcription runs. On a narration dense source it passes
    most of the file, which the run log states honestly.
    Returns (windows, stats) where windows are (start, end) seconds.
    """
    import wave as wavmod
    import numpy as np
    with wavmod.open(wav, "rb") as wf:
        sr = wf.getframerate()
        x = np.frombuffer(wf.readframes(wf.getnframes()),
                          dtype=np.int16).astype(np.float32) / 32768.0
    per = []
    for s in range(int(len(x) / sr)):
        seg = x[s * sr:(s + 1) * sr]
        per.append(float(np.sqrt(np.mean(seg ** 2))))
    db = 20 * np.log10(np.array(per) + 1e-6)
    thr = float(np.median(db)) - 6.0
    passed = db > thr
    windows, start = [], None
    for i, ok in enumerate(passed):
        if ok and start is None:
            start = i
        if not ok and start is not None:
            windows.append((start, i)); start = None
    if start is not None:
        windows.append((start, len(passed)))
    merged = []
    for a, b in windows:
        a, b = max(0, a - 4), min(len(per), b + 4)
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    merged = [(a, b) for a, b in merged if b - a >= 8]
    gated_s = sum(b - a for a, b in merged)
    stats = {"total_s": int(len(per)), "gated_s": int(gated_s),
             "gate_threshold_db": round(thr, 1),
             "gate_pass_fraction": round(gated_s / max(1, len(per)), 3),
             "windows": len(merged)}
    print(f"[gate] audio energy gate: {gated_s}s of {len(per)}s pass "
          f"({stats['gate_pass_fraction']:.0%}), threshold {thr:.1f} dB, "
          f"{len(merged)} window(s). Zero tokens, zero ASR spent so far.")
    return merged, stats

# ---------- stage 2: transcribe (gated windows only) ----------
def _source_sig(path):
    try:
        dur = float(probe(path)["format"]["duration"])
    except Exception:
        dur = 0
    return {"size": os.path.getsize(path), "duration": round(dur, 1)}

def transcribe(video, use_gate=True):
    tj = os.path.join(WORK, "transcript.json")
    sig = _source_sig(video)
    if os.path.exists(tj):
        cached = json.load(open(tj))
        if cached.get("source_sig") == sig:
            print("[transcribe] cached transcript.json matches this source, reusing")
            return cached
        print("[transcribe] cached transcript belongs to a different source, "
              "transcribing fresh")
    wav = os.path.join(WORK, "audio.wav")
    sh(["ffmpeg", "-y", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", wav])
    from faster_whisper import WhisperModel
    import wave as wavmod
    import numpy as np
    model_name = os.environ.get("WHISPER_MODEL", "tiny")
    print(f"[transcribe] faster-whisper model={model_name} on CPU (int8)")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    with wavmod.open(wav, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    full = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    total_s = len(full) / sr
    if use_gate:
        windows, gate_stats = cheap_signals(wav)
    else:
        windows, gate_stats = [(0, int(total_s))], {"total_s": int(total_s),
                                                    "gated_s": int(total_s),
                                                    "gate_pass_fraction": 1.0,
                                                    "windows": 1}
    # Feed raw PCM arrays per window: avoids a PyAV version clash in
    # faster-whisper's decode_audio on this VM (av.open metadata_errors).
    segs, lang = [], "en"
    for (a, b) in windows:
        audio = full[int(a * sr):int(b * sr)]
        if len(audio) < sr:
            continue
        segments, info = model.transcribe(audio, word_timestamps=True,
                                          vad_filter=True, language="en")
        lang = info.language
        for s in segments:
            words = [{"start": round(w.start + a, 3), "end": round(w.end + a, 3),
                      "word": w.word} for w in (s.words or [])]
            segs.append({"start": round(s.start + a, 3), "end": round(s.end + a, 3),
                         "text": s.text.strip(), "avg_logprob": s.avg_logprob,
                         "words": words})
            print(f"  [{s.start + a:7.2f} -> {s.end + a:7.2f}] {s.text.strip()[:80]}")
    segs.sort(key=lambda s: s["start"])
    data = {"language": lang, "duration": total_s, "segments": segs,
            "gate": gate_stats, "source_sig": sig}
    json.dump(data, open(tj, "w"), indent=1)
    with open(os.path.join(WORK, "transcript.srt"), "w") as f:
        for i, s in enumerate(segs, 1):
            def ts(t):
                h = int(t // 3600); m = int(t % 3600 // 60); sec = t % 60
                return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")
            f.write(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n\n")
    print(f"[transcribe] {len(segs)} segments, language={info.language}")
    return data

# ---------- stage 3: highlights ----------
def score_window(win_segs):
    text = " ".join(s["text"] for s in win_segs)
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return -1e9, text
    dur = win_segs[-1]["end"] - win_segs[0]["start"]
    score = 0.0
    score += 2.0 * sum(1 for w in words if w in HOOK_WORDS) / max(1, len(words)) * 10
    score += 3.0 if "?" in text else 0.0
    score += 1.5 if "!" in text else 0.0
    score += 1.5 * min(3, len(re.findall(r"\d", text)))
    score += 2.0 if words[0] in IMPERATIVES else 0.0
    density = len(words) / max(1.0, dur)          # words per second
    score += min(4.0, density)                    # reward clear steady narration
    conf = sum(s["avg_logprob"] for s in win_segs) / len(win_segs)
    score += max(-2.0, conf)                      # whisper confidence, usually -1..0
    if dur < 20: score -= (20 - dur) * 0.3        # too short to stand alone
    if dur > 60: score -= (dur - 60) * 0.2        # too long for a short
    low = text.lower()
    if low.count("bert") > 1 or "turtle" in low and "sings" in low:
        score -= 3.0                              # sung intro, weak clip material
    return round(score, 3), text

def highlights(data, n_clips=3, min_len=25, max_len=55):
    segs = data["segments"]
    candidates = []
    for i in range(len(segs)):
        start = segs[i]["start"]
        j = i
        while j < len(segs) and segs[j]["end"] - start < min_len:
            j += 1
        while j < len(segs) and segs[j]["end"] - start <= max_len:
            win = segs[i:j + 1]
            sc, text = score_window(win)
            candidates.append({"start": round(start, 2),
                               "end": round(win[-1]["end"], 2),
                               "duration": round(win[-1]["end"] - start, 2),
                               "score": sc, "text": text})
            j += 1
    candidates.sort(key=lambda c: -c["score"])
    picked = []
    for c in candidates:
        if all(c["end"] <= p["start"] or c["start"] >= p["end"] for p in picked):
            hook = c["text"].split(".")[0].strip()
            c["hook"] = (hook[:57] + "...") if len(hook) > 60 else hook
            picked.append(c)
        if len(picked) >= n_clips:
            break
    picked.sort(key=lambda c: c["start"])
    for rank, c in enumerate(picked, 1):
        c["rank"] = rank
    json.dump(picked, open(os.path.join(WORK, "highlights.json"), "w"), indent=1)
    print(f"[highlights] picked {len(picked)} moments from {len(candidates)} candidates")
    for c in picked:
        print(f"  #{c['rank']} {c['start']:.1f}s->{c['end']:.1f}s score={c['score']} :: {c['text'][:70]}")
    return picked

# ---------- stage 3.5: clickbait title (rule based, zero tokens) ----------
CLICKBAIT_STOPWORDS = {
    "that", "this", "with", "have", "what", "when", "were", "they", "them",
    "then", "than", "just", "like", "gonna", "yeah", "right", "know", "gotta",
    "want", "going", "come", "came", "over", "under", "about", "there",
    "their", "been", "were", "will", "would", "could", "should", "really",
    "very", "much", "more", "some", "into", "from", "your", "yours", "mine",
    "ours", "lets", "let's", "dont", "don't", "cant", "can't", "aint",
    "okay", "well", "back", "here", "once", "again", "still", "even",
    "also", "because", "through", "before", "after", "while", "where",
}

def _content_words(text):
    """Content words of a transcript, most frequent first."""
    from collections import Counter
    words = [w for w in re.findall(r"[a-z']+", text.lower())
             if len(w) >= 4 and w not in CLICKBAIT_STOPWORDS]
    return [w for w, _ in Counter(words).most_common()]

def make_clickbait_title(clip, streamer=None):
    """Funny, relatively absurd clickbait title for one clip.

    Pure rules over the clip's transcript: pull the most repeated
    content words, spot numbers and questions, then fill one of a set
    of absurd templates naming the streamer (the CaseOh test titles
    Approved examples are the house style: "A TREE JUST TRIPPED CASEOH
    AND HE SURVIVED TO TELL THE TALE"). Deterministic per clip, so a
    rerun produces the same title. Zero LLM tokens.
    """
    name = (streamer or os.environ.get("STREAMER_NAME")
            or "THIS STREAMER").strip().upper()
    text = clip.get("text", "")
    words = _content_words(text)
    k1 = words[0].upper() if words else "CHAOS"
    k2 = words[1].upper() if len(words) > 1 else "DISASTER"
    templates = [
        f"THE {k1} JUST DECLARED WAR ON {name}",
        f"{name} TRUSTED THE {k1} AND THAT WAS A HUGE MISTAKE",
        f"NOBODY WARNED {name} ABOUT THE {k1}",
        f"THE {k1} INCIDENT THAT COMPLETELY BROKE {name}",
        f"{name} VS THE {k1}: A LOVE STORY NOBODY ASKED FOR",
        f"THE {k1} MET THE {k2} AND {name} PAID THE PRICE",
    ]
    # Number template only when the number is a real count: not a
    # percentage fragment, and with the noun that actually follows it
    # in the transcript ("send me seven bands" style, digit form).
    mnum = re.search(r"(\d[\d,]*)\s+([a-z']{3,})", text)
    if mnum and not text[mnum.start():].startswith(
            mnum.group(1) + "%") and "%" not in text[mnum.start(1):mnum.end(1) + 1] \
            and mnum.group(2) not in CLICKBAIT_STOPWORDS:
        templates.append(
            f"{name} FOUND {mnum.group(1)} {mnum.group(2).upper()} "
            "AND COMPLETELY LOST IT")
    if "?" in text:
        templates.append(
            f"{name} ASKED ABOUT THE {k1} AND CHAT WENT FERAL")
    pick = int(clip.get("start", 0)) + len(text)
    return templates[pick % len(templates)]

def wrap_title(title, width=CLICKBAIT_WRAP):
    lines, cur = [], ""
    for w in title.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines

# ---------- stage 4: render ----------
def ass_time(t):
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def make_ass(clip, data, path):
    start, end = clip["start"], clip["end"]
    words = []
    for s in data["segments"]:
        for w in s["words"]:
            if start - 0.01 <= w["start"] < end:
                words.append(w)
    events = []
    chunk, chunk_start = [], None
    for w in words:
        if chunk_start is None:
            chunk_start = w["start"]
        chunk.append(w)
        if len(chunk) >= 7 or (w["end"] - chunk_start) >= 2.4:
            events.append((chunk_start - start, w["end"] - start,
                           "".join(x["word"] for x in chunk).strip()))
            chunk, chunk_start = [], None
    if chunk:
        events.append((chunk_start - start, chunk[-1]["end"] - start,
                       "".join(x["word"] for x in chunk).strip()))
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,64,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,420,1
Style: Hook,DejaVu Sans,72,&H0000FFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,2,8,60,60,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    hook = clip["hook"].replace("{", "").replace("}", "").replace("\n", " ")
    lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(min(3.5, clip['duration']))},Hook,,0,0,0,,{hook}")
    for a, b, txt in events:
        txt = txt.replace("{", "").replace("}", "")
        lines.append(f"Dialogue: 0,{ass_time(max(0, a))},{ass_time(max(a + 0.3, b))},Cap,,0,0,0,,{txt}")
    open(path, "w").write("\n".join(lines) + "\n")
    return len(events)

def render(video, data, picked, layout="fit", watermark=WATERMARK_ON,
           out_dir=CLIPS, clickbait=CLICKBAIT_ON, streamer=None):
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for c in picked:
        name = f"clip{ c['rank'] }_{int(c['start'])}s.mp4"
        outp = os.path.join(out_dir, name)
        ass = os.path.join(WORK, f"clip{c['rank']}.ass")
        n_events = make_ass(c, data, ass)
        title = make_clickbait_title(c, streamer) if clickbait else None
        if title:
            print(f"[clickbait] clip {c['rank']}: {title}")
        dur = c["end"] - c["start"]
        if layout == "crop":
            vf = ("scale=1080:1920:force_original_aspect_ratio=increase,"
                  "crop=1080:1920")
        else:  # fit: blurred background fill, whole frame visible (Reelsaw "fit")
            vf = ("split[a][b];"
                  "[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
                  "gblur=sigma=55,eq=brightness=-0.12[bg];"
                  "[b]scale=1080:-2[fg];"
                  "[bg][fg]overlay=(W-w)/2:(H-h)/2")
        vf += f",subtitles='{ass}'"
        if title:
            for li, line in enumerate(wrap_title(title)):
                tf = os.path.join(WORK, f"clickbait{c['rank']}_{li}.txt")
                open(tf, "w").write(line)
                y = CLICKBAIT_Y + li * (CLICKBAIT_FONTSIZE + 16)
                vf += (f",drawtext=fontfile={CLICKBAIT_FONT}:"
                       f"textfile='{tf}':expansion=none:"
                       f"fontsize={CLICKBAIT_FONTSIZE}:"
                       "fontcolor=white:borderw=3:bordercolor=black:"
                       "box=1:boxcolor=black@0.45:boxborderw=18:"
                       f"x=(w-tw)/2:y={y}")
        if watermark:
            vf += (f",drawtext=fontfile={WATERMARK_FONT}:"
                   f"text='{WATERMARK_TEXT}':fontsize=40:"
                   "fontcolor=white@0.45:borderw=2:bordercolor=black@0.22:"
                   "x=w-tw-36:y=h-th-36")
        sh(["ffmpeg", "-y", "-v", "error", "-ss", str(c["start"]), "-i", video,
            "-t", f"{dur:.2f}", "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", outp])
        info = probe(outp)
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        rec = {"file": outp, "rank": c["rank"], "start": c["start"], "end": c["end"],
               "duration_s": round(float(info["format"]["duration"]), 2),
               "width": v["width"], "height": v["height"],
               "caption_events": n_events, "score": c["score"], "hook": c["hook"],
               "watermark": (WATERMARK_TEXT if watermark else None),
               "clickbait_title": title,
               "text": c["text"]}
        made.append(rec)
        print(f"[render] {name}: {rec['duration_s']}s {v['width']}x{v['height']}, {n_events} caption events")
    json.dump(made, open(os.path.join(out_dir, "clips-manifest.json"), "w"), indent=1)
    if os.path.abspath(out_dir) == os.path.abspath(CLIPS):
        json.dump(made, open(os.path.join(OUT, "clips-manifest.json"), "w"), indent=1)
    return made

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--input", required=True)
    r.add_argument("--clips", type=int, default=3)
    r.add_argument("--layout", choices=["fit", "crop"], default="fit")
    r.add_argument("--no-gate", action="store_true",
                   help="transcribe the whole file instead of gated windows only")
    r.add_argument("--no-watermark", action="store_true",
                   help="render without the @yourchannel watermark")
    r.add_argument("--no-clickbait", action="store_true",
                   help="render without the burned in clickbait title")
    r.add_argument("--streamer", default=None,
                   help="streamer name used in clickbait titles "
                        "(or set STREAMER_NAME)")
    r.add_argument("--out-dir", default=CLIPS,
                   help="folder for rendered clips (default: clips/)")
    args = ap.parse_args()
    video, _ = ingest(args.input)
    data = transcribe(video, use_gate=not args.no_gate)
    picked = highlights(data, n_clips=args.clips)
    made = render(video, data, picked, layout=args.layout,
                  watermark=not args.no_watermark, out_dir=args.out_dir,
                  clickbait=not args.no_clickbait, streamer=args.streamer)
    print(json.dumps(made, indent=1))
    print("DONE")

if __name__ == "__main__":
    main()
