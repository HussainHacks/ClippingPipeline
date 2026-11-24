#!/usr/bin/env python3
"""
YouTube Live Chat capture for monitor mode (no API key, HTTPS only).

Why this exists: monitor.py needs a chat feed in its JSONL format
({"ts","user","text"}). Twitch chat is IRC over raw TCP, which is
blocked in some sandboxes. YouTube Live Chat can be polled over plain
HTTPS using the same innertube endpoint the watch page itself uses:

  1. GET the watch page, pull INNERTUBE_API_KEY, the client version,
     and the first live chat continuation token out of ytInitialData.
  2. POST youtubei/v1/live_chat/get_live_chat with that continuation.
     Each response carries chat actions plus the next continuation
     token and a suggested poll timeout.
  3. Append every text message as one JSONL line and repeat until the
     duration elapses, the stream ends, or chat closes.

Zero LLM tokens; this is plain HTTP polling and JSON parsing.

Usage:
  python3 src/youtube_chat.py --video jBz7NqfeUVw --duration 600 \
      --out work/youtube-test/chat.jsonl
  # or: --url "https://www.youtube.com/watch?v=..."

Note on TLS: this sandbox egresses through a proxy that presents a
self signed certificate chain, so verification is disabled by default
here (same reason yt-dlp needs --no-check-certificates in this
environment). Pass --verify-tls on a normal network.
"""
import argparse, json, re, ssl, sys, time, urllib.request

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def http(url, data=None, verify=True, timeout=30):
    ctx = None if verify else ssl._create_unverified_context()
    req = urllib.request.Request(url, data=data, headers={
        "User-Agent": UA,
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return r.read().decode("utf-8", errors="replace")


def extract_initial(html):
    """Pull (api_key, client_version, continuation) out of a watch page."""
    key = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', html)
    ver = re.search(r'"INNERTUBE_CLIENT_VERSION":"([^"]+)"', html)
    m = re.search(r"var ytInitialData = (\{.*?\});</script>", html, re.S)
    cont = None
    if m:
        try:
            data = json.loads(m.group(1))
            lcr = (data["contents"]["twoColumnWatchNextResults"]
                   ["conversationBar"]["liveChatRenderer"])
            conts = lcr.get("continuations") or []
            if conts:
                cont = (conts[0].get("reloadContinuationData", {})
                        .get("continuation"))
        except Exception:
            cont = None
    if cont is None:  # fallback: first continuation token near the chat block
        idx = html.find("liveChatRenderer")
        if idx >= 0:
            cm = re.search(r'"continuation":"([^"]+)"', html[idx:idx + 20000])
            if cm:
                cont = cm.group(1)
    return (key.group(1) if key else None,
            ver.group(1) if ver else "2.20240101.00.00",
            cont)


def message_from_action(action):
    item = (action.get("addChatItemAction") or {}).get("item") or {}
    r = item.get("liveChatTextMessageRenderer")
    if not r:
        return None
    try:
        ts = int(r.get("timestampUsec", "0")) / 1e6
    except ValueError:
        ts = time.time()
    user = ((r.get("authorName") or {}).get("simpleText") or "?")
    runs = ((r.get("message") or {}).get("runs")) or []
    text = "".join(run.get("text", "") for run in runs)
    if not text:
        return None
    return {"ts": ts, "user": user, "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="YouTube video id of the live stream")
    ap.add_argument("--url", help="full watch URL (alternative to --video)")
    ap.add_argument("--duration", type=float, default=600,
                    help="seconds to capture (default 600)")
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--verify-tls", action="store_true",
                    help="verify TLS certificates (off by default, see docstring)")
    args = ap.parse_args()
    vid = args.video
    if args.url and not vid:
        m = re.search(r"[?&]v=([\w-]{6,})", args.url)
        vid = m.group(1) if m else None
    if not vid:
        raise SystemExit("need --video <id> or --url <watch url>")

    page = http(f"https://www.youtube.com/watch?v={vid}",
                verify=args.verify_tls)
    api_key, client_ver, cont = extract_initial(page)
    if not api_key or not cont:
        raise SystemExit(
            "could not extract innertube key or chat continuation from the "
            "watch page (chat disabled, stream not live, or page layout "
            "changed); nothing captured")
    print(f"[yt-chat] video {vid}, client {client_ver}, continuation ok")

    endpoint = ("https://www.youtube.com/youtubei/v1/live_chat/"
                f"get_live_chat?key={api_key}")
    start = time.time()
    n = 0
    with open(args.out, "w") as f:
        while time.time() - start < args.duration:
            body = json.dumps({
                "context": {"client": {"clientName": "WEB",
                                       "clientVersion": client_ver}},
                "continuation": cont,
            }).encode()
            try:
                raw = http(endpoint, data=body, verify=args.verify_tls,
                           timeout=60)
            except Exception as e:
                print(f"[yt-chat] poll error: {e}; retrying in 3 s")
                time.sleep(3)
                continue
            resp = json.loads(raw)
            lcc = ((resp.get("continuationContents") or {})
                   .get("liveChatContinuation"))
            if not lcc:
                print("[yt-chat] no liveChatContinuation in response; "
                      "chat ended or stream ended")
                break
            for action in lcc.get("actions") or []:
                msg = message_from_action(action)
                if msg:
                    f.write(json.dumps(msg) + "\n")
                    n += 1
            f.flush()
            nxt, timeout_ms = None, 2000
            for c in lcc.get("continuations") or []:
                td = c.get("timedContinuationData") or {}
                iv = c.get("invalidationContinuationData") or {}
                if td.get("continuation"):
                    nxt, timeout_ms = td["continuation"], td.get("timeoutMs", 2000)
                elif iv.get("continuation"):
                    nxt = iv["continuation"]
            if not nxt:
                print("[yt-chat] no next continuation; chat closed")
                break
            cont = nxt
            time.sleep(max(0.5, min(10.0, timeout_ms / 1000.0)))
    print(f"[yt-chat] done: {n} messages -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
