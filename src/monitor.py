#!/usr/bin/env python3
"""
Live monitor mode: chat spike detection with a rolling pre roll buffer.

Why this exists
---------------
Chat reacts LATE. Stream latency (encode, CDN, viewer delay, typing time)
means the on screen moment happens well BEFORE chat goes crazy. So the
value of a chat spike trigger is the footage from BEFORE the spike. This
module therefore keeps a rolling buffer of the stream (at least 60 s) and,
when chat spikes, cuts a window that starts a configurable pre roll
(default 20 s, and 20 s is the minimum sensible default) BEFORE the
trigger, plus a short post trigger tail so the moment is not cut off
mid action.

Token cost: zero. Detection is pure counting statistics on chat text.
No transcription and no LLM run unless a trigger fires, and then only
on the single cut window (see pipeline.py).

Chat log format (JSON Lines), one message per line:
  {"ts": 1712345678.0, "user": "someone", "text": "LETS GOOO PogChamp"}
  ts may be epoch seconds or seconds since stream start, as long as it
  is monotonic. A CSV variant "ts,user,text" is also accepted.

Trigger math
------------
Chat is bucketed into 1 second counts. A trailing baseline window
(default 120 s) gives mean and standard deviation of messages per
second. Chat counts behave roughly like a Poisson process, where the
standard deviation is close to sqrt(mean), so a spike has to clear BOTH:
  * rate test : current 3 s average rate >= spike_ratio x baseline mean
                (default spike_ratio = 3.0)
  * z test    : current 3 s average rate >= baseline mean + z x baseline std
                (default z = 6.0, far outside normal chatter noise)
  * floor     : current rate >= min_msgs_per_sec (default 5) so a tiny
                chat going from 1 to 4 msgs/s never fires.
A spam boost signal (share of messages that are mostly CAPS or contain
known emote tokens) is recorded with every trigger as supporting
evidence; it does not fire on its own.
After a trigger, a cooldown (default: clip length + 30 s) suppresses
repeat triggers for the same event, so one spike produces one clip.
"""
import argparse, csv, io, json, math, os
from collections import deque
from dataclasses import dataclass, field

EMOTES = {"pogchamp", "lul", "omegalul", "kekw", "letsgo", "let's go",
          "pog", "poggers", "gg", "w", "lmao", "clipit", "clip it"}

@dataclass
class MonitorConfig:
    baseline_window_s: int = 120      # trailing window for baseline stats
    bucket_s: float = 1.0             # counting bucket
    rate_window_s: int = 3            # seconds averaged for the current rate
    spike_ratio: float = 3.0          # must be >= ratio x baseline mean
    z_threshold: float = 6.0          # and >= mean + z * std
    min_msgs_per_sec: float = 5.0     # absolute floor
    min_baseline_s: int = 30          # seconds of history before arming
    pre_roll_s: float = 20.0          # seconds BEFORE trigger in the cut (chat lag)
    post_roll_s: float = 10.0         # short tail AFTER trigger
    buffer_s: float = 60.0            # rolling buffer kept, must exceed pre roll
    cooldown_s: float = 0.0           # 0 means auto: pre+post+30

    def __post_init__(self):
        if self.buffer_s < self.pre_roll_s + 30:
            raise ValueError("buffer_s must exceed pre_roll_s by at least 30 s "
                             "so the pre roll is always available")
        if self.pre_roll_s < 20:
            raise ValueError("pre_roll_s below 20 s defeats the point: chat lags "
                             "the stream, so the moment lives before the spike")
        if self.cooldown_s <= 0:
            self.cooldown_s = self.pre_roll_s + self.post_roll_s + 30

@dataclass
class Trigger:
    trigger_ts: float
    clip_start: float
    clip_end: float
    rate: float
    baseline_mean: float
    baseline_std: float
    caps_ratio: float
    emote_ratio: float

    def as_dict(self):
        return dict(trigger_ts=self.trigger_ts, clip_start=self.clip_start,
                    clip_end=self.clip_end, rate=round(self.rate, 2),
                    baseline_mean=round(self.baseline_mean, 2),
                    baseline_std=round(self.baseline_std, 2),
                    caps_ratio=round(self.caps_ratio, 3),
                    emote_ratio=round(self.emote_ratio, 3))

class RollingBuffer:
    """Keeps the last buffer_s seconds of stream segment references.

    In live use each entry is a small .ts segment file written by ffmpeg's
    segment muxer; here entries are (ts, ref) pairs and old ones are dropped.
    The buffer guarantees clip_start is always >= the oldest kept ts when a
    trigger fires after warm up.
    """
    def __init__(self, buffer_s):
        self.buffer_s = buffer_s
        self.entries = deque()

    def push(self, ts, ref):
        self.entries.append((ts, ref))
        while self.entries and ts - self.entries[0][0] > self.buffer_s:
            self.entries.popleft()

    def oldest(self):
        return self.entries[0][0] if self.entries else None

    def covers(self, ts):
        o = self.oldest()
        return o is not None and o <= ts

class ChatSpikeDetector:
    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg
        self.counts = {}            # bucket index -> count
        self.caps = {}              # bucket index -> caps/emote counts
        self.emotes = {}
        self.first_ts = None
        self.last_trigger_end = -1e18
        self.triggers = []

    def _bucket(self, ts):
        return int(math.floor(ts / self.cfg.bucket_s))

    def feed(self, ts, text):
        if self.first_ts is None:
            self.first_ts = ts
        b = self._bucket(ts)
        self.counts[b] = self.counts.get(b, 0) + 1
        letters = [c for c in text if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7 and len(letters) >= 3:
            self.caps[b] = self.caps.get(b, 0) + 1
        low = text.lower()
        if any(e in low for e in EMOTES):
            self.emotes[b] = self.emotes.get(b, 0) + 1

    def _stats(self, upto_bucket):
        cfg = self.cfg
        lo = upto_bucket - int(cfg.baseline_window_s / cfg.bucket_s)
        vals = [self.counts.get(b, 0) for b in range(lo, upto_bucket)]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        return mean, math.sqrt(var)

    def check(self, now_ts):
        """Call once per bucket as time advances. Returns a Trigger or None."""
        cfg = self.cfg
        if self.first_ts is None or now_ts - self.first_ts < cfg.min_baseline_s:
            return None
        cur_bucket = self._bucket(now_ts)
        rate_buckets = list(range(cur_bucket - cfg.rate_window_s + 1, cur_bucket + 1))
        rate = sum(self.counts.get(b, 0) for b in rate_buckets) / cfg.rate_window_s
        mean, std = self._stats(cur_bucket - cfg.rate_window_s + 1)
        if mean <= 0:
            return None
        rate_ok = rate >= cfg.spike_ratio * mean
        z_ok = rate >= mean + cfg.z_threshold * std
        floor_ok = rate >= cfg.min_msgs_per_sec
        in_cooldown = now_ts < self.last_trigger_end + cfg.cooldown_s
        if rate_ok and z_ok and floor_ok and not in_cooldown:
            total = sum(self.counts.get(b, 0) for b in rate_buckets) or 1
            trig = Trigger(
                trigger_ts=now_ts,
                clip_start=now_ts - cfg.pre_roll_s,
                clip_end=now_ts + cfg.post_roll_s,
                rate=rate, baseline_mean=mean, baseline_std=std,
                caps_ratio=sum(self.caps.get(b, 0) for b in rate_buckets) / total,
                emote_ratio=sum(self.emotes.get(b, 0) for b in rate_buckets) / total)
            self.triggers.append(trig)
            self.last_trigger_end = trig.clip_end
            return trig
        return None

def load_chat_log(path):
    msgs = []
    with open(path) as f:
        head = f.read(4096); f.seek(0)
        if head.lstrip().startswith("{"):
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    msgs.append((float(d["ts"]), d.get("text", "")))
        else:
            for row in csv.reader(f):
                if len(row) >= 3 and row[0].replace(".", "", 1).isdigit():
                    msgs.append((float(row[0]), row[2]))
    msgs.sort(key=lambda m: m[0])
    return msgs

def run_detector(msgs, cfg):
    """Feed a recorded chat log through the detector with a rolling buffer."""
    det = ChatSpikeDetector(cfg)
    buf = RollingBuffer(cfg.buffer_s)
    if not msgs:
        return det.triggers, buf
    t0 = msgs[0][0]
    idx = 0
    now = t0
    end = msgs[-1][0] + cfg.post_roll_s + 2
    while now <= end:
        while idx < len(msgs) and msgs[idx][0] <= now:
            det.feed(msgs[idx][0], msgs[idx][1]); idx += 1
        buf.push(now, f"segment@{now:.0f}")
        trig = det.check(now)
        if trig:
            trig.buffer_covered = buf.covers(trig.clip_start)
            if not trig.buffer_covered and (now - t0) >= cfg.buffer_s:
                raise RuntimeError("buffer failed to cover the pre roll window")
        now += cfg.bucket_s
    return det.triggers, buf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat-log", help="JSONL or CSV chat log to scan")
    ap.add_argument("--pre-roll", type=float, default=20.0)
    ap.add_argument("--post-roll", type=float, default=10.0)
    ap.add_argument("--spike-ratio", type=float, default=3.0)
    ap.add_argument("--buffer", type=float, default=60.0)
    args = ap.parse_args()
    cfg = MonitorConfig(pre_roll_s=args.pre_roll, post_roll_s=args.post_roll,
                        spike_ratio=args.spike_ratio, buffer_s=args.buffer)
    if args.chat_log:
        msgs = load_chat_log(args.chat_log)
        trigs, _ = run_detector(msgs, cfg)
        print(json.dumps([t.as_dict() for t in trigs], indent=1))
        print(f"{len(trigs)} trigger(s) from {len(msgs)} messages")
    else:
        print("no chat log given; run the synthetic test with: "
              "python3 tests/test_monitor.py")

if __name__ == "__main__":
    main()
