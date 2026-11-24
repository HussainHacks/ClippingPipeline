#!/usr/bin/env python3
"""
Synthetic chat test for the live monitor.

Simulates 300 s of chat:
  * 0 to 195 s   : normal chatter, Poisson rate 2 msgs/s, low caps/emote use
  * 200 to 206 s : injected spike, 28 msgs/s, heavy caps and emote spam
  * 207 to 300 s : back to normal chatter

Expectations:
  1. No trigger fires during normal chatter.
  2. Exactly one trigger fires for the spike (cooldown suppresses repeats).
  3. The trigger lands within a few seconds of the spike start.
  4. The clip window start lands 20 s BEFORE the spike (chat lags the
     stream, so the moment lives in the pre roll), within tolerance.
  5. The rolling buffer covers the clip start (60 s buffer).

Run:  python3 tests/test_monitor.py
"""
import os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from monitor import MonitorConfig, run_detector

def poisson(rng, lam):
    # Knuth, fine for small lambda; for the spike use normal approximation
    if lam > 12:
        return max(0, int(rng.gauss(lam, lam ** 0.5)))
    L, k, p = pow(2.718281828, -lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1

def simulate(seed=42):
    rng = random.Random(seed)
    msgs = []
    normal = ["nice play", "that was close", "hello everyone", "good stream",
              "what game is this", "lol", "that aim though", "hello"]
    spike = ["LETS GOOOO", "POGCHAMP POGCHAMP", "WHAT WAS THAT", "CLIP IT",
             "OMEGALUL", "INSANE PLAY", "GG GG GG", "NO WAYYYY"]
    for t in range(0, 300):
        if 200 <= t < 206:
            n, pool, p_spam = poisson(rng, 28), spike, 0.9
        else:
            n, pool, p_spam = poisson(rng, 2), normal, 0.08
        for _ in range(n):
            ts = t + rng.random()
            text = rng.choice(pool) if rng.random() < p_spam + 0.1 else rng.choice(normal)
            msgs.append((ts, text))
    msgs.sort(key=lambda m: m[0])
    return msgs

def main():
    cfg = MonitorConfig()   # defaults: pre roll 20 s, post roll 10 s, buffer 60 s
    msgs = simulate()
    trigs, buf = run_detector(msgs, cfg)
    print(f"simulated {len(msgs)} messages over 300 s")
    assert len(trigs) >= 1, "FAIL: spike produced no trigger"
    first = trigs[0]
    print("first trigger:", first.as_dict())
    assert len(trigs) == 1, f"FAIL: expected exactly 1 trigger, got {len(trigs)}"
    assert 199 <= first.trigger_ts <= 206, f"FAIL: trigger at {first.trigger_ts}, spike at 200"
    lead = 200 - first.clip_start
    print(f"clip start {first.clip_start:.0f} is {lead:.0f} s before the spike (target 20 s)")
    assert 16 <= lead <= 24, f"FAIL: pre roll lead {lead:.1f} s, expected about 20 s"
    assert getattr(first, "buffer_covered", False), "FAIL: rolling buffer did not cover clip start at trigger time"
    assert first.clip_end - first.trigger_ts == cfg.post_roll_s
    print("PASS: one trigger, on time, clip window starts 20 s before the spike, buffer covers it")

if __name__ == "__main__":
    main()
