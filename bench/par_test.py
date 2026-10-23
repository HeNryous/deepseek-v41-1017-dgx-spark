# par_test.py [N]: one foreground stream (priority 0, six 500-token turns with 3 s pauses) plus N background requests
# (priority 10, ~30K uncached tokens each, 300-token answers), all started together, thinking off. Per background request:
# time to first byte, longest gap without data (what an inactivity timeout sees), total time, errors. Foreground: tok/s and
# time to first token per turn. Client timeout 1,800 s. Talks to http://127.0.0.1:8000 (the head node).
import json, os, random, string, sys, threading, time, urllib.request
U = "http://127.0.0.1:8000/v1/chat/completions"
M = json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=10))["data"][0]["id"]
N_BG = int(sys.argv[1]) if len(sys.argv) > 1 else 7
BG_CHARS = int(os.getenv("BG_CHARS", "120000"))
res_bg, res_fg, lock = [], [], threading.Lock()

def request(prompt, n, prio):
    b = {"model": M, "max_tokens": n, "temperature": 1.0, "priority": prio, "stream": True,
         "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False},
         "messages": [{"role": "user", "content": prompt}]}
    t0 = time.time(); last = t0; gap = 0.0; first = None; toks = 0
    r = urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=1800)
    for line in r:
        now = time.time(); gap = max(gap, now - last); last = now
        line = line.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]": continue
        j = json.loads(line[6:])
        if j.get("choices") and first is None: first = now
        if j.get("usage"): toks = j["usage"]["completion_tokens"]
    t1 = time.time(); f = first or t1
    return {"ttft": f - t0, "gap": gap, "total": t1 - t0, "toks": toks, "tps": toks / max(t1 - f, 1e-6)}

def bg(i):
    rg = random.Random(i * 7919 + int(time.time() * 1000))
    rnd = "".join(rg.choice(string.ascii_lowercase + " ") for _ in range(BG_CHARS))
    try:
        r = request("Summarize this text in 3 sentences:\n" + rnd, 300, 10)
    except Exception as e:
        r = {"error": repr(e)[:120]}
    r["i"] = i
    with lock: res_bg.append(r)

def fg():
    time.sleep(5)
    for k in range(6):
        try:
            r = request("Write a detailed essay on the history of the railway in Germany, part %d." % (k + 1), 500, 0)
        except Exception as e:
            r = {"error": repr(e)[:120]}
        res_fg.append(r); time.sleep(3)

t0 = time.time()
ths = [threading.Thread(target=bg, args=(i,)) for i in range(N_BG)] + [threading.Thread(target=fg)]
for t in ths: t.start()
for t in ths: t.join()
wall = time.time() - t0
ok = [r for r in res_bg if "error" not in r]; fok = [r for r in res_fg if "error" not in r]
fg_mean = sum(r["tps"] for r in fok) / max(len(fok), 1)
print("PAR N_BG=%d wall %.0fs | foreground tok/s %s mean %.1f, TTFT max %.1fs, errors %d | background done %d/%d, errors %d, "
      "first byte %s s, longest gap %.0f s, longest total %.0f s" % (
      N_BG, wall, " ".join("%.1f" % r["tps"] for r in fok), fg_mean, max([r["ttft"] for r in fok] or [0]), len(res_fg) - len(fok),
      len(ok), N_BG, N_BG - len(ok), "/".join("%.0f" % r["ttft"] for r in sorted(ok, key=lambda r: r["ttft"])),
      max([r["gap"] for r in ok] or [0]), max([r["total"] for r in ok] or [0])), flush=True)
print("GATE fg_mean=%.1f fg_err=%d bg_ok=%d bg_err=%d bg_gap_max=%.0f" % (fg_mean, len(res_fg) - len(fok), len(ok), N_BG - len(ok), max([r["gap"] for r in ok] or [0])), flush=True)
for r in res_bg: 
    if "error" in r: print("bg-error", r, flush=True)
