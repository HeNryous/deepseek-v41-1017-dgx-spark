# tput.py N: N parallel priority-0 streams (500-token essays, T = 1); prints the aggregate and per-stream tok/s.
import json, os, sys, threading, time, urllib.request
U = "http://127.0.0.1:8000/v1/chat/completions"
M = json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=10))["data"][0]["id"]
N = int(sys.argv[1]); res = []
def stream(i):
    b = {"model": M, "max_tokens": 500, "temperature": 1.0, "priority": 0, "stream": True,
         "stream_options": {"include_usage": True},
         "messages": [{"role": "user", "content": "Write a detailed text on topic %d: the history of seafaring." % i}]}
    t0 = time.time(); first = None; toks = 0
    r = urllib.request.urlopen(urllib.request.Request(U, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=600)
    for line in r:
        line = line.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]": continue
        j = json.loads(line[6:])
        if j.get("choices") and first is None: first = time.time()
        if j.get("usage"): toks = j["usage"]["completion_tokens"]
    res.append((toks, time.time() - first, first - t0))
t0 = time.time(); th = [threading.Thread(target=stream, args=(i,)) for i in range(N)]
for t in th: t.start()
for t in th: t.join()
wall = time.time() - t0; tot = sum(r[0] for r in res)
print("N=%d total %.1f tok/s | per stream %s | TTFT max %.1fs | wall %.0fs" % (N, tot / wall, " ".join("%.1f" % (r[0] / r[1]) for r in res), max(r[2] for r in res), wall), flush=True)
