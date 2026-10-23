# fair_bench.py: one priority-0 stream (three 600-token answers) alone, next to 1 and next to 2 background jobs that each
# summarize ~60K uncached tokens (priority BG_PRIO, default 10). Prints the foreground tok/s and time to first token, and
# the background jobs' prefill times. Talks to http://127.0.0.1:8000 (the head node).
import json, urllib.request, time, threading, random, string
U="http://127.0.0.1:8000/v1/chat/completions"; M=json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models",timeout=10))["data"][0]["id"]
import os; PRIO={400:int(os.getenv("BG_PRIO","10"))}
def stream(prompt, n=600):
    b={"model":M,"max_tokens":n,"temperature":1.0,"priority":PRIO.get(n,0),"stream":True,"stream_options":{"include_usage":True},"messages":[{"role":"user","content":prompt}]}
    t0=time.time(); first=None; toks=0
    r=urllib.request.urlopen(urllib.request.Request(U,json.dumps(b).encode(),{"Content-Type":"application/json"}),timeout=600)
    for line in r:
        line=line.strip()
        if not line.startswith(b"data: ") or line==b"data: [DONE]": continue
        j=json.loads(line[6:])
        if j.get("choices") and first is None: first=time.time()
        if j.get("usage"): toks=j["usage"]["completion_tokens"]
    return toks/(time.time()-first), first-t0
P="Write a detailed essay on the history of the railway in Germany."
stop=False
bg_times=[]
def background():
    while not stop:
        rnd="".join(random.choice(string.ascii_lowercase+" ") for _ in range(240000))  # ~60K tokens, uncached
        t=time.time()
        try:
            r=stream("Summarize this text in 5 sentences:\n"+rnd, 400); bg_times.append((time.time()-t, r[1]))
        except Exception as e: print("bg-error", e)
def measure(name, n_bg):
    global stop; stop=False
    th=[threading.Thread(target=background, daemon=True) for _ in range(n_bg)]
    for t in th: t.start()
    if n_bg: time.sleep(8)
    rs=[stream(P) for _ in range(3)]
    stop=True
    for t in th: t.join(timeout=300)
    print("%-22s tok/s %s  mean %.1f | TTFT %s" % (name, " ".join("%.1f"%r[0] for r in rs), sum(r[0] for r in rs)/3, " ".join("%.1fs"%r[1] for r in rs)), flush=True)
measure("alone", 0); measure("next to 1 background", 1)
print("background jobs: %d, prefill (TTFT) per job: %s" % (len(bg_times), " ".join("%.0fs" % b[1] for b in bg_times)), flush=True)
bg_times.clear(); measure("next to 2 background", 2)
print("background jobs: %d, prefill (TTFT) per job: %s" % (len(bg_times), " ".join("%.0fs" % b[1] for b in bg_times)), flush=True)
