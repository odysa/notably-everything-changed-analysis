#!/usr/bin/env python3
"""Download every PDF in <workdir>/lists.tsv to <workdir>/pdfs/<corpus>/<id>.pdf.

Usage: python3 -I download.py <workdir> [--workers 6]
Skips files already present; keeps only responses that start with %PDF. Concurrency is
capped per host so no proceedings site sees more than 3 parallel requests (arXiv: 1).
"""
import collections, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

wd = sys.argv[1]
rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(wd, "lists.tsv")) if l.strip()]
# Interleave hosts so the per-host caps, not one busy site, set the pace.
by_host = collections.defaultdict(list)
for r in rows:
    by_host[urlparse(r[2]).netloc].append(r)
rows = [r for grp in __import__("itertools").zip_longest(*by_host.values()) for r in grp if r]
locks = collections.defaultdict(lambda: threading.Semaphore(3))
locks["arxiv.org"] = threading.Semaphore(1)


def fetch(row):
    corpus, pid, url = row
    d = os.path.join(wd, "pdfs", corpus)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, pid + ".pdf")
    if os.path.exists(dst):
        return "skip"
    host = urlparse(url).netloc
    with locks[host]:
        for attempt in range(3):
            tmp = dst + ".part"
            subprocess.run(["curl", "-sL", "-A", "Mozilla/5.0 (research corpus study)",
                            "--max-time", "180", "-o", tmp, url])
            ok = os.path.exists(tmp) and open(tmp, "rb").read(4) == b"%PDF"
            if ok:
                os.rename(tmp, dst)
                if host == "arxiv.org":
                    time.sleep(3)
                return "ok"
            if os.path.exists(tmp):
                os.remove(tmp)
            time.sleep(5 * (attempt + 1))
    return "fail:" + corpus + ":" + url


with ThreadPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 32) as ex:
    res = list(ex.map(fetch, rows))
c = collections.Counter(r.split(":")[0] for r in res)
print(dict(c))
for r in res:
    if r.startswith("fail"):
        print(r)
