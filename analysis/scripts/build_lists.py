#!/usr/bin/env python3
"""Build download lists (corpus \t id \t pdf_url) for each venue-year corpus.

Usage: python3 -I build_lists.py <workdir> [corpus ...]  -> writes (or appends to) <workdir>/lists.tsv
Corpora: MLSys 2019+2020 / 2025+2026, OSDI 2016/2020/2025/2026, NSDI 2016/2020/2025/2026,
ACL long 2016/2020/2025 (random sample of N_ACL, seed 0).
"""
import random, re, subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor

N_ACL = 200
N_BIG = 150      # sample size for venues with >300 papers per year
N_CTRL = 100     # sample size for the 2016 trend-control years
N_SERIES = 80    # sample size for the 2021-2024 series years
UA = "Mozilla/5.0 (research corpus study)"


def get(url):
    return subprocess.run(["curl", "-sL", "--compressed", "-A", UA, "--max-time", "60", url],
                          capture_output=True, text=True, errors="replace").stdout


def mlsys(year):
    html = get(f"https://proceedings.mlsys.org/paper_files/paper/{year}")
    out = []
    for h, suf in sorted(set(re.findall(r"/hash/([0-9a-f]{32})-Abstract(-[A-Za-z0-9]+)?\.html", html))):
        out.append((h, f"https://proceedings.mlsys.org/paper_files/paper/{year}/file/{h}-Paper{suf}.pdf"))
    return out


def usenix(conf, yy):
    html = get(f"https://www.usenix.org/conference/{conf}{yy}/technical-sessions")
    pat = rf'href="(?:https://www\.usenix\.org)?(/conference/{conf}{yy}/(?:technical-sessions/)?presentation/[^"#?]+)"'
    pres = sorted(set(re.findall(pat, html)))
    if not pres:  # older years redirect to /program
        pres = sorted(set(re.findall(pat, get(f"https://www.usenix.org/conference/{conf}{yy}/program"))))
    def one(p):
        page = get("https://www.usenix.org" + p)
        pdfs = [u for u in re.findall(r'(?:https://www\.usenix\.org)?/system/files/([^"\s]+?\.pdf)', page)
                if u.split("/")[-1].startswith(f"{conf}{yy}-") and "slides" not in u]
        return (p.rsplit("/", 1)[1], "https://www.usenix.org/system/files/" + pdfs[0]) if pdfs else None
    with ThreadPoolExecutor(8) as ex:
        return [r for r in ex.map(one, pres) if r]


def acl(year, n=None):
    if year <= 2019:
        base = "https://aclanthology.org/{}.pdf"
        vol = get(f"https://aclanthology.org/volumes/P{year % 100}-1/")
        ids = sorted(set(re.findall(rf'/(P{year % 100}-1\d{{3}})/', vol)) - {f"P{year % 100}-1000"})
    else:
        vol = get(f"https://aclanthology.org/volumes/{year}.acl-main/" if year == 2020
                  else f"https://aclanthology.org/volumes/{year}.acl-long/")
        key = "acl-main" if year == 2020 else "acl-long"
        ids = sorted(set(re.findall(rf'/({year}\.{key}\.\d+)/', vol)) - {f"{year}.{key}.0"})
        base = "https://aclanthology.org/{}.pdf"
    # 2020 acl-main mixes long and short papers: oversample, drop short ones by length later.
    random.Random(0).shuffle(ids)
    n = n or N_ACL
    take = ids[: int(n * (1.4 if year == 2020 else 1))]
    return [(i, base.format(i)) for i in take]


def sample(lst, n, seed=0):
    lst = sorted(set(lst))
    random.Random(seed).shuffle(lst)
    return lst[:n]


def neurips(year, n):
    html = get(f"https://papers.nips.cc/paper_files/paper/{year}")
    hs = re.findall(r"/hash/([0-9a-f]{32})-Abstract(-Conference)?\.html", html)
    return [(h, f"https://papers.nips.cc/paper_files/paper/{year}/file/{h}-Paper{suf}.pdf")
            for h, suf in sample(hs, n)]


def icml(vol, n):
    html = get(f"https://proceedings.mlr.press/v{vol}/")
    pdfs = [u for u in re.findall(rf'https?://(?:proceedings\.mlr\.press/v{vol}|raw\.githubusercontent\.com/mlresearch/v{vol}/main/assets)/[^"\']+?\.pdf', html)
            if "supp" not in u]
    return [(u.rsplit("/", 1)[1][:-4], u.replace("http://", "https://")) for u in sample(pdfs, n)]


def cvpr(year, n):
    html = get(f"https://openaccess.thecvf.com/CVPR{year}?day=all")
    days = [] if "_paper.pdf" in html else sorted(set(re.findall(rf"CVPR{year}(?:\.py)?\?day=([0-9-]+)", get(f"https://openaccess.thecvf.com/CVPR{year}"))))
    for d in days:
        html += get(f"https://openaccess.thecvf.com/CVPR{year}.py?day={d}")
    pdfs = re.findall(r'href="(/?content[^"]+?_paper\.pdf)"', html)
    return [(p.rsplit("/", 1)[1][:-10], "https://openaccess.thecvf.com/" + p.lstrip("/")) for p in sample(pdfs, n)]


def vldb(vol, n):
    html = get(f"https://www.vldb.org/pvldb/volumes/{vol}")
    pdfs = re.findall(rf'https?://(?:www\.)?vldb\.org/pvldb/vol{vol}/p[^"]+?\.pdf', html)
    return [(u.rsplit("/", 1)[1][:-4], u) for u in sample(pdfs, n)]


ACCEPT = re.compile(r"accept|published|camera|to appear|oral|poster|spotlight|main (?:track|conference)|proceedings", re.I)
REJECT = re.compile(r"under review|submitted to|in submission|under submission|workshop|tiny ?paper|blog ?post", re.I)


def arxiv_comment(query, n, year=None):
    """arXiv versions of papers whose comment field says they were accepted at the venue
    (e.g. 'Accepted at COLM 2025'); 'under review' and workshop papers are dropped."""
    import html as H, time, urllib.parse
    keep = []
    for start in range(0, 2400, 200):
        x = get("https://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(query)
                + f"&start={start}&max_results=200")
        entries = re.findall(r"<entry>(.*?)</entry>", x, re.S)
        for e in entries:
            m = re.search(r"<id>http://arxiv\.org/abs/([^<]+?)(?:v\d+)?</id>", e)
            c = re.search(r"<arxiv:comment[^>]*>(.*?)</arxiv:comment>", e, re.S)
            com = H.unescape(c.group(1)) if c else ""
            if m and ACCEPT.search(com) and not REJECT.search(com):
                keep.append(m.group(1))
        if len(entries) < 200:
            break
        time.sleep(3)
    return [(i.replace("/", "_"), f"https://arxiv.org/pdf/{i}") for i in sample(keep, n)]


AAAI_ISSUES = {2016: [303], 2017: [302], 2018: [301], 2019: [246]}


def aaai(year, n):
    """AAAI main technical tracks from the open OJS proceedings.

    Since 2024 some articles also carry a Video/Poster galley, so a sampled galley id can point at a
    talk page instead of the PDF. Those rows were repaired after sampling by reading each article's
    citation_pdf_url (the shipped results/paper_lists.tsv holds the repaired URLs)."""
    issues = AAAI_ISSUES.get(year)
    if not issues:
        arch = "".join(get(f"https://ojs.aaai.org/index.php/AAAI/issue/archive/{p}") for p in range(1, 15))
        tag = f"AAAI-{year % 100:02d} Technical Tracks"
        issues = sorted({int(i) for i, t in re.findall(r'issue/view/(\d+)"[^>]*>\s*([^<]+?)\s*</a>', arch) if t.strip().startswith(tag)})
    links = []
    for i in issues:
        h = get(f"https://ojs.aaai.org/index.php/AAAI/issue/view/{i}")
        links += re.findall(r"article/view/(\d+)/(\d+)", h)
    links = sorted(set(links))
    return [(f"{a}_{g}", f"https://ojs.aaai.org/index.php/AAAI/article/download/{a}/{g}") for a, g in sample(links, n)]


def main():
    wd = sys.argv[1]
    rows = []
    jobs = {
        "mlsys2019": lambda: mlsys(2019), "mlsys2020": lambda: mlsys(2020),
        "mlsys2025": lambda: mlsys(2025), "mlsys2026": lambda: mlsys(2026),
        "osdi2016": lambda: usenix("osdi", 16), "osdi2020": lambda: usenix("osdi", 20),
        "osdi2025": lambda: usenix("osdi", 25), "osdi2026": lambda: usenix("osdi", 26),
        "nsdi2020": lambda: usenix("nsdi", 20), "nsdi2025": lambda: usenix("nsdi", 25),
        "nsdi2016": lambda: usenix("nsdi", 16), "nsdi2026": lambda: usenix("nsdi", 26),
        "acl2016": lambda: acl(2016), "acl2020": lambda: acl(2020), "acl2025": lambda: acl(2025),
        "neurips2016": lambda: neurips(2016, N_CTRL), "neurips2020": lambda: neurips(2020, N_BIG),
        "neurips2024": lambda: neurips(2024, N_BIG),  # 2025 main track not yet posted (Oct 2026)
        "icml2016": lambda: icml(48, N_CTRL), "icml2020": lambda: icml(119, N_BIG),
        "icml2025": lambda: icml(267, N_BIG),
        "cvpr2016": lambda: cvpr(2016, N_CTRL), "cvpr2020": lambda: cvpr(2020, N_BIG),
        "cvpr2025": lambda: cvpr(2025, N_BIG),
        "vldb2020": lambda: vldb(13, N_BIG), "vldb2025": lambda: vldb(18, N_BIG),
        # COLM began in 2024 and lives on OpenReview (bot-blocked): use arXiv versions.
        "colm2025": lambda: arxiv_comment('co:"COLM 2025"', N_BIG),
        # EuroSys is on the ACM DL (bot-blocked); arXiv has only a few dozen recent versions.
        "eurosys2025": lambda: arxiv_comment("co:EuroSys AND (co:2025 OR co:2026)", N_BIG),
        # Year-by-year series to locate the break (gradual trend vs jump after Nov 2022).
        **{f"acl{y}": (lambda y=y: acl(y, N_SERIES)) for y in (2021, 2022, 2023, 2024)},
        **{f"icml{y}": (lambda v=v: icml(v, N_SERIES)) for y, v in
           ((2021, 139), (2022, 162), (2023, 202), (2024, 235))},
        # Round 2: yearly series 2021-2024 for every venue, to test the break year in each field.
        **{f"osdi{y}": (lambda y=y: usenix("osdi", y % 100)) for y in (2021, 2022, 2023, 2024)},
        **{f"nsdi{y}": (lambda y=y: usenix("nsdi", y % 100)) for y in (2021, 2022, 2023, 2024)},
        **{f"mlsys{y}": (lambda y=y: mlsys(y)) for y in (2021, 2022, 2023, 2024)},
        **{f"cvpr{y}": (lambda y=y: cvpr(y, N_SERIES)) for y in (2021, 2022, 2023, 2024)},
        **{f"neurips{y}": (lambda y=y: neurips(y, N_SERIES)) for y in (2021, 2022, 2023)},
        **{f"vldb{y}": (lambda v=v: vldb(v, N_SERIES)) for y, v in ((2021, 14), (2022, 15), (2023, 16), (2024, 17))},
        # Round 3: a 2018 point for every venue with pre-2019 proceedings (denser pre-LLM trend).
        "osdi2018": lambda: usenix("osdi", 18), "nsdi2018": lambda: usenix("nsdi", 18),
        "vldb2018": lambda: vldb(11, N_SERIES), "neurips2018": lambda: neurips(2018, N_SERIES),
        "icml2018": lambda: icml(80, N_SERIES), "cvpr2018": lambda: cvpr(2018, N_SERIES),
        "acl2018": lambda: acl(2018, N_SERIES),
        # Round 4: fill the remaining gaps (2017, 2019 trend years; 2026; COLM 2024/2026).
        # OSDI was biennial until 2021 (no 2017/2019); MLSys/SysML 2018 has no proceedings;
        # NeurIPS 2025 main track and the EuroSys proceedings are not downloadable yet.
        "nsdi2017": lambda: usenix("nsdi", 17), "nsdi2019": lambda: usenix("nsdi", 19),
        "vldb2016": lambda: vldb(9, N_SERIES), "vldb2017": lambda: vldb(10, N_SERIES),
        "vldb2019": lambda: vldb(12, N_SERIES), "vldb2026": lambda: vldb(19, N_BIG),
        "neurips2017": lambda: neurips(2017, N_SERIES), "neurips2019": lambda: neurips(2019, N_SERIES),
        "icml2017": lambda: icml(70, N_SERIES), "icml2019": lambda: icml(97, N_SERIES), "icml2026": lambda: icml(306, N_BIG),
        "cvpr2017": lambda: cvpr(2017, N_SERIES), "cvpr2019": lambda: cvpr(2019, N_SERIES), "cvpr2026": lambda: cvpr(2026, N_BIG),
        "acl2017": lambda: acl(2017, N_SERIES), "acl2019": lambda: acl(2019, N_SERIES), "acl2026": lambda: acl(2026, N_ACL),
        "colm2024": lambda: arxiv_comment('co:"COLM 2024"', N_BIG), "colm2026": lambda: arxiv_comment('co:"COLM 2026"', N_BIG),
        # Round 5: ICLR (OpenReview blocks scripts; arXiv versions of accepted papers) and AAAI (OJS).
        **{f"iclr{y}": (lambda y=y, n=n: arxiv_comment(f'co:"ICLR {y}"', n)) for y, n in
           ((2016, 80), (2018, 80), (2020, 150), (2021, 80), (2022, 80), (2023, 80), (2024, 80), (2025, 150), (2026, 150))},
        **{f"aaai{y}": (lambda y=y, n=n: aaai(y, n)) for y, n in
           ((2016, 80), (2018, 80), (2020, 150), (2021, 80), (2022, 80), (2023, 80), (2024, 80), (2025, 150), (2026, 150))},
    }
    only = sys.argv[2:]
    jobs = {k: v for k, v in jobs.items() if not only or k in only}
    for name, fn in jobs.items():
        lst = fn()
        print(name, len(lst), file=sys.stderr)
        rows += [(name, i, u) for i, u in lst]
    with open(os.path.join(wd, "lists.tsv"), "a" if only else "w") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")


if __name__ == "__main__":
    main()
