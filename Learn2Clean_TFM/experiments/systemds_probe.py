"""
experiments/systemds_probe.py — print the INSTALLED SystemDS `topk_cleaning` builtin signature so we
can match the driver's multi-return LHS to the installed version (the run failed with
"Undefined Variable topKScores": the builtin's return arity/order differs across versions).
Read-only; prints to stdout (captured in the run log).
"""
import glob, re, sys
from pathlib import Path

try:
    import systemds
except Exception as e:
    print("FATAL: systemds not importable:", repr(e)); sys.exit(1)

base = Path(systemds.__file__).parent
print("systemds __version__:", getattr(systemds, "__version__", "?"))
print("package base:", base)

cands = glob.glob(str(base / "**" / "topk_cleaning.dml"), recursive=True)
print("topk_cleaning.dml candidates:", cands)
for c in cands:
    txt = Path(c).read_text()
    print(f"\n===== {c} (first 40 lines) =====")
    print("\n".join(txt.splitlines()[:40]))
    m = re.search(r"function\s*\((?P<args>.*?)\)\s*return\s*\((?P<ret>.*?)\)", txt, re.S)
    if m:
        print("\n--- ARGS  ---\n", " ".join(m.group("args").split()))
        print("\n--- RETURN---\n", " ".join(m.group("ret").split()))

# also surface the eval-function contract topk_cleaning expects (it calls evaluationFunc)
ev = glob.glob(str(base / "**" / "*.dml"), recursive=True)
for f in ev:
    t = Path(f).read_text()
    if "evaluationFunc" in t and "topk" in f.lower():
        hits = [ln for ln in t.splitlines() if "evaluationFunc" in ln or "eval(" in ln]
        print(f"\n===== evaluationFunc usage in {Path(f).name} =====")
        print("\n".join(hits[:12]))
