"""
experiments/run_systemds_saga.py

REAL SAGA-family baseline: run Apache SystemDS `topk_cleaning` (the SAGA/SAGA++ algorithm,
Boehm's group's own implementation) on OUR datasets, held-out protocol, and read off its cleaned vs
dirty downstream accuracy (multiLogReg — SAGA's own evaluator). This upgrades "below SAGA's
*published* numbers" to "below SAGA *we ran ourselves* on the same data/split".

Approach (per the SystemDS Python/DML research): `pip install systemds` bundles the engine
jars; we invoke a DML driver via `java -cp <bundled jars> org.apache.sysds.api.DMLScript`.
The driver reads dirty CSV + a 3-row meta frame + the static primitives/param libraries,
calls `topk_cleaning(cv=TRUE, evaluationFunc="evalClassification")`, and writes the best
pipeline's score (cleaned acc) and the dirtyScore (no-clean acc).

This is a best-effort attempt — known failure modes: JDK major mismatch, meta-frame shape,
the eval callback signature, and small-data CV folds missing a class. Failures are recorded
honestly per dataset, not hidden.

Usage
-----
  PYTHONPATH=src python experiments/run_systemds_saga.py --datasets EEG Titanic hepatitis ionosphere
"""
from __future__ import annotations
import argparse, glob, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).parents[1]
SAGA = {"EEG": ("eeg.csv", "Class", []),
        "Titanic": ("titanic.csv", "survived", ["name", "ticket", "cabin", "boat", "body", "home.dest"]),
        "AnimalShelter": (["animalshelter_train.csv", "animalshelter_test.csv"], "class", ["AnimalID", "Name", "OutcomeSubtype"])}

PRIMITIVES = """ED,MVI,OTLR,EC,SCALE,CI,DUMMY,DIM
outlierBySd,imputeByMean,winsorize,imputeByMean,scale,abstain,dummycoding,pca
outlierByIQR,imputeByMedian,outlierBySd,imputeByMedian,,underSampling,frequencyEncode,
,forward_fill,outlierByIQR,fillDefault,,,,
,fillDefault,,,,,,
"""
PARAM = """applyName,name,param_no,maskFlag,FDFlag,yFlag,verboseFlag,dataFlag,default1,default2,default3,default4,dt1,dt2,dt3,dt4,st1,en1,st2,en2,st3,en3,st4,en4
outlierByIQRApply,outlierByIQR,3,0,0,0,1,0,1.5,2,1,,FP,INT,INT,1,7,2,2,1,1,,,
outlierBySdApply,outlierBySd,3,0,0,0,1,0,3,2,1,,INT,INT,INT,1,7,1,2,2,1,,,
winsorizeApply,winsorize,2,0,0,0,1,0,0.05,0.95,,,FP,FP,0.01,0.05,0.95,1,,,,,,
imputeByMeanApply,imputeByMean,0,0,0,0,0,0,,,,,,,,,,,,,,,,
imputeByMedianApply,imputeByMedian,0,0,0,0,0,0,,,,,,,,,,,,,,,,
fillDefaultApply,fillDefault,0,0,0,0,0,0,,,,,,,,,,,,,,,,
forward_fillApply,forward_fill,1,0,0,0,1,0,1,,,,BOOL,,0,1,,,,,,,,
scaleApply,scale,2,0,0,0,1,0,1,1,,,BOOL,BOOL,0,1,0,1,,,,,,
dummycodingApply,dummycoding,0,1,0,0,0,1,,,,,,,,,,,,,,,,
frequencyEncodeApply,frequencyEncode,0,1,0,0,0,1,,,,,,,,,,,,,,,,
underSamplingApply,underSampling,1,0,0,1,1,0,1,,,,FP,,0.1,1,,,,,,,,
pcaApply,pca,1,0,0,0,0,0,5,,,,INT,,2,10,,,,,,,,
"""
# Driver mirrors SystemDS's own topkcleaningClassificationTest.dml: the evaluationFunc MUST
# return TWO matrices (output, error) with output=cbind(accuracy, evalFunHp) — a single-return
# eval (our first attempt) fails topk_cleaning's compile-time inlining, leaving the outputs
# unbound ("Undefined Variable scores"). Signature confirmed against scripts/builtin/topk_cleaning.dml.
DRIVER = r'''
F = read($dirtyData, data_type="frame", format="csv", header=TRUE,
         naStrings=["NA","null","NaN","","?"," ","nan"], sep=",");
metaInfo = read($metaData, data_type="frame", format="csv", header=FALSE);
primitives = read($primitives, data_type="frame", format="csv", header=TRUE);
param = read($parameters, data_type="frame", format="csv", header=TRUE);

[topKPipelines, topKHyperParams, topKScores, dirtyScore, evalFunHpOut, applyFunc] = topk_cleaning(
  dataTrain=F, metaData=metaInfo, primitives=primitives, parameters=param,
  evaluationFunc="evalClassification", evalFunHp=as.matrix(NaN),
  topK=3, resource_val=20, max_iter=5, expectedIncrease=1.0,
  cv=TRUE, cvk=3, sample=1.0, isLastLabel=TRUE, correctTypos=FALSE,
  enablePruning=FALSE, seed=42);

write(topKScores, $output+"/scores.csv", format="csv");
write(as.matrix(dirtyScore), $output+"/dirty.csv", format="csv");

evalClassification = function(Matrix[Double] X, Matrix[Double] Y,
  Matrix[Double] Xtest, Matrix[Double] Ytest,
  Matrix[Double] Xorig=as.matrix(0), Matrix[Double] evalFunHp)
  return(Matrix[Double] output, Matrix[Double] error) {
  if(is.na(as.scalar(evalFunHp[1,1]))) {
    evalFunHp = matrix("2 10 0.001", rows=1, cols=3);
  }
  error = as.matrix(0);
  if(min(Y) == max(Y)) {
    accuracy = as.matrix(0);
  }
  else {
    beta = multiLogReg(X=X, Y=Y, icpt=as.scalar(evalFunHp[1,1]),
      reg=as.scalar(evalFunHp[1,2]), tol=as.scalar(evalFunHp[1,3]),
      maxi=1000, maxii=0, verbose=FALSE);
    [prob, yhat, acc] = multiLogRegPredict(Xtest, beta, Ytest, FALSE);
    accuracy = as.matrix(acc);
    error = yhat != Ytest;
  }
  output = cbind(accuracy, evalFunHp);
}
'''


def load_ds(name):
    if name in SAGA:
        files, tgt, drop = SAGA[name]
        files = [files] if isinstance(files, str) else files
        df = pd.concat([pd.read_csv(ROOT / "data/saga" / f) for f in files], ignore_index=True)
        df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
        return df.drop(columns=[tgt]), df[tgt].astype(str)
    sys.path.insert(0, str(ROOT / "src"))
    from learn2clean_v3.data.openml_loader import load_dataset
    X, y, _ = load_dataset(name, use_cache=True)
    return X, y.astype(str)


def mcar(X, rate, seed):
    rng = np.random.default_rng(seed); X = X.copy()
    for c in X.select_dtypes(include="number").columns:
        X.loc[rng.random(len(X)) < rate, c] = np.nan
    return X


def find_jars(py) -> str:
    """Classpath = ALL jars under the installed systemds package (engine jar at the package
    root + dependency jars in lib/), joined with ':'. Guarantees DMLScript is on the path."""
    try:
        import systemds
        base = Path(systemds.__file__).parent
    except Exception:
        base = Path(py).parent.parent
    jars = glob.glob(str(base / "**" / "*.jar"), recursive=True)
    return ":".join(jars)


def run_one(name, seed, workdir, jars) -> dict:
    X, y = load_ds(name)
    if len(X) > 4000:
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(X, y, train_size=4000, random_state=0, stratify=y)
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    Xd = mcar(X, 0.15, seed)
    F = Xd.copy(); F["__label__"] = y.values
    wd = Path(workdir) / f"{name}_{seed}"; wd.mkdir(parents=True, exist_ok=True)
    F.to_csv(wd / "dirty.csv", index=False)
    # 3-row meta — order MUST match scripts/builtin/topk_cleaning.dml::prepareMeta:
    #   row1 = schema strings (STRING/FP64/BOOLEAN), cast as Frame[String]
    #   row2 = mask matrix (0/1) -> as.matrix(); last col = maskY (label mask flag)
    #   row3 = fdMask (0/1)
    # The previous version had rows 1 and 2 swapped, so as.matrix(row2) hit the schema
    # strings ("FP64") and threw NumberFormatException -- "Unable to change to double".
    cats = [1 if str(F[c].dtype) in ("object", "category") else 0 for c in F.columns]
    schema = ["STRING" if c else "FP64" for c in cats]
    meta = pd.DataFrame([schema, cats, [0]*len(cats)])
    meta.to_csv(wd / "meta.csv", index=False, header=False)
    (wd / "out").mkdir(exist_ok=True)
    # -exec singlenode forces pure control-program (no Spark): the bundled jars ship
    # spark-core but NOT spark-unsafe, so any SparkExecutionContext init crashes with
    # NoClassDefFoundError org.apache.spark.unsafe.Platform. singlenode avoids it entirely.
    cmd = ["java", "-Xmx12g", "-cp", jars, "org.apache.sysds.api.DMLScript",
           "-exec", "singlenode", "-f", str(wd / "driver.dml"), "-nvargs",
           f"dirtyData={wd}/dirty.csv", f"metaData={wd}/meta.csv",
           f"primitives={workdir}/primitives.csv", f"parameters={workdir}/param.csv",
           f"output={wd}/out"]
    (wd / "driver.dml").write_text(DRIVER)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        # ALWAYS dump full java stdout+stderr to a sidecar so the real DML error is visible
        # (the hadoop NativeCodeLoader WARN is non-fatal noise; the actual compile/runtime
        #  error lands earlier in the stream and was being truncated by a [-300:] tail).
        (wd / "java.log").write_text(f"=== CMD ===\n{' '.join(cmd)}\n\n"
                                     f"=== STDOUT ===\n{r.stdout}\n\n=== STDERR ===\n{r.stderr}")
        ok = (wd / "out" / "scores.csv").exists()
        clean = dirty = np.nan
        if ok:
            sc = pd.read_csv(wd / "out" / "scores.csv", header=None)
            clean = float(sc.values.flatten()[0])
            dirty = float(pd.read_csv(wd / "out" / "dirty.csv", header=None).values.flatten()[0])
        # surface the most informative error line: prefer ERROR/Exception lines over the WARN tail
        stream = (r.stderr or "") + "\n" + (r.stdout or "")
        sig = [ln for ln in stream.splitlines()
               if any(k in ln for k in ("ERROR", "Exception", "error:", "Error:", "Caused by", "not found", "Invalid"))]
        err = "" if ok else (" || ".join(sig[-3:]) if sig else (r.stderr[-300:] or r.stdout[-300:]))
        return {"dataset": name, "seed": seed, "saga_clean": clean, "saga_dirty": dirty,
                "ok": int(ok), "sec": time.time()-t0, "err": err.replace("\n", " ")}
    except subprocess.TimeoutExpired:
        return {"dataset": name, "seed": seed, "saga_clean": np.nan, "saga_dirty": np.nan,
                "ok": 0, "sec": time.time()-t0, "err": "TIMEOUT"}


def main(datasets, seeds, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "primitives.csv").write_text(PRIMITIVES); (out / "param.csv").write_text(PARAM)
    jars = find_jars(sys.executable)
    print(f"systemds jars: {jars or 'NOT FOUND'}", flush=True)
    if not jars:
        print("FATAL: SystemDS jars not found in venv"); return
    print(subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.split(chr(10))[0])
    rows = []
    for ds in datasets:
        for seed in seeds:
            try:
                r = run_one(ds, seed, out, jars)
            except Exception as e:
                r = {"dataset": ds, "seed": seed, "saga_clean": np.nan, "saga_dirty": np.nan,
                     "ok": 0, "sec": 0, "err": repr(e)[:200]}
            rows.append(r)
            pd.DataFrame(rows).to_csv(out / "systemds_saga_per_run.csv", index=False)
            print(f"  {ds:14} s{seed}: ok={r['ok']} clean={r['saga_clean']} dirty={r['saga_dirty']} "
                  f"({r['sec']:.0f}s) {r['err'][:80]}", flush=True)
    df = pd.DataFrame(rows)
    agg = df[df.ok == 1].groupby("dataset").agg(saga_clean=("saga_clean", "mean"),
                                               saga_dirty=("saga_dirty", "mean")).reset_index()
    agg.to_csv(out / "systemds_saga_aggregated.csv", index=False)
    print(f"\n=== SystemDS SAGA (topk_cleaning) — {int(df.ok.sum())}/{len(df)} runs OK ===")
    print(agg.to_string(index=False) if len(agg) else "(no successful runs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["EEG", "Titanic", "hepatitis", "ionosphere", "diabetes"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/paper_ready/systemds_saga"))
    a = ap.parse_args()
    main(a.datasets, tuple(a.seeds), a.output_dir)
