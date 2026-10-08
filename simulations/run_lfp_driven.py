"""Over-damped vs under-damped AdEx networks driven at 55 Hz (Fig. 1g-h).

Two networks, nu_drive = 2 Hz, b = 0: Delta E_L = -10 mV (over-damped, OD) and +10 mV
(under-damped, UD), both driven at 55 Hz (the input frequency with the largest excitatory
response of the UD network in the Delta E_L sweep) with modulation amplitude A = 1.8 Hz.

  g   Fig. 1g: the drive is modulated from t = 2 s to the end of a 5 s run (the protocol of
      the frequency-response sweeps); population rates, fitted responses, spikes and LFP proxy.
      Seeds 12-21 -> data/driven/od_ud_55hz/g_{m10,p10}_f55_<seed>.pkl
  h   Fig. 1h: the modulation is on from 2 s to 5 s and then switched off; the run continues to
      8 s so that the LFP after stimulus offset can be followed.  Seed 300
      -> data/driven/od_ud_55hz/h_{m10,p10}_f55_300.pkl

Code-generation targets are those of the stored files (cython for g, numpy for h).  The
simulated rates and LFP are reproduced exactly; the fitted g responses (linear least squares)
can differ from the stored ones in the last bit depending on the BLAS kernel selected for the
CPU (with OpenBLAS, OPENBLAS_CORETYPE=Prescott reproduces the stored values bit for bit).
Runtime: g about 0.5-1 min per run (20 runs), h about 1-1.5 min per run (2 runs), one core.

    python simulations/run_lfp_driven.py h
    python simulations/run_lfp_driven.py g --workers 2
    python simulations/run_lfp_driven.py g --check
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asyncresp.paths import DRIVEN               # noqa: E402

OUT_DIR = DRIVEN / "od_ud_55hz"
DELTA_EL = [-10.0, 10.0]           # OD, UD (mV)
DRIVE_HZ = 55.0
A_HZ = 1.8                          # = 0.9 * nu_drive
NU_DRIVE, B_VAL, EL_EXC = 2, 0, -70.0
G_SEEDS = list(range(12, 22))
H_SEEDS = [300]


def _tag(d):
    return f"{d:+.0f}".replace("+", "p").replace("-", "m")


def g_name(d, seed):
    return f"g_{_tag(d)}_f{DRIVE_HZ:g}_{seed}.pkl"


def h_name(d, seed):
    return f"h_{_tag(d)}_f{DRIVE_HZ:g}_{seed}.pkl"


def run_g(d, seed):
    """Sinusoidal drive from 2 s to the end of a 5 s run (frequency-response protocol)."""
    from simulations import adex_impulse as imp
    imp.set_target("cython")
    period = 1000.0 / DRIVE_HZ
    EL_inh = EL_EXC - d
    t0 = time.time()
    out = imp.run_impulse_simulation(b_val=B_VAL, rate=NU_DRIVE, EL_inh=EL_inh, EL_exc=EL_EXC,
                                     A=A_HZ, period=period, oscillation_onset=2000, seed=seed,
                                     return_traces=True)
    rec = imp.impulse_record(B_VAL, NU_DRIVE, EL_inh, EL_EXC, A_HZ, period, out)
    rec.update(out)
    rec.update(delta_EL=d, drive_hz=DRIVE_HZ, seed=seed)
    print(f"  g dEL={d:+g} seed={seed}: amp_exc={rec['amp_exc']:.3f} Hz ({time.time() - t0:.0f} s)",
          flush=True)
    return rec


def run_h(d, seed):
    """Drive switched off at 5 s; run continues to 8 s."""
    from simulations import adex_lfp
    return adex_lfp.run_offset_persistence_sim(delta_EL=d, seed=seed, freq_hz=DRIVE_HZ, A=A_HZ,
                                               nu_0=float(NU_DRIVE), b_post=float(B_VAL),
                                               target="numpy", verbose=True)


PROTOCOLS = {"g": (run_g, g_name, G_SEEDS), "h": (run_h, h_name, H_SEEDS)}


def _job(args):
    proto, d, seed, out_dir = args
    run_fn, name_fn, _ = PROTOCOLS[proto]
    out = run_fn(d, seed)
    path = Path(out_dir) / name_fn(d, seed)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(out, f, protocol=4)
    os.replace(tmp, path)
    return path


def check(proto):
    """Re-run the OD network (first stored seed) and compare every stored field exactly."""
    run_fn, name_fn, seeds = PROTOCOLS[proto]
    ref_path = OUT_DIR / name_fn(DELTA_EL[0], seeds[0])
    with open(ref_path, "rb") as f:
        ref = pickle.load(f)
    t0 = time.time()
    out = run_fn(DELTA_EL[0], seeds[0])
    ok = True
    for k, v in ref.items():
        w = out.get(k)
        if isinstance(v, np.ndarray) or isinstance(w, np.ndarray):
            same = np.array_equal(np.asarray(v), np.asarray(w))
        else:
            same = v == w
        if isinstance(v, np.ndarray):
            print(f"  {k:14s} identical: {same}")
        elif not same:
            rel = abs(w - v) / max(abs(v), 1e-300)
            print(f"  {k:14s} identical: False  (stored {v!r}, re-run {w!r}, rel. diff {rel:.1e})")
        ok &= bool(same)
    print(f"{ref_path.name}: {'EXACT MATCH' if ok else 'MISMATCH'} over {len(ref)} stored fields "
          f"({time.time() - t0:.0f} s)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("protocol", choices=list(PROTOCOLS))
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--check", action="store_true",
                    help="re-run one stored run (OD network, first seed) and compare with the stored file")
    a = ap.parse_args()

    if a.check:
        raise SystemExit(0 if check(a.protocol) else 1)

    seeds = a.seeds if a.seeds is not None else PROTOCOLS[a.protocol][2]
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.protocol, d, s, str(a.out)) for d in DELTA_EL for s in seeds]
    print(f"{a.protocol}: {len(jobs)} runs -> {a.out}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for n, path in enumerate(pool.imap_unordered(_job, jobs), 1):
            print(f"[{n}/{len(jobs)}] {path.name}  {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
