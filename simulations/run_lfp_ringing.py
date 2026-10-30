"""LFP ringing after the offset of a sinusoidal drive, without and with a synaptic delay
(supplementary LFP-ringing figures)."""
import argparse
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asyncresp.paths import SUPP

OUT_DIR = SUPP / "lfp_ringing"
SEEDS = list(range(300, 310))
TARGET = "numpy"

PROTOCOLS = {
    "nodelay": dict(kind="driven", file="driven_r2b0_nodelay_f50.pkl", net_key="r2b0",
                    nu_0=2.0, delay_ms=0.0, freq_hz=50.0,
                    grid=[-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]),
    "delay": dict(kind="driven", file="driven_r4b0_delay2_f45.pkl", net_key="r4b0",
                  nu_0=4.0, delay_ms=2.0, freq_hz=45.0,
                  grid=[-15.0, -13.0, -11.0, -10.0, -9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0]),
    "spont_delay": dict(kind="spont", file="spont_r4b0_delay2.pkl", net_key="r4b0",
                        nu_0=4.0, delay_ms=2.0, grid=[-15.0, -10.0, 5.0]),
}
DRIVEN_KEYS = ("net_key", "nu_0", "delay_ms", "freq_hz", "period_ms", "A",
               "oscillation_onset", "oscillation_offset", "total_ms")


def run_one(proto, delta_EL, seed):
    """One simulation; returns the full record of adex_lfp."""
    from simulations import adex_lfp
    p = PROTOCOLS[proto]
    if p["kind"] == "driven":
        return adex_lfp.run_offset_persistence_sim(
            delta_EL=delta_EL, seed=seed, freq_hz=p["freq_hz"], A=0.9 * p["nu_0"], nu_0=p["nu_0"],
            b_post=0.0, delay_ms=p["delay_ms"], target=TARGET, verbose=True)
    return adex_lfp.run_spontaneous_sim(delta_EL=delta_EL, seed=seed, nu_0=p["nu_0"], b_post=0.0,
                                        total_ms=10000.0, delay_ms=p["delay_ms"], target=TARGET,
                                        verbose=True)


def _job(args):
    proto, d, s = args
    o = run_one(proto, d, s)
    return d, s, o


def assemble(proto, runs, seeds):
    """Output record from {(Delta_EL, seed): full record}."""
    p = PROTOCOLS[proto]
    grid = p["grid"]
    first = runs[(grid[0], seeds[0])]
    lfp = np.stack([np.stack([runs[(d, s)]["lfp_inh"] for s in seeds]) for d in grid])
    rec = dict(net_key=p["net_key"], delta_EL=np.array(grid), seeds=np.array(seeds), lfp_inh=lfp)
    if p["kind"] == "driven":
        rec.update({k: first[k] for k in DRIVEN_KEYS if k != "net_key"}, t_state=first["t_state"])
    else:
        rec.update(nu_0=first["nu_0"], b_post=first["b_post"], delay_ms=first["delay_ms"],
                   total_ms=first["total_ms"], dt_lfp_ms=first["dt_lfp_ms"])
    return rec


def check(proto):
    """Re-run the first Delta_EL of the grid at seed 300 and compare with the stored file."""
    p = PROTOCOLS[proto]
    with open(OUT_DIR / p["file"], "rb") as f:
        ref = pickle.load(f)
    d, s = p["grid"][0], SEEDS[0]
    i = int(np.flatnonzero(ref["delta_EL"] == d)[0])
    j = int(np.flatnonzero(ref["seeds"] == s)[0])
    t0 = time.time()
    out = run_one(proto, d, s)
    ok = np.array_equal(ref["lfp_inh"][i, j], out["lfp_inh"])
    print(f"  lfp_inh identical: {ok}")
    if p["kind"] == "driven":
        same_t = np.array_equal(ref["t_state"], out["t_state"])
        print(f"  t_state identical: {same_t}")
        ok &= same_t
    print(f"{p['file']} [Delta_EL={d:+g} mV, seed {s}]: {'EXACT MATCH' if ok else 'MISMATCH'} "
          f"({time.time() - t0:.0f} s)")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("protocol", choices=list(PROTOCOLS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--check", action="store_true",
                    help="re-run one stored run (first Delta_EL, seed 300) and compare lfp_inh")
    a = ap.parse_args()

    if a.check:
        raise SystemExit(0 if check(a.protocol) else 1)

    p = PROTOCOLS[a.protocol]
    jobs = [(a.protocol, d, s) for d in p["grid"] for s in a.seeds]
    print(f"{a.protocol}: {len(jobs)} runs", flush=True)
    t0 = time.time()
    runs = {}
    with Pool(a.workers) as pool:
        for n, (d, s, o) in enumerate(pool.imap_unordered(_job, jobs), 1):
            runs[(d, s)] = {k: o[k] for k in o if k not in ("fr_exc", "fr_inh", "t_fr",
                                                           "vm_exc_pop", "lfp_exc", "lfp_absmag")}
            print(f"[{n}/{len(jobs)}] dEL={d:+g} seed={s}  {(time.time() - t0) / 60:.1f} min", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / p["file"]
    with open(path, "wb") as f:
        pickle.dump(assemble(a.protocol, runs, list(a.seeds)), f, protocol=4)
    print("wrote", path)


if __name__ == "__main__":
    main()
