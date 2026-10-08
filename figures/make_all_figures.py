"""Build all main-text figures from the stored results in data/ and write them to output/.

Each figure is built by its own script (figures/figN/make_figN.py), run here in a separate
process so that the per-figure modules stay independent.

    python figures/make_all_figures.py            # all figures
    python figures/make_all_figures.py 1 3        # selected figures
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = sorted(int(p.name[3:]) for p in HERE.glob("fig[0-9]*") if (p / f"make_{p.name}.py").exists())


def main(selected):
    for n in selected:
        script = HERE / f"fig{n}" / f"make_fig{n}.py"
        t0 = time.time()
        print(f"Fig. {n} ...", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)
        print(f"Fig. {n} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or FIGURES)
