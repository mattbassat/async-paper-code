"""Build all main-text figures from the stored results in data/ and write them to output/."""
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


SUPP_SCRIPTS = sorted((HERE / "supp").glob("make_*.py")) + sorted((HERE / "supp").glob("*/make_*.py"))


def main_supp():
    for script in SUPP_SCRIPTS:
        t0 = time.time()
        print(f"{script.stem} ...", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)
        print(f"{script.stem} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["supp"]:
        main_supp()
    else:
        main([int(a) for a in sys.argv[1:]] or FIGURES)
