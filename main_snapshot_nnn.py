"""
main_09_snapshot_nnn.py
Snapshot del reticolo nel modello con secondi vicini (src/ising_nnn.py), J2 = -1, a temperatura BASSA e per tre
campi: h = 0.25 (stato fondamentale: strisce), h = 1.5 (plateau M = 1/2) e h = 2.5 (tutti up).
Serve a VEDERE la struttura degli stati di cui parlano le mappe di M, S, Q:
    strisce      colonne (o righe) alternate di spin su e giu':            M = 0,   S = 0,   Q = 1
    plateau      uno spin giu' ogni blocco 2x2 (reticolo di passo 2):      M = 1/2, S = 1/2, Q = 0.71
    tutti up     nessuno spin giu' (a T bassa: pochi spin giu' isolati):   M ~ 1
Ogni riga e' un h, ogni colonna un tempo (in sweep, 1 sweep = L*L tentativi di flip). Colore = valore dello spin
(scuro = su, chiaro = giu'). Sopra ogni snapshot sono scritti M, S, Q del reticolo mostrato.

Partenza (--start):
  random   (default) stesso reticolo CASUALE per i tre h: si vede come il sistema evolve verso lo stato fondamentale;
           a T bassa il Glauber puo' restare intrappolato in domini di strisce di orientamenti diversi o in
           miscele di plateau e strisce: e' un risultato fisico, non un errore.
  ground   si parte direttamente dallo stato fondamentale (strisce / plateau / tutti up): si vede quanto e'
           perfetto lo stato e come le fluttuazioni termiche lo disturbano a T.

Produce figures/fig11_snapshot_nnn_<start>.png.

Uso:
    python main_09_snapshot_nnn.py                         # L = 20, T = 0.05, partenza casuale
    python main_09_snapshot_nnn.py --start ground          # partenza dallo stato fondamentale
    python main_09_snapshot_nnn.py --T 0.1 --L 30 --times 0 10 100 1000 10000
    python main_09_snapshot_nnn.py --h 0.25 1.5 2.5 1.0025 # altri campi (una riga per ciascuno)
"""
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src import ising_nnn as nnn

J2 = -1.0
H_DEFAULT = [0.25, 1.5, 2.5]


def save_fig(fig, outdir, name):
    path = os.path.join(outdir, name)
    try:
        fig.savefig(path, dpi=150)
        print(f"   salvata: {path}")
    except OSError as err:
        base, ext = os.path.splitext(name)
        alt = os.path.join(outdir, f"{base}_{time.strftime('%H%M%S')}{ext}")
        print(f"   ATTENZIONE: non riesco a scrivere {path} ({err}). Salvo come {alt}")
        fig.savefig(alt, dpi=150)


def ground_state_name(h):
    """Stato fondamentale a (h, J2 = -1): strisce (h < 1), plateau (1 < h < 2), tutti up (h > 2)."""
    if h < -J2:
        return "stripe"
    if h < 1.0 - J2:
        return "plateau"
    return "up"


def state_label(h):
    return {"stripe": "strisce", "plateau": "plateau M=1/2", "up": "tutti up"}[ground_state_name(h)]


def run_snapshots(T, h, lattice0, times):
    """Evolve una copia di lattice0 a temperatura T e restituisce {t: reticolo} per t in times (sweep)."""
    lat = lattice0.copy()
    wanted = set(times)
    snaps = {}
    if 0 in wanted:
        snaps[0] = lat.copy()
    for t in range(1, max(times) + 1):
        nnn.sweep(lat, T, h, J2)
        if t in wanted:
            snaps[t] = lat.copy()
    return snaps


def plot_snapshots(all_snaps, hs, times, L, T, start, outdir, name):
    fig, axes = plt.subplots(len(hs), len(times), figsize=(2.7 * len(times), 3.0 * len(hs)), squeeze=False)
    cmap = ListedColormap(["#f2c14e", "#1b3a5c"])           # -1 chiaro, +1 scuro
    for r, h in enumerate(hs):
        for c, t in enumerate(times):
            lat = all_snaps[h][t]
            ax = axes[r, c]
            ax.imshow(lat, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            M = nnn.magnetization(lat)
            S = nnn.staggered_magnetization(lat)
            Q = nnn.stripe_order(lat)
            ax.set_title(f"t = {t}\nM = {M:.2f}  S = {S:.2f}  Q = {Q:.2f}", fontsize=8.5)
        axes[r, 0].set_ylabel(f"h = {h:g}\n({state_label(h)})", fontsize=11)
    handles = [Patch(color="#1b3a5c", label=r"spin su ($+1$)"), Patch(color="#f2c14e", label=r"spin giu' ($-1$)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False)
    start_txt = "stato casuale" if start == "random" else "stato fondamentale"
    fig.suptitle(f"Snapshot ({L}x{L}), J2 = {J2:g}, T = {T:g}, partenza: {start_txt}   (t in sweep)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95), h_pad=2.0)
    save_fig(fig, outdir, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=20, help="lato del reticolo (pari)")
    ap.add_argument("--T", type=float, default=0.05, help="temperatura (bassa)")
    ap.add_argument("--h", type=float, nargs="+", default=H_DEFAULT, help="campi (una riga per ciascuno)")
    ap.add_argument("--times", type=int, nargs="+", default=[0, 10, 100, 1000, 10000],
                    help="tempi (in sweep) a cui fare lo snapshot")
    ap.add_argument("--start", choices=["random", "ground"], default="random")
    ap.add_argument("--j2", type=float, default=J2)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    J2 = args.j2
    if J2 > -0.5 or abs(J2 + 1.0) > 1e-9:
        print("ATTENZIONE: le etichette degli stati (strisce / plateau / tutti up) valgono per J2 = -1.")
    times = sorted(set(args.times))
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(outdir, exist_ok=True)

    nnn.set_seed(args.seed)
    hs = list(args.h)
    lattice_random = nnn.initialize_lattice(args.L, "random")           # STESSA partenza casuale per tutti gli h

    all_snaps = {}
    t_start = time.time()
    for h in hs:
        lat0 = lattice_random if args.start == "random" else nnn.initialize_lattice(args.L, ground_state_name(h))
        all_snaps[h] = run_snapshots(args.T, h, lat0, times)
        last = all_snaps[h][times[-1]]
        print(f"   h = {h:<7g} fatto ({time.time() - t_start:5.1f} s)   a t = {times[-1]}:  "
              f"M = {nnn.magnetization(last):.3f}  S = {nnn.staggered_magnetization(last):.3f}  "
              f"Q = {nnn.stripe_order(last):.3f}", flush=True)

    plot_snapshots(all_snaps, hs, times, args.L, args.T, args.start, outdir,
                   f"fig11_snapshot_nnn_{args.start}.png")
    print(f"finito in {time.time() - t_start:.1f} s")

    if not args.no_show:
        plt.show()