"""
main_07_size_scaling.py
Dipendenza dalla taglia L del locus di equilibrio nel piano (<M>, <S>) a h fissato < h_c = 1 (default h = 0.25).

Per h < 1 c'e' una transizione ordine-disordine (Neel -> paramagnete) a una temperatura T_c(h). A L finito il
salto di S e' smussato e diventa piu' brusco al crescere di L. Qui si vede SOLO qualitativamente: per ogni L
(default 10, 30, 50, 70, 100) si simula l'equilibrio a h su una griglia di T attorno a T_c e si disegna il
locus pi(T) = (<M>, <S>) nel piano M-S, una curva per ogni L.

Produce (in figures/):
  fig10_locus_per_L.png     locus nel piano M-S, una curva per ogni L (colore = L); il quadrato marca T_max
  fig10b_S_vs_T.png         (solo con --also-st) S(T) per ogni L: lo stesso dato, ma il gradino si legge meglio
  dati_fig10_h<h>.npz       risultati numerici (salvati PRIMA dei grafici; si riusano con --load)

Note pratiche.
  * La statistica e' volutamente modesta (grafico qualitativo): vicino a T_c la dinamica di Glauber rallenta
    (tempo di correlazione ~ L^2.17 sweep) e i punti a L grande possono essere rumorosi. Termalizzazione e misura
    scalano come L^2 (--burn-base, --meas-base valgono per L = 10); se vuoi curve piu' lisce aumenta questi due
    parametri e --rep.
  * Partenza: Neel (come negli altri script per h < 1), per evitare i domini a strisce bloccati sotto T_c.
  * T_cold e T_0 stanno nella fase ordinata (S ~ 1, M ~ 0) per ogni L; T_max = 0.625 e' appena sopra la
    transizione: li' S dipende molto da L.
  * Tempi (numba): L = 10, 30, 50, 70 -> pochi minuti; L = 100 e' la parte lunga (circa 10 minuti con i default).

Uso:
    python main_07_size_scaling.py                          # h = 0.25, L = 10 30 50 70 100
    python main_07_size_scaling.py --h 0.5 --tlo 0.3 --thi 0.7
    python main_07_size_scaling.py --sizes 10 30 70          # altri L (pari)
    python main_07_size_scaling.py --also-st                 # aggiunge la figura S(T)
    python main_07_size_scaling.py --load figures/dati_fig10_h0.25.npz   # rifai solo i grafici
    python main_07_size_scaling.py --quick                   # run velocissimo per controllare che giri
"""
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from src import ising

H_DEFAULT = 0.25
T0, TMAX = 0.125, 0.625

FULL = dict(sizes=[10, 30, 50, 70, 100], n_rep=12, burn_base=100, meas_base=150)
QUICK = dict(sizes=[6, 10, 14], n_rep=3, burn_base=40, meas_base=40)


# --------------------------------------------------------------------------
# Utilita'
# --------------------------------------------------------------------------
def sem(x, axis=0):
    return x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])


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


def burn_meas(L, cfg):
    f = (L / 10.0) ** 2
    return max(int(cfg["burn_base"] * f), 20), max(int(cfg["meas_base"] * f), 20)


def make_T_grid(args):
    """Griglia 'grossolana' equispaziata + zona fine attorno a T_c (h = 0.25: T_c ~ 0.55) + T_max."""
    T = list(np.linspace(args.tlo, args.thi, args.nT))
    if not args.no_fine:
        T += list(np.linspace(args.fine[0], args.fine[1], int(args.fine[2])))
    T.append(TMAX)
    T = np.unique(np.round(np.array(T), 6))
    return T[(T >= args.tlo - 1e-9) & (T <= args.thi + 1e-9)]


# --------------------------------------------------------------------------
# Simulazione
# --------------------------------------------------------------------------
def simulate(h, sizes, T_grid, cfg):
    """Ritorna R (nL, nT, n_rep, 3) con colonne [M, S, e] (medie temporali per replica)."""
    R = np.empty((len(sizes), len(T_grid), cfg["n_rep"], 3))
    t_start = time.time()
    for a, L in enumerate(sizes):
        nb, nm = burn_meas(L, cfg)
        print(f"   L = {L:<3d} ({nb} sweep di termalizzazione + {nm} di misura, {cfg['n_rep']} repliche)", flush=True)
        for b, T in enumerate(T_grid):
            R[a, b] = ising.equilibrium_point(L, T, h, nb, nm, cfg["n_rep"], ising.INIT["neel"])
        print(f"      fatto ({time.time() - t_start:6.1f} s)", flush=True)
    return R


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def colors(n):
    cmap = plt.get_cmap("viridis")
    return [cmap(0.05 + 0.85 * k / max(n - 1, 1)) for k in range(n)]


def fig_locus(h, sizes, T, R, outdir):
    mean = R.mean(axis=2)
    err = sem(R, axis=2)
    cols = colors(len(sizes))
    j = int(np.argmin(np.abs(T - TMAX)))
    fig, ax = plt.subplots(figsize=(8.2, 6.2), layout="constrained")
    for a, L in enumerate(sizes):
        M, S = mean[a, :, 0], mean[a, :, 1]
        ax.errorbar(M, S, xerr=err[a, :, 0], yerr=err[a, :, 1], fmt="none", ecolor=cols[a], alpha=0.35,
                    elinewidth=0.8, zorder=1)
        ax.plot(M, S, "-o", color=cols[a], ms=4, lw=1.5, label=f"L = {L}", zorder=2)
        if abs(T[j] - TMAX) < 1e-6:
            ax.plot(M[j], S[j], "s", color=cols[a], mec="k", ms=9, zorder=3)
    ax.plot([], [], "s", color="w", mec="k", ms=9, label=r"$T_{max}$")
    ax.set_xlabel(r"$\langle M\rangle_{\pi(T)}$")
    ax.set_ylabel(r"$\langle S\rangle_{\pi(T)}$")
    ax.set_title(f"Locus di equilibrio nel piano M-S per diversi L   (h = {h:g})\n"
                 "T cresce dall'alto a sinistra (fase di Neel) al basso a destra (disordine)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    save_fig(fig, outdir, "fig10_locus_per_L.png")


def fig_S_vs_T(h, sizes, T, R, outdir):
    mean = R.mean(axis=2)
    err = sem(R, axis=2)
    cols = colors(len(sizes))
    fig, ax = plt.subplots(figsize=(8.2, 5.4), layout="constrained")
    for a, L in enumerate(sizes):
        ax.errorbar(T, mean[a, :, 1], yerr=err[a, :, 1], fmt="-o", ms=3.5, capsize=2, color=cols[a],
                    label=f"L = {L}")
    if T[0] <= TMAX <= T[-1]:
        ax.axvline(TMAX, color="r", ls=":", lw=1, label=r"$T_{max}$")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\langle S\rangle$")
    ax.set_title(f"S(T) per diversi L   (h = {h:g})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    save_fig(fig, outdir, "fig10b_S_vs_T.png")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=H_DEFAULT, help="campo h (< 1)")
    ap.add_argument("--sizes", type=int, nargs="+", help="lati L (pari), default 10 30 50 70 100")
    ap.add_argument("--tlo", type=float, default=0.30)
    ap.add_argument("--thi", type=float, default=0.85)
    ap.add_argument("--nT", type=int, default=12, help="punti equispaziati in [tlo, thi]")
    ap.add_argument("--fine", type=float, nargs=3, default=[0.50, 0.62, 13], metavar=("LO", "HI", "N"),
                    help="zona a griglia fine attorno a T_c: T da LO a HI con N punti (default 0.50 0.62 13)")
    ap.add_argument("--no-fine", action="store_true", help="niente zona fine")
    ap.add_argument("--rep", type=int, help="repliche per punto")
    ap.add_argument("--burn-base", type=int, help="sweep di termalizzazione per L = 10 (scala come L^2)")
    ap.add_argument("--meas-base", type=int, help="sweep di misura per L = 10 (scala come L^2)")
    ap.add_argument("--also-st", action="store_true", help="disegna anche S(T)")
    ap.add_argument("--load", help="npz salvato in precedenza: salta la simulazione e rifa' solo i grafici")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    cfg = dict(QUICK if args.quick else FULL)
    if args.sizes:
        cfg["sizes"] = sorted(set(args.sizes))
    for key, val in (("n_rep", args.rep), ("burn_base", args.burn_base), ("meas_base", args.meas_base)):
        if val is not None:
            cfg[key] = val
    if any(L % 2 for L in cfg["sizes"]):
        raise SystemExit("tutti gli L devono essere pari")
    h = args.h
    if h >= 1.0:
        raise SystemExit("questo script e' pensato per h < 1 (fase di Neel a bassa T)")
    if args.quick:
        args.nT, args.fine = 6, [0.5, 0.62, 4]

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(outdir, exist_ok=True)
    ising.set_seed(args.seed)
    t0 = time.time()

    if args.load:
        d = np.load(args.load)
        R, sizes, T, h = d["R"], [int(x) for x in d["sizes"]], d["T"], float(d["h"])
        print(f"dati letti da {args.load}  (h = {h}, L = {sizes})")
    else:
        sizes = cfg["sizes"]
        T = make_T_grid(args)
        R = simulate(h, sizes, T, cfg)
        try:
            np.savez(os.path.join(outdir, f"dati_fig10_h{h:g}.npz"), R=R, sizes=np.array(sizes), T=T, h=h)
        except OSError as e:
            print(f"   (dati non salvati: {e})")

    fig_locus(h, sizes, T, R, outdir)
    if args.also_st:
        fig_S_vs_T(h, sizes, T, R, outdir)
    print(f"finito in {time.time() - t0:.1f} s")

    if not args.no_show:
        plt.show()