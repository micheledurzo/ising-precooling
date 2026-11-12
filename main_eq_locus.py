"""
main_03_locus.py
"Equilibrium locus" nel piano (<M>, <S>) per diversi valori del campo h.
 
Per ogni h si calcola, su una griglia di temperature, il punto di equilibrio
(<M>_pi(T), <S>_pi(T)); l'insieme di questi punti al variare di T e' una curva nel
piano M-S: il LUOGO DI EQUILIBRIO (equilibrium locus) di quel valore di h.
Mettendo piu' h nello stesso piano si esplora il diagramma di fase:
    h < 1  :  ordine antiferromagnetico (Neel), S -> 1 e M -> 0 a bassa T
    h ~ 1  :  zona critica, M ~ 0.5-0.6 e S piccola (gas di quadrati duri)
    h > 1  :  ordine ferromagnetico indotto dal campo, M -> 1 e S -> 0 a bassa T
Il verso di percorrenza: ogni curva parte da T_cold (triangolo) a bassa T e sale in T
passando per T_0 (cerchio) e T_max (quadrato), fino a T = 1.2 (croce).
 
Produce (in figures/):
  fig6_locus_tutti_h.png    tutte le curve nello stesso piano (a sinistra intero, a destra zoom su S piccola)
  fig6b_locus_per_h.png     un pannello per ogni h, con barre d'errore e colore = T
  dati_fig6.npz             risultati numerici (salvati PRIMA dei grafici)
e stampa a terminale una tabella con <M>, <S> a T_cold, T_0, T_max per ogni h.
 
Uso:
    python main_03_locus.py                        # h = 0.25 0.5 0.99 1.0025 1.01 1.5 2.5
    python main_03_locus.py --h 0.25 1.01 2.5      # altri valori di h
    python main_03_locus.py --check                # solo test: l'equilibrio non dipende dalla partenza
    python main_03_locus.py --quick                # run velocissimo per controllare che giri
    python main_03_locus.py --L 20 --rep 50        # altro L (pari) / piu' repliche
"""
import argparse
import os
import time
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
 
from src import ising
 
H_PAPER = 1.0025
T0, TMAX = 0.125, 0.625
TCOLD = 0.0125                      # modificabile con --tcold
T_END = 1.2                         # ultima temperatura della griglia
H_LIST = [0.25, 0.5, 0.99, H_PAPER, 1.01, 1.5, 2.5]
 
FULL = dict(n_T=36, n_burn=1000, n_meas=3000, n_rep=30)
QUICK = dict(n_T=9, n_burn=60, n_meas=60, n_rep=4)
MARKERS = ("^", "o", "s")           # T_cold, T_0, T_max (stessi simboli del paper)
 
 
# --------------------------------------------------------------------------
# Utilita'
# --------------------------------------------------------------------------
def sem(x, axis=0):
    return x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])
 
 
def paper_T():
    return np.array([TCOLD, T0, TMAX])
 
 
def T_grid_geom(n, tmin=None, tmax=T_END):
    """Griglia geometrica di temperature che contiene esattamente T_cold, T_0, T_max."""
    tmin = TCOLD if tmin is None else tmin
    T = np.geomspace(tmin, tmax, n)
    far = np.all(np.abs(np.log(T[:, None] / paper_T()[None, :])) > 0.04, axis=1)
    return np.unique(np.concatenate([T[far], paper_T()]))
 
 
def init_for(h, init):
    """'auto': Neel per h < 1 (a bassa T lo stato di equilibrio e' ordinato, e da una
    partenza casuale il sistema resta intrappolato in domini/strisce con S ~ 0);
    casuale per h >= 1 (verificato: tutte le partenze danno lo stesso equilibrio)."""
    if init != "auto":
        return init
    return "neel" if h < 1.0 - 1e-9 else "random"
 
 
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
 
 
def scan(L, h_list, T_grid, n_burn, n_meas, n_rep, init):
    """Equilibrio su (h, T). Ritorna media ed errore, shape (nh, nT, 3) = [M, S, e]."""
    mean = np.empty((len(h_list), len(T_grid), 3))
    err = np.empty_like(mean)
    t_start = time.time()
    for a, h in enumerate(h_list):
        init_h = init_for(h, init)
        for b, T in enumerate(T_grid):
            out = ising.equilibrium_point(L, T, h, n_burn, n_meas, n_rep, ising.INIT[init_h])
            mean[a, b] = out.mean(axis=0)
            err[a, b] = sem(out)
        print(f"   h = {h:<7g} fatto (start {init_h}; {time.time() - t_start:5.1f} s)", flush=True)
    return mean, err
 
 
# --------------------------------------------------------------------------
# Controllo: l'equilibrio non deve dipendere dalla condizione iniziale
# --------------------------------------------------------------------------
def check_starts(L, h_list, cfg, init):
    nb, nm, nr = cfg["n_burn"], cfg["n_meas"], max(cfg["n_rep"], 8)
    print("\nCONTROLLO termalizzazione: <M>, <S> da tre partenze diverse "
          f"(L = {L}, {nr} repliche, {nb}+{nm} MC steps)")
    print(f"{'h':>7} {'T':>8} | {'casuale':^15} | {'tutti up':^15} | {'Neel':^15} | start usato")
    for h in h_list:
        for T in (TCOLD, 0.05, T0):
            res = {}
            for st in ("random", "up", "neel"):
                o = ising.equilibrium_point(L, T, h, nb, nm, nr, ising.INIT[st])
                res[st] = (o.mean(axis=0), sem(o))
            used = init_for(h, init)
            ref = res[used]
            ok = all(abs(ref[0][k] - res[s][0][k]) < 3 * np.hypot(ref[1][k], res[s][1][k]) + 0.01
                     for s in res for k in (0, 1))
            cells = " | ".join(f"{res[s][0][0]:6.3f} {res[s][0][1]:6.3f}  " for s in ("random", "up", "neel"))
            print(f"{h:7g} {T:8.4f} | {cells} | {used:6s} " + ("" if ok else "  <-- le partenze NON concordano"))
 
 
# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
 
def fig_per_h(L, h_list, T_grid, mean, err, outdir):
    n = len(h_list)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.3 * nrows), squeeze=False,
                             layout="constrained")
    norm = LogNorm(T_grid[0], T_grid[-1])
    sc = None
    for k, h in enumerate(h_list):
        ax = axes[k // ncols, k % ncols]
        M, S = mean[k, :, 0], mean[k, :, 1]
        ax.plot(M, S, "-", color="0.65", lw=1, zorder=1)
        ax.errorbar(M, S, xerr=err[k, :, 0], yerr=err[k, :, 1], fmt="none",
                    ecolor="0.7", elinewidth=0.7, zorder=1)
        sc = ax.scatter(M, S, c=T_grid, norm=norm, cmap="plasma", s=18, zorder=2)
        for T, mk in zip(paper_T(), MARKERS):
            i = int(np.argmin(np.abs(T_grid - T)))
            ax.plot(M[i], S[i], mk, color="none", mec="k", ms=10, mew=1.5, zorder=3)
        ax.set_title(f"h = {h:g}")
        ax.set_xlabel(r"$\langle M\rangle$")
        ax.set_ylabel(r"$\langle S\rangle$")
        ax.grid(alpha=0.3)
    for k in range(n, nrows * ncols):
        axes[k // ncols, k % ncols].axis("off")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="T", shrink=0.8)
    fig.suptitle(f"Equilibrio per h diversi (L = {L}; "
                 "triangolo = T_cold, cerchio = T_0, quadrato = T_max)")
    save_fig(fig, outdir, "fig6b_locus_per_h.png")
 
 
def print_table(h_list, T_grid, mean, err):
    print("\n" + "=" * 92)
    print("EQUILIBRIO ai tre punti del paper, per ogni h  (<M> +- err ; <S> +- err)")
    print("=" * 92)
    print(f"{'h':>7} | " + " | ".join(f"{nm + ' = ' + format(T, 'g'):^26}" for nm, T in
                                      zip(("T_cold", "T_0", "T_max"), paper_T())))
    for a, h in enumerate(h_list):
        cells = []
        for T in paper_T():
            i = int(np.argmin(np.abs(T_grid - T)))
            m, s, dm, ds = mean[a, i, 0], mean[a, i, 1], err[a, i, 0], err[a, i, 1]
            cells.append(f"{m:6.3f}+-{dm:.3f}  {s:6.3f}+-{ds:.3f}")
        print(f"{h:7g} | " + " | ".join(f"{c:^26}" for c in cells))
    print("=" * 92)
 
 
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=10, help="LATO del reticolo (pari); N = L*L spin")
    ap.add_argument("--h", type=float, nargs="+", default=H_LIST, help="valori del campo h")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--check", action="store_true", help="solo il test sulle condizioni iniziali")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--tcold", type=float, default=TCOLD)
    ap.add_argument("--init", choices=list(ising.INIT) + ["auto"], default="auto")
    ap.add_argument("--rep", type=int, help="numero di repliche per punto")
    ap.add_argument("--meas", type=int, help="MC steps di misura per punto")
    ap.add_argument("--burn", type=int, help="MC steps di termalizzazione per punto")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
 
    TCOLD = args.tcold
    cfg = dict(QUICK if args.quick else FULL)
    for key, val in (("n_rep", args.rep), ("n_meas", args.meas), ("n_burn", args.burn)):
        if val is not None:
            cfg[key] = val
    h_list = sorted(set(args.h))
 
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(outdir, exist_ok=True)
    ising.set_seed(args.seed)
 
    t0 = time.time()
    if args.check:
        check_starts(args.L, h_list, cfg, args.init)
    else:
        T_grid = T_grid_geom(cfg["n_T"])
        mean, err = scan(args.L, h_list, T_grid, cfg["n_burn"], cfg["n_meas"], cfg["n_rep"], args.init)
        try:
            np.savez(os.path.join(outdir, "dati_fig6.npz"), h=h_list, T=T_grid, mean=mean, err=err)
        except OSError as e:
            print(f"   (dati non salvati: {e})")
        print_table(h_list, T_grid, mean, err)
        fig_per_h(args.L, h_list, T_grid, mean, err, outdir)
    print(f"finito in {time.time() - t0:.1f} s")
 
    if not args.no_show and not args.check:
        plt.show()
 