"""
main_08_equilibrio_nnn.py
Studio dell'equilibrio e dell'evoluzione verso l'equilibrio del modello di Ising 2D antiferromagnetico con campo
E INTERAZIONE A SECONDI VICINI J2 (src/ising_nnn.py). Analogo di main_01_equilibrio.py.
Default: J2 = -1 (secondi vicini antiferromagnetici, forte frustrazione).

PERCHE' CAMBIA L'ANALISI. Con J2 = -1 gli stati fondamentali (verificati per enumerazione esaustiva) sono
    h < 1       strisce        M = 0,   S = 0 (!),  Q = 1
    1 < h < 2   plateau M=1/2  M = 0.5, S = sx = sy = 0.5
    h > 2       tutti up       M = 1
(in generale per J2 < -0.5: strisce per h < -J2, plateau fino a h = 1 - J2, poi tutti up). Il parametro d'ordine
di Neel S vale 0 nelle strisce, quindi qui si seguono QUATTRO osservabili:
    M   magnetizzazione,
    S   magnetizzazione staggered a scacchiera (Neel),
    Q   ordine a strisce  Q = sqrt(sx^2 + sy^2),  sx = |(1/N) sum (-1)^i s|,  sy = |(1/N) sum (-1)^j s|,
    e   energia per spin.
Livelli di uno stato casuale a L finito: S ~ 0.8/L, Q ~ 1.25/L.
Il campo del paper h = 1.0025 cade subito sopra la prima transizione (h = 1): strisce e plateau sono quasi degeneri
(differenza di energia 0.00125 per spin) e il Glauber a bassa T rimane facilmente intrappolato in miscele di domini.
Per questo le scansioni partono dallo stato fondamentale giusto ('auto') e la tabella mostra quattro partenze.

Figure (in figures/):
  fig1_rilassamento_nnn   M, S, Q, e in funzione di t (sweep) da 4 condizioni iniziali, a T_cold, T_0, T_max
  fig2_equilibrio_T_nnn   M, S, Q, e di equilibrio vs T per diversi h (riga in alto T log, in basso T lineare)
  fig3_mappa_h_T_nnn      mappe di M, S e Q nel piano (h, T)
  fig4_curva_nnn          curva di equilibrio a h = h_paper nei piani (M, S) e (M, Q)
Terminale: tabella di <M>, <S>, <Q>, e ai tre punti T_cold, T_0, T_max a h = h_paper con 4 partenze diverse
(se non concordano il sistema non ha termalizzato: aumenta --burn-pts o cambia h).
I dati numerici vengono salvati (npz) PRIMA di ogni grafico.

Uso:
    python main_08_equilibrio_nnn.py                  # J2 = -1, tutto
    python main_08_equilibrio_nnn.py --only table     # solo la tabella
    python main_08_equilibrio_nnn.py --only 1 --h 0.5 # rilassamento a h = 0.5 (strisce) invece che 1.0025
    python main_08_equilibrio_nnn.py --j2 -0.3        # altro J2 (J2 > -0.5: stato fondamentale di Neel)
    python main_08_equilibrio_nnn.py --quick          # run velocissimo per controllare che giri
    python main_08_equilibrio_nnn.py --L 20 --only 2  # altro L (pari)
"""
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from src import ising_nnn as nnn

# ---- parametri (come main_01) ---------------------------------------------
H_PAPER = 1.0025
T0, TMAX = 0.125, 0.625
TCOLD = 0.00125               # --tcold
J2 = -1.0                     # --j2
PAPER_T = np.array([TCOLD, T0, TMAX])
PAPER_NAMES = ["T_cold", "T_0", "T_max"]

FULL = dict(
    n_relax=10000, n_rep_relax=40,                                       # fig1
    H_list=[0.0, 0.5, 0.99, H_PAPER, 1.5, 1.99, 2.5],                    # fig2
    n_T=35, n_burn=3000, n_meas=2000, n_rep=30, n_T_lin=19, T_lin_lo=0.3, T_lin_hi=1.2,
    h_map=np.linspace(0.0, 3.0, 25), n_T_map=23, n_burn_map=1500, n_meas_map=1000, n_rep_map=10,   # fig3
    n_T_curve=50, n_burn_curve=5000, n_meas_curve=5000, n_rep_curve=25,  # fig4
    n_burn_pts=20000, n_meas_pts=20000, n_rep_pts=40,                    # tabella
)
QUICK = dict(
    n_relax=300, n_rep_relax=6,
    H_list=[0.0, 0.5, 0.99, H_PAPER, 1.5, 1.99, 2.5],
    n_T=8, n_burn=60, n_meas=60, n_rep=4, n_T_lin=4, T_lin_lo=0.3, T_lin_hi=1.2,
    h_map=np.linspace(0.0, 3.0, 7), n_T_map=7, n_burn_map=40, n_meas_map=40, n_rep_map=3,
    n_T_curve=10, n_burn_curve=60, n_meas_curve=60, n_rep_curve=4,
    n_burn_pts=150, n_meas_pts=150, n_rep_pts=4,
)

LABELS = [r"$\langle M\rangle$", r"$\langle S\rangle$ (Neel)", r"$\langle Q\rangle$ (strisce)", r"$e = E/N$"]
SHORT = ["M", "S", "Q", "e"]
SCAN_INIT = "auto"


# --------------------------------------------------------------------------
# Utilita'
# --------------------------------------------------------------------------
def sem(x, axis=0):
    return x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])


def ground_init(h, j2=None):
    """Condizione iniziale = stato fondamentale a (h, J2)."""
    j2 = J2 if j2 is None else j2
    if j2 > -0.5:                                 # Neel e' lo stato fondamentale a h piccolo
        return "neel" if h < 1.0 - 1e-9 else "random"
    if h < -j2 - 1e-9:
        return "stripe"
    if h < 1.0 - j2 - 1e-9:
        return "plateau"
    return "up"


def init_for(h, init):
    return ground_init(h) if init == "auto" else init


def T_grid_geom(n, tmin=None, tmax=1.2):
    """Griglia geometrica che CONTIENE esattamente T_cold, T_0, T_max."""
    tmin = TCOLD if tmin is None else tmin
    T = np.geomspace(tmin, tmax, n)
    far = np.all(np.abs(np.log(T[:, None] / PAPER_T[None, :])) > 0.04, axis=1)
    return np.unique(np.concatenate([T[far], PAPER_T]))


def edges_lin(x):
    x = np.asarray(x, float)
    mid = (x[1:] + x[:-1]) / 2
    return np.concatenate([[x[0] - (x[1] - x[0]) / 2], mid, [x[-1] + (x[-1] - x[-2]) / 2]])


def edges_log(x):
    return np.exp(edges_lin(np.log(np.asarray(x, float))))


def save_fig(fig, outdir, name):
    path = os.path.join(outdir, name)
    try:
        fig.savefig(path, dpi=150)
        print(f"   salvata: {path}")
    except OSError as err:
        base, ext = os.path.splitext(name)
        alt = os.path.join(outdir, f"{base}_{time.strftime('%H%M%S')}{ext}")
        print(f"   ATTENZIONE: non riesco a scrivere {path}  ({err}). Salvo come {alt}")
        fig.savefig(alt, dpi=150)


def save_data(outdir, name, **arrays):
    try:
        np.savez(os.path.join(outdir, name), **arrays)
    except OSError as err:
        print(f"   (dati non salvati: {err})")


def mark_paper_T(ax, labels=False):
    for T, c, nm in zip(PAPER_T, ("b", "k", "r"), PAPER_NAMES):
        ax.axvline(T, color=c, ls=":", lw=1, label=(nm if labels else None))


def scan(L, h_values, T_values, n_burn, n_meas, n_rep, init=None):
    """Equilibrio su una griglia (h, T). Ritorna media ed errore, shape (nh, nT, 4) = [M, S, Q, e]."""
    init = SCAN_INIT if init is None else init
    mean = np.empty((len(h_values), len(T_values), 4))
    err = np.empty_like(mean)
    t_start = time.time()
    for a, h in enumerate(h_values):
        init_h = init_for(h, init)
        for b, T in enumerate(T_values):
            out = nnn.equilibrium_point(L, T, h, J2, n_burn, n_meas, n_rep, nnn.INIT[init_h])
            mean[a, b] = out.mean(axis=0)
            err[a, b] = sem(out)
        print(f"   h = {h:6.4f} fatto  (start {init_h}; {time.time() - t_start:5.1f} s)", flush=True)
    return mean, err


# --------------------------------------------------------------------------
# Tabella a terminale
# --------------------------------------------------------------------------
def table_paper_points(L, cfg, h):
    nb, nm, nr = cfg["n_burn_pts"], cfg["n_meas_pts"], cfg["n_rep_pts"]
    gi = ground_init(h)
    inits = ["random", "up", "neel", gi] if gi not in ("random", "up", "neel") else ["random", "up", "neel"]
    print()
    print("=" * 92)
    print(f"PUNTI DI EQUILIBRIO a h = {h}, J2 = {J2}   (L = {L}; {nr} repliche; "
          f"{nb} sweep di termalizzazione + {nm} di misura)")
    print(f"stato fondamentale a questo (h, J2): {gi}")
    print("=" * 92)
    for T, name in zip(PAPER_T, PAPER_NAMES):
        res = {}
        for init in inits:
            out = nnn.equilibrium_point(L, T, h, J2, nb, nm, nr, nnn.INIT[init])
            res[init] = (out.mean(axis=0), sem(out))
        print(f"\n{name} = {T}")
        for init in inits:
            m, e = res[init]
            tag = "  <- stato fondamentale" if init == gi else ""
            print(f"   start {init:8s}: M = {m[0]:.4f}+-{e[0]:.4f}  S = {m[1]:.4f}+-{e[1]:.4f}  "
                  f"Q = {m[2]:.4f}+-{e[2]:.4f}  e = {m[3]:.5f}+-{e[3]:.5f}{tag}")
        ok = True
        for init in inits[1:]:
            for k in (0, 1, 2, 3):
                d = abs(res[inits[0]][0][k] - res[init][0][k])
                tol = 3 * np.hypot(res[inits[0]][1][k], res[init][1][k]) + (0.005 if k < 3 else 0.0005)
                ok &= d < tol
        print("   termalizzazione: " + ("OK (le partenze concordano)" if ok else
              "ATTENZIONE - le partenze NON concordano: il sistema e' intrappolato in stati metastabili "
              "(domini di strisce / plateau) su questa durata. Aumenta --burn-pts."))
    print(f"\n(livelli di uno stato casuale a L={L}:  S ~ 0.8/L = {np.sqrt(2 / np.pi) / L:.4f},"
          f"   Q ~ 1.25/L = {np.sqrt(np.pi / 2) / L:.4f})")
    print("=" * 92)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def fig1_rilassamento(L, cfg, outdir, h):
    """Rilassamento temporale da condizioni iniziali diverse a (h, J2): M, S, Q, e."""
    gi = ground_init(h)
    inits = [("random", "casuale"), ("up", "tutti up"), ("neel", "Neel")]
    if gi not in ("random", "up", "neel"):
        inits.append((gi, f"stato fond. ({gi})"))
    temps = [(TCOLD, r"$T_{cold}$"), (T0, r"$T_0$"), (TMAX, r"$T_{max}$")]
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
    t = np.arange(cfg["n_relax"] + 1)
    data_all = {}
    for col, (T, Tname) in enumerate(temps):
        for init, iname in inits:
            M, S, Q, E = nnn.run_relaxation(L, T, h, J2, cfg["n_relax"], cfg["n_rep_relax"], nnn.INIT[init])
            data_all[f"{init}_T{T:g}"] = np.stack([M.mean(0), S.mean(0), Q.mean(0), E.mean(0)])
            for row, data in enumerate((M, S, Q, E)):
                m, e = data.mean(axis=0), sem(data)
                axes[row, col].plot(t[1:], m[1:], label=iname)
                axes[row, col].fill_between(t[1:], (m - e)[1:], (m + e)[1:], alpha=0.25)
        axes[0, col].set_title(f"{Tname} = {T}")
        axes[1, col].axhline(np.sqrt(2 / np.pi) / L, color="k", ls=":", lw=1, label=r"casuale $\approx 0.8/L$")
        axes[2, col].axhline(np.sqrt(np.pi / 2) / L, color="k", ls=":", lw=1, label=r"casuale $\approx 1.25/L$")
        axes[3, col].set_xlabel("t  (sweep = L*L tentativi di flip)")
    for row in range(4):
        axes[row, 0].set_ylabel(LABELS[row])
        for col in range(3):
            axes[row, col].set_xscale("log")
            axes[row, col].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    axes[2, 0].legend(fontsize=8)
    fig.suptitle(f"Equilibrio vs T  (L = {L}, J2 = {J2}, h = {h})")
    fig.tight_layout()
    save_data(outdir, "dati_fig1_nnn.npz", t=t, h=h, J2=J2, **data_all)
    save_fig(fig, outdir, "fig1_rilassamento_nnn.png")


def fig2_equilibrio_T(L, cfg, outdir):
    """M, S, Q, e di equilibrio vs T per diversi h: riga in alto T log, riga in basso T lineare."""
    T_grid = np.unique(np.concatenate([T_grid_geom(cfg["n_T"]),
                                       np.linspace(cfg["T_lin_lo"], cfg["T_lin_hi"], cfg["n_T_lin"])]))
    mean, err = scan(L, cfg["H_list"], T_grid, cfg["n_burn"], cfg["n_meas"], cfg["n_rep"])
    save_data(outdir, "dati_fig2_nnn.npz", h=cfg["H_list"], T=T_grid, mean=mean, err=err, J2=J2)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for row, scale in enumerate(("log", "linear")):
        for a, h in enumerate(cfg["H_list"]):
            for k in range(4):
                axes[row, k].errorbar(T_grid, mean[a, :, k], yerr=err[a, :, k], marker="o", ms=3, capsize=2,
                                      label=f"h = {h}")
        for k in range(4):
            ax = axes[row, k]
            ax.set_xscale(scale)
            ax.set_xlabel("T" + ("  (scala log)" if scale == "log" else "  (scala lineare)"))
            ax.set_ylabel(LABELS[k])
            ax.grid(alpha=0.3, which="both")
            mark_paper_T(ax, labels=(row == 0 and k == 0))
            if scale == "linear":
                ax.set_xlim(0, T_grid[-1] * 1.02)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Equilibrio vs T  (L = {L}, J2 = {J2}; linee verticali: T_cold, T_0, T_max)")
    fig.tight_layout()
    save_fig(fig, outdir, "fig2_equilibrio_T_nnn.png")


def fig3_mappa(L, cfg, outdir):
    """Mappe di <M>, <S>, <Q> nel piano (h, T)."""
    h_map = np.unique(np.concatenate([cfg["h_map"], [H_PAPER]]))
    T_map = T_grid_geom(cfg["n_T_map"], tmax=1.0)
    mean, _ = scan(L, h_map, T_map, cfg["n_burn_map"], cfg["n_meas_map"], cfg["n_rep_map"])
    save_data(outdir, "dati_fig3_nnn.npz", h=h_map, T=T_map, mean=mean, J2=J2)
    h_e = edges_lin(h_map)
    h_e[0] = max(h_e[0], 0.0)
    T_e = edges_log(T_map)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), layout="constrained")
    for ax, k, cmap in ((axes[0], 0, "viridis"), (axes[1], 1, "magma"), (axes[2], 2, "cividis")):
        pc = ax.pcolormesh(h_e, T_e, mean[:, :, k].T, shading="flat", cmap=cmap)
        fig.colorbar(pc, ax=ax, label=LABELS[k], pad=0.02)
        ax.plot(H_PAPER, T0, "ko", ms=7, mfc="w")
        ax.plot(H_PAPER, TMAX, "rs", ms=7, mfc="w")
        ax.plot(H_PAPER, TCOLD, "b^", ms=7, mfc="w")
        ax.axvline(H_PAPER, color="w", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("h")
        ax.set_ylabel("T")
        ax.set_xlim(h_e[0], h_e[-1])
        ax.set_ylim(T_e[0], T_e[-1])
    fig.suptitle(f"Mappa di equilibrio  (L = {L}, J2 = {J2};  tratteggiata: h = {H_PAPER})")
    save_fig(fig, outdir, "fig3_mappa_h_T_nnn.png")


def fig4_curva(L, cfg, outdir, h):
    """Curva di equilibrio pi(T) a (h, J2) nei piani (M, S) e (M, Q)."""
    T_grid = T_grid_geom(cfg["n_T_curve"])
    mean, err = scan(L, [h], T_grid, cfg["n_burn_curve"], cfg["n_meas_curve"], cfg["n_rep_curve"])
    M, S, Q = mean[0, :, 0], mean[0, :, 1], mean[0, :, 2]
    dM, dS, dQ = err[0, :, 0], err[0, :, 1], err[0, :, 2]
    save_data(outdir, "dati_fig4_nnn.npz", T=T_grid, M=M, S=S, Q=Q, dM=dM, dS=dS, dQ=dQ, h=h, J2=J2)
    print(f"   errore statistico medio: M +- {dM.mean():.4f}  S +- {dS.mean():.4f}  Q +- {dQ.mean():.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    style = {"T_cold": ("b", "^"), "T_0": ("k", "o"), "T_max": ("r", "s")}
    for ax, Y, dY, name, floor in ((axes[0], S, dS, "S", np.sqrt(2 / np.pi) / L),
                                   (axes[1], Q, dQ, "Q", np.sqrt(np.pi / 2) / L)):
        ax.plot(M, Y, "-", color="0.6", lw=1, zorder=1)
        ax.errorbar(M, Y, xerr=dM, yerr=dY, fmt="none", ecolor="0.6", elinewidth=0.8, zorder=1)
        sc = ax.scatter(M, Y, c=T_grid, norm=LogNorm(T_grid[0], T_grid[-1]), cmap="plasma", s=25, zorder=2)
        for T, nm in zip(PAPER_T, PAPER_NAMES):
            i = int(np.argmin(np.abs(T_grid - T)))
            c, mk = style[nm]
            ax.plot(M[i], Y[i], mk, color=c, ms=10, mec=c, zorder=3)
            ax.annotate(nm, (M[i], Y[i]), textcoords="offset points", xytext=(8, 6), fontsize=10)
        ax.axhline(floor, color="k", ls=":", lw=1, label="livello di uno stato casuale")
        ax.set_xlabel(r"$\langle M\rangle_{\pi(T)}$")
        ax.set_ylabel(rf"$\langle {name}\rangle_{{\pi(T)}}$")
        ax.set_title(f"piano M-{name}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.colorbar(sc, ax=ax, label="T")
    fig.suptitle(f"Curva di equilibrio  h = {h},  J2 = {J2},  L = {L}")
    save_fig(fig, outdir, "fig4_curva_nnn.png")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=10, help="lato del reticolo (pari)")
    ap.add_argument("--j2", type=float, default=J2, help="J2 (stessa convenzione di J = -1); default -1")
    ap.add_argument("--h", type=float, default=H_PAPER, help="campo per fig1, fig4 e tabella (default 1.0025)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--only", choices=["1", "2", "3", "4", "table"], help="esegui solo una parte")
    ap.add_argument("--burn-pts", type=int, help="sweep di termalizzazione per la tabella")
    ap.add_argument("--meas-pts", type=int, help="sweep di misura per la tabella")
    ap.add_argument("--rep-pts", type=int, help="repliche per la tabella")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tcold", type=float, default=TCOLD)
    ap.add_argument("--init", choices=list(nnn.INIT) + ["auto"], default="auto",
                    help="condizione iniziale per le scansioni 2, 3, 4 ('auto' = stato fondamentale)")
    args = ap.parse_args()

    J2 = args.j2
    TCOLD = args.tcold
    PAPER_T = np.array([TCOLD, T0, TMAX])
    SCAN_INIT = args.init
    cfg = dict(QUICK if args.quick else FULL)
    for key, val in (("n_burn_pts", args.burn_pts), ("n_meas_pts", args.meas_pts), ("n_rep_pts", args.rep_pts)):
        if val is not None:
            cfg[key] = val

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(outdir, exist_ok=True)
    nnn.set_seed(args.seed)

    t0 = time.time()
    run = lambda tag: args.only in (None, tag)
    if run("table"):
        table_paper_points(args.L, cfg, args.h)
    if run("1"):
        print("fig1: rilassamento temporale ...")
        fig1_rilassamento(args.L, cfg, outdir, args.h)
    if run("2"):
        print("fig2: equilibrio vs T ...")
        fig2_equilibrio_T(args.L, cfg, outdir)
    if run("3"):
        print("fig3: mappa (h, T) ...")
        fig3_mappa(args.L, cfg, outdir)
    if run("4"):
        print("fig4: curva di equilibrio ...")
        fig4_curva(args.L, cfg, outdir, args.h)
    print(f"finito in {time.time() - t0:.1f} s  ->  cartella '{outdir}'")

    if not args.no_show and args.only != "table":
        plt.show()