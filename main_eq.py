"""
main_01_equilibrio.py   (versione 2)
Studio dell'equilibrio del modello di Ising 2D antiferromagnetico con campo
(Gal & Raz, PRL 124, 060602) su un reticolo LxL.
 
Figure (salvate in figures/):
  fig1_rilassamento   M, S, e in funzione di t da 3 condizioni iniziali, a T_cold, T_0, T_max
  fig2_equilibrio_T   M, S, e di equilibrio in funzione di T per diversi h: riga in alto T in scala
                      logaritmica, riga in basso T in scala lineare (6 pannelli)
  fig3_mappa_h_T      mappe di M e S nel piano (h, T) (asse T logaritmico)
  fig4_curva_MS       curva di equilibrio nel piano (<M>, <S>) a h = h_paper  (come Fig. 3 del paper)
Terminale:
  tabella con <M>, <S>, e ai tre punti T_cold, T_0, T_max a h = h_paper, con verifica
  di termalizzazione (3 condizioni iniziali diverse devono dare lo stesso risultato)
  e confronto con i valori letti dalla Fig. 3 del paper.
 
Uso:
    python main_01_equilibrio.py                 # tutto
    python main_01_equilibrio.py --only table    # solo la tabella (veloce)
    python main_01_equilibrio.py --only 4        # solo la curva M-S
    python main_01_equilibrio.py --quick         # run velocissimo per controllare che giri
    python main_01_equilibrio.py --tcold 0.0125  # altra T_cold (default 0.00125)
    python main_01_equilibrio.py --L 70 --only table --rep-pts 10 --burn-pts 50000 --meas-pts 20000
"""
import argparse
import os
import time
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
 
from src import ising
 
# ---- parametri del paper (Fig. 3) ----------------------------------------
H_PAPER = 1.0025
T0, TMAX = 0.125, 0.625
 
# T_cold usata nelle simulazioni: 0.00125 (a quella T il modello da' M ~ 0.966 come il punto blu del
# paper). TCOLD_CAPTION e' il valore scritto nella didascalia della Fig. 3 (la tabella lo riporta in una
# riga a parte). Si puo' cambiare da terminale con --tcold.
TCOLD_CAPTION = 0.0125
TCOLD = 0.00125
PAPER_T = np.array([TCOLD, T0, TMAX])
PAPER_NAMES = ["T_cold", "T_0", "T_max"]
 
# Valori letti A OCCHIO dalla Fig. 3 del paper (L = 70): (M, S).
# Incertezza di lettura ~ +-0.005 su M e ~ +-0.0005 su S.
PAPER_REF = {
    "T_cold": (0.966, 0.0193),
    "T_0":    (0.555, 0.0233),
    "T_max":  (0.495, 0.0198),
}
 
# ---- configurazioni ------------------------------------------------------
FULL = dict(
    n_relax=10000, n_rep_relax=40,                               # fig1
    H_list=[0.0, 0.25, 0.5, 0.99, H_PAPER, 1.5, 2.0],            # fig2
    n_T=35, n_burn=3000, n_meas=2000, n_rep=30,
    n_T_lin=19, T_lin_lo=0.3, T_lin_hi=1.2,                      # fig2: punti extra, equispaziati, attorno a T_c(h)
    n_T_curve=50, n_burn_curve=5000, n_meas_curve=5000, n_rep_curve=25,   # fig4
    h_map=np.linspace(0.0, 2.0, 21),                             # fig3
    n_T_map=23, n_burn_map=1500, n_meas_map=1000, n_rep_map=10,
    n_burn_pts=20000, n_meas_pts=20000, n_rep_pts=40,            # tabella
)
QUICK = dict(
    n_relax=300, n_rep_relax=6,
    H_list=[0.0, 0.25, 0.5, 0.99, H_PAPER, 1.5, 2.0],
    n_T=8, n_burn=60, n_meas=60, n_rep=4,
    n_T_lin=4, T_lin_lo=0.3, T_lin_hi=1.2,
    n_T_curve=10, n_burn_curve=60, n_meas_curve=60, n_rep_curve=4,
    h_map=np.linspace(0.0, 2.0, 7),
    n_T_map=7, n_burn_map=40, n_meas_map=40, n_rep_map=3,
    n_burn_pts=150, n_meas_pts=150, n_rep_pts=4,
)
 
LABELS = [r"$\langle M\rangle$ ",
          r"$\langle S\rangle$",
          r"$e = E/N$"]
 
 
# --------------------------------------------------------------------------
# Utilita'
# --------------------------------------------------------------------------
def sem(x, axis=0):
    """errore standard della media sulle repliche"""
    return x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])
 
 
def T_grid_geom(n, tmin=None, tmax=1.2):
    """Griglia di temperature geometrica (fitta a basse T) che CONTIENE sempre
    esattamente T_cold, T_0, T_max (i punti geometrici troppo vicini vengono tolti)."""
    tmin = TCOLD if tmin is None else tmin
    T = np.geomspace(tmin, tmax, n)
    far = np.all(np.abs(np.log(T[:, None] / PAPER_T[None, :])) > 0.04, axis=1)
    return np.unique(np.concatenate([T[far], PAPER_T]))
 
 
def edges_lin(x):
    """Bordi delle celle per pcolormesh, a partire dai centri (punti equispaziati o no)."""
    x = np.asarray(x, float)
    mid = (x[1:] + x[:-1]) / 2
    return np.concatenate([[x[0] - (x[1] - x[0]) / 2], mid, [x[-1] + (x[-1] - x[-2]) / 2]])
 
 
def edges_log(x):
    """Come edges_lin ma geometrici: servono per un asse T logaritmico."""
    return np.exp(edges_lin(np.log(np.asarray(x, float))))
 
 
def save_fig(fig, outdir, name):
    """Salva la figura; se il file non e' scrivibile (tipico su Windows: file aperto
    nell'anteprima di VS Code / in un visualizzatore, o cartella sincronizzata da OneDrive)
    la salva con un altro nome invece di far crollare tutto il programma."""
    path = os.path.join(outdir, name)
    try:
        fig.savefig(path, dpi=150)
        print(f"   salvata: {path}")
    except OSError as err:
        base, ext = os.path.splitext(name)
        alt = os.path.join(outdir, f"{base}_{time.strftime('%H%M%S')}{ext}")
        print(f"   ATTENZIONE: non riesco a scrivere {path}  ({err}).\n"
              f"   Probabilmente il file e' aperto in un altro programma (chiudi l'anteprima in VS Code).\n"
              f"   Salvo come: {alt}")
        fig.savefig(alt, dpi=150)
 
 
def save_data(outdir, name, **arrays):
    """Salva i risultati numerici (.npz) PRIMA di fare il grafico: se il grafico fallisce
    non si perdono i minuti di simulazione. Ricaricabili con np.load(file)."""
    path = os.path.join(outdir, name)
    try:
        np.savez(path, **arrays)
    except OSError as err:
        print(f"   (dati non salvati: {err})")
 
 
SCAN_INIT = "auto"            # condizione iniziale per le scansioni (modificabile con --init)
 
 
def init_for(h, init):
    """'auto': per h < h_c lo stato di equilibrio a bassa T e' il Neel, ma partendo da uno
    stato casuale a T bassa il sistema resta intrappolato in domini/strisce di Neel (stati
    metastabili con S ~ 0) e la media sulle repliche e' sballata e rumorosa.
    Per h >= h_c lo stato casuale equilibra senza problemi (verificato)."""
    if init != "auto":
        return init
    return "neel" if h < 1.0 - 1e-9 else "random"
 
 
def scan(L, h_values, T_values, n_burn, n_meas, n_rep, init=None):
    """Equilibrio su una griglia (h, T). Ritorna media ed errore, shape (nh, nT, 3)
    con le tre colonne [M, S, e]."""
    init = SCAN_INIT if init is None else init
    mean = np.empty((len(h_values), len(T_values), 3))
    err = np.empty_like(mean)
    t_start = time.time()
    for a, h in enumerate(h_values):
        init_h = init_for(h, init)
        for b, T in enumerate(T_values):
            out = ising.equilibrium_point(L, T, h, n_burn, n_meas, n_rep, ising.INIT[init_h])
            mean[a, b] = out.mean(axis=0)
            err[a, b] = sem(out)
        print(f"   h = {h:6.4f} fatto  (start {init_h}; {time.time() - t_start:5.1f} s)", flush=True)
    return mean, err
 
 
def mark_paper_T(ax, orient="v", labels=False):
    """Linee di riferimento a T_cold, T_0, T_max."""
    for T, c, nm in zip(PAPER_T, ("b", "k", "r"), PAPER_NAMES):
        kw = dict(color=c, ls=":", lw=1, label=(nm if labels else None))
        (ax.axvline if orient == "v" else ax.axhline)(T, **kw)
 
 
# --------------------------------------------------------------------------
# Tabella a terminale
# --------------------------------------------------------------------------
def table_paper_points(L, cfg):
    """<M>, <S>, e ai tre punti del paper, con test di termalizzazione."""
    nb, nm, nr = cfg["n_burn_pts"], cfg["n_meas_pts"], cfg["n_rep_pts"]
    print()
    print("=" * 78)
    print(f"PUNTI DI EQUILIBRIO a h = {H_PAPER}   (L = {L}; {nr} repliche; "
          f"{nb} MC steps di termalizzazione + {nm} di misura)")
    print("=" * 78)
    inits = ["random", "up", "neel"]
    rows = list(zip(PAPER_T, PAPER_NAMES))
    if not np.isclose(TCOLD, TCOLD_CAPTION):       # riga extra col valore scritto nella didascalia
        rows.insert(0, (TCOLD_CAPTION, "T_cold(didascalia)"))
    for T, name in rows:
        res = {}
        for init in inits:
            out = ising.equilibrium_point(L, T, H_PAPER, nb, nm, nr, ising.INIT[init])
            res[init] = (out.mean(axis=0), sem(out))
        print(f"\n{name} = {T}")
        for init in inits:
            m, e = res[init]
            tag = "  <- come nel paper" if init == "random" else ""
            print(f"   start {init:6s}:  <M> = {m[0]:.4f} +- {e[0]:.4f}   "
                  f"<S> = {m[1]:.4f} +- {e[1]:.4f}   e = {m[2]:.4f}{tag}")
        # verdetto: le tre condizioni iniziali devono essere statisticamente compatibili
        ok = True
        for init in inits[1:]:
            for k in (0, 1):
                d = abs(res["random"][0][k] - res[init][0][k])
                tol = 3 * np.hypot(res["random"][1][k], res[init][1][k]) + 0.005
                ok &= d < tol
        print("   termalizzazione: " + ("OK (le 3 partenze concordano)" if ok else
              "ATTENZIONE - le partenze NON concordano: aumenta --burn-pts"))
        if name in PAPER_REF:
            pm, ps = PAPER_REF[name]
            print(f"   paper (L=70, letto dal grafico):  <M> ~ {pm:.3f}   <S> ~ {ps:.4f}")
            if name == "T_cold" and np.isclose(T, 0.0125):
                print("   nota: a T = 0.0125 il modello da' M ~ 0.61 (indipendente da L), non 0.966. "
                      "A T = 0.00125 si ha M ~ 0.966 ma, a L = 70, S ~ 0.003 e non 0.0193: "
                      "discrepanza con il paper NON risolta (vedi --tcold)")
        else:
            print("   paper: nessun punto del grafico a questa T")
    print(f"\n(livello di S per uno stato disordinato a L={L}: ~ 0.8/L = {np.sqrt(2/np.pi)/L:.4f})")
    print("=" * 78)
 
 
# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def fig1_rilassamento(L, cfg, outdir):
    """Rilassamento temporale da condizioni iniziali diverse, a h = H_PAPER."""
    inits = [("random", "casuale"), ("up", "tutti up"), ("neel", "Neel")]
    temps = [(TCOLD, r"$T_{cold}$"), (T0, r"$T_0$"), (TMAX, r"$T_{max}$")]
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
    t = np.arange(cfg["n_relax"] + 1)
    for col, (T, Tname) in enumerate(temps):
        for init, iname in inits:
            M, S, E = ising.run_relaxation(L, T, H_PAPER, cfg["n_relax"],
                                           cfg["n_rep_relax"], ising.INIT[init])
            for row, data in enumerate((M, S, E)):
                m, e = data.mean(axis=0), sem(data)
                axes[row, col].plot(t[1:], m[1:], label=iname)
                axes[row, col].fill_between(t[1:], (m - e)[1:], (m + e)[1:], alpha=0.25)
        axes[0, col].set_title(f"{Tname} = {T},  h = {H_PAPER},  L = {L}")
        axes[1, col].axhline(np.sqrt(2 / np.pi) / L, color="k", ls=":", lw=1,
                             label=r"stato casuale $\approx 0.8/L$")
        axes[2, col].set_xlabel("t  (MC steps)")
    for row in range(3):
        axes[row, 0].set_ylabel(LABELS[row])
        for col in range(3):
            axes[row, col].set_xscale("log")
            axes[row, col].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, outdir, "fig1_rilassamento.png")
 
 
def fig2_equilibrio_T(L, cfg, outdir):
    """<M>, <S>, <e> di equilibrio vs T per diversi h: riga in alto T logaritmica, riga in basso T lineare.
    La griglia di T e' quella geometrica (fitta a bassa T) piu' alcuni punti equispaziati tra T_lin_lo e
    T_lin_hi, per risolvere la transizione ordine-disordine a T_c(h) (h < 1), dove S cambia molto in poco T."""
    T_grid = np.unique(np.concatenate([T_grid_geom(cfg["n_T"]),
                                       np.linspace(cfg["T_lin_lo"], cfg["T_lin_hi"], cfg["n_T_lin"])]))
    mean, err = scan(L, cfg["H_list"], T_grid, cfg["n_burn"], cfg["n_meas"], cfg["n_rep"])
    save_data(outdir, "dati_fig2.npz", h=cfg["H_list"], T=T_grid, mean=mean, err=err)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row in range(2):
        for a, h in enumerate(cfg["H_list"]):
            for k in range(3):
                axes[row, k].errorbar(T_grid, mean[a, :, k], yerr=err[a, :, k],
                                      marker="o", ms=3, capsize=2, label=f"h = {h}")
    for row, scale in enumerate(("log", "linear")):
        for k in range(3):
            ax = axes[row, k]
            ax.set_xscale(scale)
            ax.set_xlabel("T" + ("  (scala log)" if scale == "log" else "  (scala lineare)"))
            ax.set_ylabel(LABELS[k])
            ax.grid(alpha=0.3, which="both")
            mark_paper_T(ax, "v", labels=(row == 0 and k == 0))
        if scale == "linear":
            for k in range(3):
                axes[row, k].set_xlim(0, T_grid[-1] * 1.02)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Equilibrio vs T (L = {L})")
    fig.tight_layout()
    save_fig(fig, outdir, "fig2_equilibrio_T.png")


def fig3_mappa(L, cfg, outdir):
    """Mappe di <M> e <S> nel piano (h, T). T in scala log, celle che coprono tutto il piano."""
    h_map = np.unique(np.concatenate([cfg["h_map"], [H_PAPER]]))
    T_map = T_grid_geom(cfg["n_T_map"], tmax=1.0)
    mean, _ = scan(L, h_map, T_map, cfg["n_burn_map"], cfg["n_meas_map"], cfg["n_rep_map"])
    save_data(outdir, "dati_fig3.npz", h=h_map, T=T_map, mean=mean)
    h_e = edges_lin(h_map)
    h_e[0] = max(h_e[0], 0.0)
    T_e = edges_log(T_map)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, k, cmap in ((axes[0], 0, "viridis"), (axes[1], 1, "magma")):
        pc = ax.pcolormesh(h_e, T_e, mean[:, :, k].T, shading="flat", cmap=cmap)
        fig.colorbar(pc, ax=ax, label=LABELS[k])
        ax.plot(H_PAPER, T0, "ko", ms=7, mfc="w", label=r"$T_0$")
        ax.plot(H_PAPER, TMAX, "rs", ms=7, mfc="w", label=r"$T_{max}$")
        ax.plot(H_PAPER, TCOLD, "b^", ms=7, mfc="w", label=r"$T_{cold}$")
        ax.axvline(1.0025, color="w", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("h")
        ax.set_ylabel("T")
        ax.set_xlim(h_e[0], h_e[-1])
        ax.set_ylim(T_e[0], T_e[-1])
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Mappa di equilibrio  (L = {L};  tratteggiata: $h = 1.0025$)")
    fig.tight_layout()
    save_fig(fig, outdir, "fig3_mappa_h_T.png")
 
 
def fig4_curva_MS(L, cfg, outdir):
    """Curva di equilibrio nel piano (<M>, <S>) a h = H_PAPER: ogni punto e' pi(T)."""
    T_grid = T_grid_geom(cfg["n_T_curve"])
    mean, err = scan(L, [H_PAPER], T_grid, cfg["n_burn_curve"], cfg["n_meas_curve"],
                     cfg["n_rep_curve"])
    M, S = mean[0, :, 0], mean[0, :, 1]
    dM, dS = err[0, :, 0], err[0, :, 1]
    save_data(outdir, "dati_fig4.npz", T=T_grid, M=M, S=S, dM=dM, dS=dS)
    print(f"   errore statistico medio sui punti della curva:  <M> +- {dM.mean():.4f}   "
          f"<S> +- {dS.mean():.4f}   (intervallo coperto: M {M.min():.3f}-{M.max():.3f}, "
          f"S {S.min():.3f}-{S.max():.3f})")
    print("   Se le oscillazioni della curva sono paragonabili a questi errori, sono solo rumore: "
          "aumenta --curve-rep / --curve-meas.")
 
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(M, S, "-", color="0.6", lw=1, zorder=1)
    ax.errorbar(M, S, xerr=dM, yerr=dS, fmt="none", ecolor="0.6", elinewidth=0.8, zorder=1)
    sc = ax.scatter(M, S, c=T_grid, norm=LogNorm(T_grid[0], T_grid[-1]),
                    cmap="plasma", s=25, zorder=2)
    fig.colorbar(sc, ax=ax, label="T")
 
    # i tre punti del paper, con gli stessi simboli della Fig. 3
    style = {"T_cold": ("b", "^"), "T_0": ("k", "o"), "T_max": ("r", "s")}
    for T, name in zip(PAPER_T, PAPER_NAMES):
        i = int(np.argmin(np.abs(T_grid - T)))
        c, mk = style[name]
        ax.plot(M[i], S[i], mk, color=c, ms=10, mec=c, zorder=3)
        ax.annotate(name.replace("_", "$_{") + "}$" if "_" in name else name,
                    (M[i], S[i]), textcoords="offset points", xytext=(8, 6), fontsize=10)
    ax.axhline(np.sqrt(2 / np.pi) / L, color="k", ls=":", lw=1,
               label=r"livello di uno stato disordinato $\approx 0.8/L$")
    ax.set_xlabel(r"$\langle M\rangle_{\pi(T)}$")
    ax.set_ylabel(r"$\langle S\rangle_{\pi(T)}$")
    ax.set_title(f"Curva di equilibrio $\\pi(T)$ proiettata, h = {H_PAPER},  L = {L}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_fig(fig, outdir, "fig4_curva_MS.png")
 
 
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=10, help="lato del reticolo (pari)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--only", choices=["1", "2", "3", "4", "table"],
                    help="esegui solo una parte")
    ap.add_argument("--burn-pts", type=int, help="MC steps di termalizzazione per la tabella")
    ap.add_argument("--meas-pts", type=int, help="MC steps di misura per la tabella")
    ap.add_argument("--rep-pts", type=int, help="numero di repliche per la tabella")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tcold", type=float, default=TCOLD,
                    help="T_cold usata (default 0.00125; la didascalia del paper riporta 0.0125)")
    ap.add_argument("--curve-rep", type=int, help="repliche per i punti della curva M-S (fig. 4)")
    ap.add_argument("--curve-meas", type=int, help="MC steps di misura per i punti della curva M-S")
    ap.add_argument("--init", choices=list(ising.INIT) + ["auto"], default="auto",
                    help="condizione iniziale per le scansioni delle fig. 2, 3, 4. 'auto' (default): "
                         "Neel per h < 1, casuale per h >= 1 (come nel paper a h = 1.0025)")
    args = ap.parse_args()
 
    TCOLD = args.tcold
    PAPER_T = np.array([TCOLD, T0, TMAX])
    SCAN_INIT = args.init
 
    cfg = dict(QUICK if args.quick else FULL)
    for key, val in (("n_burn_pts", args.burn_pts), ("n_meas_pts", args.meas_pts),
                     ("n_rep_pts", args.rep_pts), ("n_rep_curve", args.curve_rep),
                     ("n_meas_curve", args.curve_meas)):
        if val is not None:
            cfg[key] = val
 
    # cartella 'figures' accanto a questo file (non dipende da dove lanci il programma)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(outdir, exist_ok=True)
    ising.set_seed(args.seed)
 
    t0 = time.time()
    run = lambda tag: args.only in (None, tag)
    if run("table"):
        table_paper_points(args.L, cfg)
    if run("1"):
        print("fig1: rilassamento temporale ...")
        fig1_rilassamento(args.L, cfg, outdir)
    if run("2"):
        print("fig2: equilibrio vs T ...")
        fig2_equilibrio_T(args.L, cfg, outdir)
    if run("3"):
        print("fig3: mappa (h, T) ...")
        fig3_mappa(args.L, cfg, outdir)
    if run("4"):
        print("fig4: curva di equilibrio nel piano M-S ...")
        fig4_curva_MS(args.L, cfg, outdir)
    print(f"finito in {time.time() - t0:.1f} s  ->  cartella '{outdir}'")
 
    if not args.no_show and args.only != "table":
        plt.show()