"""
main_04_traiettorie.py
Sanity check: traiettorie FUORI equilibrio da un punto del locus a un altro.

Protocollo (un "quench" di temperatura del bagno), a h = h_paper:
  1. si prepara un insieme di n_rep realizzazioni all'equilibrio a T_i
     (come nel supplemental del paper: stato casuale + dinamica di Glauber a T_i);
  2. al tempo t = 0 la temperatura del bagno viene cambiata di colpo in T_f;
  3. si lascia evolvere (Glauber a T_f) e si calcolano, a ogni tempo, le medie sull'insieme
     <M>(t) e <S>(t).
All'inizio il sistema e' nel punto del locus pi(T_i); alla fine (t grande) in pi(T_f);
nel mezzo <M>(t), <S>(t) escono dal locus perche' M e S rilassano con tempi diversi.

Il tempo e' misurato in sweep (1 sweep = L*L tentativi di flip) e registrato con
risoluzione fine (anche sotto lo sweep); sull'asse in alto delle figure c'e' lo stesso
tempo in singoli tentativi di flip (= sweep x L*L), nel caso il "MC step" del paper
sia un singolo tentativo.

Produce (in figures/):
  fig7_traiettorie_MS.png    piano M-S: locus di equilibrio (grigio) + traiettoria (colore = tempo)
  fig7b_traiettorie_tempo.png <M>(t) e <S>(t) con i valori di equilibrio di partenza e arrivo
  dati_fig7.npz              risultati numerici (salvati PRIMA dei grafici)
e stampa a terminale una tabella con <M>, <S> (e relativi errori) al punto di partenza e di
arrivo della traiettoria, accanto ai valori del locus di equilibrio alle stesse temperature.
Nelle figure le fasce attorno alle curve (azzurra) e alle righe tratteggiate (grigia/rossa)
sono +-1 errore standard della media: partenza/arrivo e locus sono compatibili se la loro
differenza e' minore di circa 2 volte l'errore combinato (radice della somma dei quadrati),
non se la curva cade dentro la sola fascia del locus.

Uso:
    python main_04_traiettorie.py                       # L = 10, coppie T0->Tmax, T0->Tcold, Tmax->T0
    python main_04_traiettorie.py --pairs T0:max T0:0.3 # altre coppie (nomi: cold, T0, max, oppure numeri)
    python main_04_traiettorie.py --L 70 --rep 5000     # come nel paper (lento: vedi sotto)
    python main_04_traiettorie.py --locus-file figures/dati_fig4.npz   # riusa il locus gia' calcolato
    python main_04_traiettorie.py --quick               # run velocissimo per controllare che giri
Stima dei tempi con numba: L = 10 -> pochi secondi; L = 70 con 5000 realizzazioni -> alcuni minuti.
"""
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from src import ising

H_PAPER = 1.0025
T0, TMAX = 0.125, 0.625
TCOLD = 0.0125                       # modificabile con --tcold

FULL = dict(n_rep=4000, n_burn=300, t_max=200, n_rec=70,
            locus_nT=16, locus_rep=30, locus_burn=500, locus_meas=1000)
QUICK = dict(n_rep=60, n_burn=40, t_max=20, n_rec=25,
             locus_nT=8, locus_rep=4, locus_burn=40, locus_meas=40)


# --------------------------------------------------------------------------
# Utilita'
# --------------------------------------------------------------------------
def sem(x, axis=0):
    return x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])


def parse_T(tok):
    names = {"cold": TCOLD, "t_cold": TCOLD, "t0": T0, "t_0": T0, "max": TMAX, "t_max": TMAX}
    return names.get(tok.lower(), None) if tok.lower() in names else float(tok)


def T_name(T):
    for nm, val in (("T_cold", TCOLD), ("T_0", T0), ("T_max", TMAX)):
        if np.isclose(T, val):
            return nm
    return f"T={T:g}"


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


# --------------------------------------------------------------------------
# Locus di equilibrio a h = H_PAPER
# --------------------------------------------------------------------------
def compute_locus(L, T_values, cfg):
    """Punti di equilibrio (M, S) a h = H_PAPER sulle temperature date (partenza casuale)."""
    n = len(T_values)
    M = np.empty(n); S = np.empty(n); dM = np.empty(n); dS = np.empty(n)
    t_start = time.time()
    for k, T in enumerate(T_values):
        out = ising.equilibrium_point(L, T, H_PAPER, cfg["locus_burn"], cfg["locus_meas"],
                                      cfg["locus_rep"], ising.INIT["random"])
        M[k], S[k] = out[:, 0].mean(), out[:, 1].mean()
        dM[k], dS[k] = sem(out[:, 0]), sem(out[:, 1])
    print(f"   locus calcolato su {n} temperature ({time.time() - t_start:.1f} s)")
    return dict(T=np.asarray(T_values), M=M, S=S, dM=dM, dS=dS)


def locus_grid(temps, n):
    """Temperature del locus: geometrica tra min e max, contenente esattamente gli estremi."""
    lo, hi = min(temps), max(temps)
    T = np.geomspace(lo, hi, n)
    far = np.all(np.abs(np.log(T[:, None] / np.array(temps)[None, :])) > 0.04, axis=1)
    return np.unique(np.concatenate([T[far], np.array(temps)]))


def locus_at(locus, T):
    """(M, S, dM, dS) del locus a temperatura T (interpolazione in log T se T non e' in griglia)."""
    x = np.log(locus["T"])
    return tuple(float(np.interp(np.log(T), x, locus[k])) for k in ("M", "S", "dM", "dS"))


# --------------------------------------------------------------------------
# Traiettorie
# --------------------------------------------------------------------------
def run_pairs(L, pairs, cfg):
    """Per ogni coppia (T_i, T_f) calcola <M>(t), <S>(t), <e>(t) con errori."""
    N = L * L
    rec = np.unique(np.concatenate([[0], np.round(np.geomspace(1, cfg["t_max"] * N, cfg["n_rec"]))
                                    .astype(np.int64)]))
    ensembles = {}
    results = []
    for Ti, Tf in pairs:
        t_start = time.time()
        if Ti not in ensembles:
            ensembles[Ti] = ising.prepare_ensemble(L, Ti, H_PAPER, cfg["n_burn"], cfg["n_rep"],
                                                   ising.INIT["random"])
        M, S, E = ising.quench_from_ensemble(ensembles[Ti], Tf, H_PAPER, rec)
        results.append(dict(Ti=Ti, Tf=Tf, M=M.mean(0), S=S.mean(0), E=E.mean(0),
                            dM=sem(M), dS=sem(S), dE=sem(E)))
        print(f"   {T_name(Ti)} -> {T_name(Tf)} fatto ({time.time() - t_start:5.1f} s)", flush=True)
    return rec / N, results


def locus_endpoints(res, locus):
    """Valori (con errori) del locus di equilibrio a T_i e a T_f, per il confronto con la traiettoria."""
    Mi, Si, dMi, dSi = locus_at(locus, res["Ti"])
    Mf, Sf, dMf, dSf = locus_at(locus, res["Tf"])
    return dict(Mi=Mi, Si=Si, dMi=dMi, dSi=dSi, Mf=Mf, Sf=Sf, dMf=dMf, dSf=dSf)


def print_table(L, ts, results, infos):
    print("\n" + "=" * 100)
    print(f"TRAIETTORIE a h = {H_PAPER}  (L = {L})   medie sull'insieme di realizzazioni, +- errore standard")
    print("=" * 100)
    for res, inf in zip(results, infos):
        a, b = T_name(res["Ti"]), T_name(res["Tf"])
        print(f"\n{a} -> {b}")
        print(f"   traiettoria a t = 0 sweep   : <M> = {res['M'][0]:.4f} +- {res['dM'][0]:.4f}   "
              f"<S> = {res['S'][0]:.4f} +- {res['dS'][0]:.4f}")
        print(f"   locus pi({a:<6s})            : <M> = {inf['Mi']:.4f} +- {inf['dMi']:.4f}   "
              f"<S> = {inf['Si']:.4f} +- {inf['dSi']:.4f}")
        print(f"   traiettoria a t = {ts[-1]:<7g}sweep: <M> = {res['M'][-1]:.4f} +- {res['dM'][-1]:.4f}   "
              f"<S> = {res['S'][-1]:.4f} +- {res['dS'][-1]:.4f}")
        print(f"   locus pi({b:<6s})            : <M> = {inf['Mf']:.4f} +- {inf['dMf']:.4f}   "
              f"<S> = {inf['Sf']:.4f} +- {inf['dSf']:.4f}")
    print("\n(Confronta le coppie di righe: sono compatibili se la differenza e' minore di circa 2 volte")
    print(" la radice della somma dei quadrati dei due errori.)")
    print("=" * 100)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def fig_plane(L, ts, results, infos, locus, outdir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.4), squeeze=False, layout="constrained")
    norm = LogNorm(ts[1], ts[-1])
    lc = None
    for ax, res, inf in zip(axes[0], results, infos):
        Ti, Tf = res["Ti"], res["Tf"]
        lo, hi = min(Ti, Tf), max(Ti, Tf)
        sel = (locus["T"] >= lo * 0.999) & (locus["T"] <= hi * 1.001)
        ax.plot(locus["M"][sel], locus["S"][sel], "-o", color="0.55", ms=4, lw=1.5, zorder=1,
                label="locus di equilibrio")
        ax.errorbar(res["M"][::3], res["S"][::3], xerr=res["dM"][::3], yerr=res["dS"][::3], fmt="none",
                    ecolor="0.8", elinewidth=0.8, zorder=1)
        t_plot = np.where(ts > 0, ts, ts[1] / 2)               # t = 0 -> valore fittizio solo per il colore
        pts = np.column_stack([res["M"], res["S"]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", norm=norm, linewidth=2.6, zorder=3)
        lc.set_array(t_plot[1:])
        ax.add_collection(lc)
        for tk in (0.1, 1, 10, 100):                           # tempi di riferimento
            if ts[1] < tk < ts[-1]:
                i = int(np.argmin(np.abs(np.log(np.maximum(ts, 1e-12)) - np.log(tk))))
                ax.plot(res["M"][i], res["S"][i], "o", color="w", mec="k", ms=5, zorder=4)
                ax.annotate(f"t={tk:g}", (res["M"][i], res["S"][i]), textcoords="offset points",
                            xytext=(5, 5), fontsize=8)
        ax.plot(res["M"][0], res["S"][0], "o", color="k", ms=11, zorder=5, label=f"partenza: pi({T_name(Ti)})")
        ax.plot(inf["Mf"], inf["Sf"], "*", color="r", ms=15, mec="k", zorder=5,
                label=f"arrivo atteso: pi({T_name(Tf)})")
        ax.autoscale()
        ax.margins(0.08)
        ax.set_title(f"{T_name(Ti)} -> {T_name(Tf)}")
        ax.set_xlabel(r"$\langle M\rangle$")
        ax.set_ylabel(r"$\langle S\rangle$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.colorbar(lc, ax=axes[0].tolist(), label="t  (MC sweep, 1 sweep = L*L tentativi)", shrink=0.85)
    fig.suptitle(f"Traiettorie fuori equilibrio nel piano M-S  (h = {H_PAPER},  L = {L})")
    save_fig(fig, outdir, "fig7_traiettorie_MS.png")


def fig_time(L, ts, results, infos, outdir):
    n = len(results)
    N = L * L
    fig, axes = plt.subplots(2, n, figsize=(5.6 * n, 7), squeeze=False, sharex="col", layout="constrained")
    for c, (res, inf) in enumerate(zip(results, infos)):
        for r, (key, ekey, ki, kf, lab) in enumerate((("M", "dM", "Mi", "Mf", r"$\langle M\rangle(t)$"),
                                                       ("S", "dS", "Si", "Sf", r"$\langle S\rangle(t)$"))):
            ax = axes[r, c]
            y, e = res[key][1:], res[ekey][1:]
            ax.plot(ts[1:], y, color="C0", lw=2)
            ax.fill_between(ts[1:], y - e, y + e, color="C0", alpha=0.25)      # +-1 errore standard della media
            # righe tratteggiate = valore di equilibrio dal locus, con la sua incertezza (fascia grigia/rossa)
            for kk, col, T in ((ki, "k", res["Ti"]), (kf, "r", res["Tf"])):
                err_l = inf["d" + kk]
                ax.axhline(inf[kk], color=col, ls="--", lw=1, label=f"equilibrio {T_name(T)} (locus)")
                ax.axhspan(inf[kk] - err_l, inf[kk] + err_l, color=col, alpha=0.10, lw=0)
            ax.set_xscale("log")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3, which="both")
            if r == 0:
                ax.set_title(f"{T_name(res['Ti'])} -> {T_name(res['Tf'])}")
                ax.legend(fontsize=8)
                sec = ax.secondary_xaxis("top", functions=(lambda x: x * N, lambda x: x / N))
                sec.set_xlabel("t  (singoli tentativi di flip)", fontsize=8)
            if r == 1:
                ax.set_xlabel("t  (MC sweep)")
    fig.suptitle(f"Rilassamento di M e S dopo il cambio di temperatura  (h = {H_PAPER},  L = {L};  "
                 "fasce = +-1 errore standard)")
    save_fig(fig, outdir, "fig7b_traiettorie_tempo.png")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=10, help="LATO del reticolo (pari); N = L*L spin")
    ap.add_argument("--pairs", nargs="+", default=["T0:max", "T0:cold", "max:T0"],
                    help="coppie Ti:Tf (nomi cold, T0, max oppure numeri)")
    ap.add_argument("--rep", type=int, help="numero di realizzazioni")
    ap.add_argument("--tmax", type=float, help="tempo finale in sweep")
    ap.add_argument("--burn", type=int, help="sweep di preparazione a T_i")
    ap.add_argument("--locus-rep", type=int, help="repliche per i punti del locus")
    ap.add_argument("--locus-file", help="npz con T, M, S (e dM, dS) del locus, es. figures/dati_fig4.npz")
    ap.add_argument("--tcold", type=float, default=TCOLD)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    TCOLD = args.tcold
    cfg = dict(QUICK if args.quick else FULL)
    for key, val in (("n_rep", args.rep), ("t_max", args.tmax), ("n_burn", args.burn),
                     ("locus_rep", args.locus_rep)):
        if val is not None:
            cfg[key] = val
    pairs = []
    for tok in args.pairs:
        a, b = tok.split(":")
        pairs.append((parse_T(a), parse_T(b)))

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(outdir, exist_ok=True)
    ising.set_seed(args.seed)
    L = args.L

    t0 = time.time()
    # 1) locus di equilibrio
    if args.locus_file:
        d = np.load(args.locus_file)
        zeros = np.zeros_like(d["T"])
        locus = dict(T=d["T"], M=d["M"], S=d["S"], dM=d["dM"] if "dM" in d else zeros,
                     dS=d["dS"] if "dS" in d else zeros)
        print(f"   locus letto da {args.locus_file}  (verifica che sia stato calcolato con lo stesso L!)")
    else:
        print("locus di equilibrio ...")
        temps = sorted({T for p in pairs for T in p})
        locus = compute_locus(L, locus_grid(temps, cfg["locus_nT"]), cfg)

    # 2) traiettorie
    print("traiettorie ...")
    ts, results = run_pairs(L, pairs, cfg)
    infos = [locus_endpoints(r, locus) for r in results]

    try:
        np.savez(os.path.join(outdir, "dati_fig7.npz"), t=ts, locus_T=locus["T"], locus_M=locus["M"],
                 locus_S=locus["S"], pairs=np.array(pairs),
                 **{f"{k}_{i}": r[k] for i, r in enumerate(results) for k in ("M", "S", "E", "dM", "dS", "dE")})
    except OSError as e:
        print(f"   (dati non salvati: {e})")

    print_table(L, ts, results, infos)
    fig_plane(L, ts, results, infos, locus, outdir)
    fig_time(L, ts, results, infos, outdir)
    print(f"finito in {time.time() - t0:.1f} s")

    if not args.no_show:
        plt.show()