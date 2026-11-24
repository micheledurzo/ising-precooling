"""
main_05_protocolli.py
Protocolli di riscaldamento "oven" e "precooling" (Gal & Raz, PRL 124, 060602) a h = h_paper.

UNITA' DI TEMPO: un singolo MC step = UN tentativo di flip (quindi 1 sweep = L*L MC step).
Tutti i tempi, i parametri da riga di comando e gli assi dei grafici sono in MC step.

Protocolli (tutti partono da un insieme di realizzazioni all'equilibrio):
  OVEN        pi(T0) --> bagno a T_max da t = 0                         (riscaldamento diretto)
  PRECOOLING  pi(T0) --> bagno a T_cold per tau MC step, poi T_max      (il protocollo del paper)
  "T_cold -> T_max"  pi(T_cold) --> T_max                               (traiettoria grigia: oven partito
                                                                         da T_cold, riferimento)
  "resta freddo"     pi(T0) --> T_cold per sempre                       (traiettoria punteggiata: la
                                                                         prima fase del precooling, mai chiusa)
La preparazione di pi(T) e' come nel paper: stato casuale + dinamica di Glauber a T per n_burn MC step.

Misura di distanza dall'equilibrio finale (inset di Fig. 3 del paper): differenza di energia per spin
tra la media d'insieme <e>(t) e il valore di equilibrio a T_max,  Delta e(t) = <e>(t) - e_eq(T_max).
Nei grafici si usa |Delta e| con una fascia +-1 errore standard (sull'insieme, combinato con l'errore di e_eq);
sotto il "rumore statistico" la curva non e' piu' distinguibile da zero.

SCELTA DI tau. Si puo' dare a mano (--tau 1850 e' il valore del paper per L = 70), oppure (default) si fa una
SCANSIONE di tau: per ogni tau si misura la media di Delta e (e di Delta S) in una finestra di tempo di
riscaldamento [w1, w2] MC step dopo il cambio a T_max (default [2N, 6N]) e si sceglie il tau con |<Delta e>|
minimo. Idea del paper: il tau giusto annulla l'ampiezza del modo lento, quindi dopo il tempo "veloce" il
sistema e' gia' in pi(T_max) e la distanza crolla molto prima che nell'oven.

NOTA su T_cold: il default e' 0.00125. Con questo valore l'effetto del precooling e' netto e il tau ottimo
e' ~0.4 sweep ~ 1850 MC step a L = 70, come nel paper. Con T_cold = 0.0125 (--tcold 0.0125) l'effetto e'
debole (finestra quasi piatta in tau, l'oven partito da T_cold e' gia' buono quanto il precooling).
Se cambi --tcold ricalcola il locus (non riusare il file salvato).

Produce (in figures/):
  fig8_protocolli.png     sinistra: piano M-S con locus di equilibrio e le traiettorie dei protocolli
                          destra  : |<e>(t) - e_eq(T_max)| in funzione del tempo (come l'inset di Fig. 3)
  fig8b_protocolli_MS_t.png  <M>(t) e <S>(t) dei protocolli, con i valori di equilibrio a T_max
  fig8c_scan_tau.png      risultato della scansione di tau (solo se fatta)
  dati_fig8.npz           risultati numerici (salvati PRIMA dei grafici)
e stampa una tabella di Delta e e Delta S a tempi di riscaldamento fissati.

Uso:
    python main_05_protocolli.py                    # L = 10, scansione di tau, tutti i protocolli
    python main_05_protocolli.py --tau 1850         # tau fissato (MC step), niente scansione
    python main_05_protocolli.py --tau 500 1850 5000 --L 70
    python main_05_protocolli.py --tcold 0.00125    # T_cold diverso
    python main_05_protocolli.py --scan --tau 1850  # fa anche la scansione
    python main_05_protocolli.py --quick            # run velocissimo per controllare che giri
    python main_05_protocolli.py --locus-file figures/dati_locus_fig8.npz   # riusa il locus gia' calcolato
        (il locus viene salvato in dati_locus_fig8.npz a ogni run senza --locus-file; vale solo per lo
         stesso L e lo stesso T_cold: se cambi --tcold ricalcolalo)
Stima dei tempi con numba: L = 10 -> ~1 min; L = 70 con 4000 realizzazioni -> decine di minuti
(riduci --rep e --scan-rep per prove veloci).
"""
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from src import ising

H_PAPER = 1.0025
T0, TMAX = 0.125, 0.625
TCOLD = 0.00125                      # modificabile con --tcold

# Parametri "in unita' di N" dove ha senso (N = L*L): tutto in MC step (tentativi) alla fine.
FULL = dict(n_rep=6000, burn_N=200, tmax_N=200, n_rec=80, scan_rep=2000, scan_nT=25,
            tau_lo_N=0.05, tau_hi_N=30, win_lo_N=2, win_hi_N=6, win_pts=5,
            locus_nT=35, locus_rep=30, locus_burn=1500, locus_meas=2500)
LIN_MAX_N = 30      # la parte lineare della griglia dei tempi arriva a 30*N MC step
N_LIN = 90          # numero di punti della parte lineare

QUICK = dict(n_rep=150, burn_N=30, tmax_N=10, n_rec=30, scan_rep=150, scan_nT=6,
             tau_lo_N=0.05, tau_hi_N=8, win_lo_N=2, win_hi_N=6, win_pts=4,
             locus_nT=8, locus_rep=4, locus_burn=40, locus_meas=40)


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


def summarize(M, S, E):
    """Medie e errori standard sull'insieme (asse 0) di M, S, e."""
    return dict(M=M.mean(0), S=S.mean(0), E=E.mean(0), dM=sem(M), dS=sem(S), dE=sem(E))


# --------------------------------------------------------------------------
# Locus di equilibrio (serve a disegnare la curva e a dare i valori a T_max)
# --------------------------------------------------------------------------
def locus_grid(temps, n):
    lo, hi = min(temps), max(temps)
    T = np.geomspace(lo, hi, n)
    far = np.all(np.abs(np.log(T[:, None] / np.array(temps)[None, :])) > 0.04, axis=1)
    return np.unique(np.concatenate([T[far], np.array(temps)]))


def compute_locus(L, T_values, cfg):
    n = len(T_values)
    keys = ("M", "S", "e")
    val = {k: np.empty(n) for k in keys}
    err = {k: np.empty(n) for k in keys}
    t_start = time.time()
    for k, T in enumerate(T_values):
        out = ising.equilibrium_point(L, T, H_PAPER, cfg["locus_burn"], cfg["locus_meas"],
                                      cfg["locus_rep"], ising.INIT["random"])
        for j, key in enumerate(keys):
            val[key][k], err[key][k] = out[:, j].mean(), sem(out[:, j])
    print(f"   locus calcolato su {n} temperature ({time.time() - t_start:.1f} s)")
    return dict(T=np.asarray(T_values), M=val["M"], S=val["S"], e=val["e"],
                dM=err["M"], dS=err["S"], de=err["e"])


LOCUS_FILE = "dati_locus_fig8.npz"


def save_locus(outdir, L, locus):
    """Salva il locus (con errori) per poterlo riusare con --locus-file."""
    try:
        np.savez(os.path.join(outdir, LOCUS_FILE), L=L, h=H_PAPER, **locus)
        print(f"   locus salvato in {os.path.join(outdir, LOCUS_FILE)}  (riusabile con --locus-file)")
    except OSError as e:
        print(f"   (locus non salvato: {e})")


def load_locus(path, L, cfg):
    """Legge un locus salvato. Accetta il file di questo script (dati_locus_fig8.npz) oppure
    dati_fig4.npz di main_01 (senza energia: e_eq a T_max viene allora ricalcolata qui)."""
    d = np.load(path)
    if "L" in d and int(d["L"]) != L:
        raise SystemExit(f"Il locus in {path} e' stato calcolato con L = {int(d['L'])}, ma ora L = {L}.")
    if "L" not in d:
        print("   ATTENZIONE: il file non contiene L: verifica che sia stato calcolato con lo stesso L!")
    n = len(d["T"])
    zeros = np.zeros(n)
    locus = dict(T=d["T"], M=d["M"], S=d["S"], dM=d["dM"] if "dM" in d else zeros,
                 dS=d["dS"] if "dS" in d else zeros)
    if "e" in d:
        locus["e"] = d["e"]
        locus["de"] = d["de"] if "de" in d else zeros
    else:
        i = int(np.argmin(np.abs(np.log(locus["T"] / TMAX))))
        if not np.isclose(locus["T"][i], TMAX):
            raise SystemExit("Nel file non c'e' T_max nella griglia delle temperature: non posso ricavare e_eq.")
        print("   il file non contiene l'energia: ricalcolo e_eq(T_max) ...")
        out = ising.equilibrium_point(L, TMAX, H_PAPER, cfg["locus_burn"], cfg["locus_meas"],
                                      cfg["locus_rep"], ising.INIT["random"])
        locus["e"] = np.full(n, np.nan)
        locus["de"] = np.full(n, np.nan)
        locus["e"][i], locus["de"][i] = out[:, 2].mean(), sem(out[:, 2])
    return locus


def locus_at(locus, T):
    """Valori e errori (M, S, e) del locus a temperatura T (interpolazione in log T)."""
    x = np.log(locus["T"])
    out = {}
    for k in ("M", "S", "e", "dM", "dS", "de"):
        ok = np.isfinite(locus[k])
        out[k] = float(np.interp(np.log(T), x[ok], locus[k][ok]))
    return out


# --------------------------------------------------------------------------
# Simulazione dei protocolli (tempi in MC step = tentativi di flip)
# --------------------------------------------------------------------------
def time_grid(t_max, n_rec, N):
    """Griglia di tempi di riscaldamento (MC step, interi): logaritmica da 1 a t_max, piu' una parte
    lineare da 0 a 30*N (per i grafici in scala lineare)."""
    log_part = np.geomspace(1, t_max, n_rec)
    lin_part = np.linspace(0, min(t_max, LIN_MAX_N * N), N_LIN)      # serve ai grafici in scala lineare
    return np.unique(np.round(np.concatenate([log_part, lin_part])).astype(np.int64))


def run_oven(ens, Tm, rec):
    M, S, E = ising.run_schedule(ens, Tm, 0, Tm, H_PAPER, rec)
    return summarize(M, S, E)


def run_stay_cold(ens, Tc, rec):
    M, S, E = ising.run_schedule(ens, Tc, np.int64(rec[-1] + 1), Tc, H_PAPER, rec)
    return summarize(M, S, E)


def run_precooling(ens, Tc, Tm, tau, t_grid, heat_extra):
    """Precooling con tempo di raffreddamento tau. Registra: la fase fredda (t < tau), il tempo tau,
    e il riscaldamento (tau + t_heat). Ritorna il dizionario di medie con 't' = tempo totale (MC step)
    e 'th' = tempo di riscaldamento = t - tau (negativo durante la fase fredda)."""
    heat = np.unique(np.concatenate([[0], t_grid, heat_extra])).astype(np.int64)
    cool = t_grid[t_grid < tau]
    rec = np.unique(np.concatenate([[0], cool, [tau], tau + heat])).astype(np.int64)
    M, S, E = ising.run_schedule(ens, Tc, np.int64(tau), Tm, H_PAPER, rec)
    res = summarize(M, S, E)
    res["t"] = rec
    res["th"] = rec - tau
    res["tau"] = int(tau)
    return res


def window_stats(ens, Tc, tau, Tm, win_t, eeq, Seq):
    """Media in una finestra di tempi di riscaldamento win_t (MC step) di Delta e e Delta S,
    media su realizzazioni con errore standard. Per tau = 0 e' l'oven partito da 'ens'."""
    rec = (np.int64(tau) + win_t).astype(np.int64)
    _, S, E = ising.run_schedule(ens, Tc, np.int64(tau), Tm, H_PAPER, rec)
    de = E.mean(1) - eeq
    ds = S.mean(1) - Seq
    return dict(de=de.mean(), dde=sem(de), ds=ds.mean(), dds=sem(ds))


def scan_tau(L, ens0, ens_cold, Tc, Tm, taus, win_t, eeq, Seq):
    out = []
    t_start = time.time()
    for k, tau in enumerate(taus):
        out.append(window_stats(ens0, Tc, tau, Tm, win_t, eeq, Seq))
        print(f"   tau = {tau:>8d} MC step ({tau / (L * L):7.3f} sweep): "
              f"<De> = {out[-1]['de'] * 1e3:8.3f} +- {out[-1]['dde'] * 1e3:.3f} (x1e-3)   "
              f"<DS> = {out[-1]['ds'] * 1e3:8.3f} +- {out[-1]['dds'] * 1e3:.3f} (x1e-3)   "
              f"[{time.time() - t_start:5.1f} s]", flush=True)
    oven = window_stats(ens0, Tc, 0, Tm, win_t, eeq, Seq)
    gray = window_stats(ens_cold, Tc, 0, Tm, win_t, eeq, Seq)
    return out, oven, gray


# --------------------------------------------------------------------------
# Tabella
# --------------------------------------------------------------------------
def print_table(L, protos, eq, check):
    N = L * L
    print("\n" + "=" * 112)
    print(f"DISTANZA DA pi(T_max) a tempi di RISCALDAMENTO fissati  (h = {H_PAPER}, L = {L}, N = {N};"
          f" tempo in MC step = tentativi di flip)")
    print("  Delta e = <e>(t) - e_eq(T_max)   e   Delta S = <S>(t) - S_eq(T_max),  in unita' di 1e-3, +- errore standard")
    print(f"  e_eq(T_max) = {eq['e']:.5f} +- {eq['de']:.5f}      S_eq(T_max) = {eq['S']:.5f} +- {eq['dS']:.5f}")
    print("  (il tempo di riscaldamento e' contato dall'istante in cui il bagno va a T_max)")
    print("=" * 112)
    for kind, label in (("e", "Delta e  x 1e3"), ("S", "Delta S  x 1e3")):
        print(f"\n{label}")
        head = f"   {'t_riscald':>10s}  " + "  ".join(f"{p['name']:>24s}" for p in protos)
        print(head)
        for tc in check:
            row = f"   {tc:>10d}  "
            cells = []
            for p in protos:
                i = int(np.searchsorted(p["th"], tc))
                if i >= len(p["th"]) or p["th"][i] != tc:
                    cells.append(f"{'--':>24s}")
                    continue
                if kind == "e":
                    v, d = p["E"][i] - eq["e"], np.hypot(p["dE"][i], eq["de"])
                else:
                    v, d = p["S"][i] - eq["S"], np.hypot(p["dS"][i], eq["dS"])
                cells.append(f"{v * 1e3:>13.3f} +- {d * 1e3:6.3f}")
            print(row + "  ".join(cells))
    print("\n(Differenze compatibili con 0 se minori di circa 2 volte l'errore indicato.)")
    print("=" * 112)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def fig_main(L, Tc, locus, eq, oven, gray, cold, pres, outdir):
    N = L * L
    fig, (axp, axe) = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")

    # --- piano M-S ---
    axp.plot(locus["M"], locus["S"], "-o", color="0.6", ms=3.5, lw=1.5, zorder=1, label="locus di equilibrio")
    axp.plot(cold["M"], cold["S"], ":", color="tab:orange", lw=2.2, zorder=3,
             label=r"$\pi(T_0)$ accoppiato a $T_{cold}$ ")
    axp.plot(gray["M"], gray["S"], "-", color="tab:green", lw=1.8, zorder=3,
             label=r"oven da $\pi(T_{cold})$ a $T_{max}$")
    axp.plot(oven["M"], oven["S"], "-", color="tab:blue", lw=2.4, zorder=4, label=r"oven: $T_0 \to T_{max}$")
    cmap = plt.get_cmap("Reds")
    for k, pc in enumerate(pres):
        col = cmap(0.45 + 0.5 * (k + 1) / len(pres))
        axp.plot(pc["M"], pc["S"], "-", color=col, lw=2.4, zorder=5,
                 label=f"precooling (tau = {pc['tau']} MC step)")
        i = int(np.searchsorted(pc["t"], pc["tau"]))
        axp.plot(pc["M"][i], pc["S"][i], "s", color=col, mec="k", ms=7, zorder=6)
    for T, nm, mk, c in ((T0, r"$\pi(T_0)$", "o", "k"), (Tc, r"$\pi(T_{cold})$", "o", "tab:cyan"),
                         (TMAX, r"$\pi(T_{max})$", "*", "gold")):
        q = locus_at(locus, T)
        axp.plot(q["M"], q["S"], mk, color=c, mec="k", ms=13 if mk == "*" else 9, zorder=7)
        axp.annotate(nm, (q["M"], q["S"]), textcoords="offset points", xytext=(6, 6), fontsize=9)
    axp.set_xlabel(r"$\langle M\rangle$")
    axp.set_ylabel(r"$\langle S\rangle$")
    axp.set_title("piano M-S")
    axp.grid(alpha=0.3)
    axp.legend(fontsize=7.5, loc="best")

    # --- distanza di energia ---
    def curve(t, p, **kw):
        ok = t > 0
        d = p["E"] - eq["e"]
        e = np.hypot(p["dE"], eq["de"])
        y = np.abs(d)
        axe.plot(t[ok], y[ok], **kw)
        return e[ok]

    floors = []      # errori sull'ultimo punto: servono solo a fissare il limite inferiore dell'asse y
    floors.append(curve(oven["t"], oven, color="tab:blue", lw=2, label=r"oven: $T_0 \to T_{max}$"))
    floors.append(curve(cold["t"], cold, color="tab:orange", lw=1.6, ls=":", label=r"resta a $T_{cold}$"))
    for k, pc in enumerate(pres):
        col = cmap(0.45 + 0.5 * (k + 1) / len(pres))
        floors.append(curve(pc["t"], pc, color=col, lw=2, label=f"precooling, tau = {pc['tau']}"))
        axe.axvline(pc["tau"], color=col, ls="--", lw=0.9)
    floor = min(f[-1] for f in floors)
    axe.set_xscale("log")
    axe.set_yscale("log")
    axe.set_ylim(bottom=0.4 * floor)
    axe.set_xlabel("t  (MC step)")
    axe.set_ylabel(r"$|\langle e\rangle(t)-e_{eq}(T_{max})|$")
    axe.set_title("distanza di energia dall'equilibrio finale  (tau tratteggiata)")
    axe.grid(alpha=0.3, which="both")
    axe.legend(fontsize=7.5, loc="best")
    sec = axe.secondary_xaxis("top", functions=(lambda x: x / N, lambda x: x * N))
    sec.set_xlabel("t  (sweep = L*L MC step)", fontsize=8)
    fig.suptitle(f"Protocolli oven e precooling   (h = {H_PAPER},  L = {L})")
    save_fig(fig, outdir, "fig8_protocolli.png")


def fig_time(L, Tc, eq, oven, gray, cold, pres, outdir):
    N = L * L
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True, layout="constrained")
    cmap = plt.get_cmap("Reds")
    for ax, key, ekey, lab in ((axes[0], "M", "dM", r"$\langle M\rangle(t)$"),
                               (axes[1], "S", "dS", r"$\langle S\rangle(t)$")):
        items = [(oven, "tab:blue", "-", r"oven: $T_0 \to T_{max}$"),
                 (gray, "tab:green", "-", r"oven da $\pi(T_{cold})$"),
                 (cold, "tab:orange", ":", r"resta a $T_{cold}$")]
        for k, pc in enumerate(pres):
            items.append((pc, cmap(0.45 + 0.5 * (k + 1) / len(pres)), "-", f"precooling (tau = {pc['tau']})"))
        for p, col, ls, name in items:
            ok = p["t"] > 0
            ax.plot(p["t"][ok], p[key][ok], color=col, ls=ls, lw=2, label=name)
            ax.fill_between(p["t"][ok], (p[key] - p[ekey])[ok], (p[key] + p[ekey])[ok], color=col,
                            alpha=0.2, lw=0)
        for pc in pres:
            ax.axvline(pc["tau"], color="0.5", ls="--", lw=0.8)
        ax.axhline(eq[key], color="k", ls="--", lw=1, label=r"equilibrio $T_{max}$")
        ax.set_ylabel(lab)
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7.5, loc="best")
    axes[1].set_xlabel("t  (MC step)")
    sec = axes[0].secondary_xaxis("top", functions=(lambda x: x / N, lambda x: x * N))
    sec.set_xlabel("t  (sweep = L*L MC step)", fontsize=8)
    fig.suptitle(f"<M>(t) e <S>(t) nei protocolli  (h = {H_PAPER},  L = {L},  T_cold = {Tc:g})")
    save_fig(fig, outdir, "fig8b_protocolli_MS_t.png")


def fig_scan(L, Tc, taus, scan, oven_w, gray_w, win, outdir):
    N = L * L
    taus = np.asarray(taus)
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True, layout="constrained")
    for ax, k, dk, lab in ((axes[0], "de", "dde", r"$\langle\Delta e\rangle$ (x$10^{-3}$)"),
                           (axes[1], "ds", "dds", r"$\langle\Delta S\rangle$ (x$10^{-3}$)")):
        y = np.array([s[k] for s in scan]) * 1e3
        e = np.array([s[dk] for s in scan]) * 1e3
        ax.errorbar(taus, y, yerr=e, fmt="-o", color="tab:red", capsize=3, label="precooling")
        for w, col, nm in ((oven_w, "tab:blue", r"oven da $\pi(T_0)$"),
                           (gray_w, "tab:green", r"oven da $\pi(T_{cold})$")):
            ax.axhline(w[k] * 1e3, color=col, ls="--", label=nm)
            ax.axhspan((w[k] - w[dk]) * 1e3, (w[k] + w[dk]) * 1e3, color=col, alpha=0.12, lw=0)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(lab)
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("tau  (MC step)")
    sec = axes[0].secondary_xaxis("top", functions=(lambda x: x / N, lambda x: x * N))
    sec.set_xlabel("tau  (sweep)", fontsize=8)
    fig.suptitle(f"Scansione di tau")
    save_fig(fig, outdir, "fig8c_scan_tau.png")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=10, help="LATO del reticolo (pari); N = L*L spin")
    ap.add_argument("--tau", type=int, nargs="+", help="tempo/i di raffreddamento in MC step (es. 1850)")
    ap.add_argument("--scan", action="store_true", help="fai la scansione di tau anche se --tau e' dato")
    ap.add_argument("--rep", type=int, help="realizzazioni per i protocolli")
    ap.add_argument("--scan-rep", type=int, help="realizzazioni per la scansione di tau")
    ap.add_argument("--burn", type=int, help="MC step di preparazione dell'equilibrio (tentativi)")
    ap.add_argument("--tmax", type=int, help="durata del riscaldamento in MC step")
    ap.add_argument("--win", type=int, nargs=2, metavar=("W1", "W2"),
                    help="finestra di tempo di riscaldamento (MC step) per la scelta di tau")
    ap.add_argument("--locus-file", help="npz del locus da riusare, es. figures/dati_locus_fig8.npz (creato da questo script) oppure figures/dati_fig4.npz")
    ap.add_argument("--tcold", type=float, default=TCOLD)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    TCOLD = args.tcold
    cfg = dict(QUICK if args.quick else FULL)
    L = args.L
    if L % 2:
        raise SystemExit("L deve essere pari")
    N = L * L
    if args.rep:
        cfg["n_rep"] = args.rep
    if args.scan_rep:
        cfg["scan_rep"] = args.scan_rep
    n_burn_steps = args.burn if args.burn else cfg["burn_N"] * N
    n_burn_sweeps = int(np.ceil(n_burn_steps / N))            # prepare_ensemble lavora in sweep
    t_max = args.tmax if args.tmax else cfg["tmax_N"] * N
    win_lo, win_hi = args.win if args.win else (int(cfg["win_lo_N"] * N), int(cfg["win_hi_N"] * N))
    win_t = np.unique(np.round(np.linspace(win_lo, win_hi, cfg["win_pts"])).astype(np.int64))
    do_scan = args.scan or not args.tau

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(outdir, exist_ok=True)
    ising.set_seed(args.seed)
    print(f"unita' di tempo: 1 MC step = 1 tentativo di flip  (1 sweep = {N} MC step)")
    print(f"preparazione: {n_burn_steps} MC step (~{n_burn_sweeps} sweep)   riscaldamento fino a {t_max} MC step"
          f"   finestra [{win_lo}, {win_hi}]")

    t0 = time.time()
    # 1) locus di equilibrio (curva grigia e valori di equilibrio a T_max)
    if args.locus_file:
        locus = load_locus(args.locus_file, L, cfg)
        print(f"   locus letto da {args.locus_file}")
        for T in (TCOLD, T0, TMAX):
            if not np.any(np.isclose(locus["T"], T)):
                print(f"   ATTENZIONE: T = {T:g} non e' nella griglia del file (uso l'interpolazione)")
    else:
        print("locus di equilibrio ...")
        locus = compute_locus(L, locus_grid([TCOLD, T0, TMAX], cfg["locus_nT"]), cfg)
        save_locus(outdir, L, locus)
    eq = locus_at(locus, TMAX)

    # 2) insiemi iniziali
    print("preparazione degli insiemi di partenza ...")
    ens0 = ising.prepare_ensemble(L, T0, H_PAPER, n_burn_sweeps, cfg["n_rep"], ising.INIT["random"])
    ens_cold = ising.prepare_ensemble(L, TCOLD, H_PAPER, n_burn_sweeps, cfg["n_rep"], ising.INIT["random"])
    print(f"   fatto ({time.time() - t0:.1f} s)")

    # 3) scelta di tau
    scan = oven_w = gray_w = None
    taus_scan = None
    if do_scan:
        print("scansione di tau ...")
        taus_scan = np.unique(np.round(np.geomspace(cfg["tau_lo_N"] * N, cfg["tau_hi_N"] * N,
                                                     cfg["scan_nT"])).astype(np.int64))
        ens_scan0 = ens0[:cfg["scan_rep"]]
        ens_scan_c = ens_cold[:cfg["scan_rep"]]
        scan, oven_w, gray_w = scan_tau(L, ens_scan0, ens_scan_c, TCOLD, TMAX, [int(t) for t in taus_scan],
                                        win_t, eq["e"], eq["S"])
        best = int(np.argmin([abs(s["de"]) for s in scan]))
        tau_best = int(taus_scan[best])
        print(f"   oven da pi(T_0)    : <De> = {oven_w['de'] * 1e3:8.3f} +- {oven_w['dde'] * 1e3:.3f} (x1e-3)")
        print(f"   oven da pi(T_cold) : <De> = {gray_w['de'] * 1e3:8.3f} +- {gray_w['dde'] * 1e3:.3f} (x1e-3)")
        print(f"   tau scelto (|<De>| minimo): {tau_best} MC step = {tau_best / N:.3f} sweep")
    taus = sorted(set(args.tau)) if args.tau else [tau_best]

    # 4) protocolli completi
    print("protocolli ...")
    t_grid = time_grid(t_max, cfg["n_rec"], N)
    check = np.array([c for c in (N // 2, N, 2 * N, 5 * N, 10 * N) if c <= t_max], dtype=np.int64)
    rec = np.unique(np.concatenate([[0], t_grid, check])).astype(np.int64)
    oven = run_oven(ens0, TMAX, rec)
    oven.update(t=rec, th=rec, name="oven")
    print(f"   oven fatto ({time.time() - t0:.1f} s)", flush=True)
    gray = run_oven(ens_cold, TMAX, rec)
    gray.update(t=rec, th=rec, name="oven da pi(T_cold)")
    print(f"   oven da pi(T_cold) fatto ({time.time() - t0:.1f} s)", flush=True)
    cold = run_stay_cold(ens0, TCOLD, rec)
    cold.update(t=rec, th=rec, name="resta a T_cold")
    print(f"   resta a T_cold fatto ({time.time() - t0:.1f} s)", flush=True)
    pres = []
    for tau in taus:
        pc = run_precooling(ens0, TCOLD, TMAX, tau, t_grid, check)
        pc["name"] = f"precooling tau={tau}"
        pres.append(pc)
        print(f"   precooling tau = {tau} fatto ({time.time() - t0:.1f} s)", flush=True)

    # 5) salvataggio dati (prima dei grafici)
    try:
        d = dict(L=L, h=H_PAPER, T0=T0, Tcold=TCOLD, Tmax=TMAX, taus=np.array(taus),
                 locus_T=locus["T"], locus_M=locus["M"], locus_S=locus["S"], locus_e=locus["e"],
                 eq_M=eq["M"], eq_S=eq["S"], eq_e=eq["e"], eq_de=eq["de"])
        for nm, p in (("oven", oven), ("gray", gray), ("cold", cold)):
            for k in ("t", "M", "S", "E", "dM", "dS", "dE"):
                d[f"{nm}_{k}"] = p[k]
        for i, p in enumerate(pres):
            for k in ("t", "th", "M", "S", "E", "dM", "dS", "dE"):
                d[f"pre{i}_{k}"] = p[k]
        if scan is not None:
            d.update(scan_tau=taus_scan, scan_de=[s["de"] for s in scan], scan_dde=[s["dde"] for s in scan],
                     scan_ds=[s["ds"] for s in scan], scan_dds=[s["dds"] for s in scan])
        np.savez(os.path.join(outdir, "dati_fig8.npz"), **d)
    except OSError as e:
        print(f"   (dati non salvati: {e})")

    # 6) tabella e grafici
    print_table(L, [oven, gray] + pres, eq, [int(c) for c in check])
    fig_main(L, TCOLD, locus, eq, oven, gray, cold, pres, outdir)
    fig_time(L, TCOLD, eq, oven, gray, cold, pres, outdir)
    if scan is not None:
        fig_scan(L, TCOLD, taus_scan, scan, oven_w, gray_w, win_t, outdir)
    print(f"finito in {time.time() - t0:.1f} s")

    if not args.no_show:
        plt.show()