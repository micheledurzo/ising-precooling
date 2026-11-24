"""
main_06_confronto_lineare.py
Confronto oven / precooling in SCALA LINEARE, a partire dai dati salvati da main_05_protocolli.py
(figures/dati_fig8.npz): non rifa' nessuna simulazione.

Si disegna la distanza normalizzata dall'equilibrio, CON SEGNO:

    D(t) = (<e>(t) - e_eq(T_max)) / (<e>(0) - e_eq(T_max))

D = 1 e' la partenza (stesso punto pi(T0) per tutti i protocolli), D = 0 e' l'equilibrio finale.
Il tempo e' quello totale dall'inizio del protocollo (nel precooling include i tau MC step della
fase fredda, marcati da una linea verticale tratteggiata): il confronto e' quindi "alla pari".
D > 1 durante la fase fredda del precooling = il sistema si allontana da pi(T_max) (e' il "costo" del
raffreddamento); D < 0 = ha superato l'equilibrio.

Uso (dopo aver lanciato main_05_protocolli.py):
    python main_06_confronto_lineare.py
    python main_06_confronto_lineare.py --data figures/dati_fig8.npz
    python main_06_confronto_lineare.py --xmax 20000      # limite dell'asse x in MC step (default 10*N)
    python main_06_confronto_lineare.py --semilogy        # asse y logaritmico su |D| (esponenziali = rette)
    python main_06_confronto_lineare.py --semilogy --ymin 1e-2   # taglia il rumore a valori piu' alti
    python main_06_confronto_lineare.py --bands           # mostra anche +-1 errore standard
Produce figures/fig9_confronto_lineare.png.
"""
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt


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


def load(path):
    d = np.load(path)
    eq = float(d["eq_e"])
    norm = float(d["oven_E"][0]) - eq
    curves = []

    def add(name, t, E, dE, color, tau=None):
        curves.append(dict(name=name, t=np.asarray(t, float), D=(E - eq) / norm, dD=dE / abs(norm),
                           color=color, tau=tau))

    add("oven", d["oven_t"], d["oven_E"], d["oven_dE"], "tab:blue")
    taus = np.atleast_1d(d["taus"])
    cmap = plt.get_cmap("Reds")
    for i, tau in enumerate(taus):
        add(f"precooling, tau = {int(tau)}", d[f"pre{i}_t"], d[f"pre{i}_E"], d[f"pre{i}_dE"],
            cmap(0.45 + 0.5 * (i + 1) / len(taus)), tau=int(tau))
    return curves, dict(L=int(d["L"]), Tcold=float(d["Tcold"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "figures", "dati_fig8.npz"))
    ap.add_argument("--xmax", type=float, help="limite dell'asse x (MC step); default 10*N")
    ap.add_argument("--semilogy", action="store_true")
    ap.add_argument("--ymin", type=float, default=1e-3,
                    help="con --semilogy: limite inferiore dell'asse y (default 1e-3, cioe' 0.1%% della distanza iniziale)")
    ap.add_argument("--bands", action="store_true", help="aggiungi fasce +-1 errore standard")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    curves, meta = load(args.data)
    L, Tc = meta["L"], meta["Tcold"]
    N = L * L
    outdir = os.path.join(here, "figures")
    os.makedirs(outdir, exist_ok=True)

    t_end = min(c["t"][-1] for c in curves)
    xmax = min(args.xmax if args.xmax else 10 * N, t_end)

    fig, ax = plt.subplots(figsize=(9, 5.6), layout="constrained")
    for c in curves:
        ok = c["t"] <= xmax * 1.02
        y = np.abs(c["D"]) if args.semilogy else c["D"]
        ax.plot(c["t"][ok], y[ok], color=c["color"], lw=2.4 if c["tau"] is None else 2, label=c["name"])
        if args.bands:
            ax.fill_between(c["t"][ok], np.maximum((y - c["dD"])[ok], 1e-12), (y + c["dD"])[ok],
                            color=c["color"], alpha=0.15, lw=0)
        if c["tau"] is not None:
            ax.axvline(c["tau"], color=c["color"], ls="--", lw=0.9)
    ax.set_xlim(0, xmax)
    if args.semilogy:
        ax.set_yscale("log")
        top = 1.3 * max(np.abs(c["D"][c["t"] <= xmax]).max() for c in curves)
        ax.set_ylim(bottom=args.ymin, top=top)
        ax.set_ylabel(r"$|D(t)|$")
    else:
        ax.axhline(0, color="k", lw=0.9)
        ax.set_ylabel(r"$D(t)=\dfrac{\langle e\rangle(t)-e_{eq}(T_{max})}{\langle e\rangle(0)-e_{eq}(T_{max})}$")
    ax.set_xlabel("t  (MC step)")
    scala = "asse y logaritmico" if args.semilogy else "scala lineare"
    ax.set_title(f"Oven e precooling, {scala}   (L = {L},  T_cold = {Tc:g};  tau tratteggiato)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / N, lambda x: x * N))
    sec.set_xlabel("t  (sweep = L*L MC step)", fontsize=8)
    save_fig(fig, outdir, "fig9_confronto_lineare.png")
    if not args.no_show:
        plt.show()