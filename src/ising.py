"""
ising.py  --  Modello di Ising 2D antiferromagnetico con campo + dinamica di Glauber
================================================================================
Riferimento: Gal & Raz, PRL 124, 060602 (2020)

Hamiltoniana (Eq. 11 del paper)

    H = -(J/4) * sum_<ij,kl> s_ij s_kl  -  h * sum_ij s_ij ,      J = -1

La somma sui primi vicini conta OGNI LEGAME UNA VOLTA SOLA (si ricava dal
Supplemental Material, Eq. 1, dove l'energia per spin e' 1/2[-J x1 x2 - H(x1+x2)]).
Conseguenza: l'accoppiamento effettivo per legame e' |J|/4 = 0.25 e il campo
critico a T=0 e'  h_c = 4*|J|/4 = 1  (vedi lo script main_01_equilibrio.py).

Dinamica: single-spin-flip Glauber (Eq. 12)

    rate = 1 / (1 + exp(dE / T))          (kB = 1)

Un "MC step" (o sweep) = L*L tentativi di flip su siti scelti a caso.

Tutto il codice "pesante" e' compilato con numba. Se numba non e' installato
funziona lo stesso (identico risultato) ma e' decine di volte piu' lento.
"""
import numpy as np
from numba import njit

J = -1.0                     # accoppiamento antiferromagnetico (J < 0)

# codici delle condizioni iniziali (numba vuole numeri, non stringhe)
INIT = {"random": 0, "up": 1, "down": 2, "neel": 3}


# --------------------------------------------------------------------------
# Numeri casuali
# --------------------------------------------------------------------------
@njit
def _seed_numba(seed):
    np.random.seed(seed)     # dentro numba semina il generatore di numba


def set_seed(seed):
    """Semina sia il generatore di numba sia quello di numpy."""
    _seed_numba(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------
# Reticolo e osservabili
# --------------------------------------------------------------------------
@njit
def make_lattice(L, init):
    """Crea un reticolo LxL. init: 0 casuale, 1 tutti up, 2 tutti down, 3 Neel."""
    lattice = np.empty((L, L), dtype=np.int64)
    for i in range(L):
        for j in range(L):
            if init == 0:
                lattice[i, j] = 1 if np.random.random() < 0.5 else -1
            elif init == 1:
                lattice[i, j] = 1
            elif init == 2:
                lattice[i, j] = -1
            else:                                  # Neel: scacchiera +-1
                lattice[i, j] = 1 if (i + j) % 2 == 0 else -1
    return lattice


def initialize_lattice(L, init="random"):
    """Versione 'comoda' di make_lattice: init e' una stringa di INIT."""
    if L % 2 != 0:
        raise ValueError("L deve essere pari (condizioni periodiche + antiferromagnete)")
    return make_lattice(L, INIT[init])


@njit
def magnetization(lattice):
    """M = (1/N) sum s_ij"""
    L = lattice.shape[0]
    return lattice.sum() / (L * L)


@njit
def staggered_magnetization(lattice):
    """S = (1/N) |sum (-1)^(i+j) s_ij|   (Eq. 13)"""
    L = lattice.shape[0]
    total = 0
    for i in range(L):
        for j in range(L):
            if (i + j) % 2 == 0:
                total += lattice[i, j]
            else:
                total -= lattice[i, j]
    return abs(total) / (L * L)


@njit
def energy(lattice, h):
    """Energia TOTALE (Eq. 11), ogni legame contato una volta."""
    L = lattice.shape[0]
    bonds = 0
    for i in range(L):
        for j in range(L):
            s = lattice[i, j]
            bonds += s * lattice[(i + 1) % L, j] + s * lattice[i, (j + 1) % L]
    return -(J / 4.0) * bonds - h * lattice.sum()


@njit
def energy_per_spin(lattice, h):
    L = lattice.shape[0]
    return energy(lattice, h) / (L * L)


# --------------------------------------------------------------------------
# Dinamica di Glauber
# --------------------------------------------------------------------------
@njit
def delta_energy(lattice, i, j, h):
    """Variazione di energia se si flippa lo spin (i, j).

    E_locale = -(J/4) s nb - h s   --->   dopo il flip cambia segno,
    quindi  dE = (J/2) s nb + 2 h s.
    """
    L = lattice.shape[0]
    s = lattice[i, j]
    nb = (
        lattice[(i + 1) % L, j]
        + lattice[(i - 1) % L, j]
        + lattice[i, (j + 1) % L]
        + lattice[i, (j - 1) % L]
    )
    return (J / 2.0) * s * nb + 2.0 * h * s


@njit
def flip_probability(dE, T):
    """Rate di Glauber (Eq. 12)."""
    x = dE / T
    if x > 700.0:            # evita overflow di exp a T molto basse (il rate e' ~0)
        return 0.0
    return 1.0 / (1.0 + np.exp(x))


@njit
def sweep(lattice, T, h):
    """Un MC step = L*L tentativi di flip su siti casuali. Modifica lattice."""
    L = lattice.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        dE = delta_energy(lattice, i, j, h)
        if np.random.random() < flip_probability(dE, T):
            lattice[i, j] = -lattice[i, j]


# --------------------------------------------------------------------------
# Simulazioni "ensemble" (n_rep repliche indipendenti)
# --------------------------------------------------------------------------
@njit
def run_relaxation(L, T, h, n_sweeps, n_rep, init):
    """Evoluzione temporale a T e h fissati.

    Restituisce tre array di forma (n_rep, n_sweeps+1): M, S, e (energia per spin)
    a ogni MC step; la colonna 0 e' lo stato iniziale.
    """
    M = np.empty((n_rep, n_sweeps + 1))
    S = np.empty((n_rep, n_sweeps + 1))
    E = np.empty((n_rep, n_sweeps + 1))
    for r in range(n_rep):
        lattice = make_lattice(L, init)
        M[r, 0] = magnetization(lattice)
        S[r, 0] = staggered_magnetization(lattice)
        E[r, 0] = energy_per_spin(lattice, h)
        for t in range(n_sweeps):
            sweep(lattice, T, h)
            M[r, t + 1] = magnetization(lattice)
            S[r, t + 1] = staggered_magnetization(lattice)
            E[r, t + 1] = energy_per_spin(lattice, h)
    return M, S, E


@njit
def equilibrium_point(L, T, h, n_burn, n_meas, n_rep, init):
    """Stima di <M>, <S>, <e> all'equilibrio a (T, h).

    Per ogni replica: n_burn sweep scartati (termalizzazione), poi media
    temporale su n_meas sweep. Restituisce un array (n_rep, 3) con
    colonne [M, S, e]; media ed errore si fanno sulle repliche.
    """
    out = np.empty((n_rep, 3))
    for r in range(n_rep):
        lattice = make_lattice(L, init)
        for _ in range(n_burn):
            sweep(lattice, T, h)
        m = 0.0
        s = 0.0
        e = 0.0
        for _ in range(n_meas):
            sweep(lattice, T, h)
            m += magnetization(lattice)
            s += staggered_magnetization(lattice)
            e += energy_per_spin(lattice, h)
        out[r, 0] = m / n_meas
        out[r, 1] = s / n_meas
        out[r, 2] = e / n_meas
    return out


# --------------------------------------------------------------------------
# Traiettorie fuori equilibrio (quench di temperatura)
# --------------------------------------------------------------------------
@njit
def attempt(lattice, T, h):
    """UN singolo tentativo di flip (sito casuale, rate di Glauber).
    sweep() = L*L chiamate di questa funzione."""
    L = lattice.shape[0]
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    dE = delta_energy(lattice, i, j, h)
    if np.random.random() < flip_probability(dE, T):
        lattice[i, j] = -lattice[i, j]


@njit
def prepare_ensemble(L, T, h, n_burn, n_rep, init):
    """Prepara n_rep realizzazioni all'equilibrio a (T, h): partenza da 'init' e n_burn sweep.
    Come nel supplemental del paper (stato casuale + dinamica a temperatura T).
    Ritorna un array int8 di forma (n_rep, L, L)."""
    ens = np.empty((n_rep, L, L), dtype=np.int8)
    for r in range(n_rep):
        lattice = make_lattice(L, init)
        for _ in range(n_burn):
            sweep(lattice, T, h)
        for i in range(L):
            for j in range(L):
                ens[r, i, j] = lattice[i, j]
    return ens


@njit
def quench_from_ensemble(ens, T, h, rec_attempts):
    """Da ogni realizzazione di 'ens' fa evolvere il sistema a temperatura T (bagno
    cambiato di colpo) e registra M, S, e (energia per spin) quando il numero di
    tentativi di flip raggiunge i valori di rec_attempts (array int64 crescente; il primo
    puo' essere 0 = stato iniziale). Ritorna tre array (n_rep, len(rec_attempts))."""
    n_rep = ens.shape[0]
    n_rec = len(rec_attempts)
    M = np.empty((n_rep, n_rec))
    S = np.empty((n_rep, n_rec))
    E = np.empty((n_rep, n_rec))
    for r in range(n_rep):
        lattice = ens[r].astype(np.int64)
        done = 0
        for k in range(n_rec):
            while done < rec_attempts[k]:
                attempt(lattice, T, h)
                done += 1
            M[r, k] = magnetization(lattice)
            S[r, k] = staggered_magnetization(lattice)
            E[r, k] = energy_per_spin(lattice, h)
    return M, S, E