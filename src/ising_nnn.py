"""
ising_nnn.py  --  Ising 2D antiferromagnetico con campo e interazione a SECONDI VICINI + dinamica di Glauber
==============================================================================================
Estensione di src/ising.py (Gal & Raz, PRL 124, 060602 (2020)): si aggiunge un accoppiamento J2 tra gli
spin sulle DIAGONALI (secondi vicini, 4 per sito).

Hamiltoniana (stessa convenzione del modello base: ogni legame contato UNA volta, fattore 1/4)

    H = -(J /4) * sum_<ij>   s_i s_j      primi vicini (4 per sito, 2 legami per sito)
        -(J2/4) * sum_<<ij>> s_i s_j      secondi vicini (4 per sito, 2 legami per sito)
        - h * sum_i s_i

con J = -1 (primi vicini antiferromagnetici, come nel paper). J2 e' nella STESSA convenzione di J:
    J2 = 0   ->  modello del paper (questo modulo coincide con src/ising.py)
    J2 > 0   ->  secondi vicini ferromagnetici: allineano spin dello stesso sottoreticolo, quindi
                 STABILIZZANO lo stato di Neel (che ha tutti i secondi vicini allineati)
    J2 < 0   ->  secondi vicini antiferromagnetici: FRUSTRANO il Neel (competizione J1-J2).
                 Per J2 < -0.5 lo stato fondamentale a h = 0 non e' piu' il Neel ma uno stato a strisce
                 (colonne alternate): a energia per spin  +J2/2  contro  -0.5 - J2/2  del Neel.

Variazione di energia nel flip di uno spin s (si ricava come nel modello base: l'energia locale
E_loc = -(J/4) s nb - (J2/4) s nnb - h s cambia segno nel flip, quindi dE = -2 E_loc):

    dE = (J/2) s nb + (J2/2) s nnb + 2 h s

dove nb = somma dei 4 primi vicini, nnb = somma dei 4 secondi vicini (diagonali). Rate di Glauber
(kB = 1):  rate = 1 / (1 + exp(dE / T)),  un MC step = un tentativo di flip (1 sweep = L*L tentativi).

Energie per spin di tre stati utili (usate in test_ising_nnn.py):
    Neel          e = -0.5 - J2/2             (M = 0, S = 1)
    tutti up      e =  0.5 - J2/2 - h         (M = 1, S = 0)
    strisce       e =  J2/2                   (M = 0)


CASO J2 = -1 (J1 = J2, forte frustrazione). Stati fondamentali (verificati per enumerazione esaustiva su 4x4 e 4x6):
    h < 1       strisce        M = 0     e = -0.5                   (colonne o righe alternate)
    1 < h < 2   plateau M=1/2  M = 0.5   e = -h/2                   (3 spin su e 1 giu' ogni blocco 2x2)
    h > 2       tutti up       M = 1     e = 1 - h
Le transizioni a T = 0 sono a h = 1 e h = 2 (a h = 1 strisce e plateau sono degeneri: e = -0.5 per entrambi).
Il campo del paper h = 1.0025 cade quindi subito sopra la prima transizione.

OSSERVABILI. Con J2 = -1 il parametro d'ordine di Neel S (scacchiera) NON descrive piu' lo stato a strisce:
nelle strisce S = 0. Si aggiungono le magnetizzazioni staggered "a righe" e "a colonne":
    sx = |(1/N) sum (-1)^i s_ij|     (alternanza lungo le righe i)
    sy = |(1/N) sum (-1)^j s_ij|     (alternanza lungo le colonne j)
    Q  = sqrt(sx^2 + sy^2)           (ordine a strisce, indipendente dall'orientamento: strisce -> Q = 1)
Per uno stato casuale Q ~ 1.25/L (S ~ 0.8/L). Nel plateau M = 1/2 sono diversi da zero S, sx e sy (|.| = 1/2).
Per questo tutte le funzioni "ensemble" di questo modulo restituiscono anche Q: colonne [M, S, Q, e].

CONDIZIONI INIZIALI: INIT = random, up, down, neel (come il modello base) + "stripe" (colonne alternate) e
"plateau" (spin giu' sul sottoreticolo (i pari, j pari)). Il Glauber a bassa T resta facilmente intrappolato in
domini di strisce di orientamenti diversi: partire dallo stato fondamentale giusto evita il problema.

L'API ricalca src/ising.py: le funzioni hanno un argomento in piu', J2, subito dopo h. Le cose uguali al
modello base (magnetization, staggered_magnetization, flip_probability, set_seed) sono importate da li'
(il generatore di numeri casuali e' lo stesso).
"""
import numpy as np

from .ising import (J, njit, magnetization, staggered_magnetization, flip_probability,
                    set_seed)

__all__ = ["INIT", "J", "make_lattice", "initialize_lattice", "magnetization",
           "staggered_magnetization", "stripe_magnetizations", "stripe_order", "flip_probability",
           "set_seed", "energy", "energy_per_spin", "delta_energy", "attempt", "sweep", "run_relaxation",
           "equilibrium_point", "prepare_ensemble", "quench_from_ensemble", "run_schedule"]

# codici delle condizioni iniziali (numba vuole numeri, non stringhe)
INIT = {"random": 0, "up": 1, "down": 2, "neel": 3, "stripe": 4, "plateau": 5}


# --------------------------------------------------------------------------
# Reticolo e osservabili
# --------------------------------------------------------------------------
@njit
def make_lattice(L, init):
    """Reticolo LxL. init: 0 casuale, 1 tutti up, 2 tutti down, 3 Neel, 4 strisce (colonne alternate),
    5 plateau M = 1/2 (giu' sul sottoreticolo i pari, j pari)."""
    lattice = np.empty((L, L), dtype=np.int64)
    for i in range(L):
        for j in range(L):
            if init == 0:
                lattice[i, j] = 1 if np.random.random() < 0.5 else -1
            elif init == 1:
                lattice[i, j] = 1
            elif init == 2:
                lattice[i, j] = -1
            elif init == 3:
                lattice[i, j] = 1 if (i + j) % 2 == 0 else -1
            elif init == 4:
                lattice[i, j] = 1 if j % 2 == 0 else -1
            else:
                lattice[i, j] = -1 if (i % 2 == 0 and j % 2 == 0) else 1
    return lattice


def initialize_lattice(L, init="random"):
    """Versione 'comoda' di make_lattice: init e' una stringa di INIT."""
    if L % 2 != 0:
        raise ValueError("L deve essere pari (condizioni periodiche + ordini a periodo 2)")
    return make_lattice(L, INIT[init])


@njit
def stripe_magnetizations(lattice):
    """(sx, sy): |(1/N) sum (-1)^i s| e |(1/N) sum (-1)^j s|."""
    L = lattice.shape[0]
    tx = 0
    ty = 0
    for i in range(L):
        for j in range(L):
            v = lattice[i, j]
            tx += v if i % 2 == 0 else -v
            ty += v if j % 2 == 0 else -v
    return abs(tx) / (L * L), abs(ty) / (L * L)


@njit
def stripe_order(lattice):
    """Q = sqrt(sx^2 + sy^2): parametro d'ordine a strisce (1 nello stato a strisce, qualunque orientamento)."""
    sx, sy = stripe_magnetizations(lattice)
    return np.sqrt(sx * sx + sy * sy)


# --------------------------------------------------------------------------
# Energia
# --------------------------------------------------------------------------
@njit
def energy(lattice, h, J2):
    """Energia TOTALE: primi e secondi vicini, ogni legame contato una volta."""
    L = lattice.shape[0]
    nn = 0
    nnn = 0
    for i in range(L):
        for j in range(L):
            s = lattice[i, j]
            nn += s * lattice[(i + 1) % L, j] + s * lattice[i, (j + 1) % L]
            nnn += s * lattice[(i + 1) % L, (j + 1) % L] + s * lattice[(i + 1) % L, (j - 1) % L]
    return -(J / 4.0) * nn - (J2 / 4.0) * nnn - h * lattice.sum()


@njit
def energy_per_spin(lattice, h, J2):
    L = lattice.shape[0]
    return energy(lattice, h, J2) / (L * L)


# --------------------------------------------------------------------------
# Dinamica di Glauber
# --------------------------------------------------------------------------
@njit
def delta_energy(lattice, i, j, h, J2):
    """Variazione di energia se si flippa lo spin (i, j):  dE = (J/2) s nb + (J2/2) s nnb + 2 h s."""
    L = lattice.shape[0]
    s = lattice[i, j]
    nb = (
        lattice[(i + 1) % L, j]
        + lattice[(i - 1) % L, j]
        + lattice[i, (j + 1) % L]
        + lattice[i, (j - 1) % L]
    )
    nnb = (
        lattice[(i + 1) % L, (j + 1) % L]
        + lattice[(i + 1) % L, (j - 1) % L]
        + lattice[(i - 1) % L, (j + 1) % L]
        + lattice[(i - 1) % L, (j - 1) % L]
    )
    return (J / 2.0) * s * nb + (J2 / 2.0) * s * nnb + 2.0 * h * s


@njit
def attempt(lattice, T, h, J2):
    """UN singolo tentativo di flip (sito casuale, rate di Glauber). sweep() = L*L chiamate."""
    L = lattice.shape[0]
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    dE = delta_energy(lattice, i, j, h, J2)
    if np.random.random() < flip_probability(dE, T):
        lattice[i, j] = -lattice[i, j]


@njit
def sweep(lattice, T, h, J2):
    """Un MC sweep = L*L tentativi di flip su siti casuali. Modifica lattice."""
    L = lattice.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        dE = delta_energy(lattice, i, j, h, J2)
        if np.random.random() < flip_probability(dE, T):
            lattice[i, j] = -lattice[i, j]


# --------------------------------------------------------------------------
# Simulazioni "ensemble" (n_rep repliche indipendenti)
# --------------------------------------------------------------------------
@njit
def run_relaxation(L, T, h, J2, n_sweeps, n_rep, init):
    """Evoluzione temporale a T, h, J2 fissati: array (n_rep, n_sweeps+1) di M, S, Q, e a ogni sweep
    (la colonna 0 e' lo stato iniziale)."""
    M = np.empty((n_rep, n_sweeps + 1))
    S = np.empty((n_rep, n_sweeps + 1))
    Q = np.empty((n_rep, n_sweeps + 1))
    E = np.empty((n_rep, n_sweeps + 1))
    for r in range(n_rep):
        lattice = make_lattice(L, init)
        M[r, 0] = magnetization(lattice)
        S[r, 0] = staggered_magnetization(lattice)
        Q[r, 0] = stripe_order(lattice)
        E[r, 0] = energy_per_spin(lattice, h, J2)
        for t in range(n_sweeps):
            sweep(lattice, T, h, J2)
            M[r, t + 1] = magnetization(lattice)
            S[r, t + 1] = staggered_magnetization(lattice)
            Q[r, t + 1] = stripe_order(lattice)
            E[r, t + 1] = energy_per_spin(lattice, h, J2)
    return M, S, Q, E


@njit
def equilibrium_point(L, T, h, J2, n_burn, n_meas, n_rep, init):
    """Stima di <M>, <S>, <e> all'equilibrio a (T, h, J2): n_burn sweep scartati, poi media temporale su
    n_meas sweep, per ogni replica. Ritorna (n_rep, 4) con colonne [M, S, Q, e]."""
    out = np.empty((n_rep, 4))
    for r in range(n_rep):
        lattice = make_lattice(L, init)
        for _ in range(n_burn):
            sweep(lattice, T, h, J2)
        m = 0.0
        s = 0.0
        q = 0.0
        e = 0.0
        for _ in range(n_meas):
            sweep(lattice, T, h, J2)
            m += magnetization(lattice)
            s += staggered_magnetization(lattice)
            q += stripe_order(lattice)
            e += energy_per_spin(lattice, h, J2)
        out[r, 0] = m / n_meas
        out[r, 1] = s / n_meas
        out[r, 2] = q / n_meas
        out[r, 3] = e / n_meas
    return out


# --------------------------------------------------------------------------
# Traiettorie e protocolli fuori equilibrio (tempo in singoli tentativi di flip)
# --------------------------------------------------------------------------
@njit
def prepare_ensemble(L, T, h, J2, n_burn, n_rep, init):
    """n_rep realizzazioni all'equilibrio a (T, h, J2): partenza da 'init' e n_burn sweep.
    Ritorna un array int8 (n_rep, L, L)."""
    ens = np.empty((n_rep, L, L), dtype=np.int8)
    for r in range(n_rep):
        lattice = make_lattice(L, init)
        for _ in range(n_burn):
            sweep(lattice, T, h, J2)
        for i in range(L):
            for j in range(L):
                ens[r, i, j] = lattice[i, j]
    return ens


@njit
def quench_from_ensemble(ens, T, h, J2, rec_attempts):
    """Da ogni realizzazione di 'ens' evolve a temperatura T e registra M, S, Q, e quando il numero di
    tentativi di flip raggiunge rec_attempts (int64 crescente, il primo puo' essere 0).
    Ritorna quattro array (n_rep, len(rec_attempts)): M, S, Q, E."""
    n_rep = ens.shape[0]
    n_rec = len(rec_attempts)
    M = np.empty((n_rep, n_rec))
    S = np.empty((n_rep, n_rec))
    Q = np.empty((n_rep, n_rec))
    E = np.empty((n_rep, n_rec))
    for r in range(n_rep):
        lattice = ens[r].astype(np.int64)
        done = 0
        for k in range(n_rec):
            while done < rec_attempts[k]:
                attempt(lattice, T, h, J2)
                done += 1
            M[r, k] = magnetization(lattice)
            S[r, k] = staggered_magnetization(lattice)
            Q[r, k] = stripe_order(lattice)
            E[r, k] = energy_per_spin(lattice, h, J2)
    return M, S, Q, E


@njit
def run_schedule(ens, T1, t_switch, T2, h, J2, rec_attempts):
    """Protocollo a due stadi: bagno a T1 per i primi t_switch tentativi, poi a T2.
    oven: t_switch = 0, T2 = T_max;  precooling: T1 = T_cold, t_switch = tau, T2 = T_max."""
    n_rep = ens.shape[0]
    n_rec = len(rec_attempts)
    M = np.empty((n_rep, n_rec))
    S = np.empty((n_rep, n_rec))
    Q = np.empty((n_rep, n_rec))
    E = np.empty((n_rep, n_rec))
    for r in range(n_rep):
        lattice = ens[r].astype(np.int64)
        done = 0
        for k in range(n_rec):
            target = rec_attempts[k]
            while done < target and done < t_switch:
                attempt(lattice, T1, h, J2)
                done += 1
            while done < target:
                attempt(lattice, T2, h, J2)
                done += 1
            M[r, k] = magnetization(lattice)
            S[r, k] = staggered_magnetization(lattice)
            Q[r, k] = stripe_order(lattice)
            E[r, k] = energy_per_spin(lattice, h, J2)
    return M, S, Q, E