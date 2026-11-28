"""
test_ising_nnn.py  --  controlli rapidi (smoke test) di src/ising_nnn.py

  1. J2 = 0 riproduce src/ising.py: energia, delta_energy e anche la dinamica (stesso seme -> stesso reticolo)
  2. energie di stati noti con J2 != 0: Neel, tutti up, strisce; campo critico h_c = 1 indipendente da J2
  3. delta_energy coincide con la differenza E_dopo - E_prima per flip casuali, con J2 != 0
  4. un po' di dinamica a T bassa porta il sistema verso stati di energia piu' bassa
Uso:  python test_ising_nnn.py
"""
import numpy as np

from src import ising, ising_nnn as nnn

L = 8
ok = True


def check(name, cond):
    global ok
    print(("ok   " if cond else "FAIL ") + name)
    ok &= bool(cond)


rng = np.random.default_rng(1)

# ---- 1. J2 = 0 == modello base -------------------------------------------------------------
ising.set_seed(7)
lat = ising.initialize_lattice(L, "random")
h = 1.0025
check("J2 = 0: energia uguale a ising.energy", np.isclose(nnn.energy(lat, h, 0.0), ising.energy(lat, h)))
same = all(np.isclose(nnn.delta_energy(lat, i, j, h, 0.0), ising.delta_energy(lat, i, j, h))
           for i in range(L) for j in range(L))
check("J2 = 0: delta_energy uguale a ising.delta_energy su tutti i siti", same)

ising.set_seed(11)
a = ising.initialize_lattice(L, "random")
b = a.copy()
ising.set_seed(99)
for _ in range(20):
    ising.sweep(a, 0.4, h)
ising.set_seed(99)
for _ in range(20):
    nnn.sweep(b, 0.4, h, 0.0)
check("J2 = 0: stessa dinamica (stesso seme -> stesso reticolo dopo 20 sweep)", np.array_equal(a, b))

# ---- 2. stati noti --------------------------------------------------------------------------
neel = ising.initialize_lattice(L, "neel")
up = ising.initialize_lattice(L, "up")
stripes = np.where(np.arange(L)[None, :] % 2 == 0, 1, -1) * np.ones((L, 1), dtype=np.int64)   # colonne +,-,+,-
for J2 in (-0.3, 0.0, 0.2):
    for hh in (0.0, 0.7, 1.0):
        e_neel = nnn.energy_per_spin(neel, hh, J2)
        e_up = nnn.energy_per_spin(up, hh, J2)
        e_str = nnn.energy_per_spin(stripes, hh, J2)
        check(f"J2 = {J2:5.2f} h = {hh}: e_Neel = -0.5 - J2/2", np.isclose(e_neel, -0.5 - J2 / 2))
        check(f"J2 = {J2:5.2f} h = {hh}: e_up   =  0.5 - J2/2 - h", np.isclose(e_up, 0.5 - J2 / 2 - hh))
        check(f"J2 = {J2:5.2f} h = {hh}: e_strisce = J2/2", np.isclose(e_str, J2 / 2))
    check(f"J2 = {J2:5.2f}: Neel e tutti-up degeneri a h = 1 (h_c = 1)",
          np.isclose(nnn.energy_per_spin(neel, 1.0, J2), nnn.energy_per_spin(up, 1.0, J2)))
check("J2 = -0.6 < -0.5: le strisce battono il Neel a h = 0",
      nnn.energy_per_spin(stripes, 0.0, -0.6) < nnn.energy_per_spin(neel, 0.0, -0.6))
check("J2 = -0.3 > -0.5: il Neel batte le strisce a h = 0",
      nnn.energy_per_spin(neel, 0.0, -0.3) < nnn.energy_per_spin(stripes, 0.0, -0.3))
# costo di uno spin giu' isolato nello stato tutto-up: 2(h-1) + 2 J2
for J2 in (-0.3, 0.2):
    lat = up.copy()
    check(f"J2 = {J2:5.2f}: spin giu' isolato in tutto-up costa 2(h-1) + 2 J2",
          np.isclose(nnn.delta_energy(lat, 3, 3, 1.0025, J2), 2 * 0.0025 + 2 * J2))

# ---- 3. delta_energy == E_dopo - E_prima ------------------------------------------------------
worst = 0.0
for J2 in (-0.4, -0.2, 0.3):
    lat = ising.initialize_lattice(L, "random")
    for _ in range(300):
        i, j = int(rng.integers(0, L)), int(rng.integers(0, L))
        e0 = nnn.energy(lat, 0.37, J2)
        dE = nnn.delta_energy(lat, i, j, 0.37, J2)
        lat[i, j] = -lat[i, j]
        worst = max(worst, abs(nnn.energy(lat, 0.37, J2) - e0 - dE))
check(f"delta_energy == E_dopo - E_prima  (J2 != 0, 900 flip casuali; errore max {worst:.1e})", worst < 1e-9)

# ---- 4. dinamica: l'energia scende a bassa T ---------------------------------------------------
ising.set_seed(3)
lat = ising.initialize_lattice(L, "random")
e_start = nnn.energy_per_spin(lat, h, -0.2)
for _ in range(100):
    nnn.sweep(lat, 0.1, h, -0.2)
check("a T = 0.1 l'energia per spin diminuisce rispetto alla partenza casuale",
      nnn.energy_per_spin(lat, h, -0.2) < e_start)
out = nnn.equilibrium_point(L, 0.3, h, -0.2, 20, 20, 3, nnn.INIT["random"])
check("equilibrium_point restituisce (n_rep, 4) = [M, S, Q, e], finito", out.shape == (3, 4) and np.isfinite(out).all())

# ---- 5. J2 = -1: stati fondamentali, osservabili a strisce -------------------------------------
st = nnn.initialize_lattice(L, "stripe")
pl = nnn.initialize_lattice(L, "plateau")
check("init 'stripe' = colonne alternate: Q = 1, S = 0, M = 0",
      np.isclose(nnn.stripe_order(st), 1.0) and np.isclose(nnn.staggered_magnetization(st), 0.0)
      and np.isclose(nnn.magnetization(st), 0.0))
check("init 'plateau': M = 1/2, S = sx = sy = 1/2",
      np.isclose(nnn.magnetization(pl), 0.5) and np.isclose(nnn.staggered_magnetization(pl), 0.5)
      and all(np.isclose(v, 0.5) for v in nnn.stripe_magnetizations(pl)))
check("init 'neel': Q = 0", np.isclose(nnn.stripe_order(nnn.initialize_lattice(L, "neel")), 0.0))
for hh in (0.5, 1.5, 2.5):
    cand = {"stripe": nnn.energy_per_spin(st, hh, -1.0), "plateau": nnn.energy_per_spin(pl, hh, -1.0),
            "up": nnn.energy_per_spin(up, hh, -1.0), "neel": nnn.energy_per_spin(neel, hh, -1.0)}
    best = min(cand, key=cand.get)
    want = "stripe" if hh < 1 else ("plateau" if hh < 2 else "up")
    check(f"J2 = -1, h = {hh}: stato di minima energia tra i candidati = {want}", best == want)
check("J2 = -1: e_strisce = -0.5, e_plateau(h) = -h/2, e_up(h) = 1 - h",
      np.isclose(nnn.energy_per_spin(st, 1.3, -1.0), -0.5) and np.isclose(nnn.energy_per_spin(pl, 1.3, -1.0), -0.65)
      and np.isclose(nnn.energy_per_spin(up, 1.3, -1.0), 1 - 1.3))
ising.set_seed(5)
big = nnn.initialize_lattice(60, "random")
check("stato casuale L = 60: Q ~ 1.25/L (entro il 30%)", abs(nnn.stripe_order(big) - 1.2533 / 60) < 0.3 * 1.2533 / 60)

print("\nTUTTO OK" if ok else "\nCI SONO ERRORI")