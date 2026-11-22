# Precooling Strategy Allows Exponentially Faster Heating #
Il progetto ha come obiettivo lo sviluppo della numerica e delle simulazioni in Python per analizzare l'articolo di Gal & Raz PRL 124, 060602 (2020) "Precooling Strategy Allows Exponentially Faster Heating". In particolare, si applica la teoria ad un modello di Ising AFM con campo esterno e lo si riscalda tramite due protocolli ("oven" e "precooling"). Si analizza se effettivamente si riesce ad ottenere un riscaldamento esponenzialmente più veloce seguendo un preliminare protocollo di raffreddamento.\
Il codice è diviso in moduli indipendenti per migliorare la leggibilità e la chiarezza. Le immagini e i grafici principali ottenuti sono riportati di seguito.

## Struttura dei codici 
- `src/`: implementazione del modello
- `main_eq.py`: studio dell'equilibrio di Ising AFM
- `main_eq_locus.py`: implementazione del diagramma di fase e del locus di equilibrio
- `main_traiettorie.py`: studio delle traiettorie fuori equilibrio
- `main_protocols.py`: sviluppo dei protocolli oven e precooling
- `main_lineare.py`: grafici lineari e semilog degli andamenti esponenziali
- `figures/`: output delle simulazioni

## Analisi dell'equilibrio 
Studio dell'equilibrio del modello di Ising 2D AFM con campo su un reticolo LxL. Tutte le condizioni iniziali convergono allo stesso valore del punto della curva di equilibrio.\
A T_cold le configurazioni casuale e "all up" mostrano un delle curve sovrapponibili sia in M che in S. Il rilassamento è quindi dominato dalla distruzione dell'ordine staggered piuttosto che dalla crescita della magnetizzazione uniforme.

<p align="center">
  <img src="figures/ufficiali/relaxing.png" width="700">
  <br>
  <em>
  Figura 1: Rilassamento all'equilibrio
  </em>
</p>


<p align="center">
  <img src="figures/ufficiali/mappa_h_T.png" width="700">
  <br>
  <em>
  Figura 2: Mappa in funzione di h e T
  </em>
</p>

Dai grafici seguenti si nota che, per h < h_c = 1, il sistema ha una transizione di fase ordine-disordine. Sotto una temperatura critica T_c(h) è ordinato alla Néel, mentre sopra di essa il campo magnetizza gli spin aumentando M e distruggendo S. (La temperatura critica dell'antiferromagnete senza campo è 2.269·|J|/4 ≈ 0.57. Per h→1 T_c tende a zero, ecco perché per h=0.99 il salto è meno ampio)
<p align="center">
  <img src="figures/ufficiali/locus_h_70.png" width="700">
  <br>
  <em>
  Figura 3: Curve di equilibrio nel piano < M > vs < S > per diversi valori del campo h
  </em>
</p>

ALLA LUCE DI QUANTO OSSERVATO SI CERCA DI APPROFONDIRE (DA AGGIUNGERE)

<p align="center">
  <img src="figures/ufficiali/locus_equilibrio.png" width="700">
  <br>
  <em>
  Figura 4: Curva di equilibrio
  </em>
</p>

## Traiettorie fuori equilibrio 
<p align="center">
  <img src="figures/ufficiali/out_of_equilibrium.png" width="900">
  <br>
  <em>
  Figura 5: Traiettorie fuori equilibrio
  </em>
</p>


## Author 
Michele D'Urzo per l'esame di Sistemi Complessi Avanzati
