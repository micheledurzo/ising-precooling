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

Alla luce di quanto osservato per la transizione, si rimanda ad un approfondimento qualitativo sul size scaling all'appendice.

<p align="center">
  <img src="figures/ufficiali/transizione_ord_dis.png" width="800">
  <br>
  <em>
  Figura 4: Evoluzione verso l'equilibrio di M, S ed e per diversi valori del campo. Osservando S si nota la transizione di fase ordine-disordine quando h < h_c = 1.
  </em>
</p>

<p align="center">
  <img src="figures/ufficiali/locus_equilibrio.png" width="700">
  <br>
  <em>
  Figura 5: Curva di equilibrio
  </em>
</p>

## Traiettorie fuori equilibrio 
<p align="center">
  <img src="figures/ufficiali/fuori_eq_giusta.png" width="900">
  <br>
  <em>
  Figura 6: Traiettorie fuori equilibrio
  </em>
</p>


<p align="center">
  <img src="figures/ufficiali/M_S_fuori_eq.png" width="800">
  <br>
  <em>
  Figura 7: Evoluzione di M ed S durante le traiettorie fuori equilibrio 
  </em>
</p>

## Protocolli di riscaldamento
Allo scopo di cercare il tempo di precooling (tau) ottimale si effettua una scansione. Per ogni tau si misurano le medie di Delta_e e Delta_S in una finestra di tempo di riscaldamento \[2N, 6N]\ MC step dopo il cambio di temperatura a T_max. Viene scelto il tau che minimizza |<Delta_e>|.\
Come controllo di consistenza, dove |Delta_e| è minimo, |Delta_S| deve risultare molto piccolo e/o attraversare lo zero in un suo intorno.

<p align="center">
  <img src="figures/ufficiali/scan_tau.png" width="800">
  <br>
  <em>
  Figura 8: Scansione di tau (tempo di precooling)
  </em>
</p>


<p align="center">
  <img src="figures/ufficiali/giusta_lunga.png" width="800">
  <br>
  <em>
  Figura 9: Traiettorie dei protocolli rispetto al locus e distanza di energia dall'equilibrio finale
  </em>
</p>

Le fluttuazioni all'equilibrio risultano più "dense" appena raggiunto e diventano più rade oltre 10^5 MC step a causa di come è stata costruita la griglia dei tempi. Essi sono costituiti dall'unione di punti logaritmici e lineari fino a 30 N (147000 MC step), poi soltanto logaritmici fino a 200 N (980000 MC step).

<p align="center">
  <img src="figures/ufficiali/evol_prot.png" width="800">
  <br>
  <em>
  Figura 10: Evoluzione temporale di M ed S durante i protocolli
  </em>
</p>

<p align="center">
  <img src="figures/ufficiali/D_linear.png" width="800">
  <br>
  <em>
  Figura 11: Distanza di energia dall'equilibrio normalizzata
  </em>
</p>

<p align="center">
  <img src="figures/ufficiali/semilogy.png" width="800">
  <br>
  <em>
  Figura 11: Distanza di energia dall'equilibrio normalizzata in scala semilog-y
  </em>
</p>

Nel grafico accanto alle traiettorie fuori equilibrio il rumore della distanza di energia sul plateau è dell'ordine 10^(-4). In questo grafico, invece, la distanza viene normalizzata usando < e >(0), ovvero l'energia per spin media sull'insieme preparato a T0, nonché la stima di e_eq(T0). Normalizzando qui lo stesso rumore risulta amplificato di circa un fattore 26. 

## Discrepanza dal paper
Nell'articolo si utilizza T_cold = 0.0125 (non 0.00125) e tau = 1850. Facendo un'analisi con quella temperatura di precooling, però, la scansione del tau ottimale restituisce un valore decisamente più alto e non si osserva il vantaggio del protocollo a due step. Lo stesso risultato infelice si ottiene forzando tau ad essere pari a 1850 come da paper.\
Il dubbio è che nell'articolo vi sia un errore nella descrizione della didascalia che introduce il valore di T_cold. In effetti, il punto di equilibrio del locus corrispondente a T_cold = 0.00125 (usato qui) ha lo stesso valore di < M > del punto di equilibrio a T_cold = 0.0125 (usato nell'articolo), sebbene la < S > sia differente. Rimane aperta la questione.\
Di seguito vengono riportati i grafici scorretti ottenuti usando la T di precooling come da paper.

<p align="center">
  <img src="figures/prove/scan_tau_sbagliata.png" width="700">
  <br>
  <em>
  Figura 12: Scansione di tau scorretta (per T_cold = 0.0125) 
  </em>
</p>

<p align="center">
  <img src="figures/prova/Tc_0125_tau_scan.png" width="700">
  <br>
  <em>
  Figura 13: Traiettorie dei protocolli rispetto al locus e distanza di energia dall'equilibrio finale scorretti (per T_cold = 0.0125). Tau libero ottimizzato dalla scansione
  </em>
</p>

<p align="center">
  <img src="figures/prova/Tc_0125_tau_2508.png" width="700">
  <br>
  <em>
  Figura 14: Traiettorie dei protocolli rispetto al locus e distanza di energia dall'equilibrio finale scorretti (per T_cold = 0.0125). Tau fissato dall'ottimizzazione a T_cold = 0.00125
  </em>
</p>

## Appendice: size scaling qualitativo della transizione


## Author 
Michele D'Urzo per l'esame di Sistemi Complessi Avanzati
