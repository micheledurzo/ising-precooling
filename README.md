# Precooling Strategy Allows Exponentially Faster Heating #
Il progetto ha come obiettivo lo sviluppo della numerica e delle simulazioni in Python per analizzare l'articolo di Gal & Raz PRL 124, 060602 (2020) "Precooling Strategy Allows Exponentially Faster Heating". In particolare, si applica la teoria ad un modello di Ising AFM con campo esterno e lo si riscalda tramite due protocolli ("oven" e "precooling"). Si analizza se effettivamente si riesce ad ottenere un riscaldamento esponenzialmente più veloce seguendo un preliminare protocollo di raffreddamento.\
Il codice è diviso in moduli indipendenti per migliorare la leggibilità e la chiarezza. Le immagini e i grafici principali ottenuti sono riportati di seguito.

## Analisi dell'equilibrio ##
![Rilassamento all'equilibrio](figures/rilassamento.png)
![Curve di equilibrio nel piano <M> vs <S> per diversi valori del campo h](figures/eq_locus per h)
![Curva di equilibrio per L=70 e h=1.0025](figures/equilibrio L=70)

## Traiettorie fuori equilibrio ##
![Traiettorie fuori equilibrio](figures/traiettorie_fuori_eq)

## Struttura dei codici ##
- `src/`: implementazione del modello
- `main_eq.py`: studio dell'equilibrio di Ising AFM
- `main_eq_locus.py`: implementazione del diagramma di fase e del locus di equilibrio
- `main_traiettorie.py`: studio delle traiettorie fuori equilibrio
- `figures/`: output delle simulazioni
- 
## Author ##
Michele D'Urzo per l'esame di Sistemi Complessi Avanzati
