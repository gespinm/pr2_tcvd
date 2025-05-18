## Català

### Integrants del grup:
- Guillem Espín Martí

### Arxius que trobareu en aquest repositori:
- README.md: Instruccións i infomració bàsica i rellevant del projecte.
- requirements.txt: Fitxer on emmagatzemem el versionat dels paquets necessaris per fer funcionar el projecte.
- Carpeta /source: Carpeta principal d'emmagatzemament del codi en Python.
  - scrapper.py: Script scrapper per la primera practica.
  - process.py: Script de process, analisi i emmagatzament del dataset (Practica 2).
- Carpeta /dataset: Datasets d'input i output en format CSV i PNG.
  - Medicaldataset.csv: Dataset en format CSV d'input de dades.
  - dataset_clean.csv: Dataset "Medicaldataset.csv" netejat.
  - plot_clean.png: Representació grafica del dataset netejat en format png.

### Instruccións d'ús:
#### Passos:
0- Instal·lar totes les dependències amb la següent comanda:
```
pip install -r requirements.txt
```
1- Executar el script fent ús de la següent comanda:
```
python .\source\process.py --save-csv --save-png
```
2- Esperar a visualitzar el missatge final d'execució per CLI:
```
Scrapping was completed sucecssfully
  - Data was succesfully processed
```
#### Opcions:
* --path: Permet afegir un path alternatiu per a processar un csv d'input diferent al que hi ha per defecte.
* --save-csv: Flag que ens permet emmagatzemar els resultats en csv localment, per defecte no s'emmagatzemen.
* --save-png: Flag que ens permet emmagatzemar els resultats en png localment, per defecte no s'emmagatzemen.

### Origen de les dades:
* https://www.kaggle.com/datasets/fatemehmohammadinia/heart-attack-dataset-tarik-a-rashid?resource=download
