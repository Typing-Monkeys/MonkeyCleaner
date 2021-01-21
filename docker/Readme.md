# Docker Container

Il metodo piu' veloce per far funzionare `tensorflow-gpu` su Linux e' tramite un Docker Container.
Seguire la guida sul sito di tensorflow e' un po problematico dato che non spiega bene alcune cose e soprattutto l'immagine `tensorflow/tensorflow:latest-gpu` non funziona ! Ho dovuto trovare quindi una versione che funziona e aggiungere librerie python e alcuni pacchetti necessari al fuznonamento.

## Dockerfile

Il `Dockerfile` e' pronto e configurato per far funzionare questo progetto. Si basa sull'immagine `tensorflow/tensorflow:2.2.1-gpu-py3` dove:

* `2.2.1` e' la versione di tensorflow
* `gpu` inidica che tensorflow funziona su gpu (arriva con i cuda e tutto il resto pronto)
* `py3` forza l'inclusione di python3 (senno' veniva incluso python2 come interprete di default e pip non funzionava sulla versione 3)

## Script

Lo script .sh serve per avviare il container basandosi sull'immagine generata a partire dal `Dockerfile`. L'unica operazione che va fatta e' modificare il valore della variabile `progetto` con il path assoluto della cartella del progetto.

## Installazione

1. Per prima cosa vanno installati i driver video proprietari di Nvidia (Manjaro arriva gia' con i driver proprietari o comunque permette l'installazione in modo semplice, quini e' altamente consigliato). Non e' necessario installare i Cuda.
2. Installare docker. Pacchetti come `nvidia-docker` sono deprecati e nelle nuove versioni docker include il supporto alle gpu. Il parametro `--runtime=nvidia` e' delle versioni vecchie di docker, le derivate Arch (tramite AUR) hanno versioni aggiornate che usano il nuovo parametro `--gpus all`. E' anche necessario installare il pacchetto `nvidia-container-toolkit`. In caso controllare le prime sezioni della guida [Tensorflow Docker](https://www.tensorflow.org/install/docker)
3. Ricordarsi di avviare il servizio docker !
4. Generare l'immagine a partire dal `Dockerfile` con il seguente comando: `docker build -t "monkey:Dockerfile" .`
5. Avviare lo script `startdocker.sh` e verificare che tutto funzioni !

