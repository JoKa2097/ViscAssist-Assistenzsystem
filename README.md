# ViscAssist-Assistenzsystem
Im IGF-Forschungsvorhaben F23179N ViscAssist wurde ein Produktionsassistenzystem zur Prädikation und Regelung der Viskositätskurve von Polyolefinen bei Rezyklateinsatz im Compoundieren entwickelt. Das folgende Repository enthält den Aufbau des Assistenzsystems und eine Datenbasis zum testen des Assistenzssystems. 

Das Assistenzssystem besteht aus drei Komponenten um die Viskosität vorherzusagen und eine Handlungsempfehlung auszusprechen. 
1. Softsensor in Form eines KNN zur Beschreibung des Ist-Zustands (Viskositätsvorhersage)
2. Analytische Mischungsmodelle zur Beschreibung des Soll-Zustands (Zielviskositätskurve)
3. Vergleich des Ist- und Soll-Zustands; bei Hoher Abweichung Optimierung der Rezeptur und Handlungsempfehlung zur Rezepturanpassung für den Maschinenbediener

<img width="952" height="441" alt="grafik" src="https://github.com/user-attachments/assets/6ecd2931-e3e8-4e36-a145-6415ee6cc0f2" />

Es wurde eine Datenbank mit 4 Polypropylentypen, 1 Peroxidtyp und 3 PCR Polypropylentypen erarbeitet. 
Dazu wurde ein Transfermodell für Polyethylen entwickelt. Das Transfermodell wurde mit 1 Polyethylen und 2 PCR Polyethylentypen entwickelt. 

Der komplette Datensatz kann auf Anfrage erhalten werden. 

## Anleitung Nutzeroberfläche 

### Modellwahl

<img src="https://github.com/user-attachments/assets/d43b41fa-76c9-4c8a-8dc7-d529ce1d5285"
     alt="Datenbasis"
     width="200">
     
Hier ist die Polypropylen-IKV-Datenbasis hinterlegt. Hier kann auch eine eigene Datenbasis hinzugefügt werden.

### Soll-Rezeptur

<img src="https://github.com/user-attachments/assets/106ceb6c-3747-4417-bbf5-e163b1ca8be1"
     alt="Soll-Rezeptur"
     width="200">
     
Die Soll-Rezeptur ist die angestrebte Rezeptur, mit der normalerweise die Zielviskositäten erreicht werden. Die Summe der Mischungsanteile muss immer 1 ergeben. Peroxid kann prozentual hinzugefügt werden. Aus der Soll-Rezeptur wird die Soll-Kurve mithilfe analytischer Mischungsmodelle berechnet.

### Ist-Rezeptur

<img src="https://github.com/user-attachments/assets/dc129e8d-c902-47c4-9e5b-38cc3c1450c4"
     alt="Aktuelle Rezeptur"
     width="200">

Die aktuelle Rezeptur ist die Rezeptur, die gerade im Extruder verarbeitet wird. Sie dient als Information, um später Vorschläge zur Rezepturanpassung geben zu können, ist jedoch keine Berechnungsgrundlage für die Modelle.

### aktuelle Prozessdaten des Extruders

<img src="https://github.com/user-attachments/assets/382d52c7-2507-4395-8858-122724a95fe9"
     alt="Prozessdaten"
     width="200">

Hier werden die aktuellen Prozessdaten eingetragen, die am Extruder abgelesen werden. Anhand dieser Prozessdaten wird mittels eines Softsensors (KNN) die Ist-Viskositätskurve berechnet.

### Wahl des Toleranzbereichs

<img src="https://github.com/user-attachments/assets/f62f3b93-a3b2-4123-ae26-983b7fd0e120"
     alt="Toleranzbereich"
     width="200">

Hier kann der Toleranzbereich festgelegt werden, innerhalb dessen die Kurven im Mittel voneinander abweichen dürfen.

### Soll-Ist-Vergleich

<img src="https://github.com/user-attachments/assets/9b03df4e-59cc-49b4-b1fa-1ca78715ce47"
     alt="Soll-Ist-Vergleich"
     width="600">

Der Soll-Ist-Vergleich der berechneten Kurven wird grafisch dargestellt und zusätzlich quantitativ ausgewertet. 


### Rezepturanpassung

<img src="https://github.com/user-attachments/assets/0aadaed7-bdfc-4f72-95dc-5827515b1a3c"
     alt="Rezepturanpassung"
     width="400">
     
Darauf aufbauend wird eine Rezepturempfehlung iterativ berechnet, mit welcher die Soll-Viskosität errreicht werden soll. Die Suche nach einer Optimalen Rezeptur erfolgt nur in den genutzten Rezepturen.  

## Übersicht zur Installation des Systems/ Ordnerstruktur

## Aufbau des Assistenzsystems

Die Anwendung ist in die Benutzeroberfläche (`app.py`) und die eigentliche Berechnungslogik im Verzeichnis `src/` aufgeteilt. Dadurch sind Darstellung, Modelle und Berechnungen voneinander getrennt.

```text
ViscAssist/
│
├── app.py
│   └── Streamlit-Benutzeroberfläche und Ablaufsteuerung
│
├── data/
│   └── Datenbasis für die Rezepturmodelle
│
├── models/
│   └── Trainiertes KNN und zugehörige Skalierer
│
└── src/
    ├── config.py
    ├── preprocess.py
    ├── inference.py
    ├── prediction.py
    ├── comparison.py
    ├── control_logic.py
    └── plotting.py


```markdown
### `preprocess.py` – Aufbereitung der Prozessdaten

Bereitet die in der Benutzeroberfläche eingegebenen Prozessdaten für den Softsensor auf.

Für das IKV-Modell werden folgende sechs Prozessgrößen verwendet:

- mittleres Drehmoment
- mittlerer Druck
- gemessene Temperatur
- mittlere Drehzahl
- Volumenstrom
- Extrudertemperatur

Die Prozessgrößen werden in die für das KNN erforderliche Reihenfolge gebracht und mit dem beim Training verwendeten Input-Scaler skaliert.

### `inference.py` – Softsensor

Enthält das trainierte künstliche neuronale Netz zur Vorhersage der Ist-Viskositätskurve.

Das KNN erhält die sechs aufbereiteten Prozessgrößen als Eingangsgrößen und gibt die Viskosität an sechs definierten Scherraten aus. Neben der Modellarchitektur enthält das Modul die Funktionen zum Laden des trainierten Modells und der zugehörigen Skalierer sowie zur Rücktransformation der vorhergesagten Viskositätswerte.

### `prediction.py` – Berechnung der Soll-Viskositätskurve

Enthält die analytischen Mischungsmodelle zur Berechnung der Viskositätskurve aus einer vorgegebenen Rezeptur.

Aus der Datenbasis werden zunächst die Viskositätskurven der einzelnen Polymerkomponenten bestimmt. Für Polymermischungen wird daraus eine Mischungsviskosität berechnet. Zusätzlich wird der Einfluss der Peroxidzugabe (CR5P) über materialabhängige Modellparameter berücksichtigt.

Aus der in der Benutzeroberfläche angegebenen Soll-Rezeptur wird damit die Soll-Viskositätskurve bei den sechs Scherraten

51, 102, 204, 408, 815 und 1630 s⁻¹

berechnet.

### `comparison.py` – Soll-Ist-Vergleich

Vergleicht die durch den Softsensor vorhergesagte Ist-Viskositätskurve mit der aus der Soll-Rezeptur berechneten Soll-Viskositätskurve.

Hierfür werden unter anderem die absoluten und relativen Abweichungen sowie die mittlere absolute relative Abweichung berechnet. Anhand des vom Anwender vorgegebenen Toleranzbereichs wird anschließend bewertet, ob sich die Ist-Kurve noch innerhalb des zulässigen Bereichs befindet.

### `control_logic.py` – Rezepturanpassung

Verknüpft die einzelnen Modelle und enthält die Logik zur Ermittlung einer geeigneten Rezepturanpassung.

Zunächst wird aus der Soll-Rezeptur die Zielviskositätskurve berechnet. Die aktuelle Ist-Viskositätskurve wird mithilfe des Softsensors aus den Prozessdaten vorhergesagt und anschließend mit der Soll-Kurve verglichen.

Liegt die Abweichung außerhalb des vorgegebenen Toleranzbereichs, wird eine Rezepturanpassung gesucht. Die Suche erfolgt deterministisch und schrittweise von gröberen zu feineren Rezepturänderungen. Es werden ausschließlich Polymerkomponenten verändert, die auch in der Soll-Rezeptur enthalten sind. Materialien außerhalb dieses Rezepturraums bleiben unverändert.

Für mögliche Rezepturen wird die zu erwartende Viskositätskurve berechnet und bewertet. Ziel ist eine möglichst geringe Abweichung von der Soll-Viskositätskurve bei gleichzeitig möglichst kleiner Änderung der aktuellen Rezeptur.

### `plotting.py` – Visualisierung

Enthält die Funktionen zur grafischen Darstellung der berechneten Viskositätskurven.

Die Soll- und Ist-Viskositätskurven werden über der Scherrate doppellogarithmisch dargestellt und können dadurch direkt miteinander verglichen werden.



