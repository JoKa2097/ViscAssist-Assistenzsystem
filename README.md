# ViscAssist-Assistenzsystem
Im IGF-Forschungsvorhaben F23179N ViscAssist wurde ein Produktionsassistenzystem zur Prädikation und Regelung der Viskositätskurve von Polyolefinen bei Rezyklateinsatz im Compoundieren entwickelt. Das folgende Repository enthält den Aufbau des Assistenzsystems und eine Datenbasis zum testen des Assistenzssystems. 

Das Assistenzssystem besteht aus drei Komponenten um die Viskosität vorherzusagen und eine Handlungsempfehlung auszusprechen. 
1. Softsensor in Form eines KNN zur Beschreibung des Ist-Zustands (Viskositätsvorhersage)
2. Analytische Mischungsmodelle zur Beschreibung des Soll-Zustands (Zielviskositätskurve)
3. Vergleich des Ist- und Soll-Zustands; bei Hoher Abweichung Optimeirung der Rezeptur und Handlungsempfehlung zur Rezepturanpassung für den Maschinenbediener

<img width="952" height="441" alt="grafik" src="https://github.com/user-attachments/assets/6ecd2931-e3e8-4e36-a145-6415ee6cc0f2" />

Es wurde eine Datenbank mit 4 Polypropylentypen, 1 Peroxidtyp und 3 PCR Polypropylentypen erarbeitet. 
Dazu wurde ein Transfermodell für Polyethylen entwickelt. Das Transfermodell wurde mit 1 Polyethylen und 2 PCR Polyethylentypen entwickelt. 

Der komplette Datensatz kann auf Anfrage erhalten werden. 

# Anleitung Nutzeroberfläche 

<img src="https://github.com/user-attachments/assets/d43b41fa-76c9-4c8a-8dc7-d529ce1d5285"
     alt="Datenbasis"
     width="200">

Hier ist die Polypropylen-IKV-Datenbasis hinterlegt. Hier kann auch eine eigene Datenbasis hinzugefügt werden.

<img src="https://github.com/user-attachments/assets/106ceb6c-3747-4417-bbf5-e163b1ca8be1"
     alt="Soll-Rezeptur"
     width="200">

Die Soll-Rezeptur ist die angestrebte Rezeptur, mit der normalerweise die Zielviskositäten erreicht werden. Die Summe der Mischungsanteile muss immer 1 ergeben. Peroxid kann prozentual hinzugefügt werden. Aus der Soll-Rezeptur wird die Soll-Kurve mithilfe analytischer Mischungsmodelle berechnet.

<img src="https://github.com/user-attachments/assets/dc129e8d-c902-47c4-9e5b-38cc3c1450c4"
     alt="Aktuelle Rezeptur"
     width="200">

Die aktuelle Rezeptur ist die Rezeptur, die gerade im Extruder verarbeitet wird. Sie dient als Information, um später Vorschläge zur Rezepturanpassung geben zu können, ist jedoch keine Berechnungsgrundlage für die Modelle.

<img src="https://github.com/user-attachments/assets/382d52c7-2507-4395-8858-122724a95fe9"
     alt="Prozessdaten"
     width="200">

Hier werden die aktuellen Prozessdaten eingetragen, die am Extruder abgelesen werden. Anhand dieser Prozessdaten wird mittels eines Softsensors (KNN) die Ist-Viskositätskurve berechnet.

<img src="https://github.com/user-attachments/assets/f62f3b93-a3b2-4123-ae26-983b7fd0e120"
     alt="Toleranzbereich"
     width="200">

Hier kann der Toleranzbereich festgelegt werden, innerhalb dessen die Kurven im Mittel voneinander abweichen dürfen.

<img src="https://github.com/user-attachments/assets/9b03df4e-59cc-49b4-b1fa-1ca78715ce47"
     alt="Soll-Ist-Vergleich"
     width="600">

Der Soll-Ist-Vergleich der berechneten Kurven wird grafisch dargestellt und zusätzlich quantitativ ausgewertet. 

<img src="https://github.com/user-attachments/assets/0aadaed7-bdfc-4f72-95dc-5827515b1a3c"
     alt="Rezepturanpassung"
     width="400">
     
Darauf aufbauend wird eine Rezepturempfehlung iterativ berechnet, mit welcher die Soll-Viskosität errreicht werden soll. Die Suche nach einer Optimalen Rezeptur erfolgt nur in den genutzten Rezepturen.  

# Übersicht zur Installation des Systems


