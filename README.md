# ViscAssist-Assistenzsystem
Im IGF-Forschungsvorhaben F23179N ViscAssist wurde ein Produktionsassistenzystem zur Prädikation und Regelung der Viskositätskurve von Polyolefinen bei Rezyklateinsatz im Compoundieren entwickelt. Das folgende Repository enthält den Aufbau des Assistenzsystems und eine Datenbasis zum testen des Assistenzssystems. 

Das Assistenzssystem besteht aus drei Komponenten um die Viskosität vorherzusagen und eine Handlungsempfehlung auszusprechen. 
1. Softsensor in Form eines KNN zur Beschreibung des Ist-Zustands (Viskositätsvorhersage)
2. Analytische Mischungsmodelle zur Beschreibung des Soll-Zustands (Zielviskositätskurve)
3. Vergleich des Ist- und Soll-Zustands; bei Hoher Abweichung Optimeirung der Rezeptur und Handlungsempfehlung zur Rezepturanpassung für den Maschinenbediener

<img width="952" height="441" alt="grafik" src="https://github.com/user-attachments/assets/6ecd2931-e3e8-4e36-a145-6415ee6cc0f2" />

Es wurde eine Datenbank mit 4 Polypropylentypen, 1 Peroxidtyp und 3 PCR Polypropylentypen erarbeitet. Der komplette Datensatz kann auf Anfrage erhalten werden. 
Dazu wurde ein Transfermodell für Polyethylen entwickelt. Das Transfermodell wurde mit 1 Polyethylen und 2 PCR Polyethylentypen entwickelt. 
