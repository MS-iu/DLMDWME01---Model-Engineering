Die in der Hausarbeit angegebene Ordnerstruktur wird nur teilweise genutzt. 
Die Verarbeitung der Daten findet in den Python Skripten 01 bis 08 statt:

01_Datenverständnis: Laden und grundlegende Prüfung der Daten
02_Datenverständnis: Grafische Darstellung grundlegendes Datenverständnis
03_Datenverständnis: Suche nach zusammenhängenden Transaktionen / Aufteilen des Zeitstempels
04_Datenaufbereitung: Datenumwandlung / Ergänzung der Kosten / Anwendung von SMOTE
05_Datenaufbereitung: Erstellen der Korrelationsmatrix (Muss vor SMOTE ausgeführt werden)
06_Modellierung: Basismodell ohne Gewichtsfunktion
07_Modellierung: Finales Modell mit Gewichtsfunktionen, Metriken und grafischer Auswertung
08_Evaluierung: Verteilung der Zuweisung vor und nach der Gewichtsfunktion

Folgende Dateien wurde geladen:
PSP_Jan_Feb_2019.xlsx: Rohdaten
PSP_Jan_Feb_2019_cost.xlsx: Kostenstruktur

Folgende Daten wurden im Laufe der Modellentwicklung erstellt:
processed_data.xlsx: Vorverarbeitete Daten
psp_model_metrics.xlsx: Metriken des Basismodells
psp_model_metrics_costs.xlsx: Metriken des finalen Modells
results.xlsx: Detailübersicht der Ergebnisse des Modells
results_with_costs.xlsx: Detailübersicht der Ergebnisse des Modells inkl. angewendete Kostenfunktion
transformed_data: Aufbereitete Daten