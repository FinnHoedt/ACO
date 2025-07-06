# Sprechzettel: Traveling Salesman Problem (TSP) und Ant Colony Optimization (ACO)

---

## 1. Definition des Traveling Salesman Problems (TSP) (ca. 1 Minute)

Das Traveling Salesman Problem (TSP) ist ein klassisches kombinatorisches Optimierungsproblem, bei dem eine Rundtour durch eine gegebene Menge von Knoten (Städten) gesucht wird.  
Ziel ist es, eine Reihenfolge der Besuche zu finden, sodass jede Stadt genau einmal besucht wird und die Gesamtreiselänge minimiert wird.  
Aufgrund der exponentiellen Anzahl möglicher Permutationen gehört das TSP zur Klasse der NP-schweren Probleme, für die keine polynomielle Lösung bekannt ist.

---

## 2. Grundprinzip der Ant Colony Optimization (ACO) (ca. 1 Minute)

Ant Colony Optimization ist ein metaheuristisches Verfahren, das vom kollektiven Suchverhalten von Ameisen inspiriert ist.  
In der Natur markieren Ameisen Pfade mit Pheromonen, die von anderen Ameisen verstärkt werden und so den kürzesten Weg zu einer Nahrungsquelle hervorheben.  
Dieses Verhalten wird algorithmisch simuliert, um iterativ qualitativ hochwertige Lösungen für Optimierungsprobleme wie das TSP zu generieren.

---

## 3. Algorithmusbeschreibung und mathematische Formulierung (ca. 3 Minuten)

### a) Initialisierung

Der Algorithmus beginnt mit der Initialisierung des Graphen, wobei auf allen Kanten ein Anfangswert für das Pheromon τ_ij gesetzt wird.  
Die Anzahl der künstlichen Ameisen wird meist proportional zur Anzahl der Knoten gewählt. Jede Ameise startet auf einem zufällig gewählten Knoten.

### b) Tourkonstruktion

Jede Ameise konstruiert sukzessive eine Tour, wobei die Auswahl des nächsten Knotens j aus der Menge der noch nicht besuchten Knoten mit einer Wahrscheinlichkeitsverteilung erfolgt:

```
p_ij = (τ_ij^α · η_ij^β) / (Σ_k∈unbesucht τ_ik^α · η_ik^β)
```

Hierbei ist τ_ij der Pheromonwert der Kante i → j und η_ij = 1/d_ij die Heuristik basierend auf der inversen Distanz d_ij.  
Die Parameter α und β steuern den relativen Einfluss von Pheromonspur und Heuristik.

### c) Pheromonaktualisierung

Nach Abschluss aller Touren erfolgt die Aktualisierung der Pheromone in zwei Schritten:

1. **Verdunstung:**
    ```
    τ_ij ← (1 - ρ) · τ_ij
    ```

mit der Verdunstungsrate ρ ∈ (0,1). Diese verhindert eine Überakkumulation von Pheromon und fördert die Diversität.

2. **Verstärkung:**
    ```
    τ_ij ← τ_ij + Σ_k=1^m Δτ_ij^k
    ```

mit

```
Δτ_ij^k = { Q/L_k, wenn Ameise k die Kante (i,j) benutzt
          { 0,     sonst
```

wobei L_k die Gesamtlänge der Tour der k-ten Ameise und Q ein konstanter Verstärkungsfaktor ist.

---

## 4. Wesentliche Parameter und deren Einfluss (ca. 0.5 Minuten)

| Parameter      | Bedeutung                              | Wirkung auf das Suchverhalten                                           |
| -------------- | -------------------------------------- | ----------------------------------------------------------------------- |
| α              | Einfluss der Pheromonspur              | Höhere Werte fördern Exploitation bekannter Pfade                       |
| β              | Einfluss der Heuristik (1/Distanz)     | Höhere Werte fördern Exploration nahegelegener Knoten                   |
| ρ              | Verdunstungsrate                       | Steuerung der Balance zwischen Vergessen und Lernen                     |
| Q              | Pheromonverstärkungskonstante          | Skaliert die Menge an Pheromon, die für gute Lösungen hinterlassen wird |
| Anzahl Ameisen | Anzahl der parallel agierenden Agenten | Beeinflusst Diversität und Konvergenzgeschwindigkeit                    |
| Iterationen    | Anzahl der Wiederholungen              | Bestimmt die Laufzeit und Qualität der Lösungen                         |

---

## 5. Eigene Implementierung – kurze Erläuterung (ca. 4–5 Minuten)

In meiner Implementierung ist die Auswahl des nächsten Knotens gemäß der Formel

```
p_ij ∝ τ_ij^α · (1/d_ij)^β
```

realisiert, wobei ich sicherstelle, dass Distanzen stets positiv sind, um Division durch Null zu vermeiden.

Das Pheromonupdate beinhaltet eine Verdunstung mit einem Faktor ρ (Decay) und eine additive Verstärkung proportional zu α/L, wobei L die Gesamtlänge der jeweiligen Tour ist.

Der Graph ist ungerichtet, daher werden Pheromonwerte symmetrisch auf beiden Richtungen der Kante aktualisiert.

Die Parameter α, β und ρ wurden empirisch validiert, um eine gute Balance zwischen Exploitation und Exploration zu gewährleisten.

Zur Veranschaulichung zeige ich im Folgenden Auszüge meines Codes sowie Ergebnisse der Laufzeit und Qualitätsentwicklung der gefundenen Lösungen.

---

## 6. Kurzes Fazit (optional)

Die Ant Colony Optimization ist ein leistungsfähiger Metaheuristik-Ansatz für das TSP, der natürliche Prozesse simuliert und durch adaptive Lernmechanismen qualitativ hochwertige Lösungen in akzeptabler Rechenzeit generiert.  
Die sorgfältige Parametrierung und die algorithmische Implementierung sind entscheidend für den Erfolg.
