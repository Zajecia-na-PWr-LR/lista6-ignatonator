# Laboratorium: Klasyfikacja danych niezrównoważonych

Autor: Bartosz Perz

W tym laboratorium wcielisz się w rolę analityka danych pracującego dla popularnej aplikacji randkowej `Habibi`. Twoim celem będzie przewidywanie, czy dwoje ludzi przypadnie sobie do gustu (tzw. "match"). Niestety "match" to zdarzenie rzadkie, przez co Twój poprzednik, Link**In-owy influencer AI, nie podołał zadaniu i model generował tylko losowe lub większościowe predykcje.

W laboratorium wykorzystasz zbiór danych *Speed Dating*:

```
Raymond Fisman; Sheena S. Iyengar; Emir Kamenica; Itamar Simonson.
Gender Differences in Mate Selection: Evidence From a Speed Dating Experiment.
The Quarterly Journal of Economics, Volume 121, Issue 2, 1 May 2006, Pages 673–697,
https://doi.org/10.1162/qjec.2006.121.2.673
```

Laboratorium składa się z dwóch części:
1. **Walka z dysproporcją klas** – nauka metod wykorzystywanych w pracy z danymi niezbalansowanymi.
2. **Kalibracja i koszty biznesowe** – przekładanie metryk na realne zadania.

Oprócz znanych Ci już bibliotek, w laboratorium wykorzystamy `imbalanced-learn`.

---

## CZĘŚĆ 1: Walka z dysproporcją klas

### Zadanie 1.1: Pobranie Danych i EDA
- Pobierz zbiór danych *Speed Dating* korzystając z `sklearn.datasets.fetch_openml` (id=40536).
- Wykonaj analizę danych i wygeneruj wizualizacje. Zwróć szczególną uwagę na rozkład zmiennych i zależności między nimi.


### Zadanie 1.2: Przygotowanie danych
- W świecie aplikacji randkowych nie możemy korzystać z wiedzy "z przyszłości". Usuń zmienne stanowiące wyciek wiedzy (np. decyzje, wrażenia z randki, zgadywanie sympatii), pozostawiając tylko atrybuty profilowe dostępne w momencie rekomendacji. Opis zmiennych znajdziesz w pobranym zbiorze w polu 'DESCR'.
- Stwórz "odcisk palca" (id) użytkownika na podstawie profilu, by pogrupować interakcje tego samego użytkownika (lub raczej typu użytkownika o danych preferencjach). Podziel dane na uczące i testowe w taki sposób, by w zbiorze testowym dany użytkownik miał dokładnie stałą liczbę (np. 5) prób. Nie dodawaj id do danych, trzymaj je w osobnej zmiennej.


### Zadanie 1.3: Model Bazowy i Wizualizacja Problemu
- Zaadresuj braki w danych.
- Wymyśl własne dodatkowe cechy np. różnica w ocenie atrakcyjności. Wytrenuj model na danych zawierających nowe cechy.
- Wyucz naiwne predyktory – większościowy i losowy.
- Wyucz podstawowy model (np. regresję logistyczną, drzewo decyzyjne) i wygeneruj `classification_report_imbalanced` (z biblioteki `imbalanced-learn`). Zaobserwuj, jak model radzi sobie z predykcjami dla klasy mniejszościowej.


### Zadanie 1.4: Techniki Resamplingu
Porównaj poniższe metody resamplingu danych. Zwizualizuj ich działanie przy pomocy t-SNE.

- Zwizualizuj przestrzeń danych przed zastosowaniem metod resamplingu (by przyspieszyć obliczenia wykorzystaj losowy podzbiór, np. 1500 próbek).
- Zastosuj algorytm SMOTE do wykonania oversamplingu klasy mniejszościowej (generowania syntetycznych przykładów).
- Zbuduj potok (`imblearn.pipeline.Pipeline`) łączący RandomUnderSampler oraz SMOTE.
- Spróbuj metody hybrydowej np. SMOTE+ENN.

Jak działają powyższe metody? Która daje lepsze rezultaty i dlaczego?


### Zadanie 1.5: Modele Ensemble dla Danych Niezbalansowanych
- Wykorzystaj algorytmy dedykowane dla problemu niezbalansowania klas z biblioteki `imbalanced-learn`: np. `BalancedRandomForestClassifier` lub `EasyEnsembleClassifier`. Wykorzystaj też `sklearn.ensemble.RandomForestClassifier` (ustaw parametr `class_weight`). Skonfrontuj ich wyniki z poprzednimi podejściami. Które algorytmy radzą sobie lepiej i dlaczego?
- Narysuj krzywe ROC (Receiver Operating Characteristic) oraz Precision-Recall. Oblicz pola pod krzywymi (AUC, AP). Która krzywa lepiej oddaje działanie algorytmu w zadaniu rekomendacji potencjalnych par?

---

## CZĘŚĆ 2: Kalibracja Modeli i Macierz Kosztów

### Zadanie 2.1: Ewaluacja w kontekście aplikacji

W aplikacjach randkowych pokazujemy danemu użytkownikowi profile w "jakiejś" kolejności. Dobrze byłoby jednak, by użytkownik szybko zobaczył potencjalne "Matche" (zanim się zniechęci). 

- Posortuj predykcje i wylicz interesujące wskaźniki rekomendacyjne, np. **Precision@K**, **Mean Reciprocal Rank (MRR)**. 

Te wskaźniki są często ważniejsze niż globalne miary dokładności. Jak myślisz, dlaczego?

### Zadanie 2.2: Strategia "Model Romantyczny"
W tej strategii bierzemy znalezienie partnera za cel nadrzędny, bez oglądania się na koszty. Ogólnie rozumiana dokładność jest tutaj celem drugorzędnym.
- Stwórz model, który pozwoli zmaksymalizować szansę na znalezienie partnera przez użytkownika. 

### Zadanie 2.3: Strategia "Model Biznesowy"
O ile model romantyczny jest pięknym założeniem, biznes rządzi się swoimi prawami. Każdy false positive to potencjalnie niezadowolony klient, który może odejść i wystawić złą recenzję, co w efekcie obniża zyski firmy. Chociaż znalezienie partnera również może spowodować odejście użytkownika (już nie potrzebuje aplikacji), to jednak zadowolony klient może zostawić pozytywną recenzję lub wrócić w przyszłości w razie rozstania. W efekcie firma musi znaleźć balans pomiędzy optymalizacją zysków a utrzymaniem jak największej liczby zadowolonych użytkowników.

- Opracuj macierz kosztów odwzorowującą powyższe założenia. Musisz określić koszty dla:
  - **Trafnego niepolecenia (TN)**
  - **Trafnego polecenia (TP)**
  - **Chybionego polecenia (FP)**
  - **Przeoczonego dopasowania (FN)**
- Stwórz model, który **maksymalizuje zysk** (lub **minimalizuje koszt**). Przy tworzeniu modelu wykorzystaj generowane prawdopodobieństwa i ustaw własną wartość progową (threshold).

### Zadanie 2.4: Podsumowanie Wyników
- Porównaj wyniki dwóch modeli ("model romantyczny" vs "model biznesowy") w jednej tabeli zawierającej wybrane przez Ciebie metryki, np.: `Recall`, `Precision`, `F1-Score`... 
- Porównaj zysk/stratę obu modeli.
- Czy "model biznesowy" różni się od "modelu romantycznego"? W jaki sposób zachowały się metryki?


## Przydatne źródła:

- Materiały z wykładu
- https://imbalanced-learn.org/stable/user_guide.html#user-guide
- https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_6_ImbalancedLearning/Introduction.html
- https://en.wikipedia.org/wiki/Mean_reciprocal_rank
- https://machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/
- Fernández, Alberto, et al. Learning from imbalanced data sets. Vol. 10. No. 2018. Cham: Springer, 2018.