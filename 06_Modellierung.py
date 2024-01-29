import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Daten laden
df = pd.read_excel('02_data/02_processed/transformed_data.xlsx')


unique_psps = df['PSP'].unique()

# Trainings- und Testdaten aufteilen
X = df.drop(['success', 'PSP'], axis=1)
y = df['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter für Random Forest
param_distributions = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Ergebnistabelle und Metriken initialisieren
results_df = X_test.copy()
results_df['Actual_Success'] = y_test.values
psp_metrics = []

# Modelle für jeden PSP trainieren und Metriken berechnen
for psp in unique_psps:

    X_train_psp = X_train[df['PSP'] == psp]
    y_train_psp = y_train[df['PSP'] == psp]

    # RandomizedSearchCV zur Optimierung der Hyperparameter
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
    rf_random.fit(X_train_psp, y_train_psp)

    # Bestes Modell für den aktuellen PSP
    best_rf = rf_random.best_estimator_

    # Erfolgswahrscheinlichkeiten für die Testdaten berechnen
    results_df[f'Prob_Success_PSP_{psp}'] = best_rf.predict_proba(X_test)[:, 1]

    # Metriken für jedes Modell berechnen und speichern
    y_pred = best_rf.predict(X_test)
    psp_metrics.append({
        'PSP': psp,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    })

# Bestes PSP für jede Transaktion auswählen
results_df['Best_PSP'] = results_df[[f'Prob_Success_PSP_{psp}' for psp in unique_psps]].idxmax(axis=1)

# Ergebnisse speichern
results_df.to_excel('02_data/02_processed/results.xlsx', index=False)
metrics_df = pd.DataFrame(psp_metrics)
metrics_df.to_excel('02_data/02_processed/psp_model_metrics.xlsx', index=False)


print("Testdaten mit dem besten PSP und Metriken gespeichert.")
