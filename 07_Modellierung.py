import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Daten laden
df = pd.read_excel('02_data/02_processed/transformed_data.xlsx')
cost_data = pd.read_excel('02_data/01_raw/PSP_Jan_Feb_2019_cost.xlsx')  # Kosten-Daten laden

# Berechnung der Kostengewichtung mit Bevorzugung niedriger Kosten
cost_data['Cost_Weight'] = 1.5 / (cost_data['success=1'] + cost_data['success=0'])


unique_psps = cost_data['PSP'].unique()

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

    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=10, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(X_train_psp, y_train_psp)

    best_rf = rf_random.best_estimator_

    results_df[f'Prob_Success_PSP_{psp}'] = best_rf.predict_proba(X_test)[:, 1]

    y_pred = best_rf.predict(X_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    psp_metrics.append({
        'PSP': psp,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision,
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    })


metrics_df = pd.DataFrame(psp_metrics)

# Bestes PSP für jede Transaktion auswählen
prob_cols = [f'Prob_Success_PSP_{psp}' for psp in unique_psps]
results_df['Best_PSP'] = results_df[prob_cols].idxmax(axis=1)
results_df['Best_PSP'] = results_df['Best_PSP'].str.replace('Prob_Success_PSP_', '').astype(int)

# Integration der Kostengewichtung und der Präzisionswerte
results_df = results_df.merge(cost_data[['PSP', 'Cost_Weight']], how='left', left_on='Best_PSP', right_on='PSP')
results_df = results_df.merge(metrics_df[['PSP', 'Precision']], how='left', left_on='Best_PSP', right_on='PSP')

# Hinzufügen von Kostengewichtung und Präzisionswerten für jedes PSP
for psp in unique_psps:
    prob_col = f'Prob_Success_PSP_{psp}'
    cost_weight_col = f'Cost_Weight_PSP_{psp}'
    precision_col = f'Precision_PSP_{psp}'

    results_df[cost_weight_col] = cost_data.loc[cost_data['PSP'] == psp, 'Cost_Weight'].values[0]
    results_df[precision_col] = metrics_df.loc[metrics_df['PSP'] == psp, 'Precision'].values[0]

    results_df[f'Weighted_Metric_PSP_{psp}'] = results_df[prob_col] * results_df[cost_weight_col] * \
                                               (1/10*results_df[precision_col]**4)


metric_cols = [f'Weighted_Metric_PSP_{psp}' for psp in unique_psps]
results_df['Best_PSP_Cost_Precision_Adjusted'] = results_df[metric_cols].idxmax(axis=1)


results_df.to_excel('02_data/02_processed/results_with_costs.xlsx', index=False)
metrics_df.to_excel('02_data/02_processed/psp_model_metrics_with_costs.xlsx', index=False)


# Grafische Darstellung der Metriken
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Metriken für Modelle pro PSP')


sns.barplot(ax=axes[0, 0], x='PSP', y='Accuracy', data=metrics_df)
axes[0, 0].set_title('Genauigkeit')
axes[0, 0].set_xlabel('PSP')
axes[0, 0].set_ylabel('Genauigkeit')


sns.barplot(ax=axes[0, 1], x='PSP', y='Precision', data=metrics_df)
axes[0, 1].set_title('Präzision')
axes[0, 1].set_xlabel('PSP')
axes[0, 1].set_ylabel('Präzision')


sns.barplot(ax=axes[1, 0], x='PSP', y='Recall', data=metrics_df)
axes[1, 0].set_title('Recall')
axes[1, 0].set_xlabel('PSP')
axes[1, 0].set_ylabel('Recall')


sns.barplot(ax=axes[1, 1], x='PSP', y='F1 Score', data=metrics_df)
axes[1, 1].set_title('F1 Score')
axes[1, 1].set_xlabel('PSP')
axes[1, 1].set_ylabel('F1 Score')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Testdaten mit dem besten PSP und Metriken gespeichert.")