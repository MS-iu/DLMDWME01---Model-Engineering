import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Laden der Daten
results_with_costs_path = '02_data/02_processed/results_with_costs.xlsx'
results_df = pd.read_excel(results_with_costs_path)

# Berechnung der Häufigkeiten der PSPs vor und nach der Kostenanpassung
psp_distribution_before = results_df['Best_PSP'].value_counts().sort_index()
psp_distribution_after = results_df['Best_PSP_Cost_Precision_Adjusted'].apply(lambda x: int(x.split('_')[-1])).value_counts().sort_index()

# Grafische Darstellung
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)


sns.barplot(ax=axes[0], x=psp_distribution_before.index, y=psp_distribution_before.values)
axes[0].set_title('Verteilung der PSPs vor Kostenanpassung')
axes[0].set_xlabel('PSP')
axes[0].set_ylabel('Anzahl der Transaktionen')


sns.barplot(ax=axes[1], x=psp_distribution_after.index, y=psp_distribution_after.values)
axes[1].set_title('Verteilung der PSPs nach Kostenanpassung')
axes[1].set_xlabel('PSP')

plt.tight_layout()
plt.show()


