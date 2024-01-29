import pandas as pd
import matplotlib.pyplot as plt

# Laden der Daten
file_path = '02_data/01_raw/PSP_Jan_Feb_2019.xlsx'
data = pd.read_excel(file_path)

# Grafische Anzeige der Verteilung der Daten
columns_to_plot = ['country', 'PSP', '3D_secured', 'card']


fig, axs = plt.subplots(2, 2, figsize=(14, 10))


for i, column in enumerate(columns_to_plot):
    ax = axs[i//2, i%2]
    data[column].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f'Verteilung für {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Anzahl')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()


plt.show()


# Grafische Darstellung der Verteilung von amount in einem Histogram
mean_amount = data['amount'].mean()


plt.figure(figsize=(10, 6))
plt.hist(data['amount'], bins=50, alpha=0.7, label='Histogramm')
plt.axvline(mean_amount, color='r', linestyle='dashed', linewidth=2, label=f'Mittelwert: {mean_amount:.2f}')

plt.title('Histogramm der Beträge (amount) mit Mittelwert')
plt.xlabel('Betrag')
plt.ylabel('Häufigkeit')
plt.legend()
plt.grid(True)
plt.show()

# Erstellen eines 2x2 Plots mit Aufteilung nach 'success' für die angegebenen Kategorien
fig, axs = plt.subplots(2, 2, figsize=(14, 10))


categories = ['country', 'PSP', '3D_secured', 'card']


for i, category in enumerate(categories):
    ax = axs[i // 2, i % 2]


    category_counts = data.groupby([category, 'success']).size().unstack(fill_value=0)


    category_counts = category_counts.sort_values(by=1, ascending=False)


    category_counts.plot(kind='bar', stacked=True, ax=ax, color=['red', 'green'])

    ax.set_title(f'Verteilung für {category} nach Erfolg')
    ax.set_xlabel(category)
    ax.set_ylabel('Anzahl')
    ax.legend(['Kein Erfolg (0)', 'Erfolg (1)'], loc='upper right')

plt.tight_layout()
plt.show()