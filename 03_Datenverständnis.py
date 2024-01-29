import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# Laden der Daten
file_path = '02_data/01_raw/PSP_Jan_Feb_2019.xlsx'
data = pd.read_excel(file_path)

# Aufteilen des Timestamp
data['tmsp'] = pd.to_datetime(data['tmsp'])
data['minute'] = data['tmsp'].dt.minute
data['hour'] = data['tmsp'].dt.hour
data['day'] = data['tmsp'].dt.day
data['month'] = data['tmsp'].dt.month
data['year'] = data['tmsp'].dt.year
data['weekday'] = data['tmsp'].dt.weekday
data['quarter'] = data['tmsp'].dt.quarter

# Suche nach zusammenhängenden Transaktionen
data_sorted = data.sort_values(by=['country', 'amount', 'tmsp'])


data_sorted['transaction_id'] = (data_sorted['country'] != data_sorted['country'].shift()) | \
                                (data_sorted['amount'] != data_sorted['amount'].shift()) | \
                                ((data_sorted['tmsp'] - data_sorted['tmsp'].shift()) > timedelta(minutes=1))
data_sorted['transaction_id'] = data_sorted['transaction_id'].cumsum()


transaction_summary = data_sorted.groupby('transaction_id').agg(
    attempts=('success', 'count'),
    success=('success', 'max')
)


attempt_counts = transaction_summary.groupby(['attempts', 'success']).size().unstack(fill_value=0)
total_unsuccessful = transaction_summary[transaction_summary['success'] == 0].shape[0]
total_successful = transaction_summary[transaction_summary['success'] == 1].shape[0]

# Darstellung der Zusammenhänge
fig, ax = plt.subplots(figsize=(10, 6))
attempt_counts[1].plot(kind='bar', ax=ax, color='green', position=0, width=0.4, label='Erfolg')
attempt_counts[0].plot(kind='bar', ax=ax, color='red', position=1, width=0.4, label='Kein Erfolg')

ax.set_xlabel('Anzahl Versuche')
ax.set_ylabel('Anzahl Transaktionen')
ax.set_title('Anzahl zusammenhängende Transaktionsversuche')
ax.legend()
plt.xticks(rotation=0)

ax.annotate(f'Anzahl Transaktionsgruppen Erfolg: {total_unsuccessful}',
            xy=(0.5, 0.97), xycoords='axes fraction',
            ha='center', va='center',
            bbox=dict(boxstyle='round', fc='floralwhite', ec='gray'))

ax.annotate(f'Anzahl Transaktionsgruppen Kein Erfolg: {total_successful}',
            xy=(0.5, 0.92), xycoords='axes fraction',
            ha='center', va='center',
            bbox=dict(boxstyle='round', fc='floralwhite', ec='gray'))

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


fees_data = pd.read_excel(file_path, sheet_name='Tabelle1')


successful_fees = dict(zip(fees_data['PSP'], fees_data['Gebühr für erfolgreich']))
failed_fees = dict(zip(fees_data['PSP'], fees_data['Gebühr für Fehlgeschlagen']))


def calculate_cost(row):
    if row['success'] == 1:
        return successful_fees.get(row['PSP'], 0)
    else:
        return failed_fees.get(row['PSP'], 0)


data_sorted['cost'] = data_sorted.apply(calculate_cost, axis=1)


output_file_path = '02_data/02_processed/processed_data.xlsx'
data_sorted.to_excel(output_file_path, index=False)


data_sorted['cost'] = data_sorted.apply(calculate_cost, axis=1)


costs = data_sorted.groupby('success')['cost'].sum()


plt.figure(figsize=(8, 6))
bars = plt.bar(costs.index, costs.values, color=['red', 'green'])


for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')  # va: vertical alignment

plt.title('Kosten für nicht erfolgreiche vs. erfolgreiche Transaktionen')
plt.xlabel('Transaktionsstatus (0 = Nicht erfolgreich, 1 = Erfolgreich)')
plt.ylabel('Gesamtkosten')
plt.xticks([0, 1], rotation=0)  # Horizontal x labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()


plt.show()


data_sorted['cost'] = data_sorted.apply(calculate_cost, axis=1)


transaction_summary = data_sorted.groupby('transaction_id').agg(
    attempts=('success', 'count'),
    success=('success', 'max'),
    total_cost=('cost', 'sum')
)


transaction_summary_filtered = transaction_summary[transaction_summary['attempts'] <= 10]


cost_summary = transaction_summary_filtered.groupby(['attempts', 'success'])['total_cost'].sum().unstack(fill_value=0)


fig, ax = plt.subplots(figsize=(10, 6))
cost_summary[1].plot(kind='bar', ax=ax, color='green', position=0, width=0.4, label='Erfolgreich')
cost_summary[0].plot(kind='bar', ax=ax, color='red', position=1, width=0.4, label='Nicht erfolgreich')

ax.set_xlabel('Anzahl der Versuche')
ax.set_ylabel('Gesamtkosten')
ax.set_title('Gesamtkosten pro Anzahl der Versuche für Transaction IDs (1 bis 10)')
ax.legend()
plt.xticks(rotation=0)


for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print(f"Die Datei wurde erfolgreich gespeichert unter: {output_file_path}")
