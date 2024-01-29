import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Daten laden
file_path = '02_data/02_processed/processed_data.xlsx'
cost_file_path = '02_data/01_raw/PSP_Jan_Feb_2019_cost.xlsx'
output_file_path = '02_data/02_processed/transformed_data.xlsx'


data = pd.read_excel(file_path)

# Nicht benötigte Spalten entfernen
columns_to_drop = ['Unnamed: 0', 'transaction_id', 'tmsp', 'quarter', 'year', 'cost']
data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# kategoriale Spalten in Zahlen umwandeln
categorical_columns = ['country', 'PSP', 'card']
label_encoders = {column: LabelEncoder() for column in categorical_columns}
for column in categorical_columns:
    data[column] = label_encoders[column].fit_transform(data[column]) + 1

# Laden der Kosten-Daten
cost_data = pd.read_excel(cost_file_path)

# Merkmale (Features) und Zielvariable (Target) definieren
X = data.drop('success', axis=1)
y = data['success']

# Aufteilung der Daten in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Anwenden von SMOTE auf das Trainingsset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# SMOTE-transformierte Daten in DataFrame umwandeln
transformed_data = pd.DataFrame(X_train_smote, columns=X_train.columns)
transformed_data['success'] = y_train_smote


# Funktion zur Ermittlung der Kosten
def get_cost(row):
    psp = row['PSP']
    success = row['success']
    cost_row = cost_data[cost_data['PSP'] == psp]

    if success == 1:
        return cost_row['success=1'].iloc[0]
    else:
        return cost_row['success=0'].iloc[0]


# Kosteninformationen in die transformierten Daten einfügen
transformed_data['cost'] = transformed_data.apply(get_cost, axis=1)

# Speichern der transformierten Daten
transformed_data.to_excel(output_file_path, index=False)
print(
    f"Die mit SMOTE transformierten und um Kosten ergänzten Daten wurden erfolgreich gespeichert unter: {output_file_path}")

# Klassenverteilung nach der Anwendung von SMOTE anzeigen
print(y_train_smote.value_counts())
