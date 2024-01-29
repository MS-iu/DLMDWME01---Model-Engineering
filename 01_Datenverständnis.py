#Laden der Daten

import pandas as pd

def load_data(file_name):

    try:
        data = pd.read_excel(file_name)
        print("Daten erfolgreich geladen.")
        return data
    except Exception as e:
        print(f"Es gab einen Fehler beim Laden der Datei: {e}")
        return None


# Prüfen auf fehlende Werte
def data_quality_check(data):

    if data is not None:

        missing_values = data.isnull().sum()


        data_types = data.dtypes
        unique_values = data.nunique()

# Ausgabe der fehlenden Werte, der Datentypen und der eindeutigen Werte
        quality_check = pd.DataFrame({
            'Missing Values': missing_values,
            'Data Type': data_types,
            'Unique Values': unique_values
        })

        print(quality_check)
    else:
        print("Keine Daten zur Überprüfung verfügbar.")

# Main Funktion
def main():

    file_path = '02_data/01_raw/PSP_Jan_Feb_2019.xlsx'


    data = load_data(file_path)


    data_quality_check(data)

if __name__ == "__main__":
    main()