import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Erstellen der Korrelationsmatrix vor der Ausführung von SMOTE
file_path = '02_data/02_processed/transformed_data.xlsx'
data = pd.read_excel(file_path)


correlation_matrix = data.corr()


plt.figure(figsize=(12, 8))


sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})


plt.title('Korrelationsmatrix')
plt.show()
