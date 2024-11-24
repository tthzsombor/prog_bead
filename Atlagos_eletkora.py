# # Virtuális környezet létrehozása
# # python -m venv venv

# import pandas as pd  # Adatok betöltéséhez és feldolgozásához használt könyvtár
# # pip install pandas
# import matplotlib.pyplot as plt  # Adatok vizualizálásához szükséges könyvtár
# # pip install matplotlib
# from sklearn.linear_model import LinearRegression  # Lineáris regressziós modell készítéséhez
# # pip install scikit-learn  # Megjegyzés: A scikit-learn csomagot előbb telepíteni kell
# import numpy as np  # Matematikai számításokhoz szükséges könyvtár

# import matplotlib
# matplotlib.use('TkAgg')  # A TkAgg megjelenítő háttér használata interaktív grafikonhoz

# import tkinter
# print(tkinter.TkVersion)  # A tkinter telepítésének ellenőrzése, a verzió kiírásával

# # Az adatok betöltése a feltöltött fájlból
# data = pd.read_csv('nepesseg.csv', encoding='ISO-8859-2', delimiter=';')

# # Az első néhány sor megjelenítése a megfelelő struktúra ellenőrzésére
# print(data.head())  # Ellenőrzés céljából megjeleníti az adatkeret első öt sorát
# print(data.info())  # További információkat ad a fájlról

# # Releváns oszlopok kiválasztása
# # Feltételezve, hogy az 'Év' és a célváltozó (pl. népesség) az első két oszlop
# data_cleaned = data.iloc[1:, [0, 4, 5, 6]]  # Az első sor kihagyása (fejléc), és az oszlopok kiválasztása
# data_cleaned.columns = ['Év', 'Férfi', 'Nő', 'Összesen']  # Az oszlopok átnevezése

# # Az oszlopok numerikus típusúvá alakítása
# data_cleaned['Év'] = pd.to_numeric(data_cleaned['Év'], errors='coerce')  # Az 'Év' oszlop számformátumba alakítása
# # Férfi oszlop szöveggé alakítása, '..' helyettesítése 0-val, majd numerikusra konvertálás
# data_cleaned['Férfi'] = (data_cleaned['Férfi']
#     .replace('..', np.nan)  # Hiányzó adatok NaN-ra cserélése
#                .str.replace(' ', '', regex=False)  # Szóközök eltávolítása
#                .str.replace(',', '.', regex=False)  # Decimális pontot használunk a vessző helyett
#                .astype(float))  # Numerikus típusra alakítás


# # Nő oszlop szöveggé alakítása, '..' helyettesítése 0-val, majd numerikusra konvertálás
# data_cleaned['Nő'] = (data_cleaned['Nő']
#     .replace('..', np.nan)  # Hiányzó adatok NaN-ra cserélése
#                .str.replace(' ', '', regex=False)  # Szóközök eltávolítása
#                .str.replace(',', '.', regex=False)  # Decimális pontot használunk a vessző helyett
#                .astype(float))  # Numerikus típusra alakítás

# # Összesen oszlop szöveggé alakítása, '..' helyettesítése 0-val, majd numerikusra konvertálás
# data_cleaned['Összesen'] = (data_cleaned['Összesen']
#     .replace('..', np.nan)  # Hiányzó adatok NaN-ra cserélése
#                .str.replace(' ', '', regex=False)  # Szóközök eltávolítása
#                .str.replace(',', '.', regex=False)  # Decimális pontot használunk a vessző helyett
#                .astype(float))  # Numerikus típusra alakítás


# data_cleaned = data_cleaned.dropna()  # Csak a teljes adatokat tartalmazó sorokat hagyja meg




# # Az adatok előkészítése a regresszióhoz
# x = data_cleaned['Év'].values.reshape(-1, 1)  # Független változóként az 'Év' oszlopot használja

# # Lineáris regressziós modell létrehozása és betanítása a 'Férfi' változóra
# y_férfi = data_cleaned['Férfi'].values
# model_férfi = LinearRegression()
# model_férfi.fit(x, y_férfi)
# prediction_férfi_2030 = model_férfi.predict(np.array([[2030]]))[0]
# prediction_férfi_2050 = model_férfi.predict(np.array([[2050]]))[0]

# # Lineáris regressziós modell létrehozása és betanítása a 'Nő' változóra
# y_nő = data_cleaned['Nő'].values
# model_nő = LinearRegression()
# model_nő.fit(x, y_nő)
# prediction_nő_2030 = model_nő.predict(np.array([[2030]]))[0]
# prediction_nő_2050 = model_nő.predict(np.array([[2050]]))[0]

# # Lineáris regressziós modell létrehozása és betanítása az 'Összesen' változóra
# y_összesen = data_cleaned['Összesen'].values
# model_összesen = LinearRegression()
# model_összesen.fit(x, y_összesen)
# prediction_összesen_2030 = model_összesen.predict(np.array([[2030]]))[0]
# prediction_összesen_2050 = model_összesen.predict(np.array([[2050]]))[0]

# # Eredmények kiírása
# print(f"2030-ra jósolt férfiak átlag életkora: {int(prediction_férfi_2030)}")
# print(f"2030-ra jósolt nők átlag életkora: {int(prediction_nő_2030)}")
# print(f"2030-ra jósolt össz átlag életkor: {int(prediction_összesen_2030)}")
# print(f"2050-re jósolt férfiak átlag életkora: {int(prediction_férfi_2050)}")
# print(f"2050-re jósolt nők átlag életkora: {int(prediction_nő_2050)}")
# print(f"2050-re jósolt össz átlag életkor: {int(prediction_összesen_2050)}")



# # Vizualizáció
# plt.figure(figsize=(10, 6))  # Az ábra méretének beállítása
# plt.scatter(x, y_férfi, color='blue', label='Férfiak')  # Férfiak adatainak megjelenítése
# plt.scatter(x, y_nő, color='red', label='Nők')  # Nők adatainak megjelenítése
# plt.scatter(x, y_összesen, color='green', label='Összesen')  # Összesen adatainak megjelenítése

# # 2050-es évre vonatkozó előrejelzések kiírása
# plt.scatter([2050], [prediction_férfi_2050], color='blue', s=30)  # Férfiak 2050
# plt.scatter([2050], [prediction_nő_2050], color='red', s=30)  # Nők 2050
# plt.scatter([2050], [prediction_összesen_2050], color='green', s=30)  # Összesen 2050


# # Számok hozzáadása a pontok mellé (annotations)
# plt.annotate(f'{prediction_férfi_2050:.1f}', xy=(2050, prediction_férfi_2050), 
#              xytext=(5, 5), textcoords='offset points', ha='left', va='bottom', fontsize=10, color='blue')

# plt.annotate(f'{prediction_nő_2050:.1f}', xy=(2050, prediction_nő_2050), 
#              xytext=(5, 5), textcoords='offset points', ha='left', va='bottom', fontsize=10, color='red')

# plt.annotate(f'{prediction_összesen_2050:.1f}', xy=(2050, prediction_összesen_2050), 
#              xytext=(5, 5), textcoords='offset points', ha='left', va='bottom', fontsize=10, color='green')



# # Függőleges vonal húzása a 2050-es évnél
# plt.axvline(x=2050, color='black', linestyle='-')  # Vonal a 2050-es évnél

# # Szöveg hozzáadása a 2050-es évhez
# plt.xlim([data_cleaned['Év'].min(), 2060])  # Az x tengely határainak beállítása, hogy 2050 is szerepeljen
# plt.xticks(np.arange(data_cleaned['Év'].min(), 2051, step=10))  # Minden 5 évben egy címke

# #Y tengely
# plt.yticks(np.arange(int(data_cleaned['Férfi'].min()), int(prediction_összesen_2050) + 1, step=2))  # Y tengely egyesével



# plt.plot(x, model_férfi.predict(x), color='blue', linestyle='-', label='Férfiak regresszió')  # Férfiak regressziója
# plt.plot(x, model_nő.predict(x), color='red', linestyle='-', label='Nők regresszió')  # Nők regressziója
# plt.plot(x, model_összesen.predict(x), color='green', linestyle='-', label='Összesen regresszió')  # Összesen regressziója

# plt.title('Átlagos életkor nemek szerint')  # A cím
# plt.xlabel('Év')  # X tengely
# plt.ylabel('Életkor')  # Y tengely
# plt.legend()  # Jelmagyarázat
# plt.grid(True)  # Rácsvonalak
# plt.show()  # Ábra megjelenítése



