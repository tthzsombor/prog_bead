import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class PlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Népességi adatok")

        # Keret az ábra megjelenítésére
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Gombok kerete
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Gombok hozzáadása
        self.prev_button = tk.Button(
            self.button_frame, text="Előző", command=self.prev_plot)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.next_button = tk.Button(
            self.button_frame, text="Következő", command=self.next_plot)
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Ábrák
        # Ezt egészítheted ki több ábrával is.
        self.figures = [self.Nepesseg(), self.Atlageletkor()]
        self.current_page = 0

        # Első ábra megjelenítése
        self.show_plot(self.figures[self.current_page])

    def Nepesseg(self):
        # Az adatok betöltése a feltöltött fájlból
        data = pd.read_csv(
            'nepesseg.csv', encoding='ISO-8859-2', delimiter=';')
        # Az első sor kihagyása (fejléc), és az oszlopok kiválasztása
        data_cleaned = data.iloc[1:, [0, 1, 2, 3]]
        data_cleaned.columns = ['Év', 'Férfi', 'Nő',
                                'Összesen']  # Az oszlopok átnevezése

        # Az oszlopok numerikus típusúvá alakítása
        data_cleaned['Év'] = pd.to_numeric(data_cleaned['Év'], errors='coerce')
        for col in ['Férfi', 'Nő', 'Összesen']:
            data_cleaned[col] = data_cleaned[col].str.replace(
                ' ', '').apply(pd.to_numeric, errors='coerce')

        data_cleaned = data_cleaned.dropna()

        # Regresszió a különböző oszlopokra
        x = data_cleaned['Év'].values.reshape(-1, 1)
        y_férfi = data_cleaned['Férfi'].values
        y_nő = data_cleaned['Nő'].values
        y_összesen = data_cleaned['Összesen'].values

        models = {}
        for key, y in zip(['Férfi', 'Nő', 'Összesen'], [y_férfi, y_nő, y_összesen]):
            model = LinearRegression()
            model.fit(x, y)
            models[key] = model

        # Predikció a 2050-es évre
        year_2050 = np.array([[2050]])
        prediction_férfi_2050 = models['Férfi'].predict(year_2050)[0]
        prediction_nő_2050 = models['Nő'].predict(year_2050)[0]
        prediction_összesen_2050 = models['Összesen'].predict(year_2050)[0]

        # Ábra létrehozása
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y_férfi, color='blue', label='Férfiak')
        ax.scatter(x, y_nő, color='red', label='Nők')
        ax.scatter(x, y_összesen, color='green', label='Összesen')

        # Hozzáadás regressziós vonalak
        for key, color, y in zip(['Férfi', 'Nő', 'Összesen'], ['blue', 'red', 'green'], [y_férfi, y_nő, y_összesen]):
            ax.plot(x, models[key].predict(x), color=color,
                    linestyle='-', label=f'{key} regresszió')
            
        
         # 2050-es predikciók megjelenítése közvetlenül az adatpontok mellett
        ax.text(2050, prediction_férfi_2050, f'{prediction_férfi_2050:.1f}', color='blue',
                ha='left', va='bottom', fontsize=15)
        ax.text(2050, prediction_nő_2050, f'{prediction_nő_2050:.1f}', color='red',
                ha='left', va='bottom', fontsize=15)
        ax.text(2050, prediction_összesen_2050, f'{prediction_összesen_2050:.1f}', color='green',
                ha='left', va='bottom', fontsize=15)
        
        # 2050-es évre vonatkozó előrejelzések kiírása
        ax.scatter([2050], [prediction_férfi_2050],
                   color='blue', s=40,)  # Férfiak 2050
        ax.scatter([2050], [prediction_nő_2050], color='red', s=40)  # Nők 2050
        ax.scatter([2050], [prediction_összesen_2050],
                   color='green', s=40)  # Összesen 2050

        ax.set_title('Népesség száma nemek szerint')
        ax.set_xlabel('Év')
        ax.set_ylabel('Népesség (ezer fő)')
        ax.set_xlim([data_cleaned['Év'].min(), 2070])
        ax.set_xticks(np.arange(data_cleaned['Év'].min(), 2060, step=10))
        ax.axvline(x=2050, color='black', linestyle='-')
        ax.legend()
        ax.grid(True)
        return fig



    # masodik
    def Atlageletkor(self):
        # Az adatok betöltése a feltöltött fájlból
        data = pd.read_csv(
            'nepesseg.csv', encoding='ISO-8859-2', delimiter=';')
        # Az első sor kihagyása (fejléc), és az oszlopok kiválasztása
        data_cleaned = data.iloc[1:, [0, 4, 5, 6]]
        data_cleaned.columns = ['Év', 'Férfi', 'Nő',
                                'Összesen']  # Az oszlopok átnevezése

        # Az oszlopok numerikus típusúvá alakítása
        data_cleaned['Év'] = pd.to_numeric(data_cleaned['Év'], errors='coerce')
        for col in ['Férfi', 'Nő', 'Összesen']:
            data_cleaned[col] = (data_cleaned[col]
                                 # Hiányzó adatok NaN-ra cserélése
                                 .replace('..', np.nan)
                                 # Szóközök eltávolítása
                                 .str.replace(' ', '', regex=False)
                                 # Decimális pontot használunk a vessző helyett
                                 .str.replace(',', '.', regex=False)
                                 .astype(float))  # Numerikus típusra alakítás

        data_cleaned = data_cleaned.dropna()

        # Regresszió a különböző oszlopokra
        x = data_cleaned['Év'].values.reshape(-1, 1)
        y_férfi = data_cleaned['Férfi'].values
        y_nő = data_cleaned['Nő'].values
        y_összesen = data_cleaned['Összesen'].values

        models = {}
        for key, y in zip(['Férfi', 'Nő', 'Összesen'], [y_férfi, y_nő, y_összesen]):
            model = LinearRegression()
            model.fit(x, y)
            models[key] = model

        # Predikció a 2050-es évre
        year_2050 = np.array([[2050]])  # 2050-es év numpy array formában
        prediction_férfi_2050 = models['Férfi'].predict(year_2050)[0]
        prediction_nő_2050 = models['Nő'].predict(year_2050)[0]
        prediction_összesen_2050 = models['Összesen'].predict(year_2050)[0]

        # Ábra létrehozása
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y_férfi, color='blue', label='Férfiak')
        ax.scatter(x, y_nő, color='red', label='Nők')
        ax.scatter(x, y_összesen, color='green', label='Összesen')

        # Hozzáadás regressziós vonalak
        for key, color, y in zip(['Férfi', 'Nő', 'Összesen'], ['blue', 'red', 'green'], [y_férfi, y_nő, y_összesen]):
            ax.plot(x, models[key].predict(x), color=color,
                    linestyle='-', label=f'{key} regresszió')

        # 2050-es predikciók megjelenítése közvetlenül az adatpontok mellett
        ax.text(2050, prediction_férfi_2050, f'{prediction_férfi_2050:.1f}', color='blue',
                ha='left', va='bottom', fontsize=15)
        ax.text(2050, prediction_nő_2050, f'{prediction_nő_2050:.1f}', color='red',
                ha='left', va='bottom', fontsize=15)
        ax.text(2050, prediction_összesen_2050, f'{prediction_összesen_2050:.1f}', color='green',
                ha='left', va='bottom', fontsize=15)

        # 2050-es évre vonatkozó előrejelzések kiírása
        ax.scatter([2050], [prediction_férfi_2050],
                   color='blue', s=30)  # Férfiak 2050
        ax.scatter([2050], [prediction_nő_2050], color='red', s=30)  # Nők 2050
        ax.scatter([2050], [prediction_összesen_2050],
                   color='green', s=30)  # Összesen 2050

        # Y tengely lépkedése
        ax.set_yticks(np.arange(int(data_cleaned['Férfi'].min()), int(
            prediction_összesen_2050) + 1, step=2))
        # Y-tengely címkék 0-tól 50-ig, 5-ös léptékkel
        ax.set_yticks(np.arange(20, 51, step=2))

        ax.set_title('Népesség átlag életkora nemek szerint')
        ax.set_xlabel('Év')
        ax.set_ylabel('Életkor')
        ax.set_xlim([data_cleaned['Év'].min(), 2060])
        ax.set_xticks(np.arange(data_cleaned['Év'].min(), 2060, step=10))
        ax.axvline(x=2050, color='black', linestyle='-')
        ax.legend()
        ax.grid(True)
        return fig

    def show_plot(self, fig):
        # Törli a korábbi tartalmat
        for widget in self.frame.winfo_children():
            widget.destroy()

        # Ábra ágyazása Tkinterbe
        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def next_plot(self):
        self.current_page = (self.current_page + 1) % len(self.figures)
        self.show_plot(self.figures[self.current_page])

    def prev_plot(self):
        self.current_page = (self.current_page - 1) % len(self.figures)
        self.show_plot(self.figures[self.current_page])


# Tkinter ablak létrehozása
root = tk.Tk()
app = PlotApp(root)
root.mainloop()
