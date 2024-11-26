import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class PlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Népességi adatok") # Ablak címének megadása
        self.root.configure(bg="#f0f0f0")  # Világos háttérszín az ablaknak

        # Keret az ábra megjelenítésére
        self.frame = tk.Frame(self.root, bg="#ffffff", bd=10, relief="solid", padx=20, pady=20)  # Keret szín és vastagság
        self.frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)  # Tömörítés az ablak és a keret között


        # Gombok kerete
        self.button_frame = tk.Frame(self.root, bg="#f0f0f0")  # világos háttérszín
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=20)  # Tömörítés a keret és az ablak alja között

        # Gombok hozzáadása
        self.prev_button = tk.Button(self.button_frame, text="Előző", command=self.prev_plot,
                                    font=('Arial', 12, 'bold'), bg='#007BFF', fg='white', relief="flat", bd=0,
                                    padx=15, pady=10, activebackground='#0056b3', activeforeground='white', width=10)
        self.prev_button.pack(side=tk.LEFT, padx=20)

        self.next_button = tk.Button(self.button_frame, text="Következő", command=self.next_plot,
                                    font=('Arial', 12, 'bold'), bg='#28a745', fg='white', relief="flat", bd=0,
                                    padx=15, pady=10, activebackground='#218838', activeforeground='white', width=10)
        self.next_button.pack(side=tk.RIGHT, padx=20)


        # Ábrák
        self.figures = [self.Nepesseg(), self.Diagram1(),self.Atlageletkor(), self.Diagram2()]
        self.current_page = 0

        # Első ábra megjelenítése
        self.show_plot(self.figures[self.current_page])
        

    def Nepesseg(self):
        # Az adatok betöltése a feltöltött fájlból
        data = pd.read_csv('nepesseg.csv', encoding='ISO-8859-2', delimiter=';')
        # Az első sor kihagyása (fejléc), és az oszlopok kiválasztása
        data_cleaned = data.iloc[1:, [0, 1, 2, 3]]
        data_cleaned.columns = ['Év', 'Férfi', 'Nő',
                                'Összesen']  # Az oszlopok átnevezése

        # Az oszlopok numerikus típusúvá alakítása
        data_cleaned['Év'] = pd.to_numeric(data_cleaned['Év'], errors='coerce')
        for col in ['Férfi', 'Nő', 'Összesen']:
            data_cleaned[col] = data_cleaned[col].str.replace(
                ' ', '').apply(pd.to_numeric, errors='coerce')

        # NAN törlése az adatok közül
        data_cleaned = data_cleaned.dropna()    

        # Adatok osztása 1000-rel (millió fő)
        for col in ['Férfi', 'Nő', 'Összesen']:
            data_cleaned[col] = data_cleaned[col] / 1000

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

       

        # Ábra létrehozása
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y_férfi, color='blue', label='Férfiak')
        ax.scatter(x, y_nő, color='red', label='Nők')
        ax.scatter(x, y_összesen, color='purple', label='Összesen')

        # Legnagyobb értékek dinamikus keresése
        max_férfi_value = data_cleaned['Férfi'].max()
        max_nő_value = data_cleaned['Nő'].max()
        max_összesen_value = data_cleaned['Összesen'].max()

        # A legnagyobb értékekhez tartozó évek keresése
        max_férfi_year = data_cleaned[data_cleaned['Férfi']== max_férfi_value]['Év'].values[0]
        max_nő_year = data_cleaned[data_cleaned['Nő']== max_nő_value]['Év'].values[0]
        max_összesen_year = data_cleaned[data_cleaned['Összesen']== max_összesen_value]['Év'].values[0]

        # A legnagyobb értékek kiemelése
        ax.scatter([max_férfi_year], [max_férfi_value], color='orange',s=50, label=f'Legmagasabb férfi népesség({max_férfi_year})')
        ax.scatter([max_nő_year], [max_nő_value], color='green',s=50, label=f'Legmagasabb női népesség({max_nő_year})')
        ax.scatter([max_összesen_year], [max_összesen_value], color='yellow',s=50, label=f'Legmagasabb népesség összesen({max_összesen_year})')


        ax.axhline(y=max_összesen_value, color='black', linestyle='--', label=f'Legnagyobb összes népesség: {max_összesen_value} M')

        
        # Hozzáadás regressziós vonalak
        for key, color, y in zip(['Férfi', 'Nő', 'Összesen'], ['blue', 'red', 'purple'], [y_férfi, y_nő, y_összesen]):
            ax.plot(x, models[key].predict(x), color=color,linestyle='-', label=f'{key} regresszió')
            
            
            
         # Predikció a 2050-es évre
        year_2050 = np.array([[2050]])
        prediction_férfi_2050 = models['Férfi'].predict(year_2050)[0]  # Ne osszuk le itt
        prediction_nő_2050 = models['Nő'].predict(year_2050)[0]  # Ne osszuk le itt
        prediction_összesen_2050 = models['Összesen'].predict(year_2050)[0]  
        
        # 2050-es predikciók megjelenítése közvetlenül az adatpontok mellett
        ax.text(2050, prediction_férfi_2050, f'{prediction_férfi_2050:.1f}', color='blue', ha='left', va='bottom', fontsize=15)
        ax.text(2050, prediction_nő_2050, f'{prediction_nő_2050:.1f}', color='red', ha='left', va='bottom', fontsize=15)
        ax.text(2050, prediction_összesen_2050, f'{prediction_összesen_2050:.1f}', color='purple', ha='left', va='bottom', fontsize=15)
        
        # 2050-es évre vonatkozó előrejelzések kiírása
        ax.scatter([2050], [prediction_férfi_2050],color='blue', s=40) 
        ax.scatter([2050], [prediction_nő_2050], color='red', s=40)  
        ax.scatter([2050], [prediction_összesen_2050],color='purple', s=40)  
        
        ax.axvline(x=2050, color='black', linestyle='-')
        
        
        
                
        # Predikció a 2030-es évre
        year_2030 = np.array([[2030]])
        prediction_férfi_2030 = models['Férfi'].predict(year_2030)[0] 
        prediction_nő_2030 = models['Nő'].predict(year_2030)[0] 
        prediction_összesen_2030 = models['Összesen'].predict(year_2030)[0]  
        
        # 2030-es predikciók megjelenítése közvetlenül az adatpontok mellett
        ax.text(2030, prediction_férfi_2030, f'{prediction_férfi_2030:.1f}', color='blue', ha='left', va='bottom', fontsize=15)
        ax.text(2030, prediction_nő_2030, f'{prediction_nő_2030:.1f}', color='red', ha='left', va='bottom', fontsize=15)
        ax.text(2030, prediction_összesen_2030, f'{prediction_összesen_2030:.1f}', color='purple', ha='left', va='top', fontsize=15)
        
        # 2030-es évre vonatkozó előrejelzések kiírása
        ax.scatter([2030], [prediction_férfi_2030],color='blue', s=40) 
        ax.scatter([2030], [prediction_nő_2030], color='red', s=40)  
        ax.scatter([2030], [prediction_összesen_2030],color='purple', s=40) 
        
        ax.axvline(x=2030, color='black', linestyle='-')
        
          
        # Predikció a 2100-es évre
        year_2100 = np.array([[2100]])
        prediction_férfi_2100 = models['Férfi'].predict(year_2100)[0]  # Ne osszuk le itt
        prediction_nő_2100 = models['Nő'].predict(year_2100)[0]  # Ne osszuk le itt
        prediction_összesen_2100 = models['Összesen'].predict(year_2100)[0]  
        
        # 2100-es predikciók megjelenítése közvetlenül az adatpontok mellett
        ax.text(2100, prediction_férfi_2100, f'{prediction_férfi_2100:.1f}', color='blue', ha='left', va='bottom', fontsize=15)
        ax.text(2100, prediction_nő_2100, f'{prediction_nő_2100:.1f}', color='red', ha='left', va='bottom', fontsize=15)
        ax.text(2100, prediction_összesen_2100, f'{prediction_összesen_2100:.1f}', color='purple', ha='left', va='bottom', fontsize=15)
        
        # 2100-es évre vonatkozó előrejelzések kiírása
        ax.scatter([2100], [prediction_férfi_2100],color='blue', s=40) 
        ax.scatter([2100], [prediction_nő_2100], color='red', s=40)  
        ax.scatter([2100], [prediction_összesen_2100],color='purple', s=40) 
        
        ax.axvline(x=2100, color='black', linestyle='-')
        


        # Y tengely lépkedése
        ax.set_yticks(np.arange(2, 12, step=0.5))

        ax.set_title('Népesség száma nemek szerint', fontsize=30, fontweight='bold')
        ax.set_xlabel('Év', fontsize=20)  # X tengely felirat formázása
        ax.set_ylabel('Népesség (millió fő)', fontsize=20)  # Y tengely felirat formázása
        ax.set_xlim([data_cleaned['Év'].min(), 2110])
        ax.set_xticks(np.arange(data_cleaned['Év'].min(), 2110, step=10))
        ax.legend()
        ax.grid(True)
        return fig



    #Népesség az adott évben kördiagram
    def Diagram1(self):
        # Az adatok betöltése a fájlból
        data = pd.read_csv('nepesseg.csv', encoding='ISO-8859-2', delimiter=';')
        
        # Az első sor kihagyása (fejléc), és az oszlopok kiválasztása
        data_cleaned = data.iloc[1:, [0, 1, 2, 3]]
        data_cleaned.columns = ['Év', 'Férfi', 'Nő', 'Összesen']

        # Az oszlopok numerikus típusúvá alakítása, és ezerrel szorzás (ha szükséges)
        data_cleaned['Év'] = pd.to_numeric(data_cleaned['Év'], errors='coerce')
        for col in ['Férfi', 'Nő', 'Összesen']:
            data_cleaned[col] = data_cleaned[col].str.replace(
                ' ', '').apply(pd.to_numeric, errors='coerce') * 1000

        # NAN törlése az adatok közül
        data_cleaned = data_cleaned.dropna()

        # Az aktuális év adatai
        current_year = data_cleaned['Év'].max()
        current_data = data_cleaned[data_cleaned['Év'] == current_year]

        # Ellenőrzés, hogy az év adatai léteznek-e
        if current_data.empty:
            print("Az aktuális év adatai nem találhatóak.")
            return None

        # Férfi és női adatok
        férfiak_száma = current_data['Férfi'].values[0]
        nők_száma = current_data['Nő'].values[0]

        # Egyedi formázó függvény a szeletek szövegének testreszabásához
        def custom_autopct(pct, all_values):
            absolute = int(round(pct / 100. * sum(all_values)))
            formatted_absolute = f"{absolute:,.0f}".replace(",", ".")
            return f"{pct:.1f}%\n({formatted_absolute})"

        # Kördiagram készítése
        fig, ax = plt.subplots(figsize=(8, 8))  # Nagyobb méretű ábra
        labels = ['Férfiak', 'Nők']
        sizes = [férfiak_száma, nők_száma]
        colors = ['blue', 'red']

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct=lambda pct: custom_autopct(pct, sizes),
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1},
            textprops={'fontsize': 14}  # Alap betűméret a címkékhez
        )

        # A szeletek százalékos és abszolút értékeinek betűmérete
        for autotext in autotexts:
            autotext.set_fontsize(16)  # Nagyobb betűméret a szeletek szövegéhez
            
        for text in texts:
            text.set_fontsize(20)
            text.set_fontweight('bold')
        

    
        # Cím beállítása nagyobb betűmérettel
        ax.set_title(f'Népesség aránya {int(current_year)}-ben', fontsize=30, fontweight='bold')

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


        # Ábra létrehozása
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y_férfi, color='blue', label='Férfiak')
        ax.scatter(x, y_nő, color='red', label='Nők')
        ax.scatter(x, y_összesen, color='purple', label='Összesen')

        # Hozzáadás regressziós vonalak
        for key, color, y in zip(['Férfi', 'Nő', 'Összesen'], ['blue', 'red', 'purple'], [y_férfi, y_nő, y_összesen]):
            ax.plot(x, models[key].predict(x), color=color,
                    linestyle='-', label=f'{key} regresszió')

 
 
        # Predikció a 2050-es évre
        year_2050 = np.array([[2050]])
        prediction_férfi_2050 = models['Férfi'].predict(year_2050)[0]  # Ne osszuk le itt
        prediction_nő_2050 = models['Nő'].predict(year_2050)[0]  # Ne osszuk le itt
        prediction_összesen_2050 = models['Összesen'].predict(year_2050)[0]  
        
        # 2050-es predikciók megjelenítése közvetlenül az adatpontok mellett
        ax.text(2050, prediction_férfi_2050, f'{prediction_férfi_2050:.1f}', color='blue', ha='left', va='bottom', fontsize=15)
        ax.text(2050, prediction_nő_2050, f'{prediction_nő_2050:.1f}', color='red', ha='left', va='bottom', fontsize=15)
        ax.text(2050, prediction_összesen_2050, f'{prediction_összesen_2050:.1f}', color='purple', ha='left', va='bottom', fontsize=15)
        
        # 2050-es évre vonatkozó előrejelzések kiírása
        ax.scatter([2050], [prediction_férfi_2050],color='blue', s=40) 
        ax.scatter([2050], [prediction_nő_2050], color='red', s=40)  
        ax.scatter([2050], [prediction_összesen_2050],color='purple', s=40)  
        
        ax.axvline(x=2050, color='black', linestyle='-')
        
        
        
                
        # Predikció a 2030-es évre
        year_2030 = np.array([[2030]])
        prediction_férfi_2030 = models['Férfi'].predict(year_2030)[0] 
        prediction_nő_2030 = models['Nő'].predict(year_2030)[0] 
        prediction_összesen_2030 = models['Összesen'].predict(year_2030)[0]  
        
        # 2030-es predikciók megjelenítése közvetlenül az adatpontok mellett
        ax.text(2030, prediction_férfi_2030, f'{prediction_férfi_2030:.1f}', color='blue', ha='left', va='bottom', fontsize=15)
        ax.text(2030, prediction_nő_2030, f'{prediction_nő_2030:.1f}', color='red', ha='left', va='bottom', fontsize=15)
        ax.text(2030, prediction_összesen_2030, f'{prediction_összesen_2030:.1f}', color='purple', ha='left', va='bottom', fontsize=15)
        
        # 2030-es évre vonatkozó előrejelzések kiírása
        ax.scatter([2030], [prediction_férfi_2030],color='blue', s=40) 
        ax.scatter([2030], [prediction_nő_2030], color='red', s=40)  
        ax.scatter([2030], [prediction_összesen_2030],color='purple', s=40) 
        
        ax.axvline(x=2030, color='black', linestyle='-')
        
          
        # Predikció a 2100-es évre
        year_2100 = np.array([[2100]])
        prediction_férfi_2100 = models['Férfi'].predict(year_2100)[0]  # Ne osszuk le itt
        prediction_nő_2100 = models['Nő'].predict(year_2100)[0]  # Ne osszuk le itt
        prediction_összesen_2100 = models['Összesen'].predict(year_2100)[0]  
        
        # 2100-es predikciók megjelenítése közvetlenül az adatpontok mellett
        ax.text(2100, prediction_férfi_2100, f'{prediction_férfi_2100:.1f}', color='blue', ha='left', va='bottom', fontsize=15)
        ax.text(2100, prediction_nő_2100, f'{prediction_nő_2100:.1f}', color='red', ha='left', va='bottom', fontsize=15)
        ax.text(2100, prediction_összesen_2100, f'{prediction_összesen_2100:.1f}', color='purple', ha='left', va='bottom', fontsize=15)
        
        # 2100-es évre vonatkozó előrejelzések kiírása
        ax.scatter([2100], [prediction_férfi_2100],color='blue', s=40) 
        ax.scatter([2100], [prediction_nő_2100], color='red', s=40)  
        ax.scatter([2100], [prediction_összesen_2100],color='purple', s=40) 
        
        ax.axvline(x=2100, color='black', linestyle='-')
        

        # Y tengely lépkedése
        ax.set_yticks(np.arange(int(data_cleaned['Férfi'].min()), int(prediction_összesen_2050) + 1, step=2))
        ax.set_yticks(np.arange(20, 57, step=2))


        ax.set_title('Népesség átlag életkora nemek szerint', fontsize=30, fontweight='bold')
        ax.set_xlabel('Év', fontsize=20)  # X tengely felirat formázása
        ax.set_ylabel('Életkor', fontsize=20)  # Y tengely felirat formázása
        ax.set_xlim([data_cleaned['Év'].min(), 2110])
        ax.set_xticks(np.arange(data_cleaned['Év'].min(), 2110, step=10))
        ax.legend()
        ax.grid(True)
        return fig



    def Diagram2(self):
        # Az adatok betöltése a fájlból
        data = pd.read_csv('nepesseg.csv', encoding='ISO-8859-2', delimiter=';')
        
        # Az első sor kihagyása (fejléc), és az oszlopok kiválasztása
        data_cleaned = data.iloc[1:, [0, 4, 5, 6]]
        data_cleaned.columns = ['Év', 'Férfi', 'Nő', 'Átlagosan']
        
        # Az oszlopok numerikus típusúvá alakítása
        data_cleaned['Év'] = pd.to_numeric(data_cleaned['Év'], errors='coerce')
        for col in ['Férfi', 'Nő', 'Átlagosan']:
            data_cleaned[col] = (data_cleaned[col]
                                .replace('..', np.nan)
                                .str.replace(' ', '', regex=False)
                                .str.replace(',', '.', regex=False)
                                .astype(float))
        
        # NAN törlés
        data_cleaned = data_cleaned.dropna()

        # Az aktuális év adatai
        current_year = data_cleaned['Év'].max()
        current_data = data_cleaned[data_cleaned['Év'] == current_year]
        
        if current_data.empty:
            print("Az aktuális év adatai nem találhatóak.")
            return None

        férfiak_száma = current_data['Férfi'].values[0]
        nők_száma = current_data['Nő'].values[0]
        atlagosan_szama=current_data['Átlagosan'].values[0]

        # Oszlopdiagram készítése
        fig, ax = plt.subplots(figsize=(6, 6))  # Az oszlopdiagram méretének beállítása
        categories = ['Férfiak', 'Nők', 'Átlagosan']
        values = [férfiak_száma, nők_száma, atlagosan_szama]
        colors = ['blue', 'red', 'purple']  # Az oszlopok színének beállítása

        width = 0.2  
        x_positions = [0.1,0.3, 0.5]  # Az oszlopok középpontja közvetlen egymás mellett
        bars = ax.bar(x_positions, values, color=colors, edgecolor='black', width=width)

        # Az oszlopnevek betűméretének beállítása
        ax.set_xticks(x_positions)  # Az X tengelyen az oszlopok helye
        ax.set_xticklabels(categories, fontsize=20)

        # Az oszlopok címkézése (értékek hozzáadása az oszlopok tetejére)
        ax.set_title(f'Népesség átlagos életkora {int(current_year)}-ben', fontsize=30, fontweight='bold')
        ax.set_ylabel('Életkor', fontsize=20)

        # A számok hozzáadása az oszlopok tetejére
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,  
                yval/2,  
                f'{yval:,.1f}',  
                ha='center', va='bottom', fontsize=20,
            )

        ax.set_yticks(np.arange(0, max(values) + 2, 2))

        # A tengelyek betűméretének módosítása
        ax.tick_params(axis='both', labelsize=12)
        
        ax.set_xlim(0, 1.5)  # Középre igazítjuk az oszlopokat


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
