import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import os
import re
import glob

# sorter logic

def clean_numeric(series):
    return series.astype(str).str.replace(r'[^\d]', '', regex=True).pipe(pd.to_numeric, errors='coerce')

def sort_csv(filepath, col, ascending):
    df = pd.read_csv(filepath)
    temp_col = f"__sort_{col}"
    if df[col].dtype == object:
        df[temp_col] = clean_numeric(df[col])
        df = df.sort_values(by=temp_col, ascending=ascending).drop(columns=[temp_col])
    else:
        df = df.sort_values(by=col, ascending=ascending)
    df.to_csv(filepath, index=False, encoding="utf-8")
    return df

# UI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)  # go up one level to workspace root

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Sorter")
        self.root.geometry("700x520")
        self.root.configure(bg="#1e1e2e")
        self.filepath = None
        self.df = None
        self.sort_col = None
        self.ascending = tk.BooleanVar(value=True)

        self.build_ui()
        self.discover_csvs()

    def build_ui(self):
        # top bar
        top = tk.Frame(self.root, bg="#1e1e2e")
        top.pack(fill="x", padx=16, pady=12)

        tk.Label(top, text="CSV Sorter", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 14, "bold")).pack(side="left")

        tk.Button(top, text="↻ Refresh", command=self.discover_csvs,
                  bg="#313244", fg="#cdd6f4", font=("Segoe UI", 9),
                  relief="flat", padx=8, pady=4, cursor="hand2").pack(side="right")

        tk.Label(top, text=f"📁 {WORKSPACE_DIR}", bg="#1e1e2e", fg="#6c7086",
                 font=("Segoe UI", 8)).pack(side="right", padx=12)

        tk.Frame(self.root, bg="#313244", height=1).pack(fill="x", padx=16)

        # main area
        main = tk.Frame(self.root, bg="#1e1e2e")
        main.pack(fill="both", expand=True, padx=16, pady=12)

        # left: CSV file list
        left = tk.Frame(main, bg="#1e1e2e")
        left.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="CSV failai", bg="#1e1e2e", fg="#6c7086",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))

        self.csv_listbox = tk.Listbox(left, bg="#181825", fg="#cdd6f4",
                                      selectbackground="#89b4fa", selectforeground="#1e1e2e",
                                      font=("Segoe UI", 10), relief="flat", bd=0,
                                      highlightthickness=0, activestyle="none")
        self.csv_listbox.pack(fill="both", expand=True)
        self.csv_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        # middle: column list
        mid = tk.Frame(main, bg="#1e1e2e")
        mid.pack(side="left", fill="both", expand=True, padx=(12, 0))

        tk.Label(mid, text="Stulpeliai", bg="#1e1e2e", fg="#6c7086",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))

        self.col_listbox = tk.Listbox(mid, bg="#181825", fg="#cdd6f4",
                                      selectbackground="#89b4fa", selectforeground="#1e1e2e",
                                      font=("Segoe UI", 10), relief="flat", bd=0,
                                      highlightthickness=0, activestyle="none")
        self.col_listbox.pack(fill="both", expand=True)
        self.col_listbox.bind("<<ListboxSelect>>", self.on_column_select)

        # right: options
        right = tk.Frame(main, bg="#1e1e2e", width=180)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        tk.Label(right, text="Tvarka", bg="#1e1e2e", fg="#6c7086",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 8))

        tk.Radiobutton(right, text="↑  Didėjanti (A→Z)", variable=self.ascending,
                       value=True, bg="#1e1e2e", fg="#a6e3a1", selectcolor="#1e1e2e",
                       activebackground="#1e1e2e", font=("Segoe UI", 10)).pack(anchor="w", pady=3)

        tk.Radiobutton(right, text="↓  Mažėjanti (Z→A)", variable=self.ascending,
                       value=False, bg="#1e1e2e", fg="#f38ba8", selectcolor="#1e1e2e",
                       activebackground="#1e1e2e", font=("Segoe UI", 10)).pack(anchor="w", pady=3)

        tk.Frame(right, bg="#313244", height=1).pack(fill="x", pady=10)

        tk.Label(right, text="Failas:", bg="#1e1e2e", fg="#6c7086",
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.sel_file_label = tk.Label(right, text="—", bg="#1e1e2e", fg="#89b4fa",
                                       font=("Segoe UI", 9, "bold"), wraplength=160, justify="left")
        self.sel_file_label.pack(anchor="w", pady=(2, 8))

        tk.Label(right, text="Stulpelis:", bg="#1e1e2e", fg="#6c7086",
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.selected_label = tk.Label(right, text="—", bg="#1e1e2e", fg="#89b4fa",
                                       font=("Segoe UI", 9, "bold"), wraplength=160, justify="left")
        self.selected_label.pack(anchor="w", pady=(2, 0))

        # bottom
        bottom = tk.Frame(self.root, bg="#1e1e2e")
        bottom.pack(fill="x", padx=16, pady=10)

        self.status_label = tk.Label(bottom, text="", bg="#1e1e2e", fg="#a6e3a1",
                                     font=("Segoe UI", 9))
        self.status_label.pack(side="left")

        tk.Button(bottom, text="Rikiuoti ir išsaugoti", command=self.do_sort,
                  bg="#a6e3a1", fg="#1e1e2e", font=("Segoe UI", 10, "bold"),
                  relief="flat", padx=14, pady=6, cursor="hand2").pack(side="right")

    def discover_csvs(self):
        """Randa visus CSV failus workspace direktorijoje"""
        self.csv_files = [
            f for f in glob.glob(os.path.join(WORKSPACE_DIR, "**", "*.csv"), recursive=True)
            if not re.search(r'[/\\](\.?venv)[/\\]', f)
        ]
        self.csv_listbox.delete(0, tk.END)
        if not self.csv_files:
            self.csv_listbox.insert(tk.END, "  Nerasta CSV failų")
            return
        for path in self.csv_files:
            rel = os.path.relpath(path, WORKSPACE_DIR)
            self.csv_listbox.insert(tk.END, f"  {rel}")

    def on_file_select(self, event):
        sel = self.csv_listbox.curselection()
        if not sel:
            return
        self.filepath = self.csv_files[sel[0]]
        self.df = pd.read_csv(self.filepath)
        self.sel_file_label.config(text=os.path.basename(self.filepath))
        self.col_listbox.delete(0, tk.END)
        for col in self.df.columns:
            self.col_listbox.insert(tk.END, f"  {col}")
        self.sort_col = None
        self.selected_label.config(text="—")
        self.status_label.config(text="")

    def on_column_select(self, event):
        sel = self.col_listbox.curselection()
        if not sel or self.df is None:
            return
        self.sort_col = self.df.columns[sel[0]]
        self.selected_label.config(text=self.sort_col)

    def do_sort(self):
        if not self.filepath:
            messagebox.showwarning("Klaida", "Pasirinkite CSV failą!")
            return
        if not self.sort_col:
            messagebox.showwarning("Klaida", "Pasirinkite stulpelį!")
            return
        try:
            df = sort_csv(self.filepath, self.sort_col, self.ascending.get())
            direction = "didėjanti" if self.ascending.get() else "mažėjanti"
            self.status_label.config(text=f"✓ Išrikiuota pagal '{self.sort_col}' ({direction})")
            self.df = df
        except Exception as e:
            messagebox.showerror("Klaida", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()