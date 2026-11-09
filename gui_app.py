import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Ensure local imports work cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engine_v3 import process_and_fix_file, DEFAULT_KNN_BLEND, DEFAULT_ROW_BLEND

class GhostDataEngineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ghost Data Engine 3.0 — Control Console")
        self.root.geometry("600x420")
        self.root.resizable(False, False)
        
        # Style layout configuration
        style = ttk.Style()
        style.theme_use("clam")
        
        # File path storage trackers
        self.input_file_path = tk.StringVar()
        self.output_file_path = tk.StringVar()
        
        # --- UI ELEMENTS LAYOUT ---
        # Header Banner
        header = tk.Label(root, text="GHOST DATA ENGINE v3.0", font=("Arial", 16, "bold"), fg="#2c3e50")
        header.pack(pady=15)
        
        # Frame Layout wrapper
        frame = tk.Frame(root)
        frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Input file browser row
        tk.Label(frame, text="Source Dataset (CSV):", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        self.entry_in = tk.Entry(frame, textvariable=self.input_file_path, width=45)
        self.entry_in.grid(row=1, column=0, padx=5, pady=5)
        btn_browse_in = tk.Button(frame, text="Browse...", command=self.browse_input, bg="#bdc3c7")
        btn_browse_in.grid(row=1, column=1, padx=5)
        
        # Output destination browser row
        tk.Label(frame, text="Export Repair Target:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w", pady=15)
        self.entry_out = tk.Entry(frame, textvariable=self.output_file_path, width=45)
        self.entry_out.grid(row=3, column=0, padx=5, pady=5)
        btn_browse_out = tk.Button(frame, text="Save As...", command=self.browse_output, bg="#bdc3c7")
        btn_browse_out.grid(row=3, column=1, padx=5)
        
        # Parameter Settings Controls Row
        param_frame = tk.LabelFrame(frame, text=" Parameter Matrices ", padx=10, pady=10)
        param_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=20)
        
        tk.Label(param_frame, text="Neighborhood Matrix:").grid(row=0, column=0, sticky="w")
        self.conn_select = ttk.Combobox(param_frame, values=["4-way Adjacency", "8-way Adjacency"], width=18, state="readonly")
        self.conn_select.set("4-way Adjacency")
        self.conn_select.grid(row=0, column=1, padx=10)
        
        tk.Label(param_frame, text="Inward Diffusion Decay:").grid(row=0, column=2, sticky="w", padx=10)
        self.decay_select = ttk.Combobox(param_frame, values=["1.0 (No Bias/Continuous)", "0.85 (Fade to Voids)"], width=22, state="readonly")
        self.decay_select.set("1.0 (No Bias/Continuous)")
        self.decay_select.grid(row=0, column=3, padx=5)
        
        tk.Label(param_frame, text="Post-fill Blend:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.blend_select = ttk.Combobox(param_frame, values=["On (recommended)", "Off (engine only)"], width=18, state="readonly")
        self.blend_select.set("On (recommended)")
        self.blend_select.grid(row=1, column=1, padx=10, pady=(10, 0))

        # Main Execution Triggers
        self.btn_run = tk.Button(root, text="🚀 EXECUTE GDE SPATIAL REPAIR", font=("Arial", 12, "bold"), fg="white", bg="#27ae60", height=2, command=self.execute_pipeline)
        self.btn_run.pack(pady=15, fill="x", padx=40)

    def browse_input(self):
        file_selected = filedialog.askopenfilename(filetypes=[("Comma Separated Values", "*.csv")])
        if file_selected:
            self.input_file_path.set(file_selected)
            # Auto-propose an output file path destination rule match
            base, ext = os.path.splitext(file_selected)
            self.output_file_path.set(f"{base}_gde_fixed{ext}")

    def browse_output(self):
        file_selected = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("Comma Separated Values", "*.csv")])
        if file_selected:
            self.output_file_path.set(file_selected)

    def execute_pipeline(self):
        infile = self.input_file_path.get()
        outfile = self.output_file_path.get()
        
        if not infile or not os.path.exists(infile):
            messagebox.showerror("Error", "Please select a valid missing data source file.")
            return
        if not outfile:
            messagebox.showerror("Error", "Please specify an export destination file path.")
            return
            
        connectivity = 4 if "4-way" in self.conn_select.get() else 8
        inward_decay = 1.0 if "1.0" in self.decay_select.get() else 0.85
        
        try:
            # Route directly to your existing core processing loop pipeline file
            blend_kw = {} if self.blend_select.get().startswith("On") else {"knn_blend": None, "row_blend": None}
            result = process_and_fix_file(infile, outfile, connectivity=connectivity, inward_decay=inward_decay, **blend_kw)
            if result and not result.get("written", True):
                messagebox.showinfo("Nothing to repair", result.get("message", "No missing cells were found."))
            else:
                detail = f"\n{result['message']}" if result else ""
                messagebox.showinfo("Success", f"Spatial grid reconstruction complete!{detail}\nRepaired file saved to:\n{outfile}")
        except Exception as e:
            messagebox.showerror("Execution Fault", f"Pipeline calculation crash occurred:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GhostDataEngineApp(root)
    root.mainloop()
