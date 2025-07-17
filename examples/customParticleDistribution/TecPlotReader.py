##############################################################
# TecPlotReader.py
# A GUI and batch exporter for Tecplot particle distribution data.
# It allows users to visualize and save particle distributions
# from Tecplot files, with options for trimming and selecting variables.
# Usage:
#   python TecPlotReader.py -f data.pdt
#   python TecPlotReader.py -f data.pdt -i variable_name -o output.txt
#   python TecPlotReader.py  # to open the GUI
# Requires: tkinter, matplotlib, numpy, re
#####################################################################

from math import dist
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
import sys
from scipy.optimize import curve_fit


# --- Data Handling Functions ---
def read_tecplot_data(filename, thetaKey, energyKey, zone_lines_max=2):
    datafile = open(filename, "r")
    mode = -1  # 0=vars,1=zone,2=data
    zone_lines_read = 0
    variables = []
    data_dict = {}
    tmp_data = []
    var_counter = 0
    data_i = data_j = 0
    I = J = 0

    for line in datafile:
        if not line.strip():
            continue
        upper = line.upper()
        if "VARIABLES" in upper:
            mode = 0
        elif "ZONE" in upper:
            mode = 1

        if mode == 0:
            for var in re.findall(r'"([^\"]+)"', line):
                if var not in (thetaKey, energyKey):
                    var = var.split()[0]  # take only the first part
                variables.append(var)
                data_dict[var] = []
        elif mode == 1:
            if zone_lines_read == 0:
                mI = re.search(r"I=\s*(\d+)", line)
                mJ = re.search(r"J=\s*(\d+)", line)
                if mI and mJ:
                    I, J = int(mI.group(1)), int(mJ.group(1))
            zone_lines_read += 1
            if zone_lines_read == zone_lines_max:
                mode = 2
        elif mode == 2:
            var = variables[var_counter]
            parts = line.split()
            tmp_data.extend(map(float, parts))
            data_i += len(parts)
            if data_i == I:
                data_i = 0
                data_dict[var].append(tmp_data)
                tmp_data = []
                data_j += 1
                if data_j == J:
                    data_j = 0
                    var_counter += 1
    datafile.close()

    # convert to arrays
    for k in data_dict:
        data_dict[k] = np.array(data_dict[k])
    # extract 1D theta and energy
    data_dict[thetaKey] = data_dict[thetaKey][0, :]
    data_dict[energyKey] = data_dict[energyKey][:, 0]
    return data_dict, variables


def trim_data(data, energy, eps=1e-4):
    low = next((i for i in range(data.shape[0]) if np.any(data[i] >= eps)), 0)
    high = data.shape[0]
    for i in range(data.shape[0]):
        if np.all(data[i] < eps) and all(
            np.all(data[j] < eps) for j in range(i, data.shape[0])
        ):
            high = i
            break
    return data[low:high, :], energy[low:high]


def plot_data(data_dict, key, thetaKey, energyKey, eps):
    d, e = trim_data(data_dict[key], data_dict[energyKey], eps)
    xx, yy = np.meshgrid(data_dict[thetaKey], e)
    plt.figure()
    plt.pcolormesh(xx, yy, d)
    plt.colorbar()
    plt.xlabel(thetaKey)
    plt.ylabel(energyKey)
    plt.title(f"Distribution: {key}")
    plt.show()


def line_plot(data_dict, key, thetaKey, energyKey, eps):
    d, e = trim_data(data_dict[key], data_dict[energyKey], eps)
    theta = data_dict[thetaKey]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(len(e)):
        ax.plot(
            ys=theta,
            zs=d[i, :],
            xs=e[i],
            color=plt.cm.viridis(i / len(e)),
        )
    ax.set_ylabel(thetaKey)
    ax.set_zlabel(f"Distribution: {key}")
    ax.set_xlabel(energyKey)
    plt.tight_layout()
    plt.show()


def fit_data(data_dict, key, thetaKey, energyKey, eps, power_cutoff=10000):
    def angle_fit(t, power):
        f = np.cos(t) ** power
        f = np.abs(np.sin(t) * f)
        return f / np.max(f)

    def gaussian(x, mu, sigma):
        f = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        return f / np.max(f)

    def dist_fit(t, e, power, mu, sigma):
        return angle_fit(t, power) * gaussian(e, mu, sigma)

    dist, energy = trim_data(data_dict[key], data_dict[energyKey], eps)
    theta = data_dict[thetaKey]
    theta_fit = np.linspace(np.min(theta), np.max(theta), 1000)
    energy_fit = np.linspace(np.min(energy), np.max(energy), 1000)

    powers = np.zeros(len(energy))
    for i in range(len(energy)):
        sample = dist[i, :]  # / np.max(dist[i, :])
        sample_max = np.max(sample)
        popt, _ = curve_fit(
            angle_fit,
            np.deg2rad(theta),
            sample / sample_max,
            bounds=([1], [50000]),
            p0=[500],
        )
        powers[i] = popt[0]

    powers_cut = powers < power_cutoff
    popt_p, _ = curve_fit(
        lambda x, a, b: a * x + b,
        energy[powers_cut],
        powers[powers_cut],
        bounds=([0, 0], [10, 10000]),
        p0=[1, 1],
    )

    dist_t = np.sum(dist, axis=1)
    t_max = np.max(dist_t)
    popt_g, _ = curve_fit(
        gaussian, energy, dist_t / t_max, bounds=([0, 0], [100, 50]), p0=[38, 5]
    )

    xx, yy = np.meshgrid(theta_fit, energy_fit)
    d = dist_fit(np.deg2rad(xx), yy, popt_p[1], popt_g[0], popt_g[1])
    plt.figure()
    plt.pcolormesh(xx, yy, d, shading="auto")
    plt.colorbar(label="Distribution")
    plt.xlabel(thetaKey)
    plt.ylabel(energyKey)
    plt.title(
        f"Power {popt_p[1]:.2f}, Gaussian mu {popt_g[0]:.2f}, sigma {popt_g[1]:.2f}"
    )
    plt.tight_layout()
    plt.show()


def save_data(data_dict, key, thetaKey, energyKey, eps, outname):
    d, e = trim_data(data_dict[key], data_dict[energyKey], eps)
    with open(outname, "w") as f:
        f.write(f"# {key}\n")
        f.write(" ".join(map(str, e)) + "\n")
        f.write(" ".join(map(str, data_dict[thetaKey])) + "\n")
        for row in d:
            f.write(" ".join(map(str, row)) + "\n")


# --- GUI ---
class TecplotGUI(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=10)
        parent.title("Tecplot Distribution Viewer")
        parent.minsize(500, 400)
        self.data = {}
        self.vars = []

        self.tk_theta = "THETA (DEG)"
        self.tk_energy = "ENERGY (EV)"
        self.entry_theta = tk.StringVar(value=self.tk_theta)
        self.entry_energy = tk.StringVar(value=self.tk_energy)
        self.entry_zone_lines = tk.IntVar(value=2)
        self.line_plot_var = tk.BooleanVar(value=False)

        # use a built‑in theme
        style = ttk.Style()
        style.theme_use("clam")

        # File input
        ttk.Label(self, text="Filename:").pack(anchor="w")
        frame_file = ttk.Frame(self)
        self.entry_file = ttk.Entry(frame_file, width=40)
        self.entry_file.pack(side="left", padx=5)
        ttk.Button(frame_file, text="Browse", command=self.browse_file).pack(
            side="left", padx=5
        )
        ttk.Button(frame_file, text="\u21bb", command=self.load_data, width=4).pack(
            side="left", padx=5
        )
        frame_file.pack(pady=5)

        # Variable list
        ttk.Label(self, text="Select Variable:").pack(anchor="w")
        self.listbox = tk.Listbox(
            self, background="white", selectmode=tk.SINGLE, foreground="black"
        )
        self.listbox.pack(fill="both", expand=True, padx=5)

        # Trim eps
        frame_eps = ttk.Frame(self)
        ttk.Label(frame_eps, text="Trim eps:").pack(side="left")
        self.entry_eps = ttk.Entry(frame_eps, width=10)
        self.entry_eps.insert(0, "1e-4")
        self.entry_eps.pack(side="left", padx=5)
        frame_eps.pack(pady=5)

        # Buttons
        frame_buttons = ttk.Frame(self)
        ttk.Button(frame_buttons, text="Show", command=self.on_plot).pack(
            side="left", padx=5
        )
        ttk.Button(frame_buttons, text="Fit", command=self.on_fit).pack(
            side="left", padx=5
        )
        ttk.Button(frame_buttons, text="Save...", command=self.on_save).pack(
            side="left", padx=5
        )
        frame_buttons.pack(pady=10)

        # Extended options button
        btn_ext = ttk.Button(frame_buttons, text="Options", command=self.open_options)
        btn_ext.pack(side="left", padx=5)

    def browse_file(self):
        f = filedialog.askopenfilename(
            filetypes=[("TecPlot files", "*.pdt"), ("All files", "*")]
        )
        if f:
            self.entry_file.delete(0, tk.END)
            self.entry_file.insert(0, f)
            self.load_data()

    def load_data(self):
        fn = self.entry_file.get().strip()
        self.tk_theta = self.entry_theta.get().strip()
        self.tk_energy = self.entry_energy.get().strip()
        zone_lines = self.entry_zone_lines.get()
        if not fn:
            messagebox.showerror("Error", "Please provide filename.")
            return
        try:
            self.data, self.vars = read_tecplot_data(
                fn, self.tk_theta, self.tk_energy, zone_lines_max=zone_lines
            )
            # remove tk_theta and tk_energy from vars
            self.vars = [
                v for v in self.vars if v not in (self.tk_theta, self.tk_energy)
            ]
            self.listbox.delete(0, tk.END)
            for v in self.vars:
                self.listbox.insert(tk.END, v)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def on_plot(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "No variable selected.")
            return
        key = self.vars[sel[0]]
        eps = float(self.entry_eps.get())
        if self.line_plot_var.get():
            line_plot(
                self.data, key, self.entry_theta.get(), self.entry_energy.get(), eps
            )
        else:
            plot_data(
                self.data, key, self.entry_theta.get(), self.entry_energy.get(), eps
            )

    def on_save(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "No variable selected.")
            return
        key = self.vars[sel[0]]
        eps = float(self.entry_eps.get())
        out = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*")],
        )
        if out:
            save_data(
                self.data,
                key,
                self.entry_theta.get(),
                self.entry_energy.get(),
                eps,
                out,
            )
            messagebox.showinfo("Saved", f"Data saved to {out}")

    def on_fit(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "No variable selected.")
            return
        key = self.vars[sel[0]]
        eps = float(self.entry_eps.get())
        fit_data(self.data, key, self.entry_theta.get(), self.entry_energy.get(), eps)
        # messagebox.showinfo("Fit", f"Fit completed for {key}")

    def open_options(self):
        win = tk.Toplevel(self)
        win.title("Options")
        win.geometry("300x200")

        # Keys
        frame_keys = tk.Frame(win)
        tk.Label(frame_keys, text="Theta Key:").grid(row=0, column=0)
        entry_t = tk.Entry(frame_keys, textvariable=self.entry_theta)
        entry_t.grid(row=0, column=1)
        tk.Label(frame_keys, text="Energy Key:").grid(row=1, column=0)
        entry_e = tk.Entry(frame_keys, textvariable=self.entry_energy)
        entry_e.grid(row=1, column=1)
        frame_keys.pack(pady=10)

        # Zone lines
        frame_zone = tk.Frame(win)
        tk.Label(frame_zone, text="Zone Lines:").pack(side="left")
        entry_zone = tk.Entry(frame_zone, textvariable=self.entry_zone_lines, width=5)
        entry_zone.pack(side="left", padx=5)
        frame_zone.pack(pady=10)

        # Checkbox for line plot
        chk_line_plot = ttk.Checkbutton(
            win,
            text="Line Plot",
            variable=self.line_plot_var,
            command=self.toggle_line_plot,
        )
        chk_line_plot.pack(anchor="w", padx=15, pady=5)

        # Done button to close
        ttk.Button(win, text="OK", command=win.destroy).pack(pady=10)

    def toggle_line_plot(self):
        # Toggle the line plot option
        if self.line_plot_var.get():
            self.line_plot_var.set(True)
        else:
            self.line_plot_var.set(False)


def main():
    parser = argparse.ArgumentParser(
        description="Tecplot distribution GUI or batch exporter"
    )
    parser.add_argument("-f", "--file", help="input Tecplot file", metavar="FILE")
    parser.add_argument(
        "-i", "--input", help="variable name to extract", metavar="VAR", default="ALL"
    )
    parser.add_argument(
        "-o", "--output", help="output filename", metavar="OUT", default="output.txt"
    )
    parser.add_argument("--eps", help="trim epsilon", type=float, default=1e-4)

    args = parser.parse_args()

    # no input file → open GUI
    if not (args.file):
        root = tk.Tk()
        app = TecplotGUI(root)
        app.pack(fill="both", expand=True)
        root.mainloop()
        return

    # batch mode: load, trim, save, then exit
    # defaults for theta and energy keys
    theta_key = "THETA (DEG)"
    energy_key = "ENERGY (EV)"
    try:
        data_dict, variables = read_tecplot_data(
            args.file,
            thetaKey=theta_key,
            energyKey=energy_key,
        )

        outFileName = args.output
        if outFileName.endswith(".txt"):
            outFileName = outFileName[:-4]

        if args.input == "ALL":
            keys = [v for v in variables if v not in (theta_key, energy_key)]
        else:
            keys = [args.input] if args.input in variables else []
        if not keys:
            print(
                f"Error: variable '{args.input}' not found in {args.file}",
                file=sys.stderr,
            )
            sys.exit(1)

        for key in keys:
            save_data(
                data_dict,
                key=key,
                thetaKey=theta_key,
                energyKey=energy_key,
                eps=args.eps,
                outname=outFileName + f"_{key}.txt",
            )
            print(f"Saved {key} → {outFileName}_{key}.txt")
    except Exception as e:
        print(f"Batch error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
