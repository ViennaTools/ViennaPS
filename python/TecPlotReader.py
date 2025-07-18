##############################################################
# TecPlotReader.py
# A GUI and batch exporter for Tecplot particle distribution data.
# It allows users to visualize and save particle distributions
# from Tecplot files, with options for trimming and selecting variables.
# Usage:
#   python TecPlotReader.py -f data.pdt
#   python TecPlotReader.py -f data.pdt -i variable_name -o output.txt
#   python TecPlotReader.py -f data.pdt -i variable_name --fit
#   python TecPlotReader.py  # to open the GUI
# Requires: tkinter, matplotlib, numpy, scipy
#####################################################################

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import re
import argparse
import sys
from scipy.optimize import curve_fit
from scipy import stats


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
                mode = 2  # switch to data mode
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

    if thetaKey not in data_dict or energyKey not in data_dict:
        raise ValueError(
            f"Required keys '{thetaKey}' or '{energyKey}' not found in the data."
        )

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
    plt.colorbar(label="Distribution")
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


def fit_data(
    data_dict,
    key,
    thetaKey,
    energyKey,
    eps,
    angle_p_cutoff=10000,
    useSin=False,
    verbose=False,
    vm=False,
):
    def _powerCos(t, power):
        f = np.cos(t) ** power
        if useSin:
            f = np.abs(np.sin(t) * f)
        return f / np.max(f)

    def _vonMises(t, kappa):
        f = stats.vonmises.pdf(t, kappa=kappa, loc=0, scale=np.pi / 2)
        if useSin:
            f = np.abs(np.sin(t) * f)
        return f / np.max(f)

    def _gaussian(x, mu, sigma):
        f = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        return f / np.max(f)

    angle_fit = _vonMises if vm else _powerCos
    angle_param = "Kappa" if vm else "Power"

    def dist_fit(t, e, p, mu, sigma):
        return angle_fit(t, p) * _gaussian(e, mu, sigma)

    dist, energy = trim_data(data_dict[key], data_dict[energyKey], eps)
    theta = data_dict[thetaKey]
    theta_fit = np.linspace(np.min(theta), np.max(theta), 1000)
    energy_fit = np.linspace(np.min(energy), np.max(energy), 1000)

    # fit angle distribution
    angle_p = np.zeros(len(energy))
    for i in range(len(energy)):
        sample = dist[i, :]
        sample_max = np.max(sample)
        if sample_max > 0:
            popt, _ = curve_fit(
                angle_fit,
                np.deg2rad(theta),
                sample / sample_max,
                bounds=([1], [angle_p_cutoff]),
                p0=[100],
            )
            angle_p[i] = popt[0]
        else:
            angle_p[i] = angle_p_cutoff

    if verbose:
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.3)  # leave room at bottom
        [line_dist] = ax.plot(theta, dist[0, :], label="Distribution")
        [line_fit] = ax.plot(
            theta_fit,
            angle_fit(np.deg2rad(theta_fit), angle_p[0]) * np.max(dist[0, :]),
            label="Fit",
        )
        ax.set_xlabel(thetaKey)
        ax.set_title(f"Energy = {energy[0]:.2f} eV, {angle_param} = {angle_p[0]:.2f}")

        # TextBox for the integer, centered
        axbox = plt.axes([0.4, 0.15, 0.2, 0.05])
        text_box = TextBox(axbox, "", initial=str(0))

        # “–” button
        axdec = plt.axes([0.33, 0.15, 0.05, 0.05])
        btn_dec = Button(axdec, "-")

        # “+” button
        axinc = plt.axes([0.62, 0.15, 0.05, 0.05])
        btn_inc = Button(axinc, "+")

        def update_plot():
            try:
                f = int(text_box.text)
                if f < 0 or f >= len(energy):
                    raise ValueError("Index out of bounds")
            except ValueError:
                return
            line_dist.set_ydata(dist[f, :])
            line_fit.set_ydata(
                angle_fit(np.deg2rad(theta_fit), angle_p[f]) * np.max(dist[f, :])
            )
            ax.set_title(
                f"Energy = {energy[f]:.2f} eV, {angle_param} = {angle_p[f]:.2f}"
            )
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()

        def on_inc(event):
            v = min(int(text_box.text) + 1, len(energy) - 1)
            text_box.set_val(str(v))
            update_plot()

        def on_dec(event):
            v = max(int(text_box.text) - 1, 0)
            text_box.set_val(str(v))
            update_plot()

        # wire up callbacks
        btn_inc.on_clicked(on_inc)
        btn_dec.on_clicked(on_dec)
        text_box.on_submit(lambda text: update_plot())

    # fit resulting parameter distribution
    angle_p_cut = angle_p < (angle_p_cutoff - 1)
    popt_p, _ = curve_fit(
        lambda x, a, b: a * x + b,
        energy[angle_p_cut],
        angle_p[angle_p_cut],
        bounds=([0, 0], [10, 10000]),
        p0=[1, 1],
    )

    if verbose:
        plt.figure()
        plt.plot(energy, angle_p, "-x", label=angle_param)
        plt.plot(
            energy[angle_p_cut],
            popt_p[0] * energy[angle_p_cut] + popt_p[1],
            "o--",
            label=f"{angle_param} Fit",
        )
        plt.axhline(angle_p_cutoff, color="red", linestyle="--", label="Cutoff")
        plt.xlabel(energyKey)
        plt.ylabel(angle_param)
        plt.title(f"{angle_param} Fit: a={popt_p[0]:.2f}, b={popt_p[1]:.2f}")
        plt.legend()

    # fit energy distribution
    dist_t = np.sum(dist, axis=1)
    t_max = np.max(dist_t)
    if t_max > 0:
        mu_0 = np.trapezoid(dist_t * energy, x=energy)
        popt_g, _ = curve_fit(
            _gaussian, energy, dist_t / t_max, bounds=([0, 0], [100, 50]), p0=[mu_0, 5]
        )
    else:
        return -1

    if verbose:
        plt.figure()
        plt.plot(energy, dist_t, label="Cumulative Distribution")
        plt.plot(
            energy_fit,
            _gaussian(energy_fit, *popt_g) * t_max,
            label="Gaussian Fit",
        )
        plt.xlabel(energyKey)
        plt.ylabel("Cumulative Distribution")
        plt.title(f"Gaussian Fit: mu={popt_g[0]:.2f}, sigma={popt_g[1]:.2f}")
        plt.legend()

    print(
        f"{angle_param} {popt_p[1]:.2f}, Gaussian mu {popt_g[0]:.2f}, sigma {popt_g[1]:.2f}"
    )

    tt, ee = np.meshgrid(theta_fit, energy_fit)
    d = dist_fit(np.deg2rad(tt), ee, popt_p[1], popt_g[0], popt_g[1])
    plt.figure()
    plt.pcolormesh(tt, ee, d, shading="auto")
    plt.colorbar(label="Distribution")
    plt.xlabel(thetaKey)
    plt.ylabel(energyKey)
    plt.title(
        f"{angle_param} {popt_p[1]:.2f}, Gaussian mu {popt_g[0]:.2f}, sigma {popt_g[1]:.2f}"
    )
    plt.tight_layout()
    plt.show()

    return 0


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
        parent.minsize(400, 300)
        self.data = {}
        self.vars = []

        # use a built‑in theme
        # style = ttk.Style()
        # style.theme_use("clam")

        # Options Input
        self.tk_theta = "THETA (DEG)"
        self.tk_energy = "ENERGY (EV)"
        self.entry_theta = tk.StringVar(value=self.tk_theta)
        self.entry_energy = tk.StringVar(value=self.tk_energy)
        self.entry_zone_lines = tk.IntVar(value=2)
        self.line_plot_var = tk.BooleanVar(value=False)
        self.fit_angle_p_cutoff = tk.IntVar(value=10000)
        self.fit_use_sin = tk.BooleanVar(value=False)
        self.fit_verbose = tk.BooleanVar(value=False)
        self.fit_function = tk.StringVar(value="powerCosine")
        self.path_fn = ""

        # File input
        frame_file = ttk.Frame(self)
        ttk.Label(frame_file, text="Filename:").pack(anchor="w")
        self.entry_file = ttk.Entry(frame_file, width=20)
        self.entry_file.pack(side="left", padx=5)
        ttk.Button(frame_file, text="Browse", command=self.browse_file).pack(
            side="left", padx=5
        )
        frame_file.pack(pady=5)

        # Variable list
        frame_main = ttk.Frame(self)
        ttk.Label(frame_main, text="Select Variable:").grid(row=0, column=0)
        self.listbox = tk.Listbox(
            frame_main, background="gray", selectmode=tk.SINGLE, foreground="black"
        )
        self.listbox.grid(row=1, column=0, padx=5, rowspan=6)

        # Trim eps
        ttk.Label(frame_main, text="Trim eps:").grid(row=1, column=1, padx=5)
        self.entry_eps = ttk.Entry(frame_main, width=10)
        self.entry_eps.insert(0, "1e-4")
        self.entry_eps.grid(row=2, column=1, padx=5)

        # Buttons
        ttk.Button(frame_main, text="Show", command=self.on_plot).grid(
            row=3, column=1, padx=5
        )
        ttk.Button(frame_main, text="Fit", command=self.on_fit).grid(
            row=4, column=1, padx=5
        )
        ttk.Button(frame_main, text="Save", command=self.on_save).grid(
            row=5, column=1, padx=5
        )

        # Extended options button
        btn_ext = ttk.Button(
            frame_main, text="Options", command=self.open_options
        ).grid(row=6, column=1, padx=5)
        frame_main.pack(pady=5)

    def browse_file(self):
        f = filedialog.askopenfilename(
            filetypes=[("TecPlot files", "*.pdt"), ("All files", "*")],
            title="Select TecPlot File",
            initialdir=".",
        )
        if f:
            self.entry_file.delete(0, tk.END)
            # insert the filename into the entry
            self.path_fn = f
            f = f.split("/")[-1] if "/" in f else f.split("\\")[-1]
            self.entry_file.insert(0, f)
            self.load_data()

    def load_data(self):
        fn = self.path_fn.strip()
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
        angle_p_cutoff = self.fit_angle_p_cutoff.get()
        use_sine = self.fit_use_sin.get()
        verbose = self.fit_verbose.get()
        fit_function = self.fit_function.get()
        if fit_function == "vonMises":
            vm = True
        elif fit_function == "powerCosine":
            vm = False
        else:
            messagebox.showerror("Error", "Unknown fit function selected.")
            return
        res = fit_data(
            self.data,
            key,
            self.entry_theta.get(),
            self.entry_energy.get(),
            eps,
            angle_p_cutoff=angle_p_cutoff,
            useSin=use_sine,
            verbose=verbose,
            vm=vm,
        )
        if res == -1:
            messagebox.showerror("Error", "Failed to fit data.")

    def open_options(self):
        win = tk.Toplevel(self)
        win.title("Options")
        win.geometry("300x400")
        win.resizable(False, False)

        # Keys
        frame_keys = ttk.Frame(win)
        ttk.Label(frame_keys, text="Parsing:").grid(row=0, columnspan=2, pady=5)
        ttk.Label(frame_keys, text="Theta Key:").grid(row=1, column=0)
        entry_t = ttk.Entry(frame_keys, textvariable=self.entry_theta, width=20)
        entry_t.grid(row=1, column=1)
        ttk.Label(frame_keys, text="Energy Key:").grid(row=2, column=0)
        entry_e = ttk.Entry(frame_keys, textvariable=self.entry_energy, width=20)
        entry_e.grid(row=2, column=1)
        ttk.Label(frame_keys, text="Zone Lines:").grid(row=3, column=0)
        entry_zone = ttk.Entry(frame_keys, textvariable=self.entry_zone_lines, width=20)
        entry_zone.grid(row=3, column=1)
        frame_keys.pack(pady=10)

        # Checkbox for line plot
        frame_plot = ttk.Frame(win)
        ttk.Label(frame_plot, text="Plot Options:").pack(anchor="center")
        chk_line_plot = ttk.Checkbutton(
            frame_plot,
            text="Line Plot",
            variable=self.line_plot_var,
        )
        chk_line_plot.pack(anchor="center", pady=5)
        frame_plot.pack(pady=10)

        frame_fit = ttk.Frame(win)
        ttk.Label(frame_fit, text="Fit Options:").grid(row=0, columnspan=2)
        ttk.Combobox(
            frame_fit,
            textvariable=self.fit_function,
            values=["powerCosine", "vonMises"],
            state="readonly",
        ).grid(row=1, columnspan=2, pady=5)
        ttk.Label(frame_fit, text="Cutoff:").grid(row=2, column=0)
        entry_power = ttk.Entry(
            frame_fit, textvariable=self.fit_angle_p_cutoff, width=20
        )
        entry_power.grid(row=2, column=1)
        chk_fit_sin = ttk.Checkbutton(
            frame_fit,
            text="Use Sine in Fit",
            variable=self.fit_use_sin,
        )
        chk_fit_sin.grid(row=3, columnspan=2, pady=5)
        chk_fit_verbose = ttk.Checkbutton(
            frame_fit,
            text="Show All Plots",
            variable=self.fit_verbose,
        )
        chk_fit_verbose.grid(row=4, columnspan=2, pady=5)
        frame_fit.pack(pady=10)

        # Done button to close
        ttk.Button(win, text="OK", command=win.destroy).pack(anchor="s", pady=5)


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
    parser.add_argument("--fit", help="Fit distribution to data", action="store_true")

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

        if args.fit:
            if args.input == "ALL":
                print("Error: Variable input need for fit.")
                return

            if args.input not in variables:
                print(
                    f"Error: variable '{args.input}' not found in {args.file}",
                )
                return

            res = fit_data(data_dict, args.input, theta_key, energy_key, args.eps)
            if res == -1:
                print("Error fitting data.")
            return

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
