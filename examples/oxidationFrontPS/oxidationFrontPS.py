import sys
import os
import viennaps as vps
import numpy as np
from scipy.sparse import coo_matrix

import geometryPS

# Material IDs
MAT_SUBSTRATE = 0
MAT_OXIDE = 1
MAT_MASK = 2
MAT_AMBIENT = 3

class OxidationSimulation:
    """
    Electric Field Driven Oxidation Model - Python implementation using ViennaPS.
    Replicates oxidationFront.py functionality.
    """
    def __init__(self, config_file):
        self.params = vps.readConfigFile(config_file)
        
        self.dim = int(self.params.get("dimensions", 3))
        vps.setDimension(self.dim)
        vps.setNumThreads(int(self.params.get('numThreads', 1)))
        
        # 1. Geometry
        ls_list, mat_map, _, _, grid_delta = geometryPS.get_geometry_domains(self.params)
        self.grid_delta = grid_delta
        
        # 2. DenseCellSet
        print("Building DenseCellSet...")
        # ViennaPS exposes DenseCellSet via vps.d2 or vps.d3 based on setDimension
        self.cell_set = vps.DenseCellSet()
        
        depth = self.params['substrateHeight'] + self.params.get('ambientHeight', grid_delta)
        self.cell_set.setCellSetPosition(True)
        self.cell_set.setCoverMaterial(MAT_AMBIENT) # Ensure gas is MAT_AMBIENT
        self.cell_set.fromLevelSets(ls_list, mat_map, depth)
        self.cell_set.buildNeighborhood()
        self.num_cells = self.cell_set.getNumberOfCells()
        
        # 3. Data Fields
        self.cell_set.addScalarData("oxidant", 0.0)
        self.cell_set.addScalarData("oxideFraction", 0.0)
        
        # Python State
        self.materials = np.array(self.cell_set.getScalarData("Material"), dtype=int)
        self.oxidant = np.zeros(self.num_cells)
        self.oxide_fraction = np.zeros(self.num_cells)
        self.e_field_1d = np.zeros(self.num_cells)

        # Pre-calculate neighbor map
        self._build_neighbor_map()
        
        # 4. Initialize oxidant in ambient cells
        is_ambient = self.materials == MAT_AMBIENT
        self.oxidant[is_ambient] = self.params['ambientOxidant']
        
        # 5. Pre-compute E-Field mapping
        self._precompute_efield_indices()

        # 6. Build Topology
        print("Building Sparse Topology...")
        self._build_topology()

        # 7. Sync initial state and write
        self._sync_data()
        self.cell_set.writeVTU("oxidation_initial.vtu")

    def _build_neighbor_map(self):
        print("Building neighbor map for vectorization...")
        self.max_neighbors = 2 * self.dim
        self.neighbor_map = np.full((self.num_cells, self.max_neighbors), -1, dtype=np.int32)

        for idx in range(self.num_cells):
            nbs = self.cell_set.getNeighbors(idx)
            count = min(len(nbs), self.max_neighbors)
            if count > 0:
                self.neighbor_map[idx, :count] = nbs[:count]

    def _sync_data(self):
        self.cell_set.setScalarData("oxidant", self.oxidant)
        self.cell_set.setScalarData("oxideFraction", self.oxide_fraction)
        self.cell_set.setScalarData("Material", self.materials.astype(float))

    def _precompute_efield_indices(self):
        csv_dx = 1.0
        csv_extent = 150.0
        nx = int(csv_extent / csv_dx)

        centers = np.array([self.cell_set.getCellCenter(i) for i in range(self.num_cells)])
        gen_x = centers[:, 0] + (csv_extent / 2.0)
        ix = np.clip((gen_x / csv_dx).astype(int), 0, nx - 1)

        if self.dim == 2:
            self.efield_grid_indices = ix
        else:
            gen_y = centers[:, 1] + (csv_extent / 2.0)
            iy = np.clip((gen_y / csv_dx).astype(int), 0, nx - 1)
            self.efield_grid_indices = iy * nx + ix

    def _load_efield_data(self, time):
        filename = self.params['EfieldFile']
        if not os.path.exists(filename):
            print(f"[ERROR] {filename} not found.")
            sys.exit(1)

        try:
            raw_data = np.genfromtxt(filename, delimiter=',')
        except Exception as e:
            print(f"[ERROR] Reading E-field: {e}")
            sys.exit(1)

        if len(raw_data.shape) == 1:
            return np.array([raw_data[-1]])

        if self.dim == 2:
            return raw_data[:, 1] if raw_data.shape[1] >= 2 else raw_data[:, -1]
        else:
            return raw_data[:, 2] if raw_data.shape[1] >= 3 else raw_data[:, -1]

    def _update_electric_field(self, time):
        e_field_values = self._load_efield_data(time)
        valid_mask = (self.efield_grid_indices >= 0) & (self.efield_grid_indices < len(e_field_values))
        self.e_field_1d[:] = 0.0
        self.e_field_1d[valid_mask] = e_field_values[self.efield_grid_indices[valid_mask]]

    def _build_topology(self):
        is_mat_active = (self.materials == MAT_SUBSTRATE) | (self.materials == MAT_OXIDE)
        cond_oxide = self.oxide_fraction > 1e-6

        valid_nb = self.neighbor_map >= 0
        safe_nbs = np.where(valid_nb, self.neighbor_map, 0)
        nb_mats = self.materials[safe_nbs]
        nb_is_ambient = (nb_mats == MAT_AMBIENT) & valid_nb
        touching_ambient = np.any(nb_is_ambient, axis=1)

        is_core_active = is_mat_active & (cond_oxide | touching_ambient)

        nb_is_core = is_core_active[safe_nbs] & valid_nb
        touching_core = np.any(nb_is_core, axis=1)

        is_active = is_mat_active & (is_core_active | touching_core)

        self.active_indices = np.where(is_active)[0]
        self.is_active_cell = is_active
        self.n_dof = len(self.active_indices)
        print(f"Active cells: {self.n_dof} / {self.num_cells}")

        self._build_diffusion_structure()

    def _build_diffusion_structure(self):
        active_idxs = self.active_indices
        self.ambient_neighbor_count = np.zeros(self.num_cells, dtype=np.int32)

        if len(active_idxs) == 0:
            self.diff_rows = np.array([], dtype=np.int64)
            self.diff_cols = np.array([], dtype=np.int64)
            return

        nbs = self.neighbor_map[active_idxs]
        valid_nb = nbs >= 0
        safe_nbs = np.where(valid_nb, nbs, 0)
        mat_nbs = self.materials[safe_nbs]

        rows_list = []
        cols_list = []

        for k in range(self.max_neighbors):
            valid = valid_nb[:, k]
            mat_n = mat_nbs[:, k]
            idx_n = safe_nbs[:, k]

            is_amb = (mat_n == MAT_AMBIENT) & valid
            np.add.at(self.ambient_neighbor_count, active_idxs[is_amb], 1)

            is_active_nb = (self.is_active_cell[idx_n] & valid & (mat_n != MAT_AMBIENT) & (mat_n != MAT_MASK))
            if np.any(is_active_nb):
                rows_list.append(active_idxs[is_active_nb])
                cols_list.append(idx_n[is_active_nb])

        self.diff_rows = np.concatenate(rows_list) if rows_list else np.array([], dtype=np.int64)
        self.diff_cols = np.concatenate(cols_list) if cols_list else np.array([], dtype=np.int64)

    def solve_step(self, dt):
        dx = self.grid_delta
        dtdx2 = dt / (dx * dx)
        ambient_oxidant = self.params['ambientOxidant']
        D_ox = self.params['oxidantDiffusivity']

        k_base = self.params['reactionRateConstant']
        alpha = self.params['eFieldInfluence']
        E_mag = np.abs(self.e_field_1d)
        available_si = np.maximum(0.0, 1.0 - self.oxide_fraction)
        reaction_rates = k_base * (1.0 + alpha * E_mag) * available_si

        active_idxs = self.active_indices
        if len(active_idxs) > 0:
            C_old = self.oxidant[active_idxs]
            k = reaction_rates[active_idxs]
            n = self.num_cells

            if len(self.diff_rows) > 0:
                D_eff = 0.5 * D_ox * (self.oxide_fraction[self.diff_rows] + self.oxide_fraction[self.diff_cols])
                diag = np.zeros(n)
                np.add.at(diag, self.diff_rows, -D_eff)
                diag[active_idxs] -= self.ambient_neighbor_count[active_idxs] * D_ox

                all_rows = np.concatenate([self.diff_rows, active_idxs])
                all_cols = np.concatenate([self.diff_cols, active_idxs])
                all_vals = np.concatenate([D_eff, diag[active_idxs]])

                A = coo_matrix((all_vals, (all_rows, all_cols)), shape=(n, n)).tocsr()
                diff_term = A.dot(self.oxidant)[active_idxs]
            else:
                diff_term = -self.ambient_neighbor_count[active_idxs] * D_ox * C_old

            diff_term += self.ambient_neighbor_count[active_idxs] * D_ox * ambient_oxidant
            self.oxidant[active_idxs] = np.maximum(0.0, C_old + dtdx2 * diff_term - dt * k * C_old)

        mask_update = ((self.materials == MAT_SUBSTRATE) | (self.materials == MAT_OXIDE)) & (self.oxidant > 1e-12)
        idxs_update = np.where(mask_update)[0]
        
        if len(idxs_update) > 0:
            C = self.oxidant[idxs_update]
            k = reaction_rates[idxs_update]
            d_frac = k * C * dt
            
            self.oxide_fraction[idxs_update] += d_frac
            self.oxide_fraction[idxs_update] = np.minimum(1.0, self.oxide_fraction[idxs_update])
            
            new_oxides = (self.oxide_fraction[idxs_update] > 0.5) & (self.materials[idxs_update] == MAT_SUBSTRATE)
            if np.any(new_oxides):
                self.materials[idxs_update[new_oxides]] = MAT_OXIDE
                return True
        return False

    def run(self):
        duration = self.params['duration']
        D_ox = self.params['oxidantDiffusivity']
        stability_factor = 2 * self.dim
        dt_base = (self.grid_delta**2 / (D_ox * stability_factor)) * self.params['timeStabilityFactor']

        time = 0.0
        step = 0
        topology_rebuild_interval = int(self.params.get('topologyRebuildInterval', 5))
        need_topology_rebuild = True
        print(f"Starting simulation. Duration: {duration}, initial dt: {dt_base:.4f}")
        
        while time < duration:
            if need_topology_rebuild or step % topology_rebuild_interval == 0:
                self._build_topology()
                need_topology_rebuild = False

            self._update_electric_field(time)

            if len(self.active_indices) > 0:
                k_base = self.params['reactionRateConstant']
                alpha = self.params['eFieldInfluence']
                E_mag = np.abs(self.e_field_1d[self.active_indices])
                avail_si = np.maximum(0.0, 1.0 - self.oxide_fraction[self.active_indices])
                rates = k_base * (1.0 + alpha * E_mag) * avail_si
                max_k = np.max(rates)
            else:
                max_k = 0.0

            dt_diff = self.grid_delta**2 / (D_ox * stability_factor)
            dt_react = (1.0 / max_k) if max_k > 1e-12 else duration
            dt = min(dt_diff, dt_react) * self.params['timeStabilityFactor']

            if time + dt > duration:
                dt = duration - time

            material_changed = self.solve_step(dt)
            if material_changed:
                need_topology_rebuild = True
            time += dt
            step += 1
            
            if step % 10 == 0:
                self._sync_data()
                print(f"Step {step}, Time: {time:.4f}, dt: {dt:.4f}, max_k: {max_k:.4f}")
                self.cell_set.writeVTU(f"oxidation_step_{step}.vtu")
                
        self._sync_data()
        self.cell_set.writeVTU("oxidation_final.vtu")
        print("Simulation Done.")


if __name__ == "__main__":
    config_file = "config.txt"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    sim = OxidationSimulation(config_file)
    sim.run()