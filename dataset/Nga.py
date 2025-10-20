import os
import numpy as np
from torch.utils.data import Dataset
import torch

class NGADataset(Dataset):
    def __init__(self, root_dir, num_points, verbose=True):
        self.root_dir = root_dir
        self.num_points = num_points
        self.samples = []

        mach_list = [round(0.4 + 0.4 * i, 6) for i in range(6)]
        aoa_list = [round(-3 + 3 * i, 6) for i in range(6)]

        for sample_idx in range(1, 81):
            sample_name = f"sample-{sample_idx:03d}"
            folder = os.path.join(self.root_dir, sample_name)
            for mach in mach_list:
                for aoa in aoa_list:
                    # 只检查该组合下第1个 part 是否合法
                    filename = f"WallSurfGridFlowParameters_{mach:.6f}_0.000000_{aoa:.6f}_1_11.plt"
                    filepath = os.path.join(folder, filename)
                    if not os.path.exists(filepath):
                        if verbose:
                            print(f"[跳过] 缺失文件: {filepath}")
                        continue
                    try:
                        if self._is_invalid_file(filepath):
                            if verbose:
                                print(f"[跳过] 无效数据 (nan): {filepath}")
                            continue
                        self.samples.append((sample_name, mach, aoa))
                    except Exception as e:
                        if verbose:
                            print(f"[跳过] 解析失败: {filepath}, 错误: {e}")

        if verbose:
            print(f"有效样本数量: {len(self.samples)}")
            
        print('NGAFlight: load {} meshes for training.'.format(len(self.samples)))
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name, mach, aoa = self.samples[idx]
        folder = os.path.join(self.root_dir, sample_name)

        point_list = []
        for part_id in range(1, 12):
            filename = f"WallSurfGridFlowParameters_{mach:.6f}_0.000000_{aoa:.6f}_{part_id}_11.plt"
            filepath = os.path.join(folder, filename)
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"缺失文件: {filepath}")
            points = self._read_tecplot_cp(filepath)
            point_list.extend(points)

        point_cloud = torch.tensor(point_list, dtype=torch.float32)  # (N_total, 4)
        point_cloud[:,:3] = point_cloud[:,:3] / 1000
        point_cloud[:, 0] = point_cloud[:, 0] - (torch.max(point_cloud[:, 0]) + torch.min(point_cloud[:, 0]))/2
        point_cloud[:, 1] = point_cloud[:, 1] - (torch.max(point_cloud[:, 1]) + torch.min(point_cloud[:, 1]))/2
        point_cloud[:, 2] = point_cloud[:, 2] -  torch.min(point_cloud[:, 2])
        
        point_cloud = self.sample(point_cloud)
        
        cd = torch.tensor(float('nan'), dtype=torch.float32)

        return point_cloud[:,:3], point_cloud[:,3], torch.tensor([mach, aoa]), torch.tensor(1).long(), cd
    
    def sample(self, point_cloud):
        N = point_cloud.shape[0]
        if N > self.num_points:
            rand_idx = torch.randperm(N)
            point_cloud = point_cloud[rand_idx, :]
            return point_cloud[:self.num_points]
        elif N < self.num_points:
            idx = idx = torch.randint(0, N, (self.num_points,), device=point_cloud.device)
            return point_cloud[idx]

    def _is_invalid_file(self, filepath):
        """检查文件中 CP 是否全是 nan"""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        var_line = next((line for line in lines if line.strip().startswith("VARIABLES")), None)
        if var_line is None:
            return True

        var_names = [v.strip().strip('"') for v in var_line.split('=')[1].split(',')]
        try:
            cp_idx = var_names.index('CP')
        except ValueError:
            return True

        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("ZONE"):
                data_start = i + 1
                break

        for line in lines[data_start:]:
            if not line.strip():
                continue
            values = line.strip().split()
            if len(values) <= cp_idx:
                continue
            cp_val = values[cp_idx]
            if not ('nan' in cp_val or 'NaN' in cp_val):
                return False  # 有有效值
        return True  # 全是 nan

    def _read_tecplot_cp(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        var_line = next((line for line in lines if line.strip().startswith("VARIABLES")), None)
        var_names = [v.strip().strip('"') for v in var_line.split('=')[1].split(',')]
        x_idx = var_names.index('X')
        y_idx = var_names.index('Y')
        z_idx = var_names.index('Z')
        cp_idx = var_names.index('CP')

        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("ZONE"):
                data_start = i + 1
                break

        data = []
        for line in lines[data_start:]:
            if not line.strip():
                continue
            values = line.strip().split()
            if len(values) < len(var_names):
                continue
            try:
                point = [float(values[x_idx]), float(values[y_idx]), float(values[z_idx]), float(values[cp_idx])]
                if np.isnan(point[-1]):
                    continue  # 忽略CP为nan的点
                data.append(point)
            except:
                continue
        return data



class NGADataset_with_forces(Dataset):
    """
    Aircraft surface-pressure dataset with per-sample force/moment coefficients.

    Reads point-wise surface pressure coefficient (CP) from 11 Tecplot ASCII files per (sample, Mach, AoA),
    and also parses aggregate coefficients (CX, CY, CZ, Mx, My, Mz, CD, CL, K, Xcp) from
    "sample-xxx/sample-xxx-result.plt".

    Returns (in __getitem__):
        - xyz: (num_points, 3) float32, normalized and centered
        - cp:  (num_points,)     float32
        - flow: tensor([mach, aoa]) float32
        - label: torch.long (kept for backward-compat, value=1)
        - forces: tensor(len(force_vars),) in order of self.force_vars
    """
    def __init__(self, root_dir, num_points, verbose=True, force_vars=None):
        self.root_dir = root_dir
        self.num_points = num_points
        self.samples = []

        # force variables we will try to extract (case-insensitive match against VARIABLES line)
        self.force_vars = force_vars or ["CX","CZ","My","CD","CL"]

        mach_list = [round(0.4 + 0.4 * i, 6) for i in range(6)]
        aoa_list = [round(-3 + 3 * i, 6) for i in range(6)]

        for sample_idx in range(81, 101):
            sample_name = f"sample-{sample_idx:03d}"
            folder = os.path.join(self.root_dir, sample_name)
            for mach in mach_list:
                for aoa in aoa_list:
                    # Check the first part file for existence/validity
                    filename = f"WallSurfGridFlowParameters_{mach:.6f}_0.000000_{aoa:.6f}_1_11.plt"
                    filepath = os.path.join(folder, filename)
                    if not os.path.exists(filepath):
                        if verbose:
                            print(f"[跳过] 缺失文件: {filepath}")
                        continue
                    try:
                        if self._is_invalid_file(filepath):
                            if verbose:
                                print(f"[跳过] 无效数据 (nan): {filepath}")
                            continue
                        # Optionally verify result file exists (do not filter out if missing; we'll warn on access)
                        result_path = os.path.join(folder, f"{sample_name}-result.plt")
                        if not os.path.exists(result_path):
                            if verbose:
                                print(f"[提示] 未找到合力文件(将以NaN返回): {result_path}")
                            continue
                        self.samples.append((sample_name, mach, aoa))
                    except Exception as e:
                        if verbose:
                            print(f"[跳过] 解析失败: {filepath}, 错误: {e}")

        print('NGADFlight: load {} meshes for training.'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name, mach, aoa = self.samples[idx]
        folder = os.path.join(self.root_dir, sample_name)

        # --- read and merge 11 parts ---
        point_list = []
        for part_id in range(1, 12):
            filename = f"WallSurfGridFlowParameters_{mach:.6f}_0.000000_{aoa:.6f}_{part_id}_11.plt"
            filepath = os.path.join(folder, filename)
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"缺失文件: {filepath}")
            points = self._read_tecplot_cp(filepath)
            point_list.extend(points)

        point_cloud = torch.tensor(point_list, dtype=torch.float32)  # (N_total, 4) -> [x,y,z,cp]

        # normalize/center (kept as in original code)
        point_cloud[:, :3] = point_cloud[:, :3] / 1000.0
        point_cloud[:, 0] = point_cloud[:, 0] - (torch.max(point_cloud[:, 0]) + torch.min(point_cloud[:, 0])) / 2.0
        point_cloud[:, 1] = point_cloud[:, 1] - (torch.max(point_cloud[:, 1]) + torch.min(point_cloud[:, 1])) / 2.0
        point_cloud[:, 2] = point_cloud[:, 2] - torch.min(point_cloud[:, 2])

        point_cloud = self.sample(point_cloud)

        # --- read forces from result file ---
        result_path = os.path.join(folder, f"{sample_name}-result.plt")
        forces = self._read_result_forces(result_path, mach, aoa)
        
        # print(folder, mach, aoa, torch.max(forces), torch.min(forces))

        return point_cloud[:, :3], point_cloud[:, 3], torch.tensor([mach, aoa], dtype=torch.float32), torch.tensor(1).long(), forces

    def sample(self, point_cloud):
        N = point_cloud.shape[0]
        if N > self.num_points:
            rand_idx = torch.randperm(N)
            point_cloud = point_cloud[rand_idx, :]
            return point_cloud[:self.num_points]
        elif N < self.num_points:
            idx = torch.randint(0, N, (self.num_points,), device=point_cloud.device)
            return point_cloud[idx]
        return point_cloud

    def _is_invalid_file(self, filepath):
        """检查文件中 CP 是否全是 nan"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        var_line = next((line for line in lines if line.strip().startswith("VARIABLES")), None)
        if var_line is None:
            return True

        var_names = [v.strip().strip('"') for v in var_line.split('=')[1].split(',')]
        try:
            cp_idx = var_names.index('CP')
        except ValueError:
            return True

        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("ZONE"):
                data_start = i + 1
                break

        for line in lines[data_start:]:
            if not line.strip():
                continue
            values = line.strip().split()
            if len(values) <= cp_idx:
                continue
            cp_val = values[cp_idx]
            if not ('nan' in cp_val or 'NaN' in cp_val):
                return False  # 有有效值
        return True  # 全是 nan

    def _read_tecplot_cp(self, filename):
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        var_line = next((line for line in lines if line.strip().startswith("VARIABLES")), None)
        var_names = [v.strip().strip('"') for v in var_line.split('=')[1].split(',')]
        x_idx = var_names.index('X')
        y_idx = var_names.index('Y')
        z_idx = var_names.index('Z')
        cp_idx = var_names.index('CP')

        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("ZONE"):
                data_start = i + 1
                break

        data = []
        for line in lines[data_start:]:
            if not line.strip():
                continue
            values = line.strip().split()
            if len(values) < len(var_names):
                continue
            try:
                point = [float(values[x_idx]), float(values[y_idx]), float(values[z_idx]), float(values[cp_idx])]
                if np.isnan(point[-1]):
                    continue  # 忽略CP为nan的点
                data.append(point)
            except Exception:
                continue
        return data

    @staticmethod
    def _to_lower_map(names):
        """Helper: case-insensitive name->index mapping"""
        return {n.strip().strip('"').lower(): i for i, n in enumerate(names)}

    def _read_result_forces(self, result_filepath, mach, aoa, tol=5e-3):
        """
        Parse 'sample-xxx-result.plt' to fetch force/moment for the given (mach, aoa).
        Returns a 1D tensor(len(force_vars),) in the fixed order of self.force_vars.
        """
        out_dict = {k: float('nan') for k in self.force_vars}

        if not os.path.exists(result_filepath):
            return torch.tensor([out_dict[k] for k in self.force_vars], dtype=torch.float32)

        with open(result_filepath, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')

        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        # find VARIABLES line
        var_line = None
        for ln in lines:
            if ln.upper().startswith("VARIABLES"):
                var_line = ln
                break
        if var_line is None or '=' not in var_line:
            return torch.tensor([out_dict[k] for k in self.force_vars], dtype=torch.float32)

        # parse variable names
        var_part = var_line.split('=', 1)[1]
        var_names = [v.strip().strip('"') for v in var_part.split(',')]
        name2idx = self._to_lower_map(var_names)

        # ensure columns exist
        need_cols = {
            'ma': None,
            'alpha': None,
        }
        for k in list(need_cols.keys()):
            if k in name2idx:
                need_cols[k] = name2idx[k]
        if need_cols['ma'] is None or need_cols['alpha'] is None:
            return torch.tensor([out_dict[k] for k in self.force_vars], dtype=torch.float32)

        # scan data lines after ZONE
        data_rows = []
        after_zone = False
        for ln in lines:
            if ln.upper().startswith("ZONE"):
                after_zone = True
                continue
            if not after_zone:
                continue
            parts = ln.split()
            if len(parts) < len(var_names):
                continue
            try:
                vals = [float(p) for p in parts[:len(var_names)]]
                data_rows.append(vals)
            except ValueError:
                continue

        # find matching row
        target_row = None
        for vals in data_rows:
            ma_val = vals[name2idx['ma']]
            aoa_val = vals[name2idx['alpha']]
            if abs(ma_val - float(mach)) <= tol and abs(aoa_val - float(aoa)) <= tol:
                target_row = vals
                break

        if target_row is not None:
            for key in self.force_vars:
                k_lower = key.lower()
                if k_lower in name2idx:
                    out_dict[key] = target_row[name2idx[k_lower]]

        return torch.tensor([out_dict[k] for k in self.force_vars], dtype=torch.float32)


if __name__ == '__main__':
    dataset = NGADataset_with_forces('/data/home/zdhs0017/zoujunhong/data/Flight/nga', num_points=8192, verbose=False)
    for i in range(dataset.__len__()):
        dataset.__getitem__(i)