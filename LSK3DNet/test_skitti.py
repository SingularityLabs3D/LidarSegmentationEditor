import os, sys, struct
import random
import requests
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import shutil
from easydict import EasyDict
from tqdm import tqdm

import yaml
from utils.load_util import load_yaml
from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name, get_pc_model_class
from dataloader.dataset2 import get_dataset_class, get_collate_class
from network.largekernel_model import get_model_class

from utils.load_save_util import load_checkpoint_old, load_checkpoint_model_mask
from utils.erk_sparse_core import Masking, CosineDecay
import open3d as o3d
import matplotlib.pyplot as plt
import warnings
from copy import deepcopy

from itertools import product
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from converter import run_conversion_pipeline

import json 
from sklearn.cluster import DBSCAN

# fastapi
import os, sys, struct
from fastapi import FastAPI
import uvicorn


# =========================
# FastAPI service wrapper
# =========================
app = FastAPI(title="Segmentation Service")


@app.post("/segment")
def segment():
    global configs
    """
    Trigger segmentation for a given PCD path.
    Copies the file into $WORKDIR/points_uncompressed.pcd and runs the pipeline.
    Returns output artifact paths.
    """
    main(configs)

    # workdir = os.getenv("WORKDIR", ".")
    # os.makedirs(workdir, exist_ok=True)
    # dst = os.path.join(workdir, "points_uncompressed.pcd")

    # # copy source into expected location
    # shutil.copy2(path, dst)

    # # run single-GPU worker inline
    # main_worker(0, 1, configs)

    # out_seg = os.path.join(workdir, "points_segmented.pcd")
    # out_clean = os.path.join(workdir, "points_cleaned.pcd")
    return {
        "status": "ok",
    }


instances = []
@app.get("/detections")
def segment():
    global instances
    return instances


@app.post("/delete")
def segment():
    global instances
    print("Saved to output_normalized.ply/pcd")
    params = {"path": os.path.join(os.getenv("WORKDIR"), "points_cleaned.pcd")}
    headers = {"accept": "application/json"}
    print("fetching...")
    timeout = 120
    r = requests.get(f"http://converter:8000/convert", params=params, headers=headers, timeout=timeout)
    print("fetched...")
    instances = []

    return {
        "status": "ok",
    }


warnings.filterwarnings("ignore")

DYNAMIC_MAPPING = {
    "car": "dynamic",
    "bicycle": "dynamic",
    "motorcycle": "dynamic",
    "motorcyclist": "dynamic",
    "bus": "dynamic",
    "bicyclist": "dynamic",
    "person": "dynamic",
    "truck": "dynamic"
}

args = {
    'config_path': "./config/lk-semantickitti_sub_tta.yaml",
    'ip': '127.0.0.1',
    'port': '3023',
    'num_vote': 8
}
config_path = args['config_path']
configs = load_yaml(config_path)
configs.update(args)  # override the configuration using the value in args
configs = EasyDict(configs)

configs['dataset_params']['val_data_loader']["batch_size"] = configs.num_vote
configs['dataset_params']['val_data_loader']["num_workers"] = configs.num_vote
if configs.num_vote > 1:
    configs['dataset_params']['val_data_loader']["rotate_aug"] = True
    configs['dataset_params']['val_data_loader']["flip_aug"] = True
    configs['dataset_params']['val_data_loader']["scale_aug"] = True
    configs['dataset_params']['val_data_loader']["transform_aug"] = True
elif configs.num_vote == 1:
    configs['dataset_params']['val_data_loader']["rotate_aug"] = False
    configs['dataset_params']['val_data_loader']["flip_aug"] = False
    configs['dataset_params']['val_data_loader']["scale_aug"] = False
    configs['dataset_params']['val_data_loader']["transform_aug"] = False

exp_dir_root = configs['model_params']['model_load_path'].split('/')
exp_dir_root = exp_dir_root[0] if len(exp_dir_root) > 1 else ''
exp_dir = './'+ exp_dir_root +'/'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# shutil.copy('test_skitti.py', str(exp_dir))
# shutil.copy('config/lk-semantickitti_sub_tta.yaml', str(exp_dir))


def main(configs):
    configs.nprocs = torch.cuda.device_count()
    configs.train_params.distributed = True if configs.nprocs > 1 else False
    if configs.train_params.distributed:
        mp.spawn(main_worker, nprocs=configs.nprocs, args=(configs.nprocs, configs))
    else:
        main_worker(0, 1, configs)

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

WHITESPACE = {9, 10, 13, 32}  # \t \n \r space

def _read_header(f):
    hdr, data_fmt = {}, None
    while True:
        b = f.readline()
        if not b:
            raise ValueError("EOF while reading header")
        line = b.decode("ascii", "ignore").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key, vals = parts[0].upper(), parts[1:]
        if key == "DATA":
            data_fmt = vals[0].lower()
            break
        hdr[key] = vals
    return hdr, data_fmt, f.tell()

def _ints(xs): return [int(x) for x in xs]
def _maybe_int(h,k): return int(h[k][0]) if k in h and h[k] else None

def _point_step_from_sizecount(h):
    if "SIZE" not in h: return None
    sizes = _ints(h["SIZE"])
    counts = _ints(h.get("COUNT", ["1"] * len(sizes)))
    if len(counts) != len(sizes):
        raise ValueError("COUNT length != SIZE length")
    return sum(s * c for s, c in zip(sizes, counts))

def _has_only_trailing_ws(f, extra_bytes):
    if extra_bytes <= 0: return True
    take = min(8, extra_bytes)
    f.seek(-take, os.SEEK_END)
    tail = f.read(take)
    return all(b in WHITESPACE for b in tail)

def validate_pcd(path, allow_trailing_ws=True):
    with open(path, "rb") as f:
        hdr, data_fmt, data_off = _read_header(f)

        pts = _maybe_int(hdr, "POINTS")
        w, h = _maybe_int(hdr, "WIDTH"), _maybe_int(hdr, "HEIGHT")
        if pts is None and (w is not None and h is not None):
            pts = w * h
        if pts is None:
            return False, "Missing POINTS and/or WIDTH×HEIGHT"

        point_step_hdr = _maybe_int(hdr, "POINT_STEP")
        row_step_hdr   = _maybe_int(hdr, "ROW_STEP")
        point_step_szc = _point_step_from_sizecount(hdr)

        fsz = os.fstat(f.fileno()).st_size
        data_len = fsz - data_off

        if data_fmt == "ascii":
            fields_total = sum(_ints(hdr.get("COUNT", ["1"] * len(hdr.get("SIZE", [])))))
            lines = bad = 0
            for b in f:
                s = b.strip()
                if not s: continue
                lines += 1
                if fields_total and len(s.split()) != fields_total:
                    bad += 1
            if lines != pts: return False, f"ASCII rows {lines} != POINTS {pts}"
            if bad: return False, f"ASCII rows with wrong column count: {bad}"
            return True, "OK (ascii)"

        elif data_fmt == "binary_compressed":
            if data_len < 8: return False, "Too short for compressed header"
            f.seek(data_off)
            comp_sz, uncomp_sz = struct.unpack("<II", f.read(8))
            payload = data_len - 8
            if payload != comp_sz:
                return False, f"Compressed payload {payload} != comp_sz {comp_sz}"
            expected_uncomp = []
            if point_step_hdr: expected_uncomp.append(("POINT_STEP", pts * point_step_hdr))
            if row_step_hdr and h: expected_uncomp.append(("ROW_STEP", row_step_hdr * h))
            if point_step_szc: expected_uncomp.append(("SIZE×COUNT", pts * point_step_szc))
            for label, expect in expected_uncomp:
                if uncomp_sz == expect:
                    return True, f"OK (binary_compressed; via {label})"
            return False, f"Uncompressed {uncomp_sz} != any of {[e for _, e in expected_uncomp]}"

        elif data_fmt == "binary":
            # base expectation from SIZE×COUNT (or POINT_STEP if present)
            if point_step_hdr:
                expect = pts * point_step_hdr
            elif point_step_szc:
                expect = pts * point_step_szc
            else:
                return False, "Cannot determine point size (no SIZE/COUNT or POINT_STEP)"

            if data_len == expect:
                return True, "OK (binary; exact)"

            extra = data_len - expect
            # tolerate trailing \n or \r\n
            if allow_trailing_ws and extra in (1, 2) and _has_only_trailing_ws(f, extra):
                return True, f"OK (binary; tolerated trailing {extra} byte(s))"

            # infer per-row padding if organized (WIDTH×HEIGHT)
            if h and extra > 0 and extra % h == 0:
                per_row_pad = extra // h
                # accept reasonable alignment paddings (multiples of 4 up to 512 bytes)
                if per_row_pad % 4 == 0 and per_row_pad <= 512:
                    return True, f"OK (binary; inferred {per_row_pad} bytes of row padding × {h} rows)"

            return False, (f"Size mismatch: data={data_len} bytes, expected={expect} "
                           f"(diff={extra}); header at offset {data_off})")

        else:
            return False, f"Unknown DATA format: {data_fmt}"


def _chunk_worker(args):
    (x_min, x_max, y_min, y_max, z_min, z_max), shm_name, shape, dtype = args
    shm = SharedMemory(name=shm_name)
    points = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    idxs = np.nonzero(mask)[0]
    shm.close()
    return idxs  # may be empty

import os
import numpy as np
from typing import List, Tuple, Optional
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory

# --- worker for parallel path ---
def _slice_worker(args):
    shm_name, shape, dtype, start, end = args
    shm = SharedMemory(name=shm_name)
    try:
        base = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        sub = base[start:end].copy()             # materialize the chunk in worker
        idxs = np.arange(start, end, dtype=np.int64)
        return (sub, idxs)
    finally:
        shm.close()

# def split_pointcloud_spatial(points: np.ndarray,
#                              chunk_size: int,
#                              voxel: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray]]:
#     """
#     Spatially group points by voxel grid, then split into size-based chunks.
#     Returns (sub_points, original_indices) for each chunk.
#     """
#     if not isinstance(points, np.ndarray) or points.ndim < 2 or points.shape[0] == 0:
#         return []
#     N = points.shape[0]
# 
#     # Quantize to a voxel grid
#     pmin = points[:, :3].min(axis=0)
#     ijk = np.floor((points[:, :3] - pmin) / float(voxel)).astype(np.int64)  # (N,3)
#     ix, iy, iz = ijk[:,0], ijk[:,1], ijk[:,2]
# 
#     # Build a lexicographic key (x-major, then y, then z)
#     Y = int(iy.max()) + 1
#     Z = int(iz.max()) + 1
#     key = (ix * Y + iy) * Z + iz  # 64-bit safe for typical ranges
# 
#     order = np.argsort(key, kind="mergesort")          # stable
#     pts_sorted = points[order]
#     idx_sorted = order
# 
#     # Slice into consecutive chunks (now spatially grouped)
#     starts = np.arange(0, N, chunk_size, dtype=np.int64)
#     ends = np.minimum(starts + chunk_size, N)
# 
#     chunks = [(pts_sorted[s:e], idx_sorted[s:e]) for s, e in zip(starts, ends)]
#     return chunks


def split_pointcloud_spatial(points: np.ndarray,
                             chunk_size: int,
                             voxel: float = 1.0,
                             overlap: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Spatially group points by voxel grid, then split into size-based chunks with optional overlap.
    Returns (sub_points, original_indices) for each chunk.

    Args:
        points: (N, D) array (first 3 columns are XYZ).
        chunk_size: max points per chunk.
        voxel: voxel size used only to build a spatial sort key (no downsampling).
        overlap: number of points to overlap between consecutive chunks (0 = no overlap).

    Notes:
        - Overlap is along the spatially-sorted sequence, so overlapped regions are near each other in space.
        - Points will appear in multiple chunks when overlap > 0.
    """
    if not isinstance(points, np.ndarray) or points.ndim < 2 or points.shape[0] == 0:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    N = points.shape[0]

    # Quantize to voxel grid to obtain a spatially-coherent sort order
    pmin = points[:, :3].min(axis=0)
    ijk = np.floor((points[:, :3] - pmin) / float(voxel)).astype(np.int64)  # (N,3)
    ix, iy, iz = ijk[:, 0], ijk[:, 1], ijk[:, 2]

    # Build lexicographic key (x-major, then y, then z)
    Y = int(iy.max()) + 1
    Z = int(iz.max()) + 1
    key = (ix * Y + iy) * Z + iz

    order = np.argsort(key, kind="mergesort")  # stable
    pts_sorted = points[order]
    idx_sorted = order

    # Sliding window with overlap
    step = chunk_size - overlap
    starts = np.arange(0, N, step, dtype=np.int64)
    ends = np.minimum(starts + chunk_size, N)

    chunks = [(pts_sorted[s:e], idx_sorted[s:e]) for s, e in zip(starts, ends)]
    return chunks


# def split_pointcloud(points: np.ndarray,
#                      chunk_size: int,
#                      processes: Optional[int] = None,
#                      task_chunk: int = 64,
#                      copy_on_single: bool = False
#                      ) -> List[Tuple[np.ndarray, np.ndarray]]:
#     """
#     Split (N, 3[+...]) points into consecutive chunks of `chunk_size`.
#     Ensures every point belongs to exactly one output chunk.
# 
#     Returns: list of (sub_points, idxs) where:
#       - sub_points: (M, D) view (single-process) or copy (multiprocess)
#       - idxs:       (M,) indices into the original array
#     """
#     if not isinstance(points, np.ndarray):
#         raise TypeError("points must be a numpy.ndarray")
#     if points.ndim < 2 or points.shape[0] == 0:
#         return []
#     if chunk_size <= 0:
#         raise ValueError("chunk_size must be > 0")
# 
#     points = np.ascontiguousarray(points)
#     N = points.shape[0]
#     D = points.shape[1]
# 
#     # Compute chunk boundaries
#     starts = np.arange(0, N, chunk_size, dtype=np.int64)
#     ends = np.minimum(starts + chunk_size, N)
#     ranges = list(zip(starts.tolist(), ends.tolist()))
# 
#     procs = processes or 1
#     out: List[Tuple[np.ndarray, np.ndarray]] = []
# 
#     if procs <= 1:
#         # Fast path: return views (no copy). Optionally copy if requested.
#         for s, e in ranges:
#             sub = points[s:e]
#             if copy_on_single:
#                 sub = sub.copy()
#             idxs = np.arange(s, e, dtype=np.int64)
#             out.append((sub, idxs))
#         return out
# 
#     # Parallel path: use shared memory to avoid serializing the whole array
#     shm = SharedMemory(create=True, size=points.nbytes)
#     try:
#         shm_np = np.ndarray(points.shape, dtype=points.dtype, buffer=shm.buf)
#         shm_np[:] = points  # one copy into shared memory
# 
#         tasks = [(shm.name, points.shape, points.dtype, s, e) for (s, e) in ranges]
# 
#         with Pool(processes=procs) as pool:
#             # chunksize controls how many slice-jobs each worker pulls at once
#             for sub, idxs in pool.imap_unordered(_slice_worker, tasks, chunksize=task_chunk):
#                 out.append((sub, idxs))
#     finally:
#         shm.close()
#         shm.unlink()
# 
#     # Note: imap_unordered returns in arbitrary order; restore original order if desired
#     # (Only needed if you rely on order of chunks; indices remain correct either way.)
#     out.sort(key=lambda t: t[1][0])
#     return out


# def split_pointcloud(points, cube_size=50.0, overlap=5.0, processes=None, task_chunk=64):
#     """
#     Parallel split of (N,3) points into overlapping cubes (cube_size).
#     Returns list of (sub_points, idxs).
#     """
#     if overlap >= cube_size:
#         raise ValueError("overlap must be < cube_size")
# 
#     points = np.ascontiguousarray(points)
#     min_xyz = points.min(axis=0)
#     max_xyz = points.max(axis=0)
# 
#     step = cube_size - overlap
#     xs = np.arange(min_xyz[0], max_xyz[0] + step, step)
#     ys = np.arange(min_xyz[1], max_xyz[1] + step, step)
#     zs = np.arange(min_xyz[2], max_xyz[2] + step, step)
# 
#     # Prepare shared memory for points
#     shm = SharedMemory(create=True, size=points.nbytes)
#     shm_np = np.ndarray(points.shape, dtype=points.dtype, buffer=shm.buf)
#     shm_np[:] = points  # copy once
# 
#     bounds = []
#     for x, y, z in product(xs, ys, zs):
#         bounds.append(((x, x + cube_size, y, y + cube_size, z, z + cube_size),
#                        shm.name, points.shape, points.dtype))
# 
#     procs = processes or os.cpu_count() or 1
#     chunks = []
#     try:
#         with Pool(processes=procs) as pool:
#             for idxs in pool.imap_unordered(_chunk_worker, bounds, chunksize=task_chunk):
#                 if idxs.size:
#                     chunks.append((points[idxs], idxs))
#     finally:
#         shm.close()
#         shm.unlink()
# 
#     return chunks


class SphereCrop(object):
    def __init__(self, point_max=120000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = 120000
        # point_max = int(self.sample_rate * data_dict["points"].shape[0]) if self.sample_rate is not None else self.point_max
        assert "points" in data_dict.keys()
        
        if data_dict["points"].shape[0] > point_max:
            if self.mode == "random":
                center_idx = np.random.randint(data_dict["points"].shape[0])
                center = data_dict["points"][center_idx]
            elif self.mode == "center":
                center_idx = data_dict["points"].shape[0] // 2
                center = data_dict["points"][center_idx]
            else:
                raise NotImplementedError
            # Преобразование в NumPy для вычислений
            points_np = data_dict["points"][:,:3].cpu().numpy()
            center_np = center[:3].cpu().numpy()
    
            # Вычисляем расстояния и сортируем
            distances = np.sum(np.square(points_np - center_np), 1)
            idx_crop = np.argsort(distances)[:point_max]  # Индексы в пределах исходного размера
            # Применяем индексы ко всем полям
            data_dict["points"] = data_dict["points"][idx_crop]
            data_dict["labels"] = data_dict["labels"][idx_crop] if "labels" in data_dict else torch.zeros_like(data_dict["points"][:, 0])
            data_dict["normal"] = data_dict["normal"][idx_crop] if "normal" in data_dict else torch.zeros_like(data_dict["points"][:, :3])
            data_dict["ref_index"] = data_dict["ref_index"][idx_crop] if "ref_index" in data_dict else torch.arange(point_max)
            data_dict["point_num"] = data_dict["points"].shape[0]
        
        return data_dict


def get_dynamic_instances(points, labels, unique_label_str, dynamic_mapping, eps=0.5, min_samples=5):
    """
    Разделяет сегментированные точки на отдельные объекты для динамических классов.
    
    Args:
        points (np.ndarray): Массив координат точек (N, 3), где N - количество точек.
        labels (np.ndarray): Массив меток классов для каждой точки (N,).
        unique_label_str (list): Список названий классов, где индекс соответствует ID класса.
        dynamic_mapping (dict): Словарь динамических классов (ключи - названия классов).
        eps (float): Параметр DBSCAN для максимального расстояния между точками в кластере.
        min_samples (int): Минимальное количество точек для формирования кластера в DBSCAN.
    
    Returns:
        dict: Словарь, где ключи - названия динамических классов, значения - списки вида
              [[size_vector, center_vector], ...], где size_vector = [dx, dy, dz],
              center_vector = [cx, cy, cz].
    """
    instances = []
    dynamic_classes = list(dynamic_mapping.keys())
    cnt = 1
    for cls_name in dynamic_classes:
        if cls_name not in unique_label_str:
            continue
        cls_id = unique_label_str.index(cls_name)
        
        mask = (labels == cls_id)
        if not np.any(mask):
            continue
        
        cls_points = points[mask]
        
        # Кластеризация с помощью DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(cls_points)
        cluster_labels = db.labels_
        
        # Игнорируем шум (-1)
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters != -1]
        
        inst_list = []
        for cl_id in unique_clusters:
            cl_mask = (cluster_labels == cl_id)
            cl_points = cls_points[cl_mask]
            
            if cl_points.shape[0] < min_samples:
                continue
            
            # Вычисление bounding box
            min_xyz = cl_points.min(axis=0)
            max_xyz = cl_points.max(axis=0)
            size = (max_xyz - min_xyz).tolist()  # [dx, dy, dz]
            
            # Центр как среднее значение координат точек в кластере
            center = cl_points.mean(axis=0).tolist()  # [cx, cy, cz]
            
            inst_list.append([size, center])

            instances.append({"id": cnt, "type": cls_name, "coords": center, "zoomLevel" : 4})
            cnt += 1
        
        # if inst_list:
        #     instances[cls_name] = inst_list
    
    return instances


def main_worker(local_rank, nprocs, configs):
    global instances
    torch.autograd.set_detect_anomaly(True)

    dataset_config = configs['dataset_params']
    model_config = configs['model_params']
    train_hypers = configs['train_params']
    train_hypers.local_rank = local_rank
    train_hypers.world_size = nprocs
    configs.train_params.world_size = nprocs
    
    if train_hypers['distributed']:
        init_method = 'tcp://' + args.ip + ':' + args.port
        dist.init_process_group(backend='nccl', init_method=init_method, world_size=nprocs, rank=local_rank)
        dataset_config.val_data_loader.batch_size = dataset_config.val_data_loader.batch_size // nprocs

    pytorch_device = torch.device('cuda:' + str(local_rank))
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)

    seed = train_hypers.seed + local_rank * dataset_config.val_data_loader.num_workers * train_hypers['max_num_epochs']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))
    print(unique_label)
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label]
    print(unique_label_str)
    
    my_model = get_model_class(model_config['model_architecture'])(configs)

    if train_hypers['distributed']:
        my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)

    model_load_path = "output_skitti/opensource_9ks_s030_w64_finetuned_new0.pt"
    # model_load_path = "output_skitti/opensource_9ks_s030_w64_finetuned0_old.pt"
    if os.path.exists(model_load_path):
        print('pre-train')
        try:
            my_model, pre_weight = load_checkpoint_model_mask(model_load_path, my_model, pytorch_device)
            # my_model, pre_weight = load_checkpoint_model_mask(model_config['model_load_path'], my_model, pytorch_device)
        except:
            my_model = load_checkpoint_old(model_load_path, my_model)

    my_model.to(pytorch_device)
    
    if train_hypers['distributed']:
        train_hypers.local_rank = train_hypers.local_rank % torch.cuda.device_count()
        my_model= DistributedDataParallel(my_model,device_ids=[train_hypers.local_rank],find_unused_parameters=False)


    # prepare dataset
    val_dataloader_config = dataset_config['val_data_loader']
    data_path = val_dataloader_config["data_path"]
    val_imageset = val_dataloader_config["imageset"]

    label_mapping = dataset_config["label_mapping"]

    with open(dataset_config['label_mapping'], 'r') as stream:
        mapfile = yaml.safe_load(stream)

    valid_labels = np.vectorize(mapfile['learning_map_inv'].__getitem__)

    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

    val_pt_dataset = SemKITTI(data_path, imageset=val_imageset, label_mapping=label_mapping, num_vote = configs.num_vote)

    val_dataset = get_dataset_class(dataset_config['dataset_type'])(
        val_pt_dataset,
        config=dataset_config,
        loader_config=val_dataloader_config,
        num_vote = configs.num_vote)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=train_hypers.world_size, rank=train_hypers.local_rank, shuffle=False)
    print("BATCH_SIZE", val_dataloader_config["batch_size"], flush=True)
    # val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                                 batch_size=val_dataloader_config["batch_size"],
    #                                                 collate_fn=get_collate_class(dataset_config['collate_type']),
    #                                                 num_workers=val_dataloader_config["num_workers"],
    #                                                 pin_memory=True,
    #                                                 drop_last=False,
    #                                                 shuffle = False,
    #                                                 sampler=val_sampler)


    if train_hypers.local_rank == 0:
        # validation
        print('*'*80)
        print('Test network performance on validation split')
        print('*'*80)
        # pbar = tqdm(total=len(val_dataset_loader), ncols=80)
        pbar = None
    else:
        pbar = None

    # Load single .pcd fille
    # pcd_path =  "/workdir/lidar/LSK3DNet/cond_0_new_refine.ply"s
    # pcd_path = "/work/" + sys.argv[1]
    # pcd_path = os.path.join(os.getenv("WORKDIR"), "points.pcd")
    pcd_path = os.path.join(os.getenv("WORKDIR"), "points_uncompressed.pcd")
    print(pcd_path)
    # while True:
    #     ok, msg = validate_pcd(pcd_path)
    #     if ok:  # File appeared
    #         break
    #     
    #     print(f"Waiting for file, {msg}", flush=True)
    #     time.sleep(5)

    while True:
        if os.path.isfile(pcd_path):  # File appeared
            break
        
        print(f"Waiting for file", flush=True)
        time.sleep(5)

    print(f"Got file: {pcd_path}", flush=True)
    time.sleep(3)

    # Read with tensor API to keep extra attributes (e.g., "intensity")
    pcd_t = o3d.t.io.read_point_cloud(pcd_path)
    
    xyz = pcd_t.point["positions"].numpy().astype(np.float32)  # (N, 3)
    # max_xyz = xyz.max(axis=0)
    # min_xyz = xyz.min(axis=0)
    # # xyz = xyz - min_xyz - ((max_xyz-min_xyz)/2)
    # xyz -= max_xyz
    
    if False and "intensity" in pcd_t.point:
        sig = pcd_t.point["intensity"].numpy().astype(np.float32).reshape(-1, 1)  # (N, 1)
    else:
        sig = np.zeros((xyz.shape[0], 1), np.float32)

    downsample_transform = SphereCrop(point_max=configs['dataset_params']['val_data_loader']['d_point_num'], mode="random")
    
    my_model.eval()
    hist_list = []
    time_list = []
    
    
    N_total = xyz.shape[0]
    num_classes = int(model_config["num_classes"])

    # Preallocate global accumulators (fixed!)
    full_logits = np.zeros((N_total, num_classes), dtype=np.float32)
    counts = np.zeros((N_total,), dtype=np.int32)

    # Split by size
    chunks = split_pointcloud_spatial(xyz, chunk_size=120_000, voxel=1.0, overlap=50_000)
    # chunks = split_pointcloud(xyz, chunk_size=120_000)
    print("LEN CHUNKS", len(chunks))

    for i_iter, (sub_points, idxs) in enumerate(chunks):
        # Build per-chunk input features [x,y,z,intensity]
        sub_sig = sig[idxs]  # (M,1)
        xyz_chunk = sub_points.astype(np.float32, copy=False)

        # cur_min = xyz_chunk.min(axis=0)
        # cur_max = xyz_chunk.max(axis=0)
        # cur_span = np.maximum(cur_max - cur_min, 1e-6)  # avoid div by zero
        # target_min = np.array([-50, -50, -4])
        # target_max = np.array([50, 50, 2])
        # tgt_span = (target_max - target_min)
        # pts_scaled = (xyz_chunk - cur_min) / cur_span * tgt_span + target_min
        # feats_np = np.hstack([pts_scaled, sub_sig]).astype(np.float32)
        feats_np = np.hstack([xyz_chunk, sub_sig]).astype(np.float32)

        M = feats_np.shape[0]

        # Pack model dict
        val_data_dict = {
            "points": torch.from_numpy(feats_np).to(pytorch_device),                 # (M,4)
            "normal": torch.zeros((M, 3), dtype=torch.float32, device=pytorch_device),
            "ref_sub_points": torch.from_numpy(xyz_chunk).to(pytorch_device),        # (M,3)
            "batch_idx": torch.zeros((M,), dtype=torch.long, device=pytorch_device),
            "batch_size": 1,
            "labels": torch.zeros((M,), dtype=torch.long, device=pytorch_device),
            "raw_labels": torch.zeros((M,), dtype=torch.long, device=pytorch_device),
            "origin_len": int(M),
            "indices": torch.arange(M, dtype=torch.long, device=pytorch_device),
            "path": pcd_path,
            "point_num": int(M),
        }
        raw_labels = val_data_dict['raw_labels'].to(pytorch_device)
        # vote_logits = torch.zeros(N, model_config['num_classes']).to(pytorch_device)
        indices = val_data_dict['indices'].to(pytorch_device)

        val_data_dict['points'] = val_data_dict['points'].to(pytorch_device)
        val_data_dict['normal'] = val_data_dict['normal'].to(pytorch_device)
        val_data_dict['batch_idx'] = val_data_dict['batch_idx'].to(pytorch_device)
        val_data_dict['labels'] = val_data_dict['labels'].to(pytorch_device)

        # Forward
        with torch.no_grad():
            out_dict = my_model(val_data_dict)    # must set out_dict['logits'] = (M, num_classes)
            logits = out_dict["logits"].detach().float().cpu().numpy()  # (M,C)

        # Accumulate into global arrays
        full_logits[idxs, :] += logits
        counts[idxs] += 1

        # cleanup GPU aggressively
        del feats_np
        # del feats_np, pts_scaled
        torch.cuda.empty_cache()

        print(f"[{i_iter+1}/{len(chunks)}] processed {M} points; classes seen:",
              set(np.argmax(logits, axis=1).tolist()), flush=True)


        # if (i_iter != 0 and i_iter % 25 == 0) or i_iter == len(chunks) - 1:

    # Average logits and finalize prediction
    counts_safe = np.maximum(counts[:, None], 1)               # (N,1)
    full_logits = full_logits / counts_safe                    # (N,C)
    final_pred = full_logits.argmax(axis=1)                    # (N,)

    # Dynamic → red; Non-dynamic → second color (green here)
    dyn_set = set(DYNAMIC_MAPPING.keys())
    print("dyn set", dyn_set)
    def _name(cid: int) -> str:
        return unique_label_str[cid] if 0 <= cid < len(unique_label_str) else f"class_{cid}"
    is_dynamic = np.fromiter((_name(int(c)) in dyn_set for c in final_pred),
                             count=final_pred.shape[0], dtype=bool)
 
    colors = np.empty((final_pred.shape[0], 3), dtype=np.float32)
    colors[:] = np.array([0.0, 1.0, 0.0], dtype=np.float32)   # non-dynamic = green
    colors[is_dynamic] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # dynamic = red

    unseen_mask = (counts == 0)
    if unseen_mask.any():
        colors[unseen_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_dir = os.getenv("WORKDIR", ".")
    new_pcd_path = os.path.join(out_dir, "points_segmented.pcd")
    o3d.io.write_point_cloud(new_pcd_path, pcd)
    print(f"Saved colored point cloud to: {new_pcd_path}")

    instances = get_dynamic_instances(xyz.astype(np.float32), final_pred, unique_label_str, DYNAMIC_MAPPING, eps=1, min_samples=500)
    print(instances)

    # Удаление динамических точек и сохранение очищенного облака
    mask_non_dynamic = ~is_dynamic  # Маска для нединамических точек

    # Фильтруем точки и цвета
    xyz_cleaned = xyz[mask_non_dynamic]
    colors_cleaned = colors[mask_non_dynamic]

    # Сохраняем очищенное облако
    pcd_cleaned = o3d.geometry.PointCloud()
    pcd_cleaned.points = o3d.utility.Vector3dVector(xyz_cleaned.astype(np.float32))
    pcd_cleaned.colors = o3d.utility.Vector3dVector(colors_cleaned.astype(np.float32))
    new_pcd_cleaned_path = os.path.join(os.getenv("WORKDIR"), "points_cleaned.pcd")
    o3d.io.write_point_cloud(new_pcd_cleaned_path, pcd_cleaned)
    print("Saved cleaned to output_normalized_cleaned.ply/pcd")

    # Send request
    print("Saved to output_normalized.ply/pcd")
    params = {"path": new_pcd_path}
    headers = {"accept": "application/json"}
    print("fetching...")
    timeout = 120
    r = requests.get(f"http://converter:8000/convert", params=params, headers=headers, timeout=timeout)
    print("fetched...")


    print("COMPLETED", flush=True)

if __name__ == '__main__':
    print(' '.join(sys.argv))
    print(configs)
    # main(configs)
    uvicorn.run("test_skitti:app", host="0.0.0.0", port=8001)
