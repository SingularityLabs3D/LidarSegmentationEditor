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

# Testing settings
# parser = argparse.ArgumentParser(description='LSKNet Testing')
# parser.add_argument('--config_path', default='./config/lk-semantickitti_sub_tta.yaml')
# parser.add_argument('--ip', default='127.0.0.1', type=str)
# parser.add_argument('--port', default='3023', type=str)
# parser.add_argument('--num_vote', type=int, default=8, help='number of voting in the test') #14
# args = parser.parse_args()
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
shutil.copy('test_skitti.py', str(exp_dir))
shutil.copy('config/lk-semantickitti_sub_tta.yaml', str(exp_dir))


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


# def split_pointcloud(points, cube_size=50.0, overlap=5.0):
#     """
#     Split (N,3) points into overlapping cubes of side cube_size.
#     Returns a list of (sub_points, idxs) for each chunk.
#     overlap: amount of overlap between cubes in all directions.
#     """
#     min_xyz = points.min(axis=0)
#     max_xyz = points.max(axis=0)

#     step = cube_size - overlap  # stride between cube starts
#     chunks = []
#     # iterate over all grid cubes
#     xs = np.arange(min_xyz[0], max_xyz[0] + step, step)
#     ys = np.arange(min_xyz[1], max_xyz[1] + step, step)
#     zs = np.arange(min_xyz[2], max_xyz[2] + step, step)

#     for x in tqdm(xs):
#         for y in ys:
#             for z in zs:
#                 # compute cube bounds
#                 x_min, x_max = x, x + cube_size
#                 y_min, y_max = y, y + cube_size
#                 z_min, z_max = z, z + cube_size

#                 mask = (
#                     (points[:,0] >= x_min) & (points[:,0] <= x_max) &
#                     (points[:,1] >= y_min) & (points[:,1] <= y_max) &
#                     (points[:,2] >= z_min) & (points[:,2] <= z_max)
#                 )
#                 idxs = np.nonzero(mask)[0]
#                 if len(idxs) == 0:
#                     continue
#                 sub_points = points[idxs]
#                 chunks.append((sub_points, idxs))
#     return chunks


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

def split_pointcloud(points, cube_size=50.0, overlap=5.0, processes=None, task_chunk=64):
    """
    Parallel split of (N,3) points into overlapping cubes (cube_size).
    Returns list of (sub_points, idxs).
    """
    if overlap >= cube_size:
        raise ValueError("overlap must be < cube_size")

    points = np.ascontiguousarray(points)
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)

    step = cube_size - overlap
    xs = np.arange(min_xyz[0], max_xyz[0] + step, step)
    ys = np.arange(min_xyz[1], max_xyz[1] + step, step)
    zs = np.arange(min_xyz[2], max_xyz[2] + step, step)

    # Prepare shared memory for points
    shm = SharedMemory(create=True, size=points.nbytes)
    shm_np = np.ndarray(points.shape, dtype=points.dtype, buffer=shm.buf)
    shm_np[:] = points  # copy once

    bounds = []
    for x, y, z in product(xs, ys, zs):
        bounds.append(((x, x + cube_size, y, y + cube_size, z, z + cube_size),
                       shm.name, points.shape, points.dtype))

    procs = processes or os.cpu_count() or 1
    chunks = []
    try:
        with Pool(processes=procs) as pool:
            for idxs in pool.imap_unordered(_chunk_worker, bounds, chunksize=task_chunk):
                if idxs.size:
                    chunks.append((points[idxs], idxs))
    finally:
        shm.close()
        shm.unlink()

    return chunks


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


def main_worker(local_rank, nprocs, configs):
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

    if os.path.exists(model_config['model_load_path']):
        print('pre-train')
        try:
            my_model, pre_weight = load_checkpoint_model_mask(model_config['model_load_path'], my_model, pytorch_device)
        except:
            my_model = load_checkpoint_old(model_config['model_load_path'], my_model)

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


    if val_imageset == 'val':
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
        
        
        with torch.no_grad():
            chunks = split_pointcloud(xyz, cube_size=140.0, overlap=70.0)
        
            # full_logits = np.zeros((120000 * len(chunks), model_config['num_classes']), dtype=np.float32)
            # counts = np.zeros((120000 * len(chunks), 1), dtype=np.float32)
            # full_logits = np.ndarray([], dtype=np.float32)
            # counts = np.ndarray([], dtype=np.float32)
            full_logits = []
            counts = []
            xyz_downsampled = []
        
            for i_iter_val, (sub_points, idxs) in enumerate(chunks):
                # if i_iter_val > 30:
                #    break
                
                # Slice intensity to the same subset and append as 4th feature
                i_sig = sig[idxs]  # (N,1) aligned with sub_points
        
                N = sub_points.shape[0]
                print(N, flush=True)
                NEW_N = min(sub_points.shape[0], 120000)
                val_data_dict = {
                    "points": torch.from_numpy(np.hstack([sub_points, i_sig])).float(),
                    "normal": torch.zeros((N, 3), dtype=torch.float32),
                    "ref_sub_points": torch.from_numpy(sub_points.astype(np.float32)),
                    "batch_idx": torch.zeros((NEW_N,), dtype=torch.long),
                    "batch_size": 1,
                    "labels": torch.zeros((N,), dtype=torch.long),
                    "raw_labels": torch.zeros((N,), dtype=torch.long),
                    "origin_len": N,
                    "indices": torch.arange(NEW_N, dtype=torch.long),
                    "path": pcd_path,
                    "point_num": NEW_N,
                }

                val_data_dict = downsample_transform(val_data_dict)
                idx_downsampled = range(len(xyz_downsampled), len(xyz_downsampled) + NEW_N)
                xyz_new = val_data_dict["points"][:, :3].cpu().numpy()
                xyz_downsampled.append(deepcopy(xyz_new))
                # print(val_data_dict["points"][:, :3].shape)
                N = NEW_N

                # Norm
                current_min = xyz_new.min(axis=0)
                current_max = xyz_new.max(axis=0)
                target_min = np.array([-50, -50, -4])
                target_max = np.array([50, 50, 2])
                current_span = current_max - current_min
                target_span = target_max - target_min              
                pts_scaled = (xyz_new - current_min) / current_span * target_span + target_min
                unscaled_points = val_data_dict["points"]
                val_data_dict["points"] = torch.from_numpy(np.hstack([pts_scaled, sig[idx_downsampled]])).float()
                
                # if sub_points.shape[0] <= 10000:
                #     continue
            
                raw_labels = val_data_dict['raw_labels'].to(pytorch_device)
                vote_logits = torch.zeros(N, model_config['num_classes']).to(pytorch_device)
                indices = val_data_dict['indices'].to(pytorch_device)
            
                val_data_dict['points'] = val_data_dict['points'].to(pytorch_device)
                val_data_dict['normal'] = val_data_dict['normal'].to(pytorch_device)
                val_data_dict['batch_idx'] = val_data_dict['batch_idx'].to(pytorch_device)
                val_data_dict['labels'] = val_data_dict['labels'].to(pytorch_device)
            
                with torch.no_grad():
                    val_data_dict = my_model(val_data_dict)
                logits = val_data_dict['logits']
                vote_logits.index_add_(0, indices, logits)
            
                predict_labels = torch.argmax(vote_logits, dim=1).cpu().numpy()
                print(set(list(predict_labels)), flush=True)
            
                # full_logits[idx_downsampled] += logits.cpu().numpy()
                # counts[idx_downsampled] += 1
                full_logits.append(logits.cpu().numpy())


                # if i_iter_val > 0 and i_iter_val % 50 == 0:


            print("Sending to fronend")
            # average and take argmax
            # full_logits /= np.maximum(counts, 1)
            merged_full_logits = np.vstack(full_logits).astype(np.float32)
            final_pred = merged_full_logits.argmax(1)
            # В main_worker, после создания unique_label_str
            
            # Add "unlabeled" if model predicts it
            if model_config['num_classes'] > len(unique_label_str):
                unique_label_str = ['unlabeled'] + unique_label_str
            
            # Print class-to-color mapping
            print("\nClass to Color Mapping:")
            print(f"{'ID':<5} {'Class Name':<20} {'RGB Color':<25}")
            print("-" * 50)
            for i in range(len(unique_label_str)):
                # label = DYNAMIC_MAPPING.get(unique_label_str[i], "static")
                is_dynamic = int(unique_label_str[i] in DYNAMIC_MAPPING.keys())
                rgb = plt.get_cmap('tab20')(is_dynamic % 20)[:3]  # Get RGB for class i
                rgb_str = f"({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f})"
                print(f"{i:<5} {unique_label_str[i]:<20} {rgb_str:<25}, {is_dynamic}")
            
            # Save colored point cloud
            colors = plt.get_cmap('tab20')(final_pred % 20)[:, :3]
            pcd = o3d.geometry.PointCloud()
            stacked = np.vstack(xyz_downsampled)
            pcd.points = o3d.utility.Vector3dVector(stacked.astype(np.float32))
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
            new_pcd_path = os.path.join(os.getenv("WORKDIR"), "points_segmented.pcd")
            o3d.io.write_point_cloud(new_pcd_path, pcd)

            # Удаление динамических точек и сохранение очищенного облака
            dynamic_classes = set(DYNAMIC_MAPPING.keys())
            is_dynamic = np.array([unique_label_str[pred] in dynamic_classes for pred in final_pred])
            mask_non_dynamic = ~is_dynamic  # Маска для нединамических точек

            # Фильтруем точки и цвета
            stacked_cleaned = stacked[mask_non_dynamic]
            colors_cleaned = colors[mask_non_dynamic]

            # Сохраняем очищенное облако
            pcd_cleaned = o3d.geometry.PointCloud()
            pcd_cleaned.points = o3d.utility.Vector3dVector(stacked_cleaned.astype(np.float32))
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


        # if train_hypers.local_rank == 0:
        #     pbar.close()
            # print('Predicted test labels are saved in %s. Need to be shifted to original label format before submitting to the Competition website.' % exp_dir)
            # print('Remapping script can be found in semantic-kitti-api.')

        print("COMPLETED", flush=True)

if __name__ == '__main__':
    print(' '.join(sys.argv))
    print(configs)
    main(configs)
