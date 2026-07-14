import os
import json
import torch
import faiss
import numpy as np
import cv2
import tifffile as tiff
from tqdm import tqdm
from src.utils import augment_image, dists2map
from src.post_eval import mean_top1p

def _load_or_extract_features(model, image=None, image_path=None, feature_path=None, feature_list=None,
                              feature_fuse=False, masking=False):
    os.makedirs(feature_path, exist_ok=True)
    grid_json = os.path.join(feature_path, "grid_size.json")
    feature_pt = os.path.join(feature_path, "feature.pt")
    feature_fuse_pt = os.path.join(feature_path, "feature_fuse.pt")
    mask_path = os.path.join(feature_path, "mask.pt")
    if image is None and image_path is not None:
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    need_extract = not os.path.exists(grid_json) or \
                   (feature_fuse and not os.path.exists(feature_fuse_pt)) or \
                   (not feature_fuse and not os.path.exists(feature_pt))

    if need_extract:
        image_tensor, grid_size = model.prepare_image(image)
        with open(grid_json, "w") as f:
            json.dump({"grid_height": grid_size[0], "grid_width": grid_size[1]}, f)

        if feature_fuse:
            features = model.extract_features(image_tensor, feature_list)
            torch.save(features, feature_fuse_pt)
            feature_origin = features[-1].squeeze().cpu().numpy()
            features = model.MLMP(features, grid_size=grid_size)
        else:
            features = model.extract_features(image_tensor)
            torch.save(features, feature_pt)
            feature_origin = features
    else:
        with open(grid_json, "r") as f:
            g = json.load(f)
        grid_size = (g["grid_height"], g["grid_width"])
        if feature_fuse:
            features = torch.load(feature_fuse_pt)
            feature_origin = features[-1].squeeze().cpu().numpy()
            features = model.MLMP(features, grid_size=grid_size)
        else:
            features = torch.load(feature_pt)
            feature_origin = features

    if masking:       
        if not os.path.exists(mask_path):
            mask = model.compute_background_mask(feature_origin, grid_size, threshold=10,
                                                 masking_type=masking)
            torch.save(mask, mask_path)
        else:
            mask = torch.load(mask_path)
    else:
        mask = np.ones(features.shape[0], dtype=bool)

    return features, feature_origin, mask, grid_size


def _build_faiss_index(features_ref, features_origin_ref=None, feature_fuse=False, faiss_on_cpu=False):
    features_dim = features_ref.shape[1]

    if faiss_on_cpu:
        knn_index = faiss.IndexIDMap(faiss.IndexFlatL2(features_dim))
        class_index = None
        if feature_fuse and features_origin_ref is not None:
            class_index = faiss.IndexIDMap(faiss.IndexFlatL2(features_origin_ref.shape[1]))
    else:
        res1 = faiss.StandardGpuResources()
        knn_index = faiss.GpuIndexFlatL2(res1, features_dim)


        class_index = None
        if feature_fuse and features_origin_ref is not None:
            res2 = faiss.StandardGpuResources()
            class_index = faiss.GpuIndexFlatL2(res2, features_dim)


    return knn_index, class_index



def _compute_anomaly_score(features2, feature_origin, knn_index, class_index, feature_fuse, knn_neighbors=1):
    faiss.normalize_L2(features2)
    distances, _ = knn_index.search(features2, k=knn_neighbors)
    if knn_neighbors > 1:
        distances = distances.mean(axis=1)
    distances = distances / 2

    if feature_fuse and feature_origin is not None:
        faiss.normalize_L2(feature_origin)
        distances2, _ = class_index.search(feature_origin, k=knn_neighbors)
        if knn_neighbors > 1:
            distances2 = distances2.mean(axis=1)
        distances2 = distances2 / 2
        return distances, distances2

    return distances, None

def _save_anomaly_map(d_masked, d_masked_class, image_shape, plots_dir,
                      object_name, type_anomaly, img_name,
                      save_patch_dists=True, save_tiffs=False,
                      feature_fuse=False):

    base = os.path.join(plots_dir, "Anomaly_maps", object_name, "test", type_anomaly)
    os.makedirs(base, exist_ok=True)
    img_base = os.path.splitext(img_name)[0]

    if save_patch_dists:
        np.save(os.path.join(base, f"{img_base}.npy"), d_masked)
        if feature_fuse and d_masked_class is not None:
            np.save(os.path.join(base, f"{img_base}_class.npy"), d_masked_class)

    if save_tiffs:
        full_map = dists2map(d_masked, image_shape)
        tiff.imwrite(os.path.join(base, f"{img_base}.tiff"), full_map)

class ReRemAnomalyDetector:
    def __init__(self, model, object_name, data_root, features_dir, feature_fuse, feature_list, object_anomalies, knn_metric, knn_neighbors, faiss_on_cpu, masking, mask_ref_images, rotation, top_p):
        self.model = model
        self.object_name = object_name
        self.object_anomalies = object_anomalies[object_name] + ['good']
        self.object_anomalies = list(dict.fromkeys(self.object_anomalies))

        self.features_dir = os.path.join(features_dir, object_name, "test")
        self.data_root = data_root
        self.feature_list = feature_list
        self.feature_fuse = feature_fuse
        self.masking = masking
        self.mask_ref_images = mask_ref_images
        self.rotation = rotation
        # self.flip = flip
        self.top_p = top_p
        self.faiss_on_cpu = faiss_on_cpu
        self.knn_metric = knn_metric
        self.knn_neighbors = knn_neighbors

        self.img_cnt = 1
        self.knn_index = None
        self.class_index = None
        self.features_ref_len = 0
        self.results = {}
    def init_reference_memory(self, img_ref_samples):
        self.results["img_ref_samples"] = img_ref_samples
        features_ref = []
        features_origin_ref = []
        for img_ref_n in tqdm(img_ref_samples, desc="Building memory bank"):
            ref_img_path = os.path.join(self.data_root, self.object_name, "test", img_ref_n)
            ref_feat_path = os.path.join(
                self.features_dir, self.object_name, "test", 
                img_ref_n.replace('.jpg', '').replace('.png', '')
            )
            image_ref = cv2.cvtColor(cv2.imread(ref_img_path), cv2.COLOR_BGR2RGB)
            img_augmented = augment_image(image_ref) if self.rotation else [image_ref]

            for i, img in enumerate(img_augmented):
                feat_path = os.path.join(ref_feat_path, str(i))
                features, feature_origin, mask, _ = _load_or_extract_features(
                    self.model, img, None, feat_path, self.feature_list, self.feature_fuse, self.mask_ref_images
                )
                features_ref.append(features[mask])
                if self.feature_fuse:
                    features_origin_ref.append(feature_origin[mask])
            self.img_cnt += 1

        # 拼接所有特征
        features_ref = np.concatenate(features_ref, axis=0).astype("float32")
        self.features_ref_len = features_ref.shape[0]
        if self.feature_fuse:
            features_origin_ref = np.concatenate(features_origin_ref, axis=0).astype("float32")

        # 初始化 FAISS 索引
        self.knn_index, self.class_index = _build_faiss_index(
            features_ref, 
            features_origin_ref if self.feature_fuse else None,
            not self.faiss_on_cpu
        )

        # 可选归一化
        if self.knn_metric == "L2_normalized":
            faiss.normalize_L2(features_ref)
            if self.feature_fuse:
                faiss.normalize_L2(features_origin_ref)

        # 添加特征到索引（不带 ID）
        self.knn_index.add(features_ref)
        if self.feature_fuse:
            self.class_index.add(features_origin_ref)


    def add_reference_samples(self, img_ref_samples):
        # # 更新总参考特征数
        # self.features_ref_len += new_features.shape[0]
        features_ref = []
        features_origin_ref = []
        # 对每个参考图像进行特征提取
        for img_ref_n in tqdm(img_ref_samples, desc="Building memory bank"):
            ref_img_path = os.path.join(self.data_root, self.object_name, "test", img_ref_n)
            ref_feat_path = os.path.join(
                self.features_dir, self.object_name, "test", 
                img_ref_n.replace('.jpg', '').replace('.png', '')
            )
            image_ref = cv2.cvtColor(cv2.imread(ref_img_path), cv2.COLOR_BGR2RGB)
            img_augmented = augment_image(image_ref) if self.rotation else [image_ref]

            for i, img in enumerate(img_augmented):
                feat_path = os.path.join(ref_feat_path, str(i))
                features, feature_origin, mask, _ = _load_or_extract_features(
                    self.model, img, None, feat_path, self.feature_list, self.feature_fuse, self.mask_ref_images
                )

                features_ref.append(features[mask])
                if self.feature_fuse:
                    features_origin_ref.append(feature_origin[mask])
            self.img_cnt += 1

        features_ref = np.concatenate(features_ref, axis=0).astype("float32")
        if self.feature_fuse:
            features_origin_ref = np.concatenate(features_origin_ref, axis=0).astype("float32")


        if self.knn_metric == "L2_normalized":
            faiss.normalize_L2(features_ref)
            if self.feature_fuse and features_origin_ref is not None:
                faiss.normalize_L2(features_origin_ref)

        self.knn_index.add(features_ref)

        # 如果开启 feature_fuse，并且提供了原始特征，则添加到 class_index
        if self.feature_fuse and features_origin_ref is not None:
            self.class_index.add(features_origin_ref)
    
    def add_reference_samples_greedy(self, img_ref_samples):
        # self.features_ref_len += new_features.shape[0]
        features_ref = []
        features_origin_ref = []
        for img_ref_n in tqdm(img_ref_samples, desc="Building memory bank"):
            ref_img_path = os.path.join(self.data_root, self.object_name, "test", img_ref_n)
            ref_feat_path = os.path.join(
                self.features_dir, self.object_name, "test", 
                img_ref_n.replace('.jpg', '').replace('.png', '')
            )
            image_ref = cv2.cvtColor(cv2.imread(ref_img_path), cv2.COLOR_BGR2RGB)
            img_augmented = augment_image(image_ref) if self.rotation else [image_ref]

            for i, img in enumerate(img_augmented):
                feat_path = os.path.join(ref_feat_path, str(i))
                features, feature_origin, mask, _ = _load_or_extract_features(
                    self.model, img, None, feat_path, self.feature_list, self.feature_fuse, self.masking
                )
                features2 = features[mask]

                d, _ = _compute_anomaly_score(features2, None, self.knn_index, self.class_index,
                                               self.feature_fuse, self.knn_neighbors)
                
                k = int(d.shape[0] * self.top_p)
                d = d.flatten()
                topk_indices = np.argpartition(-d, k)[:k]  # 取最大值的索引
                features_top = features2[topk_indices]
                features_top_origin = feature_origin[topk_indices] if self.feature_fuse else None
                features_ref.append(features_top)
                if self.feature_fuse and features_top_origin is not None:
                    features_origin_ref.append(features_top_origin)
            self.img_cnt += 1

        features_ref = np.concatenate(features_ref, axis=0).astype("float32")
        if self.feature_fuse:
            features_origin_ref = np.concatenate(features_origin_ref, axis=0).astype("float32")


        if self.knn_metric == "L2_normalized":
            faiss.normalize_L2(features_ref)
            if self.feature_fuse and features_origin_ref is not None:
                faiss.normalize_L2(features_origin_ref)

        self.knn_index.add(features_ref)

        if self.feature_fuse and features_origin_ref is not None:
            self.class_index.add(features_origin_ref)


    def run_inference(self, overall_output_dir, out_samples=None, save_patch_dists=True, save_tiffs=False):
        anomaly_scores = {}
        for type_anomaly in tqdm(self.object_anomalies, desc=f"Processing test samples ({self.object_name})"):
            test_dir = os.path.join(self.data_root, self.object_name, "test", type_anomaly)
            feat_dir = os.path.join(self.features_dir, self.object_name, "test", type_anomaly)

            for img_name in sorted(os.listdir(test_dir)):
                if out_samples is not None:
                    temp_samples = f"{type_anomaly}/img_name"
                    if temp_samples in out_samples:
                        continue
                img_path = os.path.join(test_dir, img_name)
                feat_path = os.path.join(feat_dir, img_name.replace(".jpg", "").replace(".png", ""), "0")
                image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

                features, feature_origin, mask, grid_size = _load_or_extract_features(
                    self.model, image, None, feat_path, self.feature_list, self.feature_fuse, self.masking
                )
                features2 = features[mask]
                feature_origin = feature_origin[mask] if self.feature_fuse else None

                d, d2 = _compute_anomaly_score(features2, feature_origin, self.knn_index, self.class_index,
                                               self.feature_fuse, self.knn_neighbors)

                score = mean_top1p(d2 if self.feature_fuse else d)
                anomaly_scores[f"{type_anomaly}/{img_name}"] = score
                self.results["all"].append({"path": f"{type_anomaly}/{img_name}", "score": float(score)})

                d_masked = np.zeros_like(mask, dtype=float)
                d_masked_class = np.zeros_like(mask, dtype=float)
                d_masked[mask] = d.squeeze()
                d_masked_reshape = d_masked.reshape(grid_size)
                d_masked_class_reshape = None

                if self.feature_fuse and d2 is not None:
                    d_masked_class[mask] = d2.squeeze()
                    d_masked_class_reshape = d_masked_class.reshape(grid_size)

                _save_anomaly_map(d_masked_reshape, d_masked_class_reshape, image.shape,
                                  overall_output_dir, self.object_name, type_anomaly, img_name,
                                  save_patch_dists, save_tiffs, self.feature_fuse)

        self.results["all"] = sorted(self.results["all"], key=lambda x: x["score"])
        return anomaly_scores, self.results