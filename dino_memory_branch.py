#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
from tqdm import tqdm

from relation_features import sample_key


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EPS = 1e-12


def image_paths(root):
    root = Path(root)
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_dinov2(model_name, device):
    cache_repo = Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
    if cache_repo.exists():
        model = torch.hub.load(str(cache_repo), model_name, source="local", pretrained=True)
    else:
        model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    model.eval().to(device)
    return model


def preprocess(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


@torch.inference_mode()
def extract_patch_features(model, path, transform, device):
    image = Image.open(path).convert("RGB")
    batch = transform(image).unsqueeze(0).to(device)
    if hasattr(model, "forward_features"):
        features = model.forward_features(batch)
        if isinstance(features, dict):
            if "x_norm_patchtokens" in features:
                patches = features["x_norm_patchtokens"]
            elif "x_prenorm" in features:
                patches = features["x_prenorm"][:, 1:, :]
            else:
                raise RuntimeError(f"Unsupported DINOv2 feature keys: {sorted(features.keys())}")
        else:
            patches = features
    else:
        raise RuntimeError("DINOv2 model does not expose forward_features().")

    patches = F.normalize(patches.squeeze(0), dim=1)
    return patches.detach().cpu().numpy().astype(np.float32)


def feature_cache_path(path, args):
    if not args.feature_cache_dir:
        return None
    category_root = Path(args.data_root) / args.category
    relative = Path(path).relative_to(category_root).with_suffix(".npy")
    cache_root = Path(args.feature_cache_dir) / args.category / f"{args.model}_{args.image_size}"
    return cache_root / relative


def load_cached_features(path, args):
    cache_path = feature_cache_path(path, args)
    if cache_path is None or not cache_path.exists():
        return None
    return np.load(cache_path).astype(np.float32)


def save_cached_features(path, features, args):
    cache_path = feature_cache_path(path, args)
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features.astype(np.float32))


def extract_dataset_features(paths, args, device, desc):
    features = []
    model = None
    transform = preprocess(args.image_size)
    cache_hits = 0
    cache_misses = 0
    for path in tqdm(paths, desc=desc):
        cached = load_cached_features(path, args)
        if cached is not None:
            features.append(cached)
            cache_hits += 1
            continue
        if model is None:
            model = load_dinov2(args.model, device)
        extracted = extract_patch_features(model, path, transform, device)
        save_cached_features(path, extracted, args)
        features.append(extracted)
        cache_misses += 1
    return features, {"cache_hits": cache_hits, "cache_misses": cache_misses}


def defect_class_from_path(path):
    return Path(path).parent.name


def ground_truth_from_path(path):
    return 0 if defect_class_from_path(path) == "good" else 1


def build_memory(train_features, max_memory_patches, seed):
    memory = np.concatenate(train_features, axis=0)
    if max_memory_patches and len(memory) > max_memory_patches:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(memory), size=max_memory_patches, replace=False)
        memory = memory[indices]
    return memory.astype(np.float32)


def normalize_rows(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, EPS)


def transform_features_with_pca(train_features, score_features, args):
    if args.pca_dim <= 0:
        return train_features, score_features, None

    train_matrix = np.concatenate(train_features, axis=0)
    input_dim = train_matrix.shape[1]
    n_components = min(args.pca_dim, input_dim, len(train_matrix))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=args.seed)
    pca.fit(train_matrix)

    def transform(feature_list):
        transformed = []
        for features in feature_list:
            projected = pca.transform(features).astype(np.float32)
            if args.pca_renorm:
                projected = normalize_rows(projected).astype(np.float32)
            transformed.append(projected)
        return transformed

    metadata = {
        "pca_dim_requested": args.pca_dim,
        "pca_dim": n_components,
        "pca_input_dim": input_dim,
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "pca_renorm": bool(args.pca_renorm),
    }
    return transform(train_features), transform(score_features), metadata


def class_memory_limit(args):
    if args.coreset_per_class > 0:
        return args.coreset_per_class
    return args.max_memory_patches


def memory_bytes(memory):
    if isinstance(memory, dict):
        return int(sum(value.nbytes for value in memory.values()))
    return int(memory.nbytes)


def infer_patch_grid(num_patches):
    side = int(np.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"Expected square DINO patch grid, got {num_patches} patches")
    return side, side


def composition_path_for_image(image_path, data_root, composition_root, category):
    image_path = Path(image_path)
    category_root = Path(data_root) / category
    relative = image_path.relative_to(category_root)
    return Path(composition_root) / category / relative


def patch_classes_from_composition_map(composition_path, image_size, patch_grid):
    grid_h, grid_w = patch_grid
    label_image = Image.open(composition_path)
    label_image = label_image.resize((image_size, image_size), Image.Resampling.NEAREST)
    labels = np.asarray(label_image, dtype=np.int64)

    if image_size % grid_h != 0 or image_size % grid_w != 0:
        small = Image.fromarray(labels.astype(np.uint8)).resize((grid_w, grid_h), Image.Resampling.NEAREST)
        return np.asarray(small, dtype=np.int64).reshape(-1)

    patch_h = image_size // grid_h
    patch_w = image_size // grid_w
    patch_labels = []
    for row in range(grid_h):
        y0 = row * patch_h
        y1 = (row + 1) * patch_h
        for col in range(grid_w):
            x0 = col * patch_w
            x1 = (col + 1) * patch_w
            values = labels[y0:y1, x0:x1].reshape(-1)
            patch_labels.append(int(np.bincount(values).argmax()))
    return np.asarray(patch_labels, dtype=np.int64)


def build_class_conditional_memory(paths, train_features, args):
    memory_by_class = defaultdict(list)
    patch_grid = infer_patch_grid(train_features[0].shape[0])
    for path, patch_features in tqdm(
        zip(paths, train_features),
        total=len(paths),
        desc="Building class-conditional DINO memory",
    ):
        composition_path = composition_path_for_image(
            path, args.data_root, args.composition_root, args.category
        )
        if not composition_path.exists():
            raise FileNotFoundError(f"Missing composition map for {path}: {composition_path}")
        patch_classes = patch_classes_from_composition_map(
            composition_path, args.image_size, patch_grid
        )
        if len(patch_classes) != len(patch_features):
            raise ValueError(
                f"Patch/class mismatch for {path}: {len(patch_features)} features, "
                f"{len(patch_classes)} class labels"
            )
        if args.foreground_only:
            foreground = patch_classes != args.background_class
            patch_classes = patch_classes[foreground]
            patch_features = patch_features[foreground]
            if len(patch_classes) == 0:
                continue
        for class_id in np.unique(patch_classes):
            memory_by_class[int(class_id)].append(patch_features[patch_classes == class_id])

    rng = np.random.default_rng(args.seed)
    class_memory = {}
    coreset_limit = class_memory_limit(args)
    for class_id, chunks in sorted(memory_by_class.items()):
        memory = np.concatenate(chunks, axis=0).astype(np.float32)
        if coreset_limit and len(memory) > coreset_limit:
            indices = rng.choice(len(memory), size=coreset_limit, replace=False)
            memory = memory[indices]
        class_memory[class_id] = memory
    return class_memory


def pool_distances(distances, pool, topk_percent):
    distances = np.asarray(distances, dtype=np.float32)
    mean_score = float(distances.mean())
    max_score = float(distances.max())
    if pool == "mean":
        pooled_score = mean_score
    elif pool == "max":
        pooled_score = max_score
    elif pool == "topk":
        k = max(1, int(np.ceil(len(distances) * topk_percent)))
        pooled_score = float(np.partition(distances, -k)[-k:].mean())
    else:
        raise ValueError(f"Unsupported pooling mode: {pool}")
    return pooled_score, mean_score, max_score


def score_features(features, nearest_neighbors, k):
    rows = []
    for patch_features in tqdm(features, desc="Scoring images"):
        distances, _ = nearest_neighbors.kneighbors(patch_features, n_neighbors=k)
        nearest = distances.mean(axis=1)
        rows.append((float(nearest.mean()), float(nearest.max())))
    return rows


def score_class_conditional_features(paths, features, class_neighbors, global_neighbors, args):
    rows = []
    patch_grid = infer_patch_grid(features[0].shape[0])
    for path, patch_features in tqdm(
        zip(paths, features),
        total=len(paths),
        desc="Scoring class-conditional images",
    ):
        composition_path = composition_path_for_image(
            path, args.data_root, args.composition_root, args.category
        )
        if not composition_path.exists():
            raise FileNotFoundError(f"Missing composition map for {path}: {composition_path}")
        patch_classes = patch_classes_from_composition_map(
            composition_path, args.image_size, patch_grid
        )
        score_mask = np.ones(len(patch_features), dtype=bool)
        if args.foreground_only:
            score_mask = patch_classes != args.background_class
            if not np.any(score_mask):
                score_mask = np.ones(len(patch_features), dtype=bool)
        scored_features = patch_features[score_mask]
        scored_classes = patch_classes[score_mask]
        distances = np.empty(len(scored_features), dtype=np.float32)
        missing_class_patches = 0
        for class_id in np.unique(scored_classes):
            mask = scored_classes == class_id
            neighbors = class_neighbors.get(int(class_id))
            if neighbors is None:
                neighbors = global_neighbors
                missing_class_patches += int(mask.sum())
            class_distances, _ = neighbors.kneighbors(scored_features[mask])
            distances[mask] = class_distances.mean(axis=1)

        pooled_score, mean_score, max_score = pool_distances(
            distances, args.pool, args.topk_percent
        )
        rows.append((pooled_score, mean_score, max_score, missing_class_patches))
    return rows


def main():
    parser = argparse.ArgumentParser(description="DINOv2 patch-memory anomaly scoring for MVTec LOCO images.")
    parser.add_argument("--category", default="breakfast_box")
    parser.add_argument("--data_root", default="data/mvtec_loco")
    parser.add_argument("--composition_root", "--seg_root", default="data/mvtec_loco_composition_maps")
    parser.add_argument("--composition_num_classes", type=int, default=None)
    parser.add_argument("--train_split", default="train/good")
    parser.add_argument("--score_split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata_output", default=None)
    parser.add_argument("--model", default="dinov2_vits14")
    parser.add_argument("--image_size", type=int, default=252)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--class_conditional", action="store_true")
    parser.add_argument("--pool", "--pooling", choices=("mean", "max", "topk"), default="mean")
    parser.add_argument("--topk_percent", type=float, default=0.01)
    parser.add_argument("--max_memory_patches", type=int, default=0)
    parser.add_argument("--coreset_per_class", type=int, default=0)
    parser.add_argument("--pca_dim", type=int, default=0)
    parser.add_argument("--pca_renorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument("--foreground_only", action="store_true")
    parser.add_argument("--background_class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.topk_percent <= 0 or args.topk_percent > 1:
        raise ValueError("--topk_percent must be in (0, 1]")
    if args.max_memory_patches < 0:
        raise ValueError("--max_memory_patches must be non-negative")
    if args.coreset_per_class < 0:
        raise ValueError("--coreset_per_class must be non-negative")
    if args.pca_dim < 0:
        raise ValueError("--pca_dim must be non-negative")

    category_root = Path(args.data_root) / args.category
    train_root = category_root / args.train_split
    score_root = category_root / args.score_split
    train_paths = image_paths(train_root)
    score_paths = image_paths(score_root)
    if not train_paths:
        raise FileNotFoundError(f"No training images found under {train_root}")
    if not score_paths:
        raise FileNotFoundError(f"No score images found under {score_root}")

    run_start = time.perf_counter()
    timings = {}
    device = torch.device(args.device)

    step_start = time.perf_counter()
    train_features, train_cache = extract_dataset_features(
        train_paths, args, device, "Extracting train DINOv2 patches"
    )
    timings["train_feature_time_sec"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    score_features_list, score_cache = extract_dataset_features(
        score_paths, args, device, "Extracting score DINOv2 patches"
    )
    timings["score_feature_time_sec"] = time.perf_counter() - step_start

    step_start = time.perf_counter()
    train_features, score_features_list, pca_metadata = transform_features_with_pca(
        train_features, score_features_list, args
    )
    timings["pca_time_sec"] = time.perf_counter() - step_start

    feature_dim = int(train_features[0].shape[1])
    patches_per_image = int(train_features[0].shape[0])

    if args.class_conditional:
        step_start = time.perf_counter()
        class_memory = build_class_conditional_memory(train_paths, train_features, args)
        if not class_memory:
            raise RuntimeError("Class-conditional memory is empty.")
        global_memory = np.concatenate(list(class_memory.values()), axis=0).astype(np.float32)
        timings["memory_build_time_sec"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        global_k = min(args.k, len(global_memory))
        global_neighbors = NearestNeighbors(n_neighbors=global_k, metric="euclidean", algorithm="brute")
        global_neighbors.fit(global_memory)
        class_neighbors = {}
        for class_id, memory in sorted(class_memory.items()):
            n_neighbors = min(args.k, len(memory))
            neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="brute")
            neighbors.fit(memory)
            class_neighbors[class_id] = neighbors
        timings["neighbor_fit_time_sec"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        scores = score_class_conditional_features(
            score_paths, score_features_list, class_neighbors, global_neighbors, args
        )
        timings["scoring_time_sec"] = time.perf_counter() - step_start
        memory_sizes = {class_id: len(memory) for class_id, memory in sorted(class_memory.items())}
        memory_patches_total = int(sum(memory_sizes.values()))
        memory_size_bytes = memory_bytes(class_memory)
    else:
        step_start = time.perf_counter()
        memory = build_memory(train_features, args.max_memory_patches, args.seed)
        timings["memory_build_time_sec"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        nearest_neighbors = NearestNeighbors(n_neighbors=args.k, metric="euclidean", algorithm="brute")
        nearest_neighbors.fit(memory)
        timings["neighbor_fit_time_sec"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        scores = score_features(score_features_list, nearest_neighbors, args.k)
        timings["scoring_time_sec"] = time.perf_counter() - step_start
        memory_sizes = None
        memory_patches_total = int(len(memory))
        memory_size_bytes = memory_bytes(memory)

    timings["total_time_sec"] = time.perf_counter() - run_start

    rows = []
    for path, score_values in zip(score_paths, scores):
        defect_class = defect_class_from_path(path)
        if args.class_conditional:
            pooled_score, mean_score, max_score, missing_class_patches = score_values
        else:
            mean_score, max_score = score_values
            pooled_score = mean_score
            missing_class_patches = 0
        row = {
            "Sample_Key": sample_key(str(path), defect_class),
            "Image_Path": str(path),
            "Defect_Class": defect_class,
            "Ground_Truth": ground_truth_from_path(path),
            "DINO_Memory_Score": pooled_score,
            "DINO_Memory_Mean_Score": mean_score,
            "DINO_Memory_Max_Score": max_score,
            "DINO_Memory_Feature_Dim": feature_dim,
            "DINO_Memory_Total_Patches": memory_patches_total,
            "DINO_Memory_Memory_MB": memory_size_bytes / (1024 ** 2),
            "DINO_Memory_PCA_Dim": args.pca_dim,
            "DINO_Memory_Coreset_Per_Class": class_memory_limit(args) if args.class_conditional else args.max_memory_patches,
            "DINO_Memory_Foreground_Only": bool(args.foreground_only),
        }
        if args.class_conditional:
            row.update({
                "DINO_Memory_Mode": "class_conditional",
                "DINO_Memory_Pool": args.pool,
                "DINO_Memory_TopK_Percent": args.topk_percent,
                "DINO_Memory_Missing_Class_Patches": missing_class_patches,
            })
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} DINO memory scores to {output}")
    if args.class_conditional:
        print(f"Class-conditional memory patches: {memory_sizes}")
    else:
        print(f"Memory patches: {len(memory)}")
    print(f"Total runtime: {timings['total_time_sec']:.4f}s")
    print(f"Runtime per scored image: {timings['total_time_sec'] / len(score_paths):.6f}s")

    if args.metadata_output:
        metadata = {
            "category": args.category,
            "train_split": args.train_split,
            "score_split": args.score_split,
            "model": args.model,
            "image_size": args.image_size,
            "device": args.device,
            "class_conditional": bool(args.class_conditional),
            "pool": args.pool,
            "topk_percent": args.topk_percent,
            "k": args.k,
            "max_memory_patches": args.max_memory_patches,
            "coreset_per_class": class_memory_limit(args) if args.class_conditional else 0,
            "foreground_only": bool(args.foreground_only),
            "background_class": args.background_class,
            "n_train_images": len(train_paths),
            "n_score_images": len(score_paths),
            "patches_per_image": patches_per_image,
            "feature_dim": feature_dim,
            "memory_patches_total": memory_patches_total,
            "memory_patches_by_class": memory_sizes,
            "memory_mb": memory_size_bytes / (1024 ** 2),
            "timings_sec": timings,
            "runtime_per_score_image_sec": timings["total_time_sec"] / len(score_paths),
            "scoring_runtime_per_score_image_sec": timings["scoring_time_sec"] / len(score_paths),
            "feature_cache": {
                "directory": args.feature_cache_dir,
                "train": train_cache,
                "score": score_cache,
            },
            "pca": pca_metadata,
        }
        metadata_output = Path(args.metadata_output)
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote metadata to {metadata_output}")


if __name__ == "__main__":
    main()
