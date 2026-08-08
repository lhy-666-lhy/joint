"""Datasets over the shared-PCD joint door zarr."""

from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

from joint_train.utils.pc_utils import augment_xyz_m2ae, pc_normalize
from vendor.dp3.normalizer import LinearNormalizer
from vendor.dp3.sampler import create_indices, downsample_mask


class AffordanceCloudDataset(Dataset):
    """Stage-1: one sample = one replay point cloud (N,4).

    Preprocess/augment aligned with Point-M2AE ``PcdAffordanceDataset``:
    pc_normalize + (train) Z-rot ±30° + jitter 0.005.
    Extra scale/shift augment happens in the train loop via ``provider``.
    """

    def __init__(
        self,
        zarr_path: str,
        split: str = "train",
        augment: bool = True,
        normalize: bool = True,
        view_mode: str = "primary",
        label_source: str = "updated",
        num_points: Optional[int] = None,
    ):
        root = zarr.open(zarr_path, mode="r")
        view_mode = str(view_mode).lower()
        if view_mode not in {"primary", "augmentation", "combined"}:
            raise ValueError(f"unknown Stage-1 view_mode={view_mode}")
        label_source = str(label_source).lower()
        if label_source not in {"updated", "initial"}:
            raise ValueError(f"unknown Stage-1 label_source={label_source}")
        clouds = [root["data"]["point_cloud"]]
        labels = [
            root["data"].get("affordance_updated", root["data"]["point_cloud"][:, :, 3])
            if label_source == "updated"
            else root["data"]["affordance_initial"]
        ]
        split_arrays = [np.asarray(root["meta"]["replay_split"][:])]
        if view_mode in {"augmentation", "combined"}:
            if "stage1_aug_point_cloud" not in root["data"]:
                raise ValueError("zarr has no data/stage1_aug_point_cloud")
            clouds.append(root["data"]["stage1_aug_point_cloud"])
            labels.append(
                root["data"].get("stage1_aug_affordance_updated", root["data"]["stage1_aug_point_cloud"][:, :, 3])
                if label_source == "updated"
                else root["data"]["stage1_aug_affordance_initial"]
            )
            split_arrays.append(np.asarray(root["meta"]["stage1_aug_replay_split"][:]))
        source_ids = [0] if view_mode == "primary" else ([1] if view_mode == "augmentation" else [0, 1])
        self.clouds = clouds
        self.labels = labels
        self.indices: list[tuple[int, int]] = []
        for cloud_id in source_ids:
            splits = split_arrays[cloud_id]
            if split == "train":
                ids = np.nonzero(splits == 0)[0]
            elif split in ("val", "test"):
                ids = np.nonzero(splits == 1)[0]
            else:
                ids = np.arange(len(splits))
            self.indices.extend((cloud_id, int(index)) for index in ids.tolist())
        self.augment = bool(augment) and split == "train"
        self.normalize = bool(normalize)
        available_points = int(self.clouds[0].shape[1])
        self.num_points = available_points if num_points is None else int(num_points)
        if not 512 <= self.num_points <= available_points:
            raise ValueError(
                f"num_points must be in [512, {available_points}], got {self.num_points}"
            )
        # The source clouds are already FPS samples. Evenly selecting their
        # fixed ordering gives every run the same nested point subsets.
        self.point_indices = np.linspace(
            0, available_points - 1, self.num_points, dtype=np.int64
        )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        cloud_id, replay_id = self.indices[i]
        pc = np.asarray(self.clouds[cloud_id][replay_id], dtype=np.float32)[self.point_indices]
        xyz = pc[:, :3].copy()
        scores = np.clip(
            np.asarray(self.labels[cloud_id][replay_id], dtype=np.float32)[self.point_indices],
            0.0,
            1.0,
        )
        if self.normalize:
            xyz = pc_normalize(xyz)
        if self.augment:
            xyz = augment_xyz_m2ae(xyz)
        return {
            "xyz": torch.from_numpy(xyz),  # N,3
            "affordance": torch.from_numpy(scores),  # N
        }


class JointActionDataset(Dataset):
    """Stage-2: sequence samples with shared replay point cloud."""

    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        split: str = "train",
        seed: int = 42,
        max_train_episodes: Optional[int] = None,
        use_replay_split: bool = True,
        max_train_objects: Optional[int] = None,
        traj_per_object: Optional[int] = None,
        max_val_episodes: Optional[int] = None,
        max_val_objects: Optional[int] = None,
        val_traj_per_object: Optional[int] = None,
        random_train_objects: bool = False,
        train_trajectory_fraction: float = 1.0,
        random_val_objects: bool = False,
        sequence_stride: int = 1,
        view_mode: str = "primary",
        episode_ids: Optional[list[int]] = None,
        affordance_label_source: str = "updated",
        contact_sidecar: Optional[str] = None,
    ):
        self.root = zarr.open(zarr_path, mode="r")
        self.state = self.root["data"]["state"]
        self.action = self.root["data"]["action"]
        self.point_cloud = self.root["data"]["point_cloud"]
        self.affordance_label_source = str(affordance_label_source).lower()
        if self.affordance_label_source not in {"updated", "initial"}:
            raise ValueError(f"unknown affordance_label_source={affordance_label_source}")
        self.affordance_initial = self.root["data"].get("affordance_initial")
        if self.affordance_label_source == "initial" and self.affordance_initial is None:
            raise ValueError("zarr has no data/affordance_initial")
        self.view_mode = str(view_mode).lower()
        if self.view_mode not in {"primary", "augmentation", "combined"}:
            raise ValueError(f"unknown Stage-2 view_mode={view_mode}")
        self.aug_point_cloud = None
        self.aug_ids_by_source: dict[int, list[int]] = {}
        if self.view_mode != "primary":
            if "stage1_aug_point_cloud" not in self.root["data"]:
                raise ValueError("zarr has no target-aware augmentation point clouds")
            self.aug_point_cloud = self.root["data"]["stage1_aug_point_cloud"]
            source_ids = np.asarray(self.root["meta"]["stage1_aug_source_replay_id"][:], dtype=np.int32)
            for view_id, replay_id in enumerate(source_ids.tolist()):
                self.aug_ids_by_source.setdefault(int(replay_id), []).append(int(view_id))
        self.episode_ends = np.asarray(self.root["meta"]["episode_ends"][:], dtype=np.int64)
        self.episode_replay_ids = np.asarray(
            self.root["meta"]["episode_replay_ids"][:], dtype=np.int32
        )
        self.replay_split = np.asarray(self.root["meta"]["replay_split"][:], dtype=np.int8)
        self.replay_obj_keys = [str(k) for k in self.root["meta"]["replay_obj_keys"][:]]
        self.contact_sidecar = None
        if contact_sidecar is not None:
            sidecar = np.load(str(contact_sidecar), allow_pickle=False)
            if sidecar["episode_id"].shape != (len(self.episode_ends),):
                raise ValueError("contact sidecar episode count does not match zarr")
            if not np.array_equal(sidecar["episode_id"], np.arange(len(self.episode_ends))):
                raise ValueError("contact sidecar episode_id is not index-aligned")
            self.contact_sidecar = {key: sidecar[key] for key in sidecar.files}
        self.subset_detail = []

        n_ep = len(self.episode_ends)
        if use_replay_split:
            ep_split = self.replay_split[self.episode_replay_ids]
            if split == "train":
                mask = ep_split == 0
            elif split in ("val", "test"):
                mask = ep_split == 1
            else:
                mask = np.ones(n_ep, dtype=bool)
        else:
            mask = np.ones(n_ep, dtype=bool)

        # An explicit manifest takes precedence over random object/trajectory sampling.
        if episode_ids is not None:
            selected_ids = np.asarray(sorted(set(int(item) for item in episode_ids)), dtype=np.int64)
            if selected_ids.size == 0:
                raise ValueError("episode_ids must not be empty")
            if selected_ids[0] < 0 or selected_ids[-1] >= n_ep:
                raise ValueError("episode_ids contains an out-of-range episode index")
            if not np.all(mask[selected_ids]):
                raise ValueError("episode_ids contains an episode outside the requested split")
            explicit_mask = np.zeros_like(mask)
            explicit_mask[selected_ids] = True
            mask = explicit_mask
            grouped: dict[str, list[int]] = {}
            for episode_id in selected_ids.tolist():
                obj = self.replay_obj_keys[int(self.episode_replay_ids[episode_id])]
                grouped.setdefault(obj, []).append(episode_id)
            self.subset_detail = [
                {"obj": obj, "n_traj": len(ids), "episode_ids": ids}
                for obj, ids in grouped.items()
            ]

        # Subset: selected train objects × up to T random trajectories each.
        if episode_ids is None and split == "train" and max_train_objects is not None and int(max_train_objects) > 0:
            tpo = int(traj_per_object) if traj_per_object is not None else 10
            mask = self._subset_by_objects(
                mask,
                max_objects=int(max_train_objects),
                traj_per_object=tpo,
                seed=int(seed),
                random_objects=bool(random_train_objects),
            )

        if episode_ids is None and split == "train" and max_train_episodes is not None:
            mask = downsample_mask(mask, max_n=max_train_episodes, seed=seed)

        if episode_ids is None and split == "train":
            fraction = float(train_trajectory_fraction)
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"train_trajectory_fraction must be in (0, 1], got {fraction}")
            if fraction < 1.0:
                mask = self._sample_trajectory_fraction(mask, fraction=fraction, seed=int(seed))

        # Val quick check: keep first N val episodes (in episode-index order)
        if episode_ids is None and split in ("val", "test") and max_val_episodes is not None and int(max_val_episodes) > 0:
            keep_ids = np.nonzero(mask)[0][: int(max_val_episodes)]
            new_mask = np.zeros_like(mask)
            new_mask[keep_ids] = True
            mask = new_mask
            self.subset_detail = [
                {
                    "episode_id": int(ei),
                    "replay_id": int(self.episode_replay_ids[ei]),
                    "obj": self.replay_obj_keys[int(self.episode_replay_ids[ei])],
                }
                for ei in keep_ids.tolist()
            ]

        if episode_ids is None and split in ("val", "test") and max_val_objects is not None and int(max_val_objects) > 0:
            tpo = int(val_traj_per_object) if val_traj_per_object is not None else 1
            mask = self._subset_by_objects(
                mask,
                max_objects=int(max_val_objects),
                traj_per_object=tpo,
                seed=int(seed),
                random_objects=bool(random_val_objects),
            )

        self.train_mask = mask
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        if np.any(mask):
            self.indices = create_indices(
                self.episode_ends,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=mask,
            )
        else:
            self.indices = np.zeros((0, 4), dtype=np.int64)

        self.sequence_stride = max(1, int(sequence_stride))
        if split == "train" and self.sequence_stride > 1:
            self.indices = self._stride_indices(self.indices, self.sequence_stride)

        # map buffer index -> episode id for replay lookup
        self._step_to_episode = np.empty(int(self.episode_ends[-1]), dtype=np.int32)
        start = 0
        for ei, end in enumerate(self.episode_ends):
            self._step_to_episode[start:end] = ei
            start = end

        self.subset_info = {
            "n_episodes": int(mask.sum()),
            "n_sequences": int(len(self.indices)),
            "max_train_objects": max_train_objects,
            "traj_per_object": traj_per_object,
            "max_val_episodes": max_val_episodes,
            "max_val_objects": max_val_objects,
            "val_traj_per_object": val_traj_per_object,
            "random_train_objects": random_train_objects,
            "train_trajectory_fraction": train_trajectory_fraction,
            "random_val_objects": random_val_objects,
            "sequence_stride": self.sequence_stride,
            "explicit_episode_ids": episode_ids is not None,
            "affordance_label_source": self.affordance_label_source,
            "objects": [d.get("obj") for d in self.subset_detail],
        }

    def _subset_by_objects(
        self,
        mask: np.ndarray,
        *,
        max_objects: int,
        traj_per_object: int,
        seed: int,
        random_objects: bool,
    ) -> np.ndarray:
        """Keep target-balanced episode subset with deterministic sampling."""
        obj_order: list[str] = []
        for ei in np.nonzero(mask)[0].tolist():
            k = self.replay_obj_keys[int(self.episode_replay_ids[ei])]
            if k not in obj_order:
                obj_order.append(k)
        rng = np.random.default_rng(seed)
        if random_objects:
            chosen_ids = np.sort(rng.choice(len(obj_order), size=min(max_objects, len(obj_order)), replace=False))
            keep_objs = [obj_order[int(i)] for i in chosen_ids]
        else:
            keep_objs = obj_order[:max_objects]

        obj_to_eps: dict[str, list[int]] = {k: [] for k in keep_objs}
        for ei in np.nonzero(mask)[0].tolist():
            rid = int(self.episode_replay_ids[ei])
            k = self.replay_obj_keys[rid]
            if k in obj_to_eps:
                obj_to_eps[k].append(int(ei))

        keep = np.zeros(len(mask), dtype=bool)
        selected = []
        for k in keep_objs:
            eps = obj_to_eps.get(k, [])
            if not eps:
                continue
            n = min(int(traj_per_object), len(eps))
            chosen = [int(x) for x in rng.choice(eps, size=n, replace=False)]
            selected.append({"obj": k, "n_traj": n, "episode_ids": chosen})
            keep[chosen] = True

        self.subset_detail = selected
        return keep

    def _sample_trajectory_fraction(
        self,
        mask: np.ndarray,
        *,
        fraction: float,
        seed: int,
    ) -> np.ndarray:
        """Randomly retain a fixed trajectory fraction within every selected target."""
        obj_to_eps: dict[str, list[int]] = {}
        obj_order: list[str] = []
        for ei in np.nonzero(mask)[0].tolist():
            obj = self.replay_obj_keys[int(self.episode_replay_ids[ei])]
            if obj not in obj_to_eps:
                obj_to_eps[obj] = []
                obj_order.append(obj)
            obj_to_eps[obj].append(int(ei))

        rng = np.random.default_rng(seed)
        keep = np.zeros_like(mask)
        selected = []
        for obj in obj_order:
            episodes = obj_to_eps[obj]
            n_keep = max(1, int(np.ceil(len(episodes) * fraction)))
            chosen = [int(x) for x in rng.choice(episodes, size=n_keep, replace=False)]
            keep[chosen] = True
            selected.append({"obj": obj, "n_traj": n_keep, "episode_ids": chosen})
        self.subset_detail = selected
        return keep

    def _stride_indices(self, indices: np.ndarray, stride: int) -> np.ndarray:
        """Keep every ``stride``-th window per episode, including each endpoint."""
        if len(indices) == 0:
            return indices
        episode_ids = np.searchsorted(self.episode_ends, indices[:, 0], side="right")
        keep = np.zeros(len(indices), dtype=bool)
        for ep_id in np.unique(episode_ids):
            positions = np.nonzero(episode_ids == ep_id)[0]
            keep[positions[::stride]] = True
            keep[positions[-1]] = True
        return indices[keep]

    def get_validation_dataset(self) -> "JointActionDataset":
        other = copy.copy(self)
        other.train_mask = ~self.train_mask
        if np.any(other.train_mask):
            other.indices = create_indices(
                self.episode_ends,
                sequence_length=self.horizon,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=other.train_mask,
            )
        else:
            other.indices = np.zeros((0, 4), dtype=np.int64)
        return other

    def get_normalizer(self, mode: str = "limits") -> LinearNormalizer:
        """Fit on selected train episodes when a subset mask is active; else full buffers."""
        if np.any(self.train_mask) and not np.all(self.train_mask):
            chunks_s, chunks_a = [], []
            start = 0
            for ei, end in enumerate(self.episode_ends):
                if self.train_mask[ei]:
                    chunks_s.append(np.asarray(self.state[start:end], dtype=np.float32))
                    chunks_a.append(np.asarray(self.action[start:end], dtype=np.float32))
                start = int(end)
            state = np.concatenate(chunks_s, axis=0) if chunks_s else np.asarray(self.state[:])
            action = np.concatenate(chunks_a, axis=0) if chunks_a else np.asarray(self.action[:])
        else:
            state = self.state
            action = self.action
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"action": action, "state": state},
            last_n_dims=1,
            mode=mode,
        )
        return normalizer

    def __len__(self) -> int:
        return int(len(self.indices))

    def _pad_sequence(self, arr: np.ndarray, sample_start: int, sample_end: int) -> np.ndarray:
        data = np.zeros((self.horizon,) + arr.shape[1:], dtype=arr.dtype)
        if sample_start > 0:
            data[:sample_start] = arr[0]
        if sample_end < self.horizon:
            data[sample_end:] = arr[-1]
        data[sample_start:sample_end] = arr
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        b0, b1, s0, s1 = self.indices[idx]
        state = np.asarray(self.state[b0:b1], dtype=np.float32)
        action = np.asarray(self.action[b0:b1], dtype=np.float32)
        state = self._pad_sequence(state, s0, s1)
        action = self._pad_sequence(action, s0, s1)

        ep_id = int(self._step_to_episode[b0])
        rid = int(self.episode_replay_ids[ep_id])
        if self.view_mode == "primary":
            pc = np.asarray(self.point_cloud[rid], dtype=np.float32)
        else:
            aug_ids = self.aug_ids_by_source.get(rid, [])
            if self.view_mode == "augmentation" and aug_ids:
                pc = np.asarray(self.aug_point_cloud[aug_ids[idx % len(aug_ids)]], dtype=np.float32)
            elif self.view_mode == "combined" and aug_ids and idx % (len(aug_ids) + 1) > 0:
                pc = np.asarray(self.aug_point_cloud[aug_ids[(idx - 1) % len(aug_ids)]], dtype=np.float32)
            else:
                pc = np.asarray(self.point_cloud[rid], dtype=np.float32)
        output = {
            "point_cloud_xyz": torch.from_numpy(pc[:, :3]),
            "affordance_gt": torch.from_numpy(
                np.asarray(self.affordance_initial[rid], dtype=np.float32)
                if self.affordance_label_source == "initial"
                else pc[:, 3]
            ),
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "replay_id": torch.tensor(rid, dtype=torch.long),
        }
        if self.contact_sidecar is not None:
            contact = np.asarray(
                self.contact_sidecar["contact_xyz_world"][ep_id], dtype=np.float32
            )
            valid = bool(np.isfinite(contact).all())
            output.update(
                {
                    "contact_xyz_world": torch.from_numpy(
                        contact if valid else np.zeros(3, dtype=np.float32)
                    ),
                    "contact_visible_5cm": torch.tensor(
                        float(self.contact_sidecar["visible_5cm"][ep_id]), dtype=torch.float32
                    ),
                    "contact_valid": torch.tensor(float(valid), dtype=torch.float32),
                    "episode_id": torch.tensor(ep_id, dtype=torch.long),
                }
            )
        return output
