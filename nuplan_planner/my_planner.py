from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, List, Optional, Sequence, Type
import math
import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.observation.observation_type import Observation, Sensors
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class MyPlanner(AbstractPlanner):
    """
    Planner wrapper that runs DrivingWorld inside the official nuPlan planner API.

    The key design choice is to keep nuPlan's planner shell unchanged and only
    replace the trajectory generation logic with the repo's original world-model
    rollout:
      1. consume log-based front-camera history from nuPlan,
      2. predict next relative ego motion with DrivingWorld,
      3. generate the matching next image token,
      4. slide the window and repeat until a trajectory horizon is produced.
    """

    def __init__(
        self,
        horizon_seconds: float = 8.0,
        sampling_time: float = 0.2,
        fallback_speed_mps: float = 2.0,
        repo_root: Optional[str] = None,
        model_config_path: str = "configs/drivingworld_v1/nuplan_action_eval.py",
        world_model_path: str = "pretrained_models/world_model.pth",
        vq_model_path: Optional[str] = None,
        device: str = "cuda:0",
        camera_channel: str = "CAM_F0",
        use_bfloat16: bool = True,
        strict_sensor_check: bool = False,
        sampling_mtd: Optional[str] = None,
        top_k: int = 30,
        temperature_k: float = 1.0,
        top_p: float = 0.8,
        temperature_p: float = 1.0,
    ):
        self.horizon_seconds = horizon_seconds
        self.sampling_time = sampling_time
        self.fallback_speed_mps = fallback_speed_mps
        self.repo_root = repo_root
        self.model_config_path = model_config_path
        self.world_model_path = world_model_path
        self.vq_model_path = vq_model_path
        self.device = device
        self.camera_channel = camera_channel
        self.use_bfloat16 = use_bfloat16
        self.strict_sensor_check = strict_sensor_check
        self.sampling_mtd = sampling_mtd
        self.top_k = top_k
        self.temperature_k = temperature_k
        self.top_p = top_p
        self.temperature_p = temperature_p

        self._initialization: Optional[PlannerInitialization] = None
        self._repo_root_path: Optional[Path] = None
        self._device: Optional[torch.device] = None
        self._args = None
        self._model = None
        self._tokenizer = None
        self._load_parameters = None
        self._indices_to_pose = None
        self._indices_to_yaws = None

    def name(self) -> str:
        return "DrivingWorldPlanner"

    def observation_type(self) -> Type[Observation]:
        # DrivingWorld requires front-camera history, so planner input must be sensor-based.
        return Sensors

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._initialization = initialization
        self._repo_root_path = self._resolve_repo_root()
        self._ensure_repo_imports()
        self._load_world_model()

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        nuPlan calls this once per simulation step.

        Compared with the original toy planner, the only thing that changes is
        how future states are produced: here we build DrivingWorld's history
        window from nuPlan buffers, run autoregressive rollout, and then wrap
        the predicted states back into an InterpolatedTrajectory.
        """
        ego_state: EgoState = current_input.history.current_state[0]

        try:
            history_bundle = self._build_history_bundle(current_input)
            predicted_xy, predicted_yaw_deg = self._rollout_with_world_model(
                history_bundle["images"],
                history_bundle["poses"],
                history_bundle["yaws"],
            )
            states = self._relative_motion_to_states(
                ego_state,
                predicted_xy,
                predicted_yaw_deg,
            )
            return InterpolatedTrajectory(states)
        except Exception as exc:
            if self.strict_sensor_check:
                raise
            warnings.warn(
                f"DrivingWorld planner fallback engaged because rollout failed: {exc}",
                RuntimeWarning,
            )
            return self._constant_velocity_fallback(ego_state)

    def _resolve_repo_root(self) -> Path:
        if self.repo_root is not None:
            return Path(self.repo_root).expanduser().resolve()
        env_root = os.getenv("DRIVINGWORLD_ROOT")
        if env_root:
            return Path(env_root).expanduser().resolve()
        return Path(__file__).resolve().parents[2]

    def _ensure_repo_imports(self) -> None:
        repo_root = str(self._repo_root_path)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    def _resolve_path(self, path_like: Optional[str]) -> Optional[Path]:
        if path_like is None:
            return None
        candidate = Path(path_like).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self._repo_root_path / candidate).resolve()

    def _load_world_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        from models.model import TrainTransformers
        from modules.tokenizers.model_tokenizer import Tokenizer
        from modules.tokenizers.pose_tokenizer import indices_to_pose, indices_to_yaws
        from utils.config_utils import Config
        from utils.running import load_parameters

        if not torch.cuda.is_available():
            raise RuntimeError("DrivingWorld planner requires CUDA because the model uses CUDA-only buffers.")

        self._device = torch.device(self.device)
        if self._device.type != "cuda":
            raise RuntimeError(f"Expected a CUDA device, got '{self._device}'.")
        torch.cuda.set_device(self._device)

        config_path = self._resolve_path(self.model_config_path)
        world_model_path = self._resolve_path(self.world_model_path)
        vq_model_path = self._resolve_path(self.vq_model_path) if self.vq_model_path is not None else None

        args = Config.fromfile(str(config_path))
        args.merge_from_dict(
            {
                "sampling_mtd": self.sampling_mtd,
                "top_k": self.top_k,
                "temperature_k": self.temperature_k,
                "top_p": self.top_p,
                "temperature_p": self.temperature_p,
                "use_bfloat16": self.use_bfloat16,
            }
        )
        if vq_model_path is not None:
            args.vq_ckpt = str(vq_model_path)

        model_dt = 1.0 / float(args.downsample_fps)
        if abs(self.sampling_time - model_dt) > 1e-6:
            warnings.warn(
                f"sampling_time={self.sampling_time} does not match model step {model_dt:.3f}s. "
                f"Using model step for faithful rollout.",
                RuntimeWarning,
            )
            self.sampling_time = model_dt

        model = TrainTransformers(args, local_rank=self._device.index, condition_frames=args.condition_frames)
        checkpoint = torch.load(str(world_model_path), map_location="cpu")
        model.model = load_parameters(model.model, checkpoint)
        model = model.to(self._device)
        model.eval()

        tokenizer = Tokenizer(args, self._device.index)

        self._args = args
        self._model = model
        self._tokenizer = tokenizer
        self._load_parameters = load_parameters
        self._indices_to_pose = indices_to_pose
        self._indices_to_yaws = indices_to_yaws

    def _autocast_context(self):
        if self._device.type != "cuda" or not bool(getattr(self._args, "use_bfloat16", False)):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def _build_history_bundle(self, current_input: PlannerInput) -> dict:
        ego_states = list(current_input.history.ego_states)
        observations = list(current_input.history.observations)

        if not ego_states or not observations:
            raise RuntimeError("Planner history is empty; cannot build DrivingWorld input.")

        history_len = min(len(ego_states), len(observations))
        if history_len < self._args.condition_frames:
            raise RuntimeError(
                f"Need at least {self._args.condition_frames} history steps, got {history_len}."
            )

        ego_states = ego_states[-history_len:]
        observations = observations[-history_len:]

        ego_window = ego_states[-self._args.condition_frames :]
        obs_window = observations[-self._args.condition_frames :]

        image_tensors = []
        for observation in obs_window:
            frame = self._extract_front_image(observation)
            image_tensors.append(self._preprocess_front_image(frame))

        pose_window, yaw_window = self._ego_states_to_relative_motion(ego_window)
        return {
            "images": torch.stack(image_tensors, dim=0),
            "poses": pose_window,
            "yaws": yaw_window,
        }

    def _extract_front_image(self, observation: Observation) -> np.ndarray:
        images = getattr(observation, "images", None)
        if images is None:
            raise RuntimeError(
                f"Observation does not contain sensor images. Check that planner observation_type is Sensors."
            )

        camera_value = None
        if self.camera_channel in images:
            camera_value = images[self.camera_channel]
        if camera_value is None:
            for key, value in images.items():
                key_name = getattr(key, "name", str(key))
                if key_name == self.camera_channel or str(key) == self.camera_channel:
                    camera_value = value
                    break
        if camera_value is None:
            available = [getattr(key, "name", str(key)) for key in images.keys()]
            raise RuntimeError(
                f"Camera channel '{self.camera_channel}' not found in observation. Available keys: {available}"
            )
        return self._sensor_image_to_numpy(camera_value)

    def _sensor_image_to_numpy(self, image_obj: Any) -> np.ndarray:
        if isinstance(image_obj, np.ndarray):
            image = image_obj
        elif isinstance(image_obj, Image.Image):
            image = np.asarray(image_obj)
        elif hasattr(image_obj, "as_numpy"):
            image = image_obj.as_numpy() if callable(image_obj.as_numpy) else image_obj.as_numpy
        elif hasattr(image_obj, "array"):
            image = image_obj.array() if callable(image_obj.array) else image_obj.array
        elif hasattr(image_obj, "image"):
            raw_image = image_obj.image() if callable(image_obj.image) else image_obj.image
            image = np.asarray(raw_image)
        else:
            raise RuntimeError(f"Unsupported sensor image type: {type(image_obj)}")

        if image.ndim != 3:
            raise RuntimeError(f"Expected an HWC image, got shape {image.shape}.")
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _preprocess_front_image(self, image: np.ndarray) -> torch.Tensor:
        target_h, target_w = self._args.image_size
        h, w, _ = image.shape
        if 2 * h < w:
            resized_w = round(w / h * target_h)
            resized = Image.fromarray(image).resize((resized_w, target_h), resample=Image.BICUBIC)
        else:
            resized_h = round(h / w * target_w)
            resized = Image.fromarray(image).resize((target_w, resized_h), resample=Image.BICUBIC)
        resized = np.asarray(resized)

        new_h, new_w, _ = resized.shape
        if new_w == target_w:
            x0 = int(new_h / 2 - target_h / 2)
            y0 = 0
        else:
            x0 = 0
            y0 = int(new_w / 2 - target_w / 2)
        cropped = resized[x0 : x0 + target_h, y0 : y0 + target_w, :]

        tensor = torch.from_numpy(cropped.copy()).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - 0.5) * 2.0
        return tensor

    def _ego_states_to_relative_motion(self, ego_states: Sequence[EgoState]) -> tuple[torch.Tensor, torch.Tensor]:
        relative_xy = np.zeros((len(ego_states), 2), dtype=np.float32)
        relative_yaw = np.zeros((len(ego_states), 1), dtype=np.float32)
        for idx in range(1, len(ego_states)):
            prev = ego_states[idx - 1].rear_axle
            curr = ego_states[idx].rear_axle
            delta_x_world = curr.x - prev.x
            delta_y_world = curr.y - prev.y
            cos_h = math.cos(prev.heading)
            sin_h = math.sin(prev.heading)
            relative_xy[idx, 0] = cos_h * delta_x_world + sin_h * delta_y_world
            relative_xy[idx, 1] = -sin_h * delta_x_world + cos_h * delta_y_world
            relative_yaw[idx, 0] = math.degrees(self._wrap_angle(curr.heading - prev.heading))
        return (
            torch.from_numpy(relative_xy).float(),
            torch.from_numpy(relative_yaw).float(),
        )

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2 * math.pi) - math.pi

    def _prepare_condition_tokens(self, image_window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_window = image_window.unsqueeze(0).to(self._device)
        padded_window = torch.cat([image_window, torch.zeros_like(image_window[:, :1])], dim=1)
        start_token_wpad0, _ = self._tokenizer.encode_to_z(padded_window)
        start_token = start_token_wpad0[:, : self._args.condition_frames]
        start_feature = self._tokenizer.vq_model.quantize.embedding(start_token)
        return start_token, start_feature

    def _rollout_with_world_model(
        self,
        image_window: torch.Tensor,
        pose_window: torch.Tensor,
        yaw_window: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        rollout_steps = int(round(self.horizon_seconds / self.sampling_time))
        if rollout_steps <= 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        start_token, start_feature = self._prepare_condition_tokens(image_window)
        pose_history = pose_window.clone().to(self._device)
        yaw_history = yaw_window.clone().to(self._device)

        predicted_xy: List[List[float]] = []
        predicted_yaw_deg: List[float] = []

        with torch.no_grad():
            for _ in range(rollout_steps):
                pose_cond = pose_history[-self._args.condition_frames :].clone().unsqueeze(0)
                yaw_cond = yaw_history[-self._args.condition_frames :].clone().unsqueeze(0)
                pose_cond[:, 0] = 0.0
                yaw_cond[:, 0] = 0.0

                with self._autocast_context():
                    motion_outputs = self._model.generate_next_pose_yaw(
                        start_feature,
                        pose_cond,
                        yaw_cond,
                        sampling_mtd=self._args.sampling_mtd,
                        temperature_k=self._args.temperature_k,
                        top_k=self._args.top_k,
                        temperature_p=self._args.temperature_p,
                        top_p=self._args.top_p,
                    )

                pred_pose_tensor, pred_yaw_tensor = self._decode_next_motion(motion_outputs)
                predicted_xy.append(
                    [
                        float(pred_pose_tensor[0, 0, 0].detach().cpu().item()),
                        float(pred_pose_tensor[0, 0, 1].detach().cpu().item()),
                    ]
                )
                predicted_yaw_deg.append(float(pred_yaw_tensor[0, 0, 0].detach().cpu().item()))

                with self._autocast_context():
                    next_img_indices = self._model.generate_gt_pose_gt_yaw(
                        start_token,
                        start_feature,
                        pose_cond,
                        yaw_cond,
                        pred_pose_tensor,
                        pred_yaw_tensor,
                        self._tokenizer.vq_model.quantize.embedding,
                        sampling_mtd=self._args.sampling_mtd,
                        temperature_k=self._args.temperature_k,
                        top_k=self._args.top_k,
                        temperature_p=self._args.temperature_p,
                        top_p=self._args.top_p,
                    )

                next_img_indices = next_img_indices.reshape(1, 1, -1)
                start_token = torch.cat([start_token[:, 1:], next_img_indices], dim=1)
                start_feature = self._tokenizer.vq_model.quantize.embedding(start_token)
                pose_history = torch.cat([pose_history[1:], pred_pose_tensor[0]], dim=0)
                yaw_history = torch.cat([yaw_history[1:], pred_yaw_tensor[0]], dim=0)

        return (
            np.asarray(predicted_xy, dtype=np.float32),
            np.asarray(predicted_yaw_deg, dtype=np.float32),
        )

    def _decode_next_motion(self, motion_outputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        yaw_idx = motion_outputs["yaw_indices"][:, 0, 0]
        pose_x_idx = motion_outputs["pose_indices"][:, 0, 0]
        pose_y_idx = motion_outputs["pose_indices"][:, 0, 1]

        pred_yaw = self._indices_to_yaws(yaw_idx, division=self._args.yaw_vocab_size).float()
        pred_x, pred_y = self._indices_to_pose(
            pose_x_idx,
            pose_y_idx,
            x_division=self._args.pose_x_vocab_size,
            y_divisions=self._args.pose_y_vocab_size,
        )

        pred_pose_tensor = torch.stack([pred_x.float(), pred_y.float()], dim=-1).unsqueeze(1)
        pred_yaw_tensor = pred_yaw.unsqueeze(1).unsqueeze(2)
        return pred_pose_tensor, pred_yaw_tensor

    def _relative_motion_to_states(
        self,
        ego_state: EgoState,
        relative_xy: np.ndarray,
        relative_yaw_deg: np.ndarray,
    ) -> List[EgoState]:
        states: List[EgoState] = [ego_state]

        x = ego_state.rear_axle.x
        y = ego_state.rear_axle.y
        heading = ego_state.rear_axle.heading
        vehicle_parameters = getattr(
            getattr(ego_state, "car_footprint", None),
            "vehicle_parameters",
            get_pacifica_parameters(),
        )
        tire_angle = getattr(ego_state, "tire_steering_angle", 0.0)

        for step_idx, ((dx_local, dy_local), dyaw_deg) in enumerate(
            zip(relative_xy, relative_yaw_deg),
            start=1,
        ):
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            world_dx = cos_h * float(dx_local) - sin_h * float(dy_local)
            world_dy = sin_h * float(dx_local) + cos_h * float(dy_local)
            x += world_dx
            y += world_dy
            heading = self._wrap_angle(heading + math.radians(float(dyaw_deg)))

            dt = step_idx * self.sampling_time
            next_state = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x, y, heading),
                rear_axle_velocity_2d=StateVector2D(world_dx / self.sampling_time, world_dy / self.sampling_time),
                rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
                tire_steering_angle=tire_angle,
                time_point=TimePoint(int(ego_state.time_us + dt * 1e6)),
                vehicle_parameters=vehicle_parameters,
                angular_vel=math.radians(float(dyaw_deg)) / self.sampling_time,
                angular_accel=0.0,
            )
            states.append(next_state)

        return states

    def _constant_velocity_fallback(self, ego_state: EgoState) -> AbstractTrajectory:
        states: List[EgoState] = [ego_state]
        rear_axle = ego_state.rear_axle
        heading = rear_axle.heading
        x = rear_axle.x
        y = rear_axle.y

        current_velocity = getattr(getattr(ego_state, "dynamic_car_state", None), "rear_axle_velocity_2d", None)
        if current_velocity is not None:
            speed = math.hypot(float(current_velocity.x), float(current_velocity.y))
        else:
            speed = 0.0
        if speed < 1e-3:
            speed = self.fallback_speed_mps

        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)
        vehicle_parameters = getattr(
            getattr(ego_state, "car_footprint", None),
            "vehicle_parameters",
            get_pacifica_parameters(),
        )
        tire_angle = getattr(ego_state, "tire_steering_angle", 0.0)
        num_steps = int(round(self.horizon_seconds / self.sampling_time))

        for step_idx in range(1, num_steps + 1):
            dt = step_idx * self.sampling_time
            next_state = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x + vx * dt, y + vy * dt, heading),
                rear_axle_velocity_2d=StateVector2D(vx, vy),
                rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
                tire_steering_angle=tire_angle,
                time_point=TimePoint(int(ego_state.time_us + dt * 1e6)),
                vehicle_parameters=vehicle_parameters,
                angular_vel=0.0,
                angular_accel=0.0,
            )
            states.append(next_state)

        return InterpolatedTrajectory(states)
