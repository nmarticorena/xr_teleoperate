import numpy as np
import hashlib
import msgspec
from msgspec.structs import replace
from typing import Optional, Union, Any, Annotated, Literal
import os

VERSION = "2.0.0beta"
AUTHOR = "Unitree Robotics"
LICENSE = "CC-BY-NC-SA 4.0"
BODY_JOINT_NAMES = {
    "G1_29": {
        "left_leg": ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", 
                     "left_knee", "left_ankle_pitch", "left_ankle_roll"],
        "right_leg": ["right_hip_pitch", "right_hip_roll", "right_hip_yaw", 
                      "right_knee", "right_ankle_pitch", "right_ankle_roll"],
        "waist": ["waist_yaw", "waist_roll", "waist_pitch"],
        "left_arm": ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", 
                     "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"],
        "right_arm": ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", 
                      "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]
    },
    "G1_23": {
        "left_leg": ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", 
                     "left_knee", "left_ankle_pitch", "left_ankle_roll"],
        "right_leg": ["right_hip_pitch", "right_hip_roll", "right_hip_yaw", 
                      "right_knee", "right_ankle_pitch", "right_ankle_roll"],
        "waist": ["waist_yaw"],
        "left_arm": ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", 
                     "left_elbow", "left_wrist_roll"],
        "right_arm": ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", 
                      "right_elbow", "right_wrist_roll"]
    },
    "G1_D": { 
        "waist": ["waist_yaw", "waist_pitch"],
        "left_arm": ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", 
                     "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"],
        "right_arm": ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", 
                      "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]
    },
    "H1_2": {
        "left_leg": ["left_hip_yaw", "left_hip_pitch", "left_hip_roll", 
                     "left_knee", "left_ankle_pitch", "left_ankle_roll"],
        "right_leg": ["right_hip_yaw", "right_hip_pitch", "right_hip_roll", 
                      "right_knee", "right_ankle_pitch", "right_ankle_roll"],
        "waist": ["torso"],
        "left_arm": ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", 
                     "left_elbow_pitch", "left_elbow_roll", "left_wrist_pitch", "left_wrist_yaw"],
        "right_arm": ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", 
                      "right_elbow_pitch", "right_elbow_roll", "right_wrist_pitch", "right_wrist_yaw"]
    },
    # "H1": { ... } # TODO
}
EE_JOINT_NAMES = {
    "dex1": {
        "left_ee": ["Joint1"],
        "right_ee": ["Joint0"]
    },
    "dex3": {
        "left_ee": ["left_hand_thumb_0", "left_hand_thumb_1", "left_hand_thumb_2",
                    "left_hand_middle_0", "left_hand_middle_1",
                    "left_hand_index_0", "left_hand_index_1"],
        "right_ee": ["right_hand_thumb_0", "right_hand_thumb_1", "right_hand_thumb_2",
                     "right_hand_index_0", "right_hand_index_1",
                     "right_hand_middle_0", "right_hand_middle_1"]
    },
    "inspire_dfx": {
        "left_ee": ['L_thumb_proximal_yaw', 'L_thumb_proximal_pitch', 'L_index_proximal', 
                    'L_middle_proximal', 'L_ring_proximal', 'L_pinky_proximal'],
        "right_ee": ['R_thumb_proximal_yaw', 'R_thumb_proximal_pitch', 'R_index_proximal', 
                     'R_middle_proximal', 'R_ring_proximal', 'R_pinky_proximal']
    },
    "inspire_ftp": {
        "left_ee": ['L_thumb_proximal_yaw', 'L_thumb_proximal_pitch', 'L_index_proximal', 
                    'L_middle_proximal', 'L_ring_proximal', 'L_pinky_proximal'],
        "right_ee": ['R_thumb_proximal_yaw', 'R_thumb_proximal_pitch', 'R_index_proximal', 
                     'R_middle_proximal', 'R_ring_proximal', 'R_pinky_proximal']
    },
    "brainco": {
      "left_ee": ["left_thumb_metacarpal", "left_thumb_proximal", "left_index_proximal",
                  "left_middle_proximal", "left_ring_proximal", "left_pinky_proximal"],
      "right_ee": ["right_thumb_metacarpal", "right_thumb_proximal", "right_index_proximal",
                   "right_middle_proximal", "right_ring_proximal", "right_pinky_proximal"]
    }
}

def enc_hook(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Objects of type {type(obj)} are not supported")

# atomic
class Dehydratable:
    def dehydrate(self, storage_dir: str, prefix: str) -> "Dehydratable":
        raise NotImplementedError("Dehydrate method not implemented.")

class Joint(msgspec.Struct, omit_defaults=True):
    qpos: Annotated[Optional[list[float]], msgspec.Meta(description="Position", extra={"unit":"rad"})] = None
    qvel: Annotated[Optional[list[float]], msgspec.Meta(description="Velocity", extra={"unit":"rad/s"})] = None
    qtau: Annotated[Optional[list[float]], msgspec.Meta(description="Joint torque", extra={"unit":"N·m"})] = None
    name: Annotated[Optional[list[str]], msgspec.Meta(description="Ordered joint names")] = None

class EEJoint(Joint):
    pose: Annotated[Optional[list[float]], msgspec.Meta(description="Cartesian pose (x,y,z,qw,qx,qy,qz)")] = None

class Body(msgspec.Struct, omit_defaults=True):
    head: Optional[Joint] = None
    neck: Optional[Joint] = None
    left_arm: Optional[Joint] = None
    right_arm: Optional[Joint] = None
    waist: Optional[Joint] = None
    left_leg: Optional[Joint] = None
    right_leg: Optional[Joint] = None
    # Additional body parts can be added here

class EndEffector(msgspec.Struct, omit_defaults=True):
    left_ee: Optional[EEJoint] = None
    right_ee: Optional[EEJoint] = None
    # Additional end-effectors can be added here

# sensors with dehydration
class Image(msgspec.Struct, omit_defaults=True):
    raw: Annotated[Optional[bytes], msgspec.Meta(description="Raw image byte data (memory)")] = None
    path: Annotated[Optional[str], msgspec.Meta(description="File path or data URI (disk)")] = None
    format: Annotated[Optional[str], msgspec.Meta(description="raw data format, e.g., 'jpg'")] = None
    shape: Annotated[Optional[list[int]], msgspec.Meta(description="Array shape [height, width]")] = None

class Color(Image, Dehydratable):
    def dehydrate(self, storage_dir: str, prefix: str) -> "Image":
        if self.raw is None: return replace(self, raw=None, format=None, path=None)
        path = os.path.join(storage_dir, f"{prefix}.{self.format}")
        with open(path, "wb") as f: f.write(self.raw)
        return replace(self, raw=None, format=None, path=path)

class Depth(Image, Dehydratable):
    def dehydrate(self, storage_dir: str, prefix: str) -> "Image":
        if self.raw is None: return replace(self, raw=None, format=None, path=None)
        path = os.path.join(storage_dir, f"{prefix}.{self.format}")
        with open(path, "wb") as f: f.write(self.raw)
        return replace(self, raw=None, format=None, path=path)

class Colors(msgspec.Struct, omit_defaults=True):
    head_camera: Optional[Color] = None
    left_wrist_camera: Optional[Color] = None
    right_wrist_camera: Optional[Color] = None
class Depths(msgspec.Struct, omit_defaults=True):
    head_camera: Optional[Depth] = None
    left_wrist_camera: Optional[Depth] = None
    right_wrist_camera: Optional[Depth] = None

class PointCloud(msgspec.Struct, Dehydratable, omit_defaults=True):
    raw: Annotated[Any, msgspec.Meta(description="Raw point cloud data (memory)")] = None
    format: Annotated[Optional[str], msgspec.Meta(description="raw data format, e.g., 'pcd'")] = None
    path: Annotated[Optional[str], msgspec.Meta(description="File path or data URI (disk)")] = None
    shape: Annotated[Optional[list[int]], msgspec.Meta(description="Array shape")] = None
    fields: Annotated[Optional[list[str]], msgspec.Meta(description="Fields in each point")] = None
    pose: Annotated[Optional[list[float]], msgspec.Meta(description="Sensor pose (x,y,z,qw,qx,qy,qz)")] = None
    frame: Annotated[Optional[str], msgspec.Meta(description="Reference frame")] = None

    def dehydrate(self, storage_dir: str, prefix: str) -> "PointCloud":
        if self.raw is None: return self
        path = os.path.join(storage_dir, f"{prefix}.{self.format}")
        # TODO
        return replace(self, raw=None, format=None, path=path)

# audio
class Audio(msgspec.Struct, Dehydratable, omit_defaults=True):
    raw: Annotated[Any, msgspec.Meta(description="Raw audio data (memory)")] = None
    format: Annotated[Optional[str], msgspec.Meta(description="raw data format, e.g., 'wav'")] = None
    path: Annotated[Optional[str], msgspec.Meta(description="File path or data URI (disk)")] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bits: Optional[int] = None

    def dehydrate(self, storage_dir: str, prefix: str) -> "Audio":
        if self.raw is None: return self
        path = os.path.join(storage_dir, f"{prefix}.{self.format}")
        # TODO
        return replace(self, raw=None, format=None, path=path)

# chassis and odometry
class Base(msgspec.Struct, omit_defaults=True):
    linear_vel: Annotated[Optional[list[float]], msgspec.Meta(description="Linear velocity (lx,ly,lz)", extra={"unit":"m/s"})] = None
    angular_vel: Annotated[Optional[list[float]], msgspec.Meta(description="Angular velocity (ax,ay,az)", extra={"unit":"rad/s"})] = None
    height: Annotated[Optional[float], msgspec.Meta(description="body height", extra={"unit":"m"})] = None

class IMU(msgspec.Struct, omit_defaults=True):
    accel: Annotated[Optional[list[float]], msgspec.Meta(description="Linear acceleration (ax,ay,az)", extra={"unit":"m/s²"})] = None
    gyro: Annotated[Optional[list[float]], msgspec.Meta(description="Angular velocity (gx,gy,gz)", extra={"unit":"rad/s"})] = None
    rpy: Annotated[Optional[list[float]], msgspec.Meta(description="Orientation (roll,pitch,yaw)", extra={"unit":"rad"})] = None
    quat: Annotated[Optional[list[float]], msgspec.Meta(description="Orientation quaternion (qw,qx,qy,qz)")] = None

class IMUs(msgspec.Struct, omit_defaults=True):
    body: Optional[IMU] = None
    # Additional IMU sensors can be added here

class Odom(msgspec.Struct, omit_defaults=True):
    position: Annotated[Optional[list[float]], msgspec.Meta(description="Position (px,py,pz)", extra={"unit":"m"})] = None
    linear: Annotated[Optional[list[float]], msgspec.Meta(description="Linear velocity", extra={"unit":"m/s"})] = None
    rpy: Annotated[Optional[list[float]], msgspec.Meta(description="Orientation (roll,pitch,yaw)", extra={"unit":"rad"})] = None

# tactile
class Tactile(msgspec.Struct, omit_defaults=True):
    left_ee: Optional[list[float]] = None
    right_ee: Optional[list[float]] = None
    # Additional tactile sensors can be added here

class Instruction(msgspec.Struct, omit_defaults=True):
    text: Annotated[Optional[str], msgspec.Meta(description="text prompt")] = None

# composite
class Observation(msgspec.Struct, omit_defaults=True):
    body: Optional[Body] = None
    end_effector: Optional[EndEffector] = None
    color: Optional[Colors] = None
    depth: Optional[Depths] = None
    point_cloud: Optional[PointCloud] = None
    audio: Optional[Audio] = None
    imu: Optional[IMUs] = None
    base: Optional[Base] = None
    odom: Optional[Odom] = None
    tactile: Optional[Tactile] = None
    instruction: Optional[Instruction] = None

class Action(msgspec.Struct, omit_defaults=True):
    body: Optional[Body] = None
    end_effector: Optional[EndEffector] = None
    base: Optional[Base] = None

class Data(msgspec.Struct, omit_defaults=True):
    observation: Observation = msgspec.field(default_factory=Observation)
    action: Action = msgspec.field(default_factory=Action)
    simulation_state: Annotated[Optional[dict[str, Any]], msgspec.Meta(description="Simulation state for recovery")] = None
    reward: Annotated[Optional[float], msgspec.Meta(description="Environment reward")] = None
    terminated: Annotated[Optional[bool], msgspec.Meta(description="Episode termination")] = None
    truncated: Annotated[Optional[bool], msgspec.Meta(description="Episode truncation")] = None
    extras: Annotated[Optional[dict[str, Any]], msgspec.Meta(description="Additional data")] = None

class Frame(msgspec.Struct, omit_defaults=True):
    id: Optional[Annotated[int, msgspec.Meta(description="Frame index", ge=0)]] = None
    data: Optional[Data] = None

    def write(self, episode_dir: str, jsonl_path: str):
        dehydrated_frame = self.dehydrate(episode_dir)
        with open(jsonl_path, "ab") as f:
            f.write(msgspec.json.encode(dehydrated_frame, enc_hook=enc_hook) + b"\n")

    def dehydrate(self, episode_path: str) -> "Frame":
        if self.data is None: return self 
        if self.id == 0:
            Frame.make_dehydratable_dirs(self.data, episode_path)
            Frame.validate(self)
        new_data = self._recursive_dehydrate(self.data, "data", episode_path)
        return replace(self, data=new_data)
    
    @staticmethod
    def make_dehydratable_dirs(obj: Any, episode_path: str):
        if isinstance(obj, Dehydratable):
            os.makedirs(os.path.join(episode_path, type(obj).__name__.lower()), exist_ok=True)
            return
        if hasattr(obj, "__struct_fields__"):
            for field in msgspec.structs.fields(obj):
                sub_obj = getattr(obj, field.name)
                if sub_obj is not None:
                    Frame.make_dehydratable_dirs(sub_obj, episode_path)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                Frame.make_dehydratable_dirs(item, episode_path)
        elif isinstance(obj, dict):
            for val in obj.values():
                Frame.make_dehydratable_dirs(val, episode_path)

    @staticmethod
    def validate(obj: "Frame") -> "Frame":
        try:
            return msgspec.convert(msgspec.to_builtins(obj, enc_hook=enc_hook), Frame)
        except msgspec.ValidationError as e:
            raise ValueError(f"Validation error for Frame: {e}") from e

    def _recursive_dehydrate(self, obj: Any, field_name: str, episode_path: str) -> Any:
        if isinstance(obj, Dehydratable):
            dehydratable_obj_dir = os.path.join(episode_path, type(obj).__name__.lower())
            return obj.dehydrate(dehydratable_obj_dir, f"{field_name}_{self.id:06d}" if self.id is not None else field_name)

        if hasattr(obj, "__struct_fields__"):
            changed_obj = {}
            for field in msgspec.structs.fields(obj):
                sub_obj = getattr(obj, field.name)
                if sub_obj is not None:
                    new_obj = self._recursive_dehydrate(sub_obj, field.name, episode_path)
                    if new_obj is not sub_obj:
                        changed_obj[field.name] = new_obj
            
            return replace(obj, **changed_obj) if changed_obj else obj
        
        if isinstance(obj, (list, tuple)):
            new_list = [self._recursive_dehydrate(item, f"{field_name}_{i}", episode_path) for i, item in enumerate(obj)]
            if any(n is not o for n, o in zip(new_list, obj)):
                return type(obj)(new_list)
            return obj

        if isinstance(obj, dict):
            new_dict = {k: self._recursive_dehydrate(v, f"{field_name}_{k}", episode_path) for k, v in obj.items()}
            if any(new_dict[k] is not obj[k] for k in obj):
                return new_dict
            return obj

        return obj

class UnitreeSchema(msgspec.Struct, frozen=True):
    # meta
    task_goal: Annotated[Optional[str], msgspec.Meta(description = "Short task goal (one sentence)")] = None
    task_desc: Annotated[Optional[str], msgspec.Meta(description = "Detailed task description")] = None
    data_freq: Annotated[Optional[float], msgspec.Meta(description = "Data recording frequency", extra={"unit":"Hz"})] = None
    body: Annotated[Optional[str], msgspec.Meta(description="G1_29, G1_23, H1, H1_2, etc.")] = None
    end_effector: Annotated[Optional[str], msgspec.Meta(description="dex1, dex3, brainco, inspire_ftp, inspire_dfx, etc.")] = None
    author:  str = AUTHOR
    license: str = LICENSE
    version: str = VERSION
    # data
    frame: Optional[Frame] = None

    @classmethod
    def from_values(
        cls,
        # meta
        task_goal: str,
        task_desc: str,
        data_freq: float,
        body: Literal["G1_29", "G1_23", "G1_D", "H1_2"],
        end_effector: Optional[Literal['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco']] = None,
        author: str = AUTHOR,
        license: str = LICENSE,
        # data
        data: Optional[Data] = None,
    ) -> "UnitreeSchema":
        d = data or Data()
        std_body = Body(**{k: Joint(name=v) for k, v in BODY_JOINT_NAMES[body].items()})
        std_ee = EndEffector(**{k: EEJoint(name=v) for k, v in EE_JOINT_NAMES[end_effector].items()}) if end_effector else None
        frame=Frame.validate(
            Frame(
                data=replace(
                    d, 
                    observation=replace(d.observation, body=std_body, end_effector=std_ee),
                    action=replace(d.action, body=std_body, end_effector=std_ee)
                )
            )
        )

        return cls(
            task_goal=task_goal,
            task_desc=task_desc,
            data_freq=data_freq,
            body=body,
            end_effector=end_effector,
            author=author,
            license=license,
            frame=frame
        )
    
    def get_fingerprint(self) -> str:
        return hashlib.md5(msgspec.json.encode(self, enc_hook=enc_hook)).hexdigest()

    def to_meta_json(self, indent: int = 4) -> bytes:
        spec_dict = self.to_spec()
        spec_dict["_schema_hash"] = self.get_fingerprint()
        raw = msgspec.json.encode(spec_dict, enc_hook=enc_hook)
        return msgspec.json.format(raw, indent=indent)
    
    def to_spec(self) -> dict[str, Any]:
        return self._unpack(self.__class__, self)

    @staticmethod
    def _unpack(cls_t: Any, inst: Any = None) -> Any:
        if not hasattr(cls_t, "__struct_fields__"):
            return inst

        res = {}
        for f in msgspec.structs.fields(cls_t):
            val = getattr(inst, f.name) if inst else None

            desc, unit = None, None
            if hasattr(f.type, "__metadata__") and f.type.__metadata__:
                m = f.type.__metadata__[0]
                if isinstance(m, msgspec.Meta):
                    desc = m.description
                    unit = m.extra.get("unit") if m.extra else None

            base_t = f.type
            if hasattr(f.type, "__origin__") and f.type.__origin__ is Union:
                args = [a for a in f.type.__args__ if a is not type(None)]
                base_t = args[0] if args else base_t

            if hasattr(base_t, "__struct_fields__"):
                unpacked_val = UnitreeSchema._unpack(base_t, val)
                if desc:
                    res[f.name] = {"value": unpacked_val, "desc": desc}
                else:
                    res[f.name] = unpacked_val
            else:
                if desc:
                    item = {"desc": desc}
                    if val is not None: item["value"] = val
                    if unit: item["unit"] = unit
                    res[f.name] = item
                elif val is not None:
                    res[f.name] = val
                    
        return res


__all__ = [
    "VERSION", "AUTHOR", "LICENSE", "BODY_JOINT_NAMES", "EE_JOINT_NAMES",
    "UnitreeSchema",
    "Data", "Frame",
    "Body",
    "EndEffector",
    "Base",
    "Image",
    "Color", "Colors",
    "Depth", "Depths",
    "PointCloud",
    "Audio",
    "Odom",
    "IMU", "IMUs",
    "Tactile",
    "Observation",
    "Action",
    "Joint", "EEJoint"
]