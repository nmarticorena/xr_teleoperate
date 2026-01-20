import os
import time
import msgspec
from typing import Optional, Literal
from .schema import *
from .rerun_visualizer import RerunLogger
from queue import Queue, Empty, Full
from threading import Thread
from collections import deque
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

class UnitreeRecorder():
    def __init__(self,
                 save_path: str,
                 *,
                 # meta
                 task_name: str,
                 task_goal: str,
                 task_desc: str,
                 data_freq: float,
                 body: Literal['G1_29', 'G1_23', 'G1_D', 'H1_2'],
                 end_effector: Optional[Literal['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco']] = None,
                 author: str = AUTHOR,
                 license: str = LICENSE,
                 # data
                 data: Optional[Data] = None,
                 # utils
                 log: bool = True,
                 vis: bool = True
                ):
        """
        Initialize the UnitreeRecorder.
        """
        logger_mp.info("==> ⏳ UnitreeRecorder initializing...")
        # Create UnitreeSchema instance
        self.unitree_schema = UnitreeSchema.from_values(
            task_goal=task_goal,
            task_desc=task_desc,
            data_freq=data_freq,
            body=body,
            end_effector=end_effector,
            author=author,
            license=license,
            data=data
        )
        current_fingerprint = self.unitree_schema.get_fingerprint()

        # Initialize paths and directories
        self.save_path: str = save_path
        self.task_name: str = task_name
        self.data_freq: float = data_freq
        self.log: bool = log
        self.vis: bool = vis
        self.task_path: str = os.path.join(self.save_path, self.task_name)
        self.meta_json_path: str = os.path.join(self.task_path, 'meta.json')
        self.task_data_path: str = os.path.join(self.task_path, 'data')
        self.episode_path: Optional[str] = None
        self.data_jsonl_path: Optional[str] = None
        os.makedirs(self.task_path, exist_ok=True)
        os.makedirs(self.task_data_path, exist_ok=True)

        # Verify or create meta.json
        if os.path.exists(self.meta_json_path) and os.path.getsize(self.meta_json_path) > 0:
            with open(self.meta_json_path, "rb") as f:
                existing_meta = msgspec.json.decode(f.read())
                existing_fingerprint = existing_meta.get("_schema_hash")
                if existing_fingerprint != current_fingerprint:
                    # TODO: provide a utility to diff the two metadata
                    logger_mp.error(f"==> ❌ Meta mismatch! Existing hash: {existing_fingerprint}, Current hash: {current_fingerprint}")
                    raise ValueError("Configuration changed! Please use a new task_name or revert changes.")
            logger_mp.info(f"==> Meta hash verified: {current_fingerprint}. Resume recording.")
        else:
            with open(self.meta_json_path, "wb") as f:
                f.write(self.unitree_schema.to_meta_json())
            logger_mp.info(f"==> 📝 New task initialized. Meta saved with hash: {current_fingerprint}")

        # Initialize episode id
        self.episode_id: int = -1
        existing_ids = []
        for entry in os.scandir(self.task_data_path):
            if entry.is_dir() and entry.name.startswith('episode_'):
                parts = entry.name.split('_')
                if parts[-1].isdigit():
                    existing_ids.append(int(parts[-1]))
        if existing_ids:
            self.episode_id = max(existing_ids)
            logger_mp.info(f"==> There are existing {len(existing_ids)} episodes. Next episode id: {self.episode_id + 1}")

        # Initialize frame queue and worker thread
        self.is_available: bool = True  # Indicates whether the UnitreeRecorder is available for new operations
        self.stop_worker: bool = False  # Flag to stop the worker thread
        self.need_save: bool = False    # Flag to indicate when save_episode is triggered
        self.frame_queue: Queue[Frame] = Queue(maxsize=int(data_freq * 10))
        self.worker_thread: Thread = Thread(target=self._process_frame_queue, daemon=True)
        self.worker_thread.start()

        if self.log:
            self.fps_monitor = SimpleFPSMonitor(window_size=int(data_freq * 0.3))
        if self.vis:
            logger_mp.info("==> ⏳ RerunLogger initializing...")
            self.rerun_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit = "300MB")
            logger_mp.info("==> ✅ RerunLogger initializing ok.")

        logger_mp.info("==> ✅ UnitreeRecorder initialized successfully.")

    def is_ready(self) -> bool:
        return self.is_available
 
    def create_episode(self) -> bool:
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_available:
            logger_mp.info("==> ⏳ Busy...")
            return False
        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved

        # Reset episode-related data and create necessary directories
        self.frame_id: int = -1
        self.episode_id += 1
        self.episode_path = os.path.join(self.task_data_path, f"episode_{self.episode_id:06d}")
        os.makedirs(self.episode_path, exist_ok=True)
        self.data_jsonl_path = os.path.join(self.episode_path, 'data.jsonl')
        self.data_jsonl_wrtier = open(self.data_jsonl_path, "wb")
        logger_mp.info(f"==> New episode created: {self.episode_path}")

        if self.log:
            self.fps_monitor.reset()
        if self.vis:
            self.online_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit="300MB")
        return True  # Return True if the episode is successfully created
    
    def add_frame(self, data: Data):
        self.frame_id += 1
        try:
            self.frame_queue.put_nowait(Frame(id=self.frame_id, data=data))
        except Full:
            logger_mp.warning("==> ⚠️ Frame queue is full. Dropping frame.")

        if self.log:
            self.fps_monitor.tick()
            logger_mp.info(f"==> episode_id:{self.episode_id}  step_id:{self.frame_id}" 
                        + (f" fps:{self.fps_monitor.fps:.2f}" if self.fps_monitor.fps > 0 else " fps: calculating..."))

    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        self.need_save = True  # Set the save flag
        logger_mp.info(f"==> Episode saved start...")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self.frame_queue.join()
        if not self.is_available:  # If self.is_available is False, it means there is still data not saved.
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()
        Frame.close()

    def _process_frame_queue(self):
        while not self.stop_worker or not self.frame_queue.empty():
            try:
                new_frame = self.frame_queue.get(timeout=0.1)
                try:
                    new_frame.write(self.episode_path, self.data_jsonl_wrtier)
                    if self.vis:
                        self.rerun_logger.log_step_data(new_frame)
                except Exception as e:
                    logger_mp.warning(f"==> ⚠️  {e}")
                finally:
                    self.frame_queue.task_done()
            except Empty:
                if self.need_save:
                    self._save_episode()

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        self.need_save = False     # Reset the save flag
        if self.data_jsonl_wrtier:
            self.data_jsonl_wrtier.flush()
            os.fsync(self.data_jsonl_wrtier.fileno())
            self.data_jsonl_wrtier.close()
            self.data_jsonl_wrtier = None
        Frame.wait_write_complete()
        self.is_available = True   # Mark the class as available after saving
        logger_mp.info(f"==> Episode saved successfully to {self.data_jsonl_path}.")

class SimpleFPSMonitor:
    def __init__(self, window_size: int = 10):
        self._times = deque(maxlen=window_size)
        self._rolling_sum = 0.0
        self._last_tick = None
        self._fps = 0.0

    def tick(self):
        now = time.monotonic()
        if self._last_tick is not None:
            interval = now - self._last_tick
            if interval < 1e-6:
                return
            
            if len(self._times) == self._times.maxlen:
                self._rolling_sum -= self._times[0]
            
            self._times.append(interval)
            self._rolling_sum += interval
            
            self._fps = len(self._times) / self._rolling_sum
            
        self._last_tick = now
    
    def reset(self):
        self._times.clear()
        self._rolling_sum = 0.0
        self._last_tick = None
        self._fps = 0.0

    @property
    def fps(self):
        return self._fps