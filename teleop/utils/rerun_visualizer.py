import logging_mp
import uuid

import json
import os
import time
from datetime import datetime
from motion_tools.robot_gui import ReRunRobot, get_blueprint
import numpy as np
from pathlib import Path

import cv2
import rerun as rr
import rerun.blueprint as rrb

os.environ["RUST_LOG"] = "error"


class RerunEpisodeReader:
    def __init__(self, task_dir=".", json_file="data.json"):
        self.task_dir = task_dir
        self.json_file = json_file

    def return_episode_data(self, episode_idx):
        # Load episode data on-demand
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_idx:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Episode {episode_idx} data.json not found.")

        with open(json_path, "r", encoding="utf-8") as jsonf:
            json_file = json.load(jsonf)

        episode_data = []

        # Loop over the data entries and process each one
        for item_data in json_file["data"]:
            # Process images and other data
            colors = self._process_images(item_data, "colors", episode_dir)
            depths = self._process_images(item_data, "depths", episode_dir)
            audios = self._process_audio(item_data, "audios", episode_dir)

            # Append the data in the item_data list
            episode_data.append(
                {
                    "idx": item_data.get("idx", 0),
                    "colors": colors,
                    "depths": depths,
                    "states": item_data.get("states", {}),
                    "actions": item_data.get("actions", {}),
                    "tactiles": item_data.get("tactiles", {}),
                    "audios": audios,
                }
            )

        return episode_data

    def _process_images(self, item_data, data_type, dir_path):
        images = {}

        for key, file_name in item_data.get(data_type, {}).items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images[key] = image
        return images

    def _process_audio(self, item_data, data_type, episode_dir):
        audio_data = {}
        dir_path = os.path.join(episode_dir, data_type)

        for key, file_name in item_data.get(data_type, {}).items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    pass  # Handle audio data if needed
        return audio_data


class RerunLogger:
    def __init__(self, prefix="", IdxRangeBoundary=30, memory_limit=None, save_dir="utils/data/stack_blocks/rerun/"):
        self.prefix = prefix
        self.IdxRangeBoundary = IdxRangeBoundary
        self.memory_limit = memory_limit
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # These are per-episode
        self.rec = None
        self.robot = None


    def start_episode(self, prefix: str, episode_id: int, spawn_viewer: bool = True):
        self.prefix = prefix

        recording_id =  uuid.uuid4()
        self.rec = rr.RecordingStream(application_id="robot", recording_id=recording_id)


        out_path = self.save_dir / f"episode_{episode_id:04d}.rrd"
        self.rec.save(str(out_path))

        self.rec.send_recording_name(f"Episode {episode_id:04d}")

        if spawn_viewer:
            self.rec.spawn(hide_welcome_screen=True, memory_limit=self.memory_limit)

        self.rec.set_time("idx", duration=0)

        self.robot = ReRunRobot.g1(self.rec)
        self.left_hand = ReRunRobot.left_dfq_hand(self.rec)
        self.right_hand = ReRunRobot.right_dfq_hand(self.rec)

        self.robot.log_transform_named_frames(
            "/transforms",
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            parent_frame="world",
            child_frame="pelvis",
        )

        if self.IdxRangeBoundary:
            self.setup_blueprint()

    def setup_blueprint(self):
        views = []

        data_plot_paths = [
            f"{self.prefix}left_arm",
            f"{self.prefix}right_arm",
            f"{self.prefix}left_ee",
            f"{self.prefix}right_ee",
        ]
        for plot_path in data_plot_paths:
            view = rrb.TimeSeriesView(
                origin=plot_path,
                plot_legend=rrb.PlotLegend(visible=True),
            )
            views.append(view)

        grid = rrb.Grid(
            contents=views,
            grid_columns=2,
            column_shares=[1, 1],
            row_shares=[1, 1],
        )
        views.append(rr.blueprint.SelectionPanel(state=rrb.PanelState.Collapsed))
        views.append(rr.blueprint.TimePanel(state=rrb.PanelState.Collapsed))
        views.append(get_blueprint("world"))
        self.rec.send_blueprint(grid)

    def log_item_data(self, item_data: dict):
        self.rec.set_time("idx", duration = item_data.get("idx", 0))

        # Log states
        states = item_data.get("states", {}) or {}
        for part, state_info in states.items():
            if part != "body" and state_info:
                values = state_info.get("qpos", [])
                for idx, val in enumerate(values):
                    self.rec.log(f"{self.prefix}{part}/states/qpos/{idx}", rr.Scalars(val))
            if part == "body":
                self.robot.log(state_info.get("qpos")[:29])
                # self.robot.log_transform_named_frames(
                #     "/transforms",
                #     np.array([0,0,0]),
                #     np.array([0,0,0,1]),
                #     parent_frame="world",
                #     child_frame="pelvis"
                # )
            if part == "left_ee":
                self.left_hand.log(state_info.get("qpos"))
            if part == "right_ee":
                self.right_hand.log(state_info.get("qpos"))
                

        # Log actions
        actions = item_data.get("actions", {}) or {}
        for part, action_info in actions.items():
            if part != "body" and action_info:
                values = action_info.get("qpos", [])
                for idx, val in enumerate(values):
                    self.rec.log(f"{self.prefix}{part}/actions/qpos/{idx}", rr.Scalars(val))


    def log_episode_data(self, episode_data: list):
        for item_data in episode_data:
            self.log_item_data(item_data)


if __name__ == "__main__":
    import os
    import zipfile

    import gdown
    import logging_mp

    logger_mp = logging_mp.get_logger(__name__, level=logging_mp.INFO)

    zip_file = "rerun_testdata.zip"
    zip_file_download_url = "https://drive.google.com/file/d/1f5UuFl1z_gaByg_7jDRj1_NxfJZh2evD/view?usp=sharing"
    unzip_file_output_dir = "./testdata"
    if not os.path.exists(os.path.join(unzip_file_output_dir, "episode_0006")):
        if not os.path.exists(zip_file):
            file_id = zip_file_download_url.split("/")[5]
            gdown.download(id=file_id, output=zip_file, quiet=False)
            logger_mp.info("download ok.")
        if not os.path.exists(unzip_file_output_dir):
            os.makedirs(unzip_file_output_dir)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(unzip_file_output_dir)
        logger_mp.info("uncompress ok.")
        os.remove(zip_file)
        logger_mp.info("clean file ok.")
    else:
        logger_mp.info("rerun_testdata exits.")

    episode_reader = RerunEpisodeReader(task_dir=unzip_file_output_dir)
    # TEST EXAMPLE 1 : OFFLINE DATA TEST
    user_input = input(
        "Please enter the start signal (enter 'off' or 'on' to start the subsequent program):\n"
    )
    if user_input.lower() == "off":
        episode_data6 = episode_reader.return_episode_data(6)
        logger_mp.info("Starting offline visualization...")
        offline_logger = RerunLogger(prefix="offline/")
        offline_logger.log_episode_data(episode_data6)
        logger_mp.info("Offline visualization completed.")

    # TEST EXAMPLE 2 : ONLINE DATA TEST, SLIDE WINDOW SIZE IS 60, MEMORY LIMIT IS 50MB
    if user_input.lower() == "on":
        episode_data8 = episode_reader.return_episode_data(8)
        logger_mp.info("Starting online visualization with fixed idx size...")
        online_logger = RerunLogger(
            prefix="online/", IdxRangeBoundary=60, memory_limit="50MB"
        )
        for item_data in episode_data8:
            online_logger.log_item_data(item_data)
            time.sleep(0.033)  # 30hz
        logger_mp.info("Online visualization completed.")

    # # TEST DATA OF data_dir
    # data_dir = "./data"
    # episode_data_number = 10
    # episode_reader2 = RerunEpisodeReader(task_dir = data_dir)
    # user_input = input("Please enter the start signal (enter 'on' to start the subsequent program):\n")
    # episode_data8 = episode_reader2.return_episode_data(episode_data_number)
    # if user_input.lower() == 'on':
    #     # Example 2: Offline Visualization with Fixed Time Window
    #     logger_mp.info("Starting offline visualization with fixed idx size...")
    #     online_logger = RerunLogger(prefix="offline/", IdxRangeBoundary = 60)
    #     for item_data in episode_data8:
    #         online_logger.log_item_data(item_data)
    #         time.sleep(0.033) # 30hz
    #     logger_mp.info("Offline visualization completed.")
