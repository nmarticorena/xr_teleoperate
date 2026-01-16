```bash
.
├── save_path/                    # dataset save root directory
│   ├── task_name/                # task_path = os.path.join(save_path, task_name)
│   │   ├── meta.json             # meta_json_path = os.path.join(task_path, 'meta.json')
│   │   └── data                  # task_data_path = os.path.join(task_path, 'data')
│   │      ├── episode_000000/    # episode_path = os.path.join(task_data_path, f"episode_{self.episode_id:06d}")
│   │      │   ├── data.jsonl     # data_jsonl_path = os.path.join(episode_path, 'data.jsonl')
│   │      │   ├── color          # color raw images from different views
│   │      │   │   ├── head_camera_000000.jpg        # Head camera file
│   │      │   │   ├── head_camera_000001.jpg     
│   │      │   │   ├── ...   
│   │      │   │   ├── left_wrist_camera_000000.jpg  # Left wrist camera file
│   │      │   │   ├── left_wrist_camera_000001.jpg
│   │      │   │   ├── ...
│   │      │   │   ├── right_wrist_camera_000000.jpg # Right wrist camera file
│   │      │   │   ├── right_wrist_camera_000001.jpg
│   │      │   │   └── ...
│   │      │   ├── depth          # depth raw images from different views
│   │      │   │   └── ...
│   │      │   ├── audio          # audio raw data
│   │      │   │   └── ...
│   │      │   ├── pointcloud     # point cloud raw data
│   │      │   │   └── ...
│   │      │   └── ...            # other sensor raw data directories   
│   │      ├── episode_000001/
│   │      └── ...
│   ├── task_name_2/
│   └── ...
├── convert_to_lerobot.py         # Conversion script
└── README.md
```