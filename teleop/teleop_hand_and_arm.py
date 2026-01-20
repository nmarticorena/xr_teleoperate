import time
import argparse
import numpy as np
from multiprocessing import Value, Array, Lock
import threading
import logging_mp
logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleimager.image_client import ImageClient
import unitree_schema as us
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }

class FixedRateTimer:
    def __init__(self, fps: float, busy_wait_ms: float = 10.0):
        self.period = 1.0 / fps
        self._busy_wait = busy_wait_ms / 1000.0
        self._next_tick = None

    def wait(self) -> bool:
        """
        Wait until the next tick to maintain a fixed rate.
        Returns True if waited successfully, False if the wait was skipped due to loop overtime.
        """
        now = time.perf_counter()
        if self._next_tick is None:
            self._next_tick = now + self.period
            return True

        remaining = self._next_tick - now
        if remaining <= 0:
            self._next_tick = now + self.period
            return False

        if remaining > self._busy_wait:
            time.sleep(remaining - self._busy_wait)
        while time.perf_counter() < self._next_tick:
            pass

        self._next_tick += self.period
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--body', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select Body')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--save-path', type = str, default = './data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'Primary objective of the task (one sentence)')
    parser.add_argument('--task-desc', type = str, default = 'pick up red cube placed on...', help = 'Detailed task description, instructions, and notes')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")
    ratetimer = FixedRateTimer(fps=args.frequency)

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        if args.body == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.body == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.body == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.body == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)

        # end-effector
        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_ee_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_ee_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_ee_state_array, dual_ee_action_array, simulation_mode=args.sim)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_ee_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_ee_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_ee_state_array, dual_ee_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_ee_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_ee_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_ee_state_array, dual_ee_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_ee_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_ee_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_ee_state_array, dual_ee_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_ee_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_ee_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_ee_state_array, dual_ee_action_array, simulation_mode=args.sim)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            head_img_size = None
            left_wrist_img_size = None
            right_wrist_img_size = None
            if camera_config['head_camera']['enable_zmq']:
                head_img_size = camera_config['head_camera']['image_shape']
            if camera_config['left_wrist_camera']['enable_zmq']:
                left_wrist_img_size = camera_config['left_wrist_camera']['image_shape']
            if camera_config['right_wrist_camera']['enable_zmq']:
                right_wrist_img_size = camera_config['right_wrist_camera']['image_shape']

            recorder = us.UnitreeRecorder(save_path=args.save_path,
                                          task_name=args.task_name,
                                          task_goal=args.task_goal,
                                          task_desc=args.task_desc,
                                          data_freq=args.frequency,
                                          body=args.body,
                                          end_effector=args.ee,
                                          data=us.Data(
                                            observation=us.Observation(
                                                color=us.Colors(
                                                    head_camera=us.Color(shape=(head_img_size)),
                                                    left_wrist_camera=us.Color(shape=left_wrist_img_size),
                                                    right_wrist_camera=us.Color(shape=right_wrist_img_size),
                                                ),
                                            ),
                                          ),
                                          # visualizer
                                          log=True,
                                          vis=args.headless)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()
        # main loop. robot start to follow VR user's motion
        while not STOP:
            # get image
            if camera_config['head_camera']['enable_zmq'] and args.record:
                head_frame = img_client.get_head_frame()
            if camera_config['left_wrist_camera']['enable_zmq'] and args.record:
                left_wrist_frame = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq'] and args.record:
                right_wrist_frame = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            
            # high level control
            loco_wrapper_vx, loco_wrapper_vy, loco_wrapper_vyaw = 0.0, 0.0, 0.0
            if args.input_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper_vx   = -tele_data.left_ctrl_thumbstickValue[1] * 0.3
                loco_wrapper_vy   = -tele_data.left_ctrl_thumbstickValue[0] * 0.3
                loco_wrapper_vyaw = -tele_data.right_ctrl_thumbstickValue[0] * 0.3
                loco_wrapper.Move(loco_wrapper_vx, loco_wrapper_vy, loco_wrapper_vyaw)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            if args.record:
                current_waist_q   = arm_ctrl.get_current_waist_q()
                current_left_leg_q  = arm_ctrl.get_current_left_leg_q()
                current_right_leg_q = arm_ctrl.get_current_right_leg_q()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.monotonic_ns()
            target_lr_arm_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q)
            time_ik_end = time.monotonic_ns()
            logger_mp.debug(f"ik:\t{(time_ik_end - time_ik_start)/1e6:.2f} ms")
            arm_ctrl.ctrl_dual_arm(target_lr_arm_q, sol_tauff)

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        current_left_ee_q = np.atleast_1d(dual_ee_state_array[:7])
                        current_right_ee_q = np.atleast_1d(dual_ee_state_array[-7:])
                        target_left_ee_q = np.atleast_1d(dual_ee_action_array[:7])
                        target_right_ee_q =  np.atleast_1d(dual_ee_action_array[-7:])
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        current_left_ee_q = np.atleast_1d(dual_ee_state_array[0])
                        current_right_ee_q = np.atleast_1d(dual_ee_state_array[1])
                        target_left_ee_q = np.atleast_1d(dual_ee_action_array[0])
                        target_right_ee_q = np.atleast_1d(dual_ee_action_array[1])
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        current_left_ee_q = np.atleast_1d(dual_ee_state_array[0])
                        current_right_ee_q = np.atleast_1d(dual_ee_state_array[1])
                        target_left_ee_q = np.atleast_1d(dual_ee_action_array[0])
                        target_right_ee_q = np.atleast_1d(dual_ee_action_array[1])
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        current_left_ee_q = np.atleast_1d(dual_ee_state_array[:6])
                        current_right_ee_q = np.atleast_1d(dual_ee_state_array[-6:])
                        target_left_ee_q = np.atleast_1d(dual_ee_action_array[:6])
                        target_right_ee_q = np.atleast_1d(dual_ee_action_array[-6:])
                else:
                    current_left_ee_q = np.array([])
                    current_right_ee_q = np.array([])
                    target_left_ee_q = np.array([])
                    target_right_ee_q = np.array([])

                # arm state and action
                if RECORD_RUNNING:
                    colors = {}
                    if camera_config['head_camera']['binocular']:
                        if head_frame is not None:
                            colors[f"head_camera"] = head_frame.jpg
                        else:
                            logger_mp.warning("Head image is None!")
                    if camera_config['left_wrist_camera']['enable_zmq']:
                        if left_wrist_frame is not None:
                            colors[f"left_wrist_camera"] = left_wrist_frame.jpg
                        else:
                            logger_mp.warning("Left wrist image is None!")
                    if camera_config['right_wrist_camera']['enable_zmq']:
                        if right_wrist_frame is not None:
                            colors[f"right_wrist_camera"] = right_wrist_frame.jpg
                        else:
                            logger_mp.warning("Right wrist image is None!")

                    if args.sim:
                        recorder.add_frame(
                                    #
                                    simulation_state=sim_state_subscriber.read_data(),
                        )
                    else:
                        recorder.add_frame(
                                data=us.Data(
                                    observation=us.Observation(
                                        body=us.Body(
                                            left_arm=us.Joint(qpos=current_lr_arm_q[:7]),
                                            right_arm=us.Joint(qpos=current_lr_arm_q[-7:]),
                                            waist=us.Joint(qpos=current_waist_q),
                                            left_leg=us.Joint(qpos=current_left_leg_q),
                                            right_leg=us.Joint(qpos=current_right_leg_q),
                                        ),
                                        color=us.Colors(
                                            head_camera=us.Color(raw=colors[f"head_camera"], format="jpg"),
                                            left_wrist_camera=us.Color(raw=colors[f"left_wrist_camera"], format="jpg"),
                                            right_wrist_camera=us.Color(raw=colors[f"right_wrist_camera"], format="jpg"),
                                        ),
                                        end_effector=us.EndEffector(
                                            left_ee=us.Joint(qpos=current_left_ee_q),
                                            right_ee=us.Joint(qpos=current_right_ee_q),
                                        ),
                                    ),
                                    action=us.Action(
                                        body=us.Body(
                                            left_arm=us.Joint(qpos=target_lr_arm_q[:7]),
                                            right_arm=us.Joint(qpos=target_lr_arm_q[-7:]),
                                        ),
                                        end_effector=us.EndEffector(
                                            left_ee=us.Joint(qpos=target_left_ee_q),
                                            right_ee=us.Joint(qpos=target_right_ee_q),
                                        ),
                                    ),
                                )
                        )
            if not ratetimer.wait():
                logger_mp.warning("Main loop overtime detected!")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record and recorder is not None:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)