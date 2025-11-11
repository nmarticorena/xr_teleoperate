import numpy as np
import threading
import time
class FixedHeightController:
    """
    Fixed Height Controller - Independent thread controls robot height to preset positions
    """
    
    def __init__(self, mobile_ctrl, target_A=0.1107, target_B=0.3868):
        """
        Initialize fixed height controller
        Args:
            mobile_ctrl: Mobile controller instance
            target_A: Target height for button A in meters
            target_B: Target height for button B in meters
        """
        self.mobile_ctrl = mobile_ctrl
        self.target_A = target_A
        self.target_B = target_B
        
        self._height_target = None
        self._height_control_active = False

        self._height_control_thread = None
        self._height_thread_running = False
        self._height_target_lock = threading.Lock()
    
    def _height_control_loop(self, max_speed=1.0, min_speed=0.2, tolerance=0.001, kp=8.0):
        """
        Height control thread function, continuously executes height control
        
        Args:
            max_speed: Maximum speed in m/s
            min_speed: Minimum speed in m/s, applied when speed is non-zero
            tolerance: Tolerance in meters
            kp: Proportional gain
        """
        
        while self._height_thread_running:
            try:
                with self._height_target_lock:
                    target = self._height_target
                    active = self._height_control_active
                
                if not active or target is None:
                    self.mobile_ctrl.g1_height_action_array_in[0] = 0.0
                    time.sleep(0.02)  # 20ms
                    continue

                current_height = self.mobile_ctrl.g1_height_state_array_out[0]
                
                height_error = target - current_height
                
                if abs(height_error) < tolerance:
                    with self._height_target_lock:
                        self._height_control_active = False
                        self._height_target = None
                        self._height_thread_running = False
                    self.mobile_ctrl.g1_height_action_array_in[0] = 0.0
                    print(f"✓ [FixedHeightController] 已到达目标高度：{current_height:+.4f} m")
                    time.sleep(0.02)
                    continue
                
                speed = kp * height_error
                
                speed = np.clip(speed, -max_speed, max_speed)
                
                if min_speed > 0.0 and abs(speed) > 0.0 and abs(speed) < min_speed:
                    speed = np.sign(speed) * min_speed
                
                self.mobile_ctrl.g1_height_action_array_in[0] = speed
                
                time.sleep(0.02)  # 50Hz
                
            except Exception as e:
                print(f"✗ [FixedHeightController] 高度控制线程错误: {e}")
                time.sleep(0.1)
        
        self.mobile_ctrl.g1_height_action_array_in[0] = 0.0
        print(">>> [FixedHeightController] Height control thread stopped")
        
        with self._height_target_lock:
            self._height_control_thread = None

    def update(self, lbutton_A, lbutton_B, max_speed=1.0, min_speed=0.2, tolerance=0.002, kp=5.0):
        """
        Update button states and set target height (Called by main thread)
        
        Args:
            lbutton_A: Left A button state
            lbutton_B: Left B button state
            max_speed: Maximum speed in m/s
            min_speed: Minimum speed in m/s, applied when speed is non-zero
            tolerance: Tolerance in meters
            kp: Proportional gain
        """
        if self.mobile_ctrl is None:
            return
        lbutton_A_pressed = lbutton_A 
        lbutton_B_pressed = lbutton_B 
        if lbutton_A_pressed:
            current_height = self.mobile_ctrl.g1_height_state_array_out[0]
            direction = "uP" if self.target_A > current_height else "DOWN" if self.target_A < current_height else "KEEP"
            print(f">>> [A key] Set height target: {direction} from {current_height:.4f}m to {self.target_A:.4f}m")
            if self._height_control_thread is not None and not self._height_control_thread.is_alive():
                self._height_control_thread = None
                self._height_thread_running = False
            with self._height_target_lock:
                self._height_target = self.target_A
                self._height_control_active = True
            
            if not self._height_thread_running:
                self._height_thread_running = True
                self._height_control_thread = threading.Thread(
                    target=self._height_control_loop,
                    args=(max_speed, min_speed, tolerance, kp),
                    daemon=True
                )
                self._height_control_thread.start()
                
        elif lbutton_B_pressed:
            current_height = self.mobile_ctrl.g1_height_state_array_out[0]
            direction = "UP" if self.target_B > current_height else "DOWN" if self.target_B < current_height else "KEEP"
            print(f">>> [B key] Set height target: {direction} from {current_height:.4f}m to {self.target_B:.4f}m")
            
            if self._height_control_thread is not None and not self._height_control_thread.is_alive():
                self._height_control_thread = None
                self._height_thread_running = False
            with self._height_target_lock:
                self._height_target = self.target_B
                self._height_control_active = True
            
            if not self._height_thread_running:
                self._height_thread_running = True
                self._height_control_thread = threading.Thread(
                    target=self._height_control_loop,
                    args=(max_speed, min_speed, tolerance, kp),
                    daemon=True
                )
                self._height_control_thread.start()
    
    def stop(self):
        """
        Stop height control thread
        """
        if self._height_thread_running:
            print(">>> [FixedHeightController] Stopping height control thread...")
            self._height_thread_running = False
            if self._height_control_thread is not None:
                self._height_control_thread.join(timeout=1.0)
                self._height_control_thread = None
            print("✓ [FixedHeightController] Height control thread stopped")
    
    def is_active(self):
        """
        Returns:
            bool: True if fixed height control is active or thread is running
        """
        return self._height_control_active or self._height_thread_running
    
    def set_targets(self, target_A, target_B):
        """
        Set target heights
        
        Args:
            target_A: Target height for button A
            target_B: Target height for button B
        """
        self.target_A = target_A
        self.target_B = target_B
        print(f"✓ [FixedHeightController] Updated target heights: A={target_A:.4f}m, B={target_B:.4f}m")
class HandleInstruction:
    def __init__(self,control_device,tv_wrapper,mobile_ctrl):
        self.control_device = control_device
        self.tv_wrapper = tv_wrapper
        self.mobile_ctrl = mobile_ctrl
    def get_instruction(self):
        if self.control_device == "unitree_handle" and self.mobile_ctrl is not None:
            lx = self.mobile_ctrl.unitree_handle_state_array_out[0]
            ly = -self.mobile_ctrl.unitree_handle_state_array_out[1]
            rx = -self.mobile_ctrl.unitree_handle_state_array_out[2]
            ry = -self.mobile_ctrl.unitree_handle_state_array_out[3]
            rbutton_A = True if int(self.mobile_ctrl.unitree_handle_state_array_out[4]) == 256 else False
            rbutton_B = True if int(self.mobile_ctrl.unitree_handle_state_array_out[4]) == 512 else False
            lbutton_A = False
            lbutton_B = False
        elif self.control_device == "other" and self.tv_wrapper is not None:
            lx = -self.tv_wrapper.get_tele_data().left_ctrl_thumbstickValue[1]
            ly = -self.tv_wrapper.get_tele_data().left_ctrl_thumbstickValue[0]
            rx = -self.tv_wrapper.get_tele_data().right_ctrl_thumbstickValue[0]
            ry = -self.tv_wrapper.get_tele_data().right_ctrl_thumbstickValue[1]
            rbutton_A = self.tv_wrapper.get_tele_data().right_ctrl_aButton
            rbutton_B = self.tv_wrapper.get_tele_data().right_ctrl_bButton
            lbutton_A = self.tv_wrapper.get_tele_data().left_ctrl_aButton
            lbutton_B = self.tv_wrapper.get_tele_data().left_ctrl_bButton
        return {'lx': lx, 'ly': ly, 'rx': rx, 'ry': ry, 'rbutton_A': rbutton_A, 'rbutton_B': rbutton_B, 'lbutton_A': lbutton_A, 'lbutton_B': lbutton_B}


class LowPassFilter:
    """Low-pass filter for smoothing data"""
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self._value = 0.0
        self._last_value = 0.0

    def update(self, new_value, max_accel=1.5):
        delta = new_value - self._last_value
        delta = np.clip(delta, -max_accel, max_accel)
        filtered = self.alpha * (self._last_value + delta) + (1 - self.alpha) * self._value
        self._last_value = filtered
        self._value = filtered
        return self._value


class ControlDataMapper:
    """
    Control data mapper for mobile base and elevation
    """
    def __init__(self):
        # Velocity filters
        self._filters = {
            'mobile_x_vel': LowPassFilter(alpha=0.15),
            'mobile_yaw_vel': LowPassFilter(alpha=0.15)
        }
        
        # Height accumulated value (remains unchanged after release)
        self._height_value = 0.0
        self.mobile_x_vel = 0
        self.mobile_yaw_vel = 0
        self.height = 0
    def update(self, lx=None, ly=None, rx=None, ry=None, rbutton_A=None, rbutton_B=None, 
               current_waist_yaw=None, current_waist_pitch=None):
        """
        Update and map control parameters
        
        Args:
            lx: Left joystick X raw value (-1 to 1)
            ly: Left joystick Y raw value (-1 to 1)
            rx: Right joystick X raw value (-1 to 1)
            ry: Right joystick Y raw value (-1 to 1)
            rbutton_A: Right button A raw value (0 or 1)
            rbutton_B: Right button B raw value (0 or 1)
            current_waist_yaw: Current waist yaw position (rad)
            current_waist_pitch: Current waist pitch position (rad)
        Returns:
            dict: Dictionary containing mobile velocities and waist positions
        """
        if lx is not None:
            # Map forward velocity 
            raw = self._map_forward_velocity(lx)
            mobile_x_vel = self._filters['mobile_x_vel'].update(raw, max_accel=1.0)
            self.mobile_x_vel = mobile_x_vel
        else:
            mobile_x_vel = self.mobile_x_vel
        if ly is not None:
            raw = self._map_lateral_velocity(ly)
            mobile_yaw_vel = self._filters['mobile_yaw_vel'].update(raw, max_accel=1.0)
            self.mobile_yaw_vel = mobile_yaw_vel
        else:
            mobile_yaw_vel = self.mobile_yaw_vel
        # Update waist yaw position based on joystick input and current position
        if rx is not None and current_waist_yaw is not None:
            waist_yaw_pos = self._update_waist_position(rx, current_waist_yaw,max_velocity=0.05,min_position=-2.5,max_position=2.5)
        elif current_waist_yaw is not None:
            # Joystick released, maintain current position
            waist_yaw_pos = current_waist_yaw
        else:
            waist_yaw_pos = 0.0
        
        # Update waist pitch position based on joystick input and current position
        if ry is not None and current_waist_pitch is not None:
            waist_pitch_pos = self._update_waist_position(ry, current_waist_pitch,max_velocity=0.01,min_position=-0.17,max_position=0.5)
        elif current_waist_pitch is not None:
            # Joystick released, maintain current position
            waist_pitch_pos = current_waist_pitch
        else:
            waist_pitch_pos = 0.0
        if rbutton_A is not None and rbutton_B is not None:
            self._update_height_button(rbutton_A,rbutton_B) 
        else:
            self._height_value = self.height
        # Update height (remains at current value after release)
        return {
            'mobile_x_vel': mobile_x_vel,
            'mobile_yaw_vel': mobile_yaw_vel,
            'waist_yaw_pos': waist_yaw_pos,
            'waist_pitch_pos': waist_pitch_pos,
            'g1_height': self._height_value
        }
        
    def _update_waist_position(self, raw_value, current_position, max_velocity,min_position,max_position):
        """
        Update waist position based on joystick input and current position
        Position increases/decreases based on joystick direction
        
        Args:
            raw_value: Raw joystick value (-1 to 1)
            current_position: Current waist position (rad)
            
        Returns:
            float: Updated waist position
        """
        deadzone = 0.05
        max_velocity = max_velocity  # Maximum position change per update (adjust for smooth control)
        min_position = min_position
        max_position = max_position
        
        if abs(raw_value) < deadzone:
            # Joystick in deadzone, maintain current position
            return current_position
        else:
            # Calculate velocity based on joystick input
            # Positive joystick -> increase position
            # Negative joystick -> decrease position
            sign = 1 if raw_value > 0 else -1
            intensity = (abs(raw_value) - deadzone) / (1.0 - deadzone)
            smooth = 6*intensity**5 - 15*intensity**4 + 10*intensity**3
            velocity = sign * smooth * max_velocity
            
            # Calculate new position based on current position
            new_position = current_position + velocity
            
            # Clamp to valid range
            new_position = np.clip(new_position, min_position, max_position)
            
            return new_position
    
    def _update_height_button(self,rbutton_A,rbutton_B):
        """
        Update height value
        Height remains unchanged when joystick is released (won't drop down)
        
        Args:
            rbutton_A: Right button A raw value (0 or 1)
            rbutton_B: Right button B raw value (0 or 1)
        """
        if rbutton_B:
            self._height_value = 0.5
        elif rbutton_A:
            self._height_value = -0.5
        else:
            self._height_value = 0.0
    def _update_height(self, raw_value):
        """
        Update height value
        Height remains unchanged when joystick is released (won't drop down)
        
        Args:
            raw_value: Raw value (-1 to 1)
        """
        deadzone = 0.05
        max_range = 1.0 
        
        if abs(raw_value) < deadzone:
            self._height_value = 0.0
        else:
            sign = 1 if raw_value > 0 else -1
            intensity = (abs(raw_value) - deadzone) / (1.0 - deadzone)
            smooth = 6*intensity**5 - 15*intensity**4 + 10*intensity**3
            height_value = sign * smooth * max_range
            
            self._height_value = height_value
    
    def _map_forward_velocity(self, value):
        return self._smooth_map(value, -0.2, 0.2)
    
    def _map_lateral_velocity(self, value):
        return self._smooth_map(value, -0.6, 0.6)
    
    def _map_yaw_velocity(self, value,min_value,max_value):
        return self._smooth_map(value,min_value,max_value)
    
    def _smooth_map(self, value, out_min, out_max, deadzone=0.05):
        """
        Smooth mapping function
        Maps input value to output range using deadzone and smooth curve
        
        Args:
            value: Input value (-1 to 1)
            out_min: Output minimum value
            out_max: Output maximum value
            deadzone: Deadzone size
        """
        if abs(value) < deadzone:
            return 0.0
        t = (abs(value) - deadzone) / (1.0 - deadzone)
        t = np.clip(t, 0.0, 1.0)
        smooth = 6 * t**5 - 15 * t**4 + 10 * t**3
        return smooth * (out_max if value > 0 else out_min)
    
    def reset_height(self, height=0.0):
        """Reset height value"""
        self._height_value = height
    
    def get_current_height(self):
        """Get current height value"""
        return self._height_value
    