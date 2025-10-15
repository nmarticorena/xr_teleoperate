import numpy as np


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
            waist_yaw_pos = self._update_waist_position(rx, current_waist_yaw)
        elif current_waist_yaw is not None:
            # Joystick released, maintain current position
            waist_yaw_pos = current_waist_yaw
        else:
            waist_yaw_pos = 0.0
        
        # Update waist pitch position based on joystick input and current position
        if ry is not None and current_waist_pitch is not None:
            waist_pitch_pos = self._update_waist_position(ry, current_waist_pitch)
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
        
    def _update_waist_position(self, raw_value, current_position):
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
        max_velocity = 0.02  # Maximum position change per update (adjust for smooth control)
        min_position = -2.5
        max_position = 2.5
        
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
        if rbutton_A:
            self._height_value = 0.2
        elif rbutton_B:
            self._height_value = -0.2
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
    