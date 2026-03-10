import time
import numpy as np
import glfw
import mujoco
from homework2 import Hw2Env

class DiscreteController:
    def __init__(self, env):
        self.env = env
        self.viewer = env.viewer
        
        self.delta = 0.05 
        
        # Initialize target as current position
        pos, _ = self.env._get_ee_pose()
        self.target_pos = pos.copy()
        
        # Fixed orientation (Gripper pointing down)
        self.fixed_rotation = [-90, 0, 180] 

        print("\n" + "="*60)
        print("🕹️  DISCRETE STEP CONTROLLER (X/Y Swapped)")
        print("="*60)
        print(f"  Step Size: {self.delta}m")
        print("  [W] : Forward (+X)")
        print("  [S] : Backward (-X)")
        print("  [A] : Left    (+Y)")
        print("  [D] : Right   (-Y)")
        print("  [Q] : Up      (+Z)")
        print("  [E] : Down    (-Z)")
        print("  [SPACE] : Print Position")
        print("  [ESC]   : Quit")
        print("="*60 + "\n")

        if hasattr(self.viewer, 'window'):
            glfw.set_key_callback(self.viewer.window, self.key_callback)

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            step = np.zeros(3)
            move_name = ""
            
            # --- SWAPPED CONTROLS ---
            if key == glfw.KEY_W:
                step[0] = self.delta   # Now controls X
                move_name = "Forward (+X)"
            elif key == glfw.KEY_S:
                step[0] = -self.delta  # Now controls X
                move_name = "Backward (-X)"
            elif key == glfw.KEY_A:
                step[1] = self.delta   # Now controls Y (Left is usually +Y)
                move_name = "Left (+Y)"
            elif key == glfw.KEY_D:
                step[1] = -self.delta  # Now controls Y (Right is usually -Y)
                move_name = "Right (-Y)"
            # ------------------------

            elif key == glfw.KEY_Q:
                step[2] = self.delta
                move_name = "Up (+Z)"
            elif key == glfw.KEY_E:
                step[2] = -self.delta
                move_name = "Down (-Z)"
            elif key == glfw.KEY_SPACE:
                self.print_state()
                return
            elif key == glfw.KEY_ESCAPE:
                self.viewer.close()
                return

            if move_name:
                self.execute_step(step, move_name)

    def execute_step(self, step_vector, move_name):
        # 1. Update Target
        self.target_pos += step_vector

        # 2. Clamp to workspace
        # X: [0.25, 0.75], Y: [-0.3, 0.3], Z: [0.8, 1.2]
        self.target_pos[0] = np.clip(self.target_pos[0], 0.25, 0.75)
        self.target_pos[1] = np.clip(self.target_pos[1], -0.3, 0.3)
        self.target_pos[2] = np.clip(self.target_pos[2], 0.8, 1.2)

        print(f"👣 Action: {move_name} -> Target: {np.round(self.target_pos, 3)}")

        # 3. Execute Move with FIXED ROTATION
        # Using the robust IK solver from BaseEnv
        self.env._set_ee_in_cartesian(
            self.target_pos, 
            rotation=self.fixed_rotation, 
            max_iters=50,
            threshold=0.01
        )

    def print_state(self):
        ee_pos, _ = self.env._get_ee_pose()
        print(f"📍 Actual EE: {np.round(ee_pos, 3)}")

    def run(self):
        # Initial move to set correct orientation
        print("⚙️  Resetting to home position...")
        self.env._set_ee_in_cartesian(
            self.target_pos, 
            rotation=self.fixed_rotation, 
            max_iters=100
        )
        
        while self.viewer.is_alive:
            mujoco.mj_step(self.env.model, self.env.data)
            self.viewer.render()

if __name__ == "__main__":
    env = Hw2Env(n_actions=8, render_mode="gui")
    env.reset()
    
    controller = DiscreteController(env)
    controller.run()
