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
        
        # --- CONFIGURATION ---
        # Table height is ~1.02m. 
        # We set Z=1.05m to hover just 3cm above it.
        self.home_pos = np.array([0.5, 0.0, 1.05]) 
        
        # [180, 0, 0] is the standard "Point Down" orientation
        self.home_rot = [180, 0, 0] 
        
        self.target_pos = self.home_pos.copy()

        print("\n" + "="*60)
        print("🕹️  DISCRETE CONTROLLER (Close & Perpendicular)")
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

    def move_to_home_immediately(self):
        """
        Teleports the robot to the home position gently before the loop starts.
        """
        print(f"⚙️  Initializing to Home: {self.home_pos}")
        
        # 1. Solve IK for the home position
        # We use a high max_iters to guarantee it finds the solution from the default pose
        self.env._set_ee_in_cartesian(
            self.home_pos, 
            rotation=self.home_rot, 
            max_iters=500, # Give it plenty of time to calculate
            threshold=0.005
        )
        
        # 2. Step physics to let the robot settle there
        mujoco.mj_step(self.env.model, self.env.data, nstep=200)
        
        # 3. Render once to update the visual before the loop
        if self.viewer:
            self.viewer.render()
        
        print("✅ Ready.")

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            step = np.zeros(3)
            move_name = ""
            
            # --- SWAPPED CONTROLS (X=Forward, Y=Left/Right) ---
            if key == glfw.KEY_W:
                step[0] = self.delta
                move_name = "Forward (+X)"
            elif key == glfw.KEY_S:
                step[0] = -self.delta
                move_name = "Backward (-X)"
            elif key == glfw.KEY_A:
                step[1] = self.delta
                move_name = "Left (+Y)"
            elif key == glfw.KEY_D:
                step[1] = -self.delta
                move_name = "Right (-Y)"
            # --------------------------------------------------

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
        # X: Reach over table
        # Y: Width of table
        # Z: [1.02, 1.3] -> Don't go below table surface (1.02)
        self.target_pos[0] = np.clip(self.target_pos[0], 0.3, 0.75)
        self.target_pos[1] = np.clip(self.target_pos[1], -0.4, 0.4)
        self.target_pos[2] = np.clip(self.target_pos[2], 1.02, 1.3)

        print(f"👣 Action: {move_name} -> Target: {np.round(self.target_pos, 3)}")

        # 3. Execute Move
        self.env._set_ee_in_cartesian(
            self.target_pos, 
            rotation=self.home_rot, 
            max_iters=50,
            threshold=0.01
        )

    def print_state(self):
        ee_pos, _ = self.env._get_ee_pose()
        print(f"📍 Actual EE: {np.round(ee_pos, 3)}")

    def run(self):
        # Move to start BEFORE entering the render loop
        self.move_to_home_immediately()
        
        while self.viewer.is_alive:
            mujoco.mj_step(self.env.model, self.env.data)
            self.viewer.render()

if __name__ == "__main__":
    env = Hw2Env(n_actions=8, render_mode="gui")
    env.reset()
    
    controller = DiscreteController(env)
    controller.run()
