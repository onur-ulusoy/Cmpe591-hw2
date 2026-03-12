import time
import torch
import torchvision.transforms as transforms
import numpy as np
import mujoco
import environment 

class Hw2Env(environment.BaseEnv):
    def __init__(self, n_actions=6, **kwargs) -> None:
        # --- 1. Define Configuration BEFORE calling super().__init__ ---
        # This is critical because super().__init__ calls reset(), 
        # and reset() uses these variables.
        self._n_actions = n_actions
        self._delta = 0.05
        
        self.home_pos = np.array([0.5, 0.0, 1.05])
        self.home_rot = [180, 0, 0] # Pointing straight down

        # Define 6 Discrete Actions (Cartesian)
        # 0: Forward (+X)
        # 1: Backward (-X)
        # 2: Left (-Y)
        # 3: Right (+Y)
        # 4: Up (+Z)
        # 5: Down (-Z)
        self._actions = {
            0: np.array([self._delta, 0, 0]),    # S (Forward)
            1: np.array([-self._delta, 0, 0]),   # W (Backward)
            2: np.array([0, -self._delta, 0]),   # A (Left)
            3: np.array([0, self._delta, 0]),    # D (Right)
            4: np.array([0, 0, self._delta]),    # Q (Up)
            5: np.array([0, 0, -self._delta]),   # E (Down)
        }

        self._goal_thresh = 0.02
        self._max_timesteps = 50

        # --- 2. Initialize Parent (which calls self.reset()) ---
        super().__init__(**kwargs)

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        
        # Randomize object position slightly
        obj_pos = [np.random.uniform(0.35, 0.65),
                   np.random.uniform(-0.2, 0.2),
                   1.03] # On table surface
        
        # Randomize goal position
        goal_pos = [np.random.uniform(0.35, 0.65),
                    np.random.uniform(-0.2, 0.2),
                    1.03]
                    
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene

    def reset(self):
        # 1. Standard MuJoCo Reset (loads model/data)
        super().reset()
        
        # 2. Force Robot to "Home/Hover" Position immediately
        self._set_ee_in_cartesian(
            self.home_pos, 
            rotation=self.home_rot, 
            max_iters=100, 
            threshold=0.01
        )
        mujoco.mj_step(self.model, self.data, nstep=50)
        
        self._t = 0
        
        # Return initial state
        if self._render_mode == "offscreen":
            return self.state()
        return self.high_level_state()

    def state(self):
        # Pixel state (for CNNs)
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        # Vector state: [EE_x, EE_y, EE_z, Obj_x, Obj_y, Obj_z, Goal_x, Goal_y, Goal_z]
        ee_pos, _ = self._get_ee_pose()
        obj_pos = self.data.body("obj1").xpos
        goal_pos = self.data.site("goal").xpos
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def reward(self):
        # Calculate distances
        ee_pos, _ = self._get_ee_pose()
        obj_pos = self.data.body("obj1").xpos
        goal_pos = self.data.site("goal").xpos
        
        # Distance from Gripper to Object
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        
        # Distance from Object to Goal
        dist_obj_goal = np.linalg.norm(obj_pos - goal_pos)
        
        # Sparse-ish reward structure
        reward = 0
        
        # 1. Approach Reward (Encourage getting close to object)
        reward -= dist_ee_obj * 2.0 
        
        # 2. Goal Reward (Encourage moving object to goal)
        if dist_ee_obj < 0.05: # If gripper is close to object
            reward += 1.0      # Bonus for grasping/being near
            reward -= dist_obj_goal * 5.0 # Strong incentive to move object
            
        # 3. Completion Bonus
        if dist_obj_goal < self._goal_thresh:
            reward += 100.0
            
        return reward

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos
        goal_pos = self.data.site("goal").xpos
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def step(self, action_id):
        # 1. Calculate Target
        current_pos, _ = self._get_ee_pose()
        step_vector = self._actions[action_id]
        target_pos = current_pos + step_vector

        # 2. Clamp Target (Workspace Limits from Debug Script)
        target_pos[0] = np.clip(target_pos[0], 0.3, 0.75)  # X
        target_pos[1] = np.clip(target_pos[1], -0.4, 0.4)  # Y
        target_pos[2] = np.clip(target_pos[2], 1.02, 1.3)  # Z (Don't hit table)

        # 3. Execute IK Move
        self._set_ee_in_cartesian(
            target_pos, 
            rotation=self.home_rot, 
            max_iters=20, # Faster for small steps
            threshold=0.02
        )
        
        self._t += 1

        # 4. Return tuple
        if self._render_mode == "offscreen":
            state = self.state()
        else:
            state = self.high_level_state()
        
        reward = self.reward()
        terminal = self.is_terminal()
        truncated = self.is_truncated()
        
        return state, reward, terminal, truncated

if __name__ == "__main__":
    # Quick Test
    env = Hw2Env(n_actions=6, render_mode="gui")
    env.reset()
    print("Environment created. Running random steps...")
    
    for i in range(50):
        action = np.random.randint(6)
        s, r, term, trunc = env.step(action)
        env.viewer.render()
        if term or trunc:
            env.reset()
