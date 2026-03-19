import wandb
import numpy as np
import torch
import collections
import tqdm
from diffusion_policy.env.maniskill_wrapper import ManiSkillEnv
from diffusion_policy.policy.base_policy import BasePolicy
from diffusion_policy.common.utils import dict_apply
from diffusion_policy.env_runner.base_runner import BaseRunner
import diffusion_policy.common.logger_utils as logger_util
from termcolor import cprint
import sapien
import os
import time

from typing import Optional

import gymnasium as gym
from mani_skill.utils import gym_utils
from mani_skill.utils import common
from mani_skill.utils.wrappers import CPUGymWrapper, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.envs.scene import ManiSkillScene

from diffusion_policy.common.observation_wrapper import FlattenPoindCloudObservationWrapper
from diffusion_policy.common.multistep_wrapper import MultiStepWrapper

def make_eval_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    n_obs_steps: int,
    n_action_steps: int,
    max_episode_steps: int,
    reward_agg_method: str,
    device: str,
    use_point_crop: bool,
    use_pc_color: bool,
    num_points: int,
    env_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list[gym.Wrapper] = [],
    projection_mode: str = "perspective",
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "cpu":
        def cpu_make_env(
            env_id, seed, video_dir=None, env_kwargs=dict()
        ):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                # This wrapper wraps any maniskill env created via gym.
                # make to ensure the outputs of env. render, env. reset, env. step are all numpy arrays and are not batched.
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                env = ManiSkillEnv(env, task_name=env_id, use_point_crop=use_point_crop, use_pc_fps=False, num_points=num_points, 
                                   resolution=(224, 224), projection_mode=projection_mode)

                if video_dir:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        source_type="3d_diffusion_policy",
                        source_desc="3d_diffusion_policy evaluation rollout",
                    )

                env = MultiStepWrapper(env=env,
                        n_obs_steps=n_obs_steps,
                        n_action_steps=n_action_steps,
                        max_episode_steps=max_episode_steps,
                        reward_agg_method=reward_agg_method)

                cprint("[ManiSkillEnv] observation mode: {}.".format(env_kwargs["obs_mode"]), "red")
                cprint("[ManiSkillEnv] action space: {}.".format(env.action_space.shape), "red")
                cprint("[ManiSkillEnv] observation space: agent_pos{}, img{}.".format(env.observation_space["agent_pos"].shape, env.observation_space["rgb"].shape), "red")

                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk()

        assert num_envs == 1
        seed = num_envs - 1
        env = cpu_make_env(
                    env_id,
                    seed,
                    video_dir if seed == 0 else None,
                    env_kwargs,
                )
    else:
        # TODO: The following code should be modified
        env = gym.make(
            env_id,
            num_envs=num_envs,
            sim_backend=sim_backend,
            reconfiguration_freq=1,
            **env_kwargs
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        # TODO: check FrameStack wrapper, if we need to change
        # Answer: Yes. This is used to stack # of 'obs_horizon' history observations
        if video_dir:
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_trajectory=False,
                save_video=True,
                source_type="3d_diffusion_policy",
                source_desc="3d_diffusion_policy evaluation rollout",
                max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillEnv(
            MultiStepWrapper(env=env,
                             n_obs_steps=n_obs_steps,
                             n_action_steps=n_action_steps,
                             max_episode_steps=max_episode_steps,
                             reward_agg_method=reward_agg_method),
            task_name=env_id,
            use_point_crop=use_point_crop,
            num_points=num_points,
            projection_mode=projection_mode,
        )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env

class ManiSkillRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 control_mode,
                 eval_episodes=20,
                 max_steps=300,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 num_eval_envs=1,
                 sim_backend='cpu',
                 exp_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 use_pc_color=True,
                 num_points=512,
                 capture_video=True,
                 projection_mode="perspective",
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.traj_visual = None  

        reward_agg_method='sum'

        env_kwargs = dict(
            control_mode=control_mode,
            reward_mode="sparse",
            obs_mode="rgbd",
            render_mode="all",
        )

        seed = 1
        if exp_name is None:
            exp_name = os.path.basename(__file__)[: -len(".py")]
            run_name = f"{task_name}__{exp_name}__{seed}__{int(time.time())}"
        else:
            run_name = exp_name

        self.env = make_eval_envs(
            task_name,
            num_eval_envs,
            sim_backend,
            n_obs_steps,
            n_action_steps,
            max_steps,
            reward_agg_method,
            device,
            use_point_crop,
            use_pc_color,
            num_points,
            env_kwargs,
            video_dir=f"./runs/{run_name}/videos" if capture_video else None,
            wrappers=[FlattenPoindCloudObservationWrapper],
            projection_mode=projection_mode,
        )

        self.eval_episodes = eval_episodes

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        for episode_idx in tqdm.tqdm(range(self.eval_episodes),
                                     desc=f"Eval in ManiSkill {self.task_name} Pointcloud Env", leave=False,
                                     mininterval=self.tqdm_interval_sec):

            # start rollout
            obs, _ = env.reset()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            count = 0
            success = False
            while not success and not done:
                count += 1
                np_obs_dict = dict(obs)
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device)) # TypeError: expected np.ndarray (got list)

                with torch.no_grad():
                    obs_dict_input = {}
                    obs_dict_input['rgb'] = obs_dict['rgb'].unsqueeze(0)
                    obs_dict_input['xy_plane'] = obs_dict['xy_plane'].unsqueeze(0)
                    obs_dict_input['xz_plane'] = obs_dict['xz_plane'].unsqueeze(0)
                    obs_dict_input['yz_plane'] = obs_dict['yz_plane'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    obs_dict_input['left_img'] = obs_dict['left_img'].unsqueeze(0)
                    obs_dict_input['right_img'] = obs_dict['right_img'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action'].squeeze(0)

                obs, reward, done, _, info = env.step(action)
                success = max(info['success'])

                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])
            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)

        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        _ = env.reset(seed=0)

        return log_data

def build_trajectory_visual(scene: ManiSkillScene, traj_points):
    """ 在 scene 里添加小球表示轨迹点，并返回可更新的 Actor 列表 """
    visual_spheres = []
    initial_pos = np.array([0.01225358, 0.03801143, 0.1821523]) 
    for i, point in enumerate(traj_points):  
        builder = scene.create_actor_builder()  # 每个小球需要一个新 builder
        builder.add_sphere_visual(
            pose=sapien.Pose(p=point),
            radius=0.01,  
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.8])  
        )
        sphere_actor = builder.build_kinematic(name=f"trajectory_sphere_{i}")  # 添加唯一索引
        sphere_actor.set_pose(sapien.Pose(p=point)) 
        visual_spheres.append(sphere_actor)  # 存储 Actor

    return visual_spheres  # 返回所有小球的 Actor 列表
 
def update_trajectory_visual(traj_visual, traj_points):
    """ 更新已有的轨迹可视化对象（更新小球位置） """
    for sphere, new_pose in zip(traj_visual, traj_points):
        sphere.set_pose(sapien.Pose(p=new_pose))  # 直接更新 Actor 的位置

