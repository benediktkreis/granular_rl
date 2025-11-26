import os
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
import numpy as np
from collections import deque, defaultdict
import gymnasium as gym
import warnings
from typing import Any, Dict, Optional, Union
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
import glob
from pathlib import Path
import sand_gym.utils.hugging_face as hugging_face
import sand_gym.utils.common as common

class SavingCallback(BaseCallback):
    def __init__(
            self, 
            log_dir, 
            save_freq, 
            online_model_bool=False, 
            online_model_params={}, 
            curriculum_learning_bool=False, 
            number_of_inter_models_to_keep=-1, 
            save_inter_replay_buffer=True, 
            wait_to_remove_newer_models=200, 
            verbose=0
            ):
        super(SavingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.wait_to_remove_newer_models = wait_to_remove_newer_models
        self.number_of_inter_models_to_keep = number_of_inter_models_to_keep
        self.save_inter_replay_buffer = save_inter_replay_buffer
        self.online_model_bool = online_model_bool
        self.online_model_params = online_model_params
        self.curriculum_learning_bool = curriculum_learning_bool

    def _on_step(self) -> bool:
        if self.n_calls == self.wait_to_remove_newer_models:
            # Deleting newer models
            # It is done in the callback and not at the programm start
            # to give the user time to stop the training
            # if the selected model to continue training was wrong.
            newer_models_list = []
            newer_models_list.extend(glob.glob(self.log_dir+'inter_model_*'))
            newer_models_list.extend(glob.glob(self.log_dir+'best_model_*'))
            newer_models_list.extend(glob.glob(self.log_dir+'final_model_*'))
            for name in newer_models_list:
                file_name = Path(name).stem
                number = common.get_integer_in_string(file_name)
                if number > self.num_timesteps:
                    print("Deleting newer model data")
                    if os.path.isfile(name):
                        print("Deleting",name)
                        os.remove(name)
                    else:
                        print("Error: %s file was not found and could not be deleted." % name)

        if self.num_timesteps % self.save_freq == 0:
            # Check if old models have to be deleted
            old_models_list = []
            old_models_list.extend(glob.glob(self.log_dir+'inter_model_*.zip'))

            old_buffer_list = []
            old_buffer_list.extend(glob.glob(self.log_dir+'inter_model_*.pkl'))

            if self.save_inter_replay_buffer == False and len(old_buffer_list) >= 1:
                path_of_replay_buffer_to_delete = old_buffer_list[0]
                print("Deleting intermediate replay buffer")
                # Delete replay buffer
                if os.path.isfile(path_of_replay_buffer_to_delete):
                    print("Deleting",path_of_replay_buffer_to_delete)
                    os.remove(path_of_replay_buffer_to_delete)
                else:
                    print("Error: %s file was not found and could not be deleted." % path_of_replay_buffer_to_delete)


            if self.number_of_inter_models_to_keep > 0:
                if len(old_models_list) >= self.number_of_inter_models_to_keep:
                    number_list = []
                    for name in old_models_list:
                        file_name = Path(name).stem
                        number_list.append(int(file_name.split("_")[2]))
                    min_index = number_list.index(min(number_list))
                    path_of_model_to_delete = old_models_list[min_index]
                    name_of_model_to_delete = Path(path_of_model_to_delete).stem
                    path_of_replay_buffer_to_delete = self.log_dir+name_of_model_to_delete+"_replay_buffer.pkl"
                    if self.curriculum_learning_bool:
                        path_of_additional_training_parameters_to_delete = self.log_dir+name_of_model_to_delete+"_additional_training_parameters.yaml"

                    print("Deleting data of oldest intermediate model")
                    # Delete model
                    if os.path.isfile(path_of_model_to_delete):
                        print("Deleting",path_of_model_to_delete)
                        os.remove(path_of_model_to_delete)
                    else:
                        print("Error: %s file was not found and could not be deleted." % path_of_model_to_delete)

                    # Delete replay buffer
                    if os.path.isfile(path_of_replay_buffer_to_delete):
                        print("Deleting",path_of_replay_buffer_to_delete)
                        os.remove(path_of_replay_buffer_to_delete)
                    else:
                        print("Error: %s file was not found and could not be deleted." % path_of_replay_buffer_to_delete)

                    # Delete additional training parameters
                    if self.curriculum_learning_bool:
                        if os.path.isfile(path_of_additional_training_parameters_to_delete):
                            print("Deleting",path_of_additional_training_parameters_to_delete)
                            os.remove(path_of_additional_training_parameters_to_delete)
                        else:
                            print("Error: %s file was not found and could not be deleted." % path_of_additional_training_parameters_to_delete)


            # Storing new intermediate model
            save_model_name = "inter_model_"+str(self.num_timesteps)
            save_replay_buffer_name = save_model_name+"_replay_buffer"
            print("Save intermediate model:", save_model_name)
            self.model.save(os.path.join(self.log_dir, save_model_name))
            print("Save replay buffer:", save_replay_buffer_name)
            self.model.save_replay_buffer(os.path.join(self.log_dir, save_replay_buffer_name))
            if self.curriculum_learning_bool:
                save_additional_training_parameters_name = save_model_name+"_additional_training_parameters"
                print("Save additional training parameters:", save_additional_training_parameters_name)
                current_curriculum_step = common.get_env_variable_value(self.training_env, self.locals, "current_curriculum_step")
                common.save_yaml(path=os.path.join(self.log_dir, save_additional_training_parameters_name+".yaml"), data={"current_curriculum_step": current_curriculum_step})

            # Upload models to Hugging Face
            if self.online_model_bool:
                # Save intermediate model
                hugging_face.push_file_to_hub(
                    repo_id=self.online_model_params["repo_id"],
                    filename=self.log_dir+save_model_name+".zip",
                    commit_message=self.online_model_params["commit_msg"]
                    )
                # Save additional training parameters
                if self.curriculum_learning_bool:
                    hugging_face.push_file_to_hub(
                        repo_id=self.online_model_params["repo_id"],
                        filename=self.log_dir+save_additional_training_parameters_name+".yaml",
                        commit_message=self.online_model_params["commit_msg"]
                        )     
        return True

class HelperEvalCallback:
    """
    A callback class to handle evaluation metrics during policy evaluation.
    """
    def __init__(self, eval_keys=["ep_rew", "ep_length"], logger=None, verbose=0):
        self.verbose = verbose
        self.eval_keys = eval_keys
        if "ep_rew" in self.eval_keys:
            self._ep_rew_buffer = []
        if "ep_length" in self.eval_keys:
            self._ep_length_buffer = []
        if "success" in self.eval_keys:
            self._success_buffer = []
        if "height_diff" in self.eval_keys:
            self._height_diff_buffer = []
        if "goal_height_diff" in self.eval_keys:
            self._goal_height_diff_buffer = []
        if "goal_area_dist" in self.eval_keys:
            self._goal_area_dist_buffer = []
        if "in_goal_cells_changed" in self.eval_keys:
            self._in_goal_cells_changed_buffer = []
        if "out_goal_cells_changed" in self.eval_keys:
            self._out_goal_cells_changed_buffer = []
        if "in_goal_mean_diff" in self.eval_keys:
            self._in_goal_mean_diff_buffer = []
        if "out_goal_mean_diff" in self.eval_keys:
            self._out_goal_mean_diff_buffer = []
        if "execution_steps" in self.eval_keys:
            self._execution_steps_buffer = []

        self.logger = logger
        
        self.reset_buffers()

    def callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the ``evaluate_policy`` function
        in order to log the success rate, etc.

        :param locals_: Local variables from the evaluation context.
        :param globals_: Global variables from the evaluation context.
        """
        info = locals_["info"]

        if locals_["done"]:
            if "ep_rew" in self.eval_keys:
                episode_reward = info["episode"]["r"]
                if episode_reward is not None:
                    self._ep_rew_buffer.append(episode_reward)
            if "ep_rew" in self.eval_keys:
                episode_length = info["episode"]["l"]
                if episode_length is not None:
                    self._ep_length_buffer.append(episode_length)
            if "success" in self.eval_keys:
                maybe_success = info.get("success")
                if maybe_success is not None:
                    self._success_buffer.append(maybe_success)
            if "height_diff" in self.eval_keys:
                height_diff = info.get("height_diff")
                if height_diff is not None:
                    self._height_diff_buffer.append(height_diff)
            if "goal_height_diff" in self.eval_keys:
                goal_height_diff = info.get("goal_height_diff")
                if goal_height_diff is not None:
                    self._goal_height_diff_buffer.append(goal_height_diff)
            if "goal_area_dist" in self.eval_keys:
                goal_area_dist = info.get("goal_area_dist")
                if goal_area_dist is not None:
                    self._goal_area_dist_buffer.append(goal_area_dist)
            if "in_goal_cells_changed" in self.eval_keys:
                in_goal_cells_changed = info.get("in_goal_cells_changed")
                if in_goal_cells_changed is not None:
                    self._in_goal_cells_changed_buffer.append(in_goal_cells_changed)
            if "out_goal_cells_changed" in self.eval_keys:
                out_goal_cells_changed = info.get("out_goal_cells_changed")
                if out_goal_cells_changed is not None:
                    self._out_goal_cells_changed_buffer.append(out_goal_cells_changed)
            if "in_goal_mean_diff" in self.eval_keys:
                in_goal_mean_diff = info.get("in_goal_mean_diff")
                if in_goal_mean_diff is not None:
                    self._in_goal_mean_diff_buffer.append(in_goal_mean_diff)
            if "out_goal_mean_diff" in self.eval_keys:
                out_goal_mean_diff = info.get("out_goal_mean_diff")
                if out_goal_mean_diff is not None:
                    self._out_goal_mean_diff_buffer.append(out_goal_mean_diff)
            if "execution_steps" in self.eval_keys:
                execution_steps = info.get("execution_steps")
                if execution_steps is not None:
                    self._execution_steps_buffer.append(execution_steps)

    def reset_buffers(self):
        """Reset the buffers for a new evaluation."""
        if "ep_rew" in self.eval_keys:
            self._ep_rew_buffer.clear()
        if "ep_length" in self.eval_keys:
            self._ep_length_buffer.clear()
        if "success" in self.eval_keys:
            self._success_buffer.clear()
        if "height_diff" in self.eval_keys:
            self._height_diff_buffer.clear()
        if "goal_height_diff" in self.eval_keys:
            self._goal_height_diff_buffer.clear()
        if "goal_area_dist" in self.eval_keys:
            self._goal_area_dist_buffer.clear()
        if "in_goal_cells_changed" in self.eval_keys:
            self._in_goal_cells_changed_buffer.clear()
        if "out_goal_cells_changed" in self.eval_keys:
            self._out_goal_cells_changed_buffer.clear()
        if "in_goal_mean_diff" in self.eval_keys:
            self._in_goal_mean_diff_buffer.clear()
        if "out_goal_mean_diff" in self.eval_keys:
            self._out_goal_mean_diff_buffer.clear()
        if "execution_steps" in self.eval_keys:
            self._execution_steps_buffer.clear()

    def get_buffer(self, buffer_name):
        if buffer_name=="ep_rew" and "ep_rew" in self.eval_keys:
            return self._ep_rew_buffer
        elif buffer_name=="ep_length" and "ep_length" in self.eval_keys:
            return self._ep_length_buffer
        elif buffer_name=="success" and "success" in self.eval_keys:
            return self._success_buffer
        elif buffer_name=="height_diff" and "height_diff" in self.eval_keys:
            return self._height_diff_buffer
        elif buffer_name=="goal_height_diff" and "goal_height_diff" in self.eval_keys:
            return self._goal_height_diff_buffer
        elif buffer_name=="goal_area_dist" and "goal_area_dist" in self.eval_keys:
            return self._goal_area_dist_buffer
        elif buffer_name=="in_goal_cells_changed" and "in_goal_cells_changed" in self.eval_keys:
            return self._in_goal_cells_changed_buffer
        elif buffer_name=="out_goal_cells_changed" and "out_goal_cells_changed" in self.eval_keys:
            return self._out_goal_cells_changed_buffer
        elif buffer_name=="in_goal_mean_diff" and "in_goal_mean_diff" in self.eval_keys:
            return self._in_goal_mean_diff_buffer
        elif buffer_name=="out_goal_mean_diff" and "out_goal_mean_diff" in self.eval_keys:
            return self._out_goal_mean_diff_buffer
        elif buffer_name=="execution_steps" and "execution_steps" in self.eval_keys:
            return self._execution_steps_buffer

    def calculate_output(self, logger_key, values, precision=2, multiplier=1, unit=""):
        if len(values) <= 0:
            return None
        else:
            mean_value = np.mean(values)
            std_value = np.std(values)
            if self.verbose >= 2:
                print(f"{logger_key}: {multiplier*mean_value:.{precision}f}{unit} +/- {multiplier*std_value:.{precision}f}")
            if self.logger is not None:
                self.logger.record("eval/"+logger_key, float(mean_value))
            return {"values": values, "mean": mean_value, "std": std_value, "precision": precision, "multiplier": multiplier, "unit": unit}

    def get_buffer_means(self, precision=2):
        output = {}
        if "ep_rew" in self.eval_keys:
            logger_key = "mean_reward"
            output[logger_key] = self.calculate_output(logger_key, self._ep_rew_buffer, precision)

        if "ep_length" in self.eval_keys:
            logger_key = "mean_ep_length"
            output[logger_key] = self.calculate_output(logger_key, self._ep_length_buffer, precision)
            
        if "success" in self.eval_keys:
            logger_key = "success_rate"
            output[logger_key] = self.calculate_output(logger_key, list(map(float, self._success_buffer)), precision, multiplier=100, unit="%")

        if "height_diff" in self.eval_keys:
            logger_key = "height_diff"
            output[logger_key] = self.calculate_output(logger_key, self._height_diff_buffer, precision, multiplier=100, unit="%")

        if "goal_height_diff" in self.eval_keys:
            logger_key = "goal_height_diff"
            output[logger_key] = self.calculate_output(logger_key, self._goal_height_diff_buffer, precision, multiplier=100, unit="%")

        if "goal_area_dist" in self.eval_keys:
            logger_key = "goal_area_dist"
            output[logger_key] = self.calculate_output(logger_key, self._goal_area_dist_buffer, precision, multiplier=1000, unit="mm")

        if "in_goal_cells_changed" in self.eval_keys:
            logger_key = "in_goal_cells_changed"
            output[logger_key] = self.calculate_output(logger_key, self._in_goal_cells_changed_buffer, precision, multiplier=100, unit="%")

        if "out_goal_cells_changed" in self.eval_keys:
            logger_key = "out_goal_cells_changed"
            output[logger_key] = self.calculate_output(logger_key, self._out_goal_cells_changed_buffer, precision, multiplier=100, unit="%")

        if "in_goal_mean_diff" in self.eval_keys:
            logger_key = "in_goal_mean_diff[mm]"
            output[logger_key] = self.calculate_output(logger_key, self._in_goal_mean_diff_buffer, precision, unit="mm")

        if "out_goal_mean_diff" in self.eval_keys:
            logger_key = "out_goal_mean_diff[mm]"
            output[logger_key] = self.calculate_output(logger_key, self._out_goal_mean_diff_buffer, precision, unit="mm")

        if "execution_steps" in self.eval_keys:
            logger_key = "execution_steps"
            output[logger_key] = self.calculate_output(logger_key, self._execution_steps_buffer, precision)

        return output

class CustomEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        eval_keys = ["ep_rew", "ep_length"],
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        save_inter_replay_buffer: bool = True,
        eval_best_model_success_threshold: float = 1.0,
        use_curriculum_learning: bool = False
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self
        self.eval_keys = eval_keys
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.best_mean_goal_height_diff = -np.inf
        self.last_mean_reward = -np.inf
        self.last_mean_goal_height_diff = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.save_inter_replay_buffer = save_inter_replay_buffer
        self.eval_best_model_success_threshold = eval_best_model_success_threshold
        self.use_curriculum_learning = use_curriculum_learning

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._success_buffer = []
        self.evaluations_successes = []
        self.old_save_model_name = ""
        self.old_save_replay_buffer_name = ""
        self.old_save_evaluations_name = ""

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn("Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)
        
        self.eval_helper_cb_instance = HelperEvalCallback(eval_keys=self.eval_keys, logger=self.logger, verbose=self.verbose)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        if self.use_curriculum_learning:
            self.current_curriculum_step = 0
            self.previous_curriculum_step = self.current_curriculum_step

    def _save_best_model(self, metric_name):
        if self.verbose >= 1:
            print("New best model for:",metric_name)
        if self.best_model_save_path is not None:
            best_model_name = "best_model_"+metric_name
            # Remove old best model
            print("Deleting old best model data")
            for name in glob.glob(self.best_model_save_path+best_model_name+"*"):
                if os.path.isfile(name):
                    print("Deleting",name)
                    os.remove(name)
                else:
                    print("Error: %s file was not found and could not be deleted." % name)

            if self.best_model_save_path is not None:
                save_model_name = best_model_name+"_"+str(self.num_timesteps)
                save_replay_buffer_name = save_model_name+"_replay_buffer"
                if self.use_curriculum_learning:
                    save_additional_training_parameters_name = save_model_name+"_additional_training_parameters.yaml"
            
            # Logs will be written in ``file_name.npz``
            if self.log_path is not None:
                save_evaluations_name = save_model_name+"_evaluations"
                log_path = os.path.join(self.log_path, save_evaluations_name)
            if log_path is not None:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)

            if log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                episode_rewards = self.eval_helper_cb_instance.get_buffer("reward")
                episode_lengths = self.eval_helper_cb_instance.get_buffer("length")
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._success_buffer) > 0:
                    self.evaluations_successes.append(self._success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            print("Save best model:",save_model_name)
            self.model.save(os.path.join(self.best_model_save_path, save_model_name))
            if self.use_curriculum_learning:
                current_curriculum_step = self.training_env.unwrapped.env_method("get_variable", "current_curriculum_step")[0]
                common.save_yaml(path=os.path.join(self.best_model_save_path, save_additional_training_parameters_name), data={"current_curriculum_step": current_curriculum_step})

            if self.save_inter_replay_buffer:
                print("Save replay buffer: ",save_replay_buffer_name)
                self.model.save_replay_buffer(os.path.join(self.best_model_save_path, save_replay_buffer_name))

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0 and self.num_timesteps >= self.model.learning_starts:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            self.eval_helper_cb_instance.reset_buffers()

            evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=False,
                warn=self.warn,
                callback=self.eval_helper_cb_instance.callback,
            )

            buffer_means = self.eval_helper_cb_instance.get_buffer_means()
            mean_reward = buffer_means["mean_reward"]["mean"]
            success_rate = buffer_means["success_rate"]["mean"]
            mean_goal_height_diff = buffer_means["goal_height_diff"]["mean"]
            
            self.last_mean_reward = mean_reward
            self.last_mean_goal_height_diff = mean_goal_height_diff
            self._success_buffer = self.eval_helper_cb_instance.get_buffer("success")

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if len(self._success_buffer) > 0:
                if success_rate >= self.eval_best_model_success_threshold:
                    if mean_reward > self.best_mean_reward:
                        self._save_best_model(metric_name="mean_reward")
                        self.best_mean_reward = mean_reward
                    if mean_goal_height_diff > self.best_mean_goal_height_diff:
                        self._save_best_model(metric_name="goal_height_diff")
                        self.best_mean_goal_height_diff = mean_goal_height_diff

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, reward_keys, stats_window_size = 100, log_interval=10, use_curriculum_learning=False, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.reward_keys = reward_keys
        self.current_episode = 0
        self.last_episode = 0
        self.stats_window_size = stats_window_size
        self.log_interval = log_interval
        self.use_curriculum_learning = use_curriculum_learning

        # Will be initialized during training start
        self.success_history = None
        self.reward_history = None
        self.rewards_history = None

    def _on_training_start(self) -> None:
        # Initialize reward history for each environment
        self.num_envs = self.training_env.num_envs
        self.success_history = [deque(maxlen=self.stats_window_size) for _ in range(self.num_envs)]
        self.reward_history = [deque(maxlen=self.stats_window_size) for _ in range(self.num_envs)]

        # Initialize rewards_history for tracking individual reward components
        self.rewards_history = [defaultdict(lambda: deque(maxlen=self.stats_window_size)) for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            for key in self.reward_keys:
                self.rewards_history[i][key] = deque(maxlen=self.stats_window_size)

    def _on_step(self) -> bool:
        # Access dones and infos
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        # Update reward history for completed episodes
        for i, done in enumerate(dones):
            if done:
                # Update total episode success
                if "success" in infos[i]:
                    self.success_history[i].append(infos[i]["success"])

                # Update total episode rewards
                if "episode_reward" in infos[i]:
                    self.reward_history[i].append(infos[i]["episode_reward"])

                # Update individual reward components
                if "episode_rewards" in infos[i]:
                    reward_components = infos[i]["episode_rewards"]
                    for key in self.reward_keys:
                        if key in reward_components:
                            self.rewards_history[i][key].append(reward_components[key])

        # Log rewards at the specified interval
        if self.num_timesteps % self.log_interval == 0:
            # Log overall mean success across all environments
            all_success = [np.mean(success) for success in self.success_history if len(success) > 0]
            overall_all_success = np.mean(all_success) if all_success else 0.0
            self.logger.record('train_all_envs/success', overall_all_success)

            # Log overall mean total rewards across all environments
            all_rewards = [np.mean(history) for history in self.reward_history if len(history) > 0]
            overall_mean_reward = np.mean(all_rewards) if all_rewards else 0.0
            self.logger.record('train_all_envs/ep_reward', overall_mean_reward)

            # Log per-environment total success
            for i in range(self.num_envs):
                env_mean_success= np.mean(self.success_history[i]) if len(self.success_history[i]) > 0 else 0.0
                self.logger.record(f'train_env_{i}/success', env_mean_success)

            # Log per-environment total rewards
            for i in range(self.num_envs):
                env_mean_reward = np.mean(self.reward_history[i]) if len(self.reward_history[i]) > 0 else 0.0
                self.logger.record(f'train_env_{i}/ep_reward', env_mean_reward)

            # Log individual reward components
            for key in self.reward_keys:
                # Log overall mean of each reward component
                all_component_rewards = [np.mean(env_rewards[key]) for env_rewards in self.rewards_history if len(env_rewards[key]) > 0]
                overall_mean_component = np.mean(all_component_rewards) if all_component_rewards else 0.0
                self.logger.record(f'train_all_envs/ep_reward/{key}', overall_mean_component)

                # Log per-environment mean of each reward component
                for i in range(self.num_envs):
                    env_component_mean = np.mean(self.rewards_history[i][key]) if len(self.rewards_history[i][key]) > 0 else 0.0
                    self.logger.record(f'train_env_{i}/ep_reward/{key}', env_component_mean)

        return True