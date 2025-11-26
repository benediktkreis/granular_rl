import robosuite as suite
import os
from multiprocessing import Process, freeze_support, set_start_method

from sand_gym.wrappers.gym_wrapper import GymWrapperDictObs, IndexEnvWrapper

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import TD3, SAC
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

import numpy as np
import time

import argparse
import glob
from pathlib import Path
import sand_gym.utils.common as common
import sand_gym.utils.hugging_face as hugging_face
import wandb
from wandb.integration.sb3 import WandbCallback
from sand_gym.utils.sb_callbacks import SavingCallback, CustomEvalCallback, HelperEvalCallback, TensorboardCallback

from sand_gym.feature_extractors.visual_features import GatedCNNFeatureExtractorWithToolMask, CNNFeatureExtractorWithMasksAndImage, CNNFeatureExtractorWithMasks
from torch.nn import ReLU

from collections import OrderedDict
import logging

from sand_gym.utils.custom_policy import RandomPolicy, CPPPolicy

class RL():
    def __init__(self):
        os.environ["TRUST_REMOTE_CODE"] = "True"

    def get_terminal_params(self):
        parser = argparse.ArgumentParser(description="Train and test sand gym RL agents.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("-d", "--dataset", type=common.comma_separated_list, default=[-1], help="Used dataset. -1 uses the one from the config file.")
        parser.add_argument("-s", "--dataset_selection", type=common.comma_separated_list_with_substrings, default=[-1], help="Used dataset selection. -1 uses the one from the config file.")
        parser.add_argument("-c", "--config", type=str, default="test_config", help="Name of the YAML configuration file.")
        parser.add_argument("-m", "--mode", type=str, default="test", choices=["train", "test", "eval", "show", "show_action", "control_action", "download"], help="Train, test or evaluation mode.")
        parser.add_argument("-rand", "--randomizations", type=common.comma_separated_list, default=[-1], help="Randomizations. -1 uses the one from the config file. Options: goal, gripper")
        parser.add_argument("-a", "--agent", type=str, default="final", help="Agent type. Options: final=final_model, best=best_model, inter_model_x=inter_model_x")
        parser.add_argument("-b", "--best_metric", type=str, default="", choices=["","mean_reward","goal_height_diff"], help="Best model metric to use. If empty: mean_reward.")
        parser.add_argument("-g", "--gui", action="store_true", help="Show GUI or not.")
        parser.add_argument("-v", "--verbose", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6], help="Verbosity level. 0=no debug information, 1=show debug prints, 2=visual GUI elements, 3=plot diff heightmap, 4=plot reward, 5=plot heightmap 3D diff., 6=plot pointcloud segmentation")
        parser.add_argument("-e", "--execute_on", type=str, default="cpu", choices=["cpu", "cuda"], help="Specify if yout want to run the code on the CPU or GPU (cuda for nvidia GPUs)")
        parser.add_argument("-n", "--num_cpu", type=int, default=1, help="Number of CPUs used for training.")
        parser.add_argument("-o", "--online_model", action="store_false", help="Allow using an online model stored on Hugging Face.")
        parser.add_argument("-w", "--wandb", action="store_false", help="Use weights and biases for online tensorboard logs.")
        parser.add_argument("-f", "--save_eval_files", action="store_true", help="Save evaluation files.")
            
        args = parser.parse_args()
        params = vars(args)
        
        common.print_dict(params, "Parameters (Terminal):")

        return params
        
    def make_robosuite_env(self, env_id, options, rank=0, seed=0):
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param options: (dict) additional arguments to pass to the specific environment class initializer
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            common.register_models(env_id, options["robots"], options["gripper_types"])
            env = suite.make(env_id, **options)
            observations_config = options["observations_config"]
            camera_config = options.get("camera_config", None)
            observation_keys = observations_config.get("observation_keys", dict())
            info_observation_keys = observations_config.get("info_observation_keys", [])
            norm_obs = observations_config.get("normalize_observations", False)
            normalization_limits = observations_config.get("normalization_limits", [-1.0, 1.0])
            env = GymWrapperDictObs(env, observation_keys, info_observation_keys, norm_obs, normalization_limits, camera_config, options["params"]["verbose"])
            env = Monitor(env)
            # Wrap the environment with IndexEnvWrapper to add the environment index
            env = IndexEnvWrapper(env, rank)
            env.reset(seed=seed + rank, options={})
            return env
        set_random_seed(seed)
        return _init

    def add_vec_wrappers(self, vec_env, robosuite_conf):
            observations_config = robosuite_conf["observations_config"]
            stack_observations = int(observations_config.get("stack_observations", 0))
            if stack_observations > 0:
                vec_env = VecFrameStack(vec_env, n_stack=stack_observations)
            return vec_env
    
    def agent_type_selection(self, params, model_path):
        config_file_name = params["config"]
        online_model = params["online_model"]
        agent = params["agent"]
        best_metric = params["best_metric"]
        model_search_name = agent+"_model_"
        model_file_ending = ".zip"
        wildcard = "*"
        best_model_search_name = model_search_name+best_metric
        model_name = ""
        file_name = ""
        continue_training = False

        if agent == "best":
            # Check if local or online model exist
            search_pattern = model_path+best_model_search_name+wildcard+model_file_ending
            model_list = glob.glob(search_pattern)
            if len(model_list) == 0 and online_model == True:
                online_model_exists, online_model = hugging_face.load_online_model(config_file_name, model_path)
                if online_model_exists:
                    model_list = glob.glob(search_pattern)
            for name in model_list:
                file_name = Path(name).stem
                if file_name != "":
                    model_name = file_name
            if len(model_list) > 0:
                continue_training = True
        elif agent == "final":
            # Check if local or online model exist
            search_pattern = model_path+model_search_name+wildcard+model_file_ending
            model_list = glob.glob(search_pattern)
            if len(model_list) == 0 and online_model == True:
                online_model_exists, online_model = hugging_face.load_online_model(config_file_name, model_path)
                if online_model_exists:
                    model_list = glob.glob(search_pattern)
            for name in model_list:
                file_name = Path(name).stem
                if file_name != "":
                    model_name = file_name
            if len(model_list) > 0:
                continue_training = True
        elif agent[:5] == "inter":
            search_pattern = model_path+agent+model_file_ending
            if os.path.exists(search_pattern):
                model_name = agent
                continue_training = True
            elif online_model == True:
                online_model_exists, online_model = hugging_face.load_online_model(config_file_name, model_path)
                if os.path.exists(search_pattern):
                    model_name = agent
                    continue_training = True
            else:
                search_pattern = model_path+model_search_name+wildcard+model_file_ending
                if len(glob.glob(search_pattern)) == 0 and online_model == True:
                    online_model_exists, online_model = hugging_face.load_online_model(config_file_name, model_path)
                inter_model_list = []
                number_list = []
                for name in glob.glob(search_pattern):
                    file_name = Path(name).stem
                    inter_model_list.append(file_name)
                    number_list.append(common.get_integer_in_string(file_name))
                if len(inter_model_list) > 0:
                    max_index = number_list.index(max(number_list))
                    model_name = inter_model_list[max_index]
                    continue_training = True
        elif agent[:6] == "policy":
            model_file_ending = ".pt"
            search_pattern = model_path+agent+model_file_ending
            if os.path.exists(search_pattern):
                model_name = agent
                continue_training = True
            elif online_model == True:
                online_model_exists, online_model = hugging_face.load_online_model(config_file_name, model_path)
                if os.path.exists(search_pattern):
                    model_name = agent
                    continue_training = True
            else:
                search_pattern = model_path+model_search_name+wildcard+model_file_ending
                if len(glob.glob(search_pattern)) == 0 and online_model == True:
                    online_model_exists, online_model = hugging_face.load_online_model(config_file_name, model_path)
                inter_model_list = []
                number_list = []
                for name in glob.glob(search_pattern):
                    file_name = Path(name).stem
                    inter_model_list.append(file_name)
                    number_list.append(common.get_integer_in_string(file_name))
                if len(inter_model_list) > 0:
                    max_index = number_list.index(max(number_list))
                    model_name = inter_model_list[max_index]
                    continue_training = True
        else:
            print("Undefined agent model.")
            raise ValueError

        return model_name, continue_training
    
    def make_callbacks(self, params, config, env, save_model_path, additional_learning_parameters=None):
        # Parameters
        online_model = params["online_model"]
        use_wandb = params["wandb"]
        config_name = params["config"]
        verbose = params["verbose"]

        callbacks = []
    
        # Settings for stable-baselines callbacks
        sb_callbacks_conf = config.get("sb_callbacks", dict())
        if "tensorboard_cb" in sb_callbacks_conf:
            tensorboard_cb_conf = sb_callbacks_conf["tensorboard_cb"]
        if "saving_cb" in sb_callbacks_conf:
            saving_cb_conf = sb_callbacks_conf["saving_cb"]
        if "eval_cb" in sb_callbacks_conf:
            eval_cb_conf = sb_callbacks_conf["eval_cb"]
        if "lr_cb" in sb_callbacks_conf:
            lr_cb_conf = sb_callbacks_conf["lr_cb"]
        if "curriculum_cb" in sb_callbacks_conf:
            curriculum_cb_conf = sb_callbacks_conf["curriculum_cb"]
            use_curriculum_learning = True
        else:
            use_curriculum_learning = False

        reward_conf = config["robosuite"]["reward_config"]

        file_handling_conf = config["file_handling"]
        wandb_conf = file_handling_conf["wandb"]
        huggingface_conf = file_handling_conf["huggingface"]
        tensorboard_conf = file_handling_conf["tensorboard"]

        # Init WandB
        run = None
        if use_wandb:
            run = wandb.init(
                project=wandb_conf["wandb_project_name"],
                group=config_name,
                config=config,
                sync_tensorboard=True
            )
            wandb_run_id = run.id
            wandb_run_name = run.name
        else:
            wandb_run_id = 1
            wandb_run_name = "without WandB"
        
        huggingface_conf["repo_id"] = huggingface_conf["huggingface_project_name"]+"/"+config_name
        huggingface_conf["commit_msg"] = "Run "+ wandb_run_name + " (ID=" + str(wandb_run_id) + ")"

        # Init callbacks
        if "saving_cb" in sb_callbacks_conf:
            saving_cb_instance = SavingCallback(
                save_model_path,
                # start_step,
                save_freq=saving_cb_conf["saving_cb_frequency"],
                curriculum_learning_bool=use_curriculum_learning,
                number_of_inter_models_to_keep=saving_cb_conf["number_of_inter_models_to_keep"],
                save_inter_replay_buffer=saving_cb_conf["save_inter_replay_buffer"],
                wait_to_remove_newer_models=200)
            callbacks.append(saving_cb_instance)
        if "tensorboard_cb" in sb_callbacks_conf:
            tensorboard_cb_instance = TensorboardCallback(
                reward_keys=reward_conf["reward_keys"],
                stats_window_size = tensorboard_cb_conf["tensorboard_cb_amount_of_episodes_for_mean"], 
                use_curriculum_learning = use_curriculum_learning)
            callbacks.append(tensorboard_cb_instance)
        
        if "eval_cb" in sb_callbacks_conf:
            eval_cb_instance = CustomEvalCallback(
                eval_env=env,
                eval_keys = eval_cb_conf["eval_keys"],
                best_model_save_path=save_model_path, 
                log_path=save_model_path, 
                eval_freq=eval_cb_conf["eval_cb_frequency"], 
                n_eval_episodes=eval_cb_conf["eval_episodes"], 
                save_inter_replay_buffer=eval_cb_conf["save_inter_replay_buffer"], 
                eval_best_model_success_threshold=eval_cb_conf["eval_best_model_success_threshold"],
                use_curriculum_learning=use_curriculum_learning)
            callbacks.append(eval_cb_instance)

        if use_wandb:
            wandb_cb_instance = WandbCallback(model_save_path=f"./wandb/{wandb_run_id}",verbose=2)
            callbacks.append(wandb_cb_instance)
        
        return callbacks, run, wandb_conf, huggingface_conf, tensorboard_conf

    def load_show(self, mode, env_id, robosuite_conf, seed=0):
            from tabulate import tabulate
            from sand_gym.utils.keyboard import Keyboard

            robosuite_conf["ignore_done"] = True

            robosuite_env = DummyVecEnv([self.make_robosuite_env(env_id, robosuite_conf, seed=seed)])
            robosuite_env = self.add_vec_wrappers(robosuite_env, robosuite_conf)
            robosuite_env.reset()

            # Get action limits
            if mode == "show_action" or mode == "control_action":
                low, high = robosuite_env.envs[0].action_spec

            print("Running... Press Ctrl+C to stop.")
            if mode == "control_action":
                keyboard = Keyboard()
            if mode == "control_action": 
                print(tabulate([['A', '-x'], ['D', '+x'], ['S', '-y'], ['W', '+y'], ['shift', '-z'], ['space', '+z'], ['R', 'reset']], headers=['Key', 'Action']) + "\n")
    
            try:
                while True:
                    if robosuite_conf['has_renderer'] == True or robosuite_conf['has_offscreen_renderer'] == True:
                        robosuite_env.render()
                    if mode == "show_action" or mode == "control_action":
                        if mode == "show_action":
                            action = np.array([np.random.uniform(low, high)])
                        elif mode == "control_action":
                            action_scale=0.5
                            action, reset = keyboard.get_action(action_scale=action_scale)
                            action = np.asarray([action])
                            if reset:
                                robosuite_env.reset()
                        observation, reward, done, info = robosuite_env.step(action)
                        if mode != "control_action" and done:
                            robosuite_env.reset()
                    else: # mode == "show"
                        time.sleep(0.001)

            except KeyboardInterrupt:
                print("KeyboardInterrupt has been caught.")
                print("Cleaning up resources...")
            finally:
                print("Exiting program.")

            return robosuite_env
    
    def main(self, params, config):
        ######################
        # Parameter settings #
        ######################
        dataset_id = params["dataset"]
        dataset_selection = params["dataset_selection"]
        config_name = params["config"]
        mode = params["mode"]
        randomizations = params["randomizations"]
        gui = params["gui"]
        verbose = params["verbose"]
        hardware = params["execute_on"]
        num_cpu = params["num_cpu"]
        # use_wandb = params["wandb"]
        save_eval_files = params["save_eval_files"]

        ##########################
        # Configuration settings #
        ##########################
        seed = config["seed"]

        # Environment specifications
        robosuite_conf = config["robosuite"]
        # Remap configs to work with Robosuite v1.5.1
        robosuite_conf["base_types"] = "NullMount"
        robosuite_conf["arena_config"].pop("mount_types")
        temp_controller_configs = robosuite_conf["controller_configs"]
        robosuite_conf.pop("controller_configs")
        robosuite_conf["controller_configs"] = {}
        robosuite_conf["controller_configs"]["body_parts"] = {}
        robosuite_conf["controller_configs"]["body_parts"]["right"] = temp_controller_configs
        robosuite_conf["controller_configs"]["body_parts"]["right"]["gripper"] = {}
        robosuite_conf["controller_configs"]["body_parts"]["right"]["gripper"]["type"] = "GRIP"
        robosuite_conf["controller_configs"]["type"] = "BASIC"
        robosuite_conf["lite_physics"] = False
        if robosuite_conf["render_camera"] == "null":
            robosuite_conf["render_camera"] = "free"

        if mode=="train" or mode=="test" or mode=="eval":
            # Settings for stable-baselines RL algorithm
            sb_config = config["sb_config"]
            total_iterations = int(float(sb_config["total_iterations"]))
            buffer_size = int(float(sb_config["buffer_size"]))
            fill_buffer_at_start = int(float(sb_config["fill_buffer_at_start"]))
            learning_rate = float(sb_config["learning_rate"])
            action_noise_sigma = float(sb_config["action_noise_sigma"])
            
            # Settings for stable-baselines policy
            sb_policy = config["sb_policy"]
            policy_type = sb_policy["type"]
            rl_algo = str(sb_policy["rl_algo"])
            rl_algo_config = sb_policy[rl_algo.lower()]
            policy_kwargs = rl_algo_config["policy_kwargs"]

        if dataset_id[0] == -1:
            if "dataset_config" not in robosuite_conf["sand_sim_config"]:
                robosuite_conf["sand_sim_config"]["dataset_config"] = None
        else:
            robosuite_conf["sand_sim_config"]["dataset_config"]["id"] = dataset_id
            if dataset_selection is None:
                robosuite_conf["sand_sim_config"]["dataset_config"]["dataset_selection"] = dataset_selection
            elif dataset_selection[0] != -1:
                robosuite_conf["sand_sim_config"]["dataset_config"]["dataset_selection"] = dataset_selection
                    
        
        if len(randomizations) > 0:
            if randomizations[0] != -1:
                if "gripper" in randomizations:
                    robosuite_conf["initial_gripper_pos_randomization"] = True
                else:
                    robosuite_conf["initial_gripper_pos_randomization"] = False
                if "goal" in randomizations:
                    robosuite_conf["goal_randomization"] = True
                else:
                    robosuite_conf["goal_randomization"] = False
        else:
            robosuite_conf["initial_gripper_pos_randomization"] = False
            robosuite_conf["goal_randomization"] = False

        if gui:
            robosuite_conf["mujoco_passive_viewer"] = True # MuJoCo 3D passive viewer
            robosuite_conf["has_renderer"] = False  # on-screen rendering
        else:
            robosuite_conf["mujoco_passive_viewer"] = False # MuJoCo 3D passive viewer
            robosuite_conf["has_renderer"] = False  # on-screen rendering

        if "observation_keys" in robosuite_conf["observations_config"]:
            if "camera" in robosuite_conf["observations_config"]["observation_keys"]:
                robosuite_conf["use_camera_obs"] = True
                robosuite_conf["has_offscreen_renderer"] = True # off-screen rendering
            else:
                robosuite_conf["use_camera_obs"] = False
                robosuite_conf["has_offscreen_renderer"] = False # off-screen rendering

        if "info_observation_keys" in robosuite_conf["observations_config"]:
            if "camera" in robosuite_conf["observations_config"]["info_observation_keys"]:
                robosuite_conf["use_camera_obs"] = True
                robosuite_conf["has_offscreen_renderer"] = True # off-screen rendering

        # Add eval camera to observations for plots
        if verbose == 7:
            robosuite_conf["camera_names"].append("right_eval")
            robosuite_conf["camera_widths"].append(1280)
            robosuite_conf["camera_heights"].append(720)
            robosuite_conf["camera_depths"].append(True)
            robosuite_conf["camera_segmentations"].append("geom")
            robosuite_conf["camera_config"]["clip_ranges"]["right_eval"] = [0.3,1.0]
            if "camera" not in robosuite_conf["observations_config"]["info_observation_keys"]:
                robosuite_conf["observations_config"]["info_observation_keys"].append("camera")

        if "object" in robosuite_conf["observations_config"]["observation_keys"]:
            robosuite_conf["use_object_obs"] = True
        else:
            robosuite_conf["use_object_obs"] = False
        
        if hardware == "cpu":
            robosuite_conf["render_gpu_device_id"] = None
        elif hardware == "cuda":
            # "render_gpu_device_id" corresponds to the GPU device id to use for offscreen rendering.
            # Defaults to -1, in which case the device will be inferred from environment variables
            # (GPUS or CUDA_VISIBLE_DEVICES).
            robosuite_conf["render_gpu_device_id"] = -1
        else:
            print("Unknown hardware for execution")
            raise ValueError

        env_id = robosuite_conf.pop("env_id")
        robosuite_conf["verbose"] = verbose

        # Paths
        file_handling_conf = config["file_handling"]
        if mode == "download":
            rl_models_path = "./"+file_handling_conf["trained_model"]["load_model_folder"]+"/"+config_name+"/"
            hugging_face.load_online_model(params["config"], rl_models_path)
            return
        elif mode == "train":
            rl_models_path = "./"+file_handling_conf["trained_model"]["save_model_folder"]+"/"+config_name+"/"
        elif mode == "test" or mode == "eval":
            rl_models_path = "./"+file_handling_conf["trained_model"]["load_model_folder"]+"/"+config_name+"/"
                
        if mode != "show" and mode != "show_action" and mode != "control_action":
            model_name, continue_training = self.agent_type_selection(params, rl_models_path)
            replay_buffer_name = "replay_buffer"
            additional_training_parameters_name = "additional_training_parameters"
            tensorboard_log_dir = file_handling_conf["tensorboard"]["tb_log_folder"]

            print("Let's", params["mode"], model_name,"\n")

        ###########################
        # Python package settings #
        ###########################
        # Pytorch settings
        common.torch_settings(hardware)

        # Seed settings
        if verbose >= 1:
            print("seed =",seed)
        common.set_random_seed(hardware, random_seed=seed)

        robosuite_conf["params"] = params
        ###############
        # RL pipeline #
        ###############
        if mode=="show" or mode=="show_action" or mode=="control_action":
            env = self.load_show(mode, env_id, robosuite_conf, seed)

        if mode == "train":
            env = SubprocVecEnv([self.make_robosuite_env(env_id, robosuite_conf, i, seed) for i in range(num_cpu)], start_method=params["multi_proc_start_method"])
            env = self.add_vec_wrappers(env, robosuite_conf)
            
            # Train new model
            if continue_training:
                continue_training_model_path = os.path.join(rl_models_path, model_name + ".zip")
                replay_buffer_path = os.path.join(rl_models_path, model_name + "_" + replay_buffer_name + ".pkl")
                if "curriculum_cb" in config["sb_callbacks"]:
                    additional_learning_parameters_path = os.path.join(rl_models_path, model_name + "_additional_training_parameters.yaml")
                    additional_learning_parameters = common.read_yaml(additional_learning_parameters_path)
                else:
                    additional_learning_parameters = None
                print(f"Continual training on model located at {continue_training_model_path}")

                # Load model
                if rl_algo == "TQC":
                    model = TQC.load(continue_training_model_path, env=env, verbose=1, device=hardware)
                elif rl_algo == "TD3":
                    model = TD3.load(continue_training_model_path, env=env, verbose=1, device=hardware)
                elif rl_algo == "SAC":
                    model = SAC.load(continue_training_model_path, env=env, verbose=1, device=hardware)
                if os.path.exists(replay_buffer_path):
                    model.load_replay_buffer(replay_buffer_path)
                
                callbacks, run, wandb_conf, huggingface_conf, tensorboard_conf = self.make_callbacks(params, config, env, rl_models_path, additional_learning_parameters)
            
            else:
                # Create models folder if it doesn't exist
                common.create_folder(rl_models_path)

                action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]), sigma=action_noise_sigma * np.ones(env.action_space.shape[-1]))

                if "features_extractor_class" in sb_policy:
                    policy_kwargs["features_extractor_class"] = globals()[sb_policy["features_extractor_class"]]
                if "features_extractor_kwargs" in sb_policy:
                    policy_kwargs["features_extractor_kwargs"] = dict(config=sb_policy["features_extractor_kwargs"])
                if "activation_fn" in policy_kwargs:
                    policy_kwargs["activation_fn"] = globals()[policy_kwargs["activation_fn"]]
                    
                # Create model
                if rl_algo == "TQC":
                    model = TQC(policy_type, env, tensorboard_log=tensorboard_log_dir,
                                buffer_size=buffer_size, batch_size=rl_algo_config["batch_size"], tau=rl_algo_config["tau"], learning_rate=learning_rate,
                                gamma=rl_algo_config["gamma"], train_freq=tuple(rl_algo_config["train_freq"]), learning_starts=fill_buffer_at_start,
                                top_quantiles_to_drop_per_net=rl_algo_config["top_quantiles_to_drop_per_net"], gradient_steps=rl_algo_config["gradient_steps"],
                                policy_kwargs=policy_kwargs, action_noise=action_noise,
                                verbose=1, device=hardware)
                elif rl_algo == "TD3":
                    model = TD3(policy_type, env, tensorboard_log=tensorboard_log_dir,
                                learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=fill_buffer_at_start,
                                batch_size=rl_algo_config["batch_size"], tau=rl_algo_config["tau"], gamma=rl_algo_config["gamma"],
                                train_freq=tuple(rl_algo_config["train_freq"]), gradient_steps=rl_algo_config["gradient_steps"],
                                action_noise=action_noise,
                                policy_delay=rl_algo_config["policy_delay"], target_policy_noise=rl_algo_config["target_policy_noise"],
                                target_noise_clip=rl_algo_config["target_noise_clip"],
                                policy_kwargs=policy_kwargs,
                                verbose=1, device=hardware)
                elif rl_algo == "SAC":
                    model = SAC(policy_type, env, tensorboard_log=tensorboard_log_dir,
                                learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=fill_buffer_at_start,
                                batch_size=rl_algo_config["batch_size"], tau=rl_algo_config["tau"], gamma=rl_algo_config["gamma"],
                                train_freq=tuple(rl_algo_config["train_freq"]), gradient_steps=rl_algo_config["gradient_steps"],
                                action_noise=action_noise,
                                policy_kwargs=policy_kwargs,
                                verbose=1, device=hardware)
                
                callbacks, run, wandb_conf, huggingface_conf, tensorboard_conf = self.make_callbacks(params, config, env, rl_models_path)
                print("Created a new model")

            # Training
            reset_tb = config["sb_callbacks"]["tensorboard_cb"]["reset_tensorboard_iterations"]
            try:
                model.learn(total_timesteps=total_iterations, tb_log_name=config_name, callback=callbacks, reset_num_timesteps=reset_tb, progress_bar=True)
            except KeyboardInterrupt:
                print("Shut down by user.")
                
            if run is not None:
                run.finish()

            # Remove old final model
            print("Deleting old final model data")
            for name in glob.glob(rl_models_path+'final_model_*'):
                if os.path.isfile(name):
                    print("Deleting",name)
                    os.remove(name)
                else:
                    print("Error: %s file was not found and could not be deleted." % name)

            # Save trained model
            for index, value in enumerate(callbacks):
                if type(value) == SavingCallback:
                    saving_cb_index = index
                    break
            end_step = callbacks[saving_cb_index].num_timesteps # from saving callback
            model_name = "final_model_" + str(end_step)
            save_model_path = os.path.join(rl_models_path, model_name + ".zip")
            save_replay_buffer_path = os.path.join(rl_models_path, model_name + "_" + replay_buffer_name + ".pkl")
            save_additional_training_parameters_path = os.path.join(rl_models_path, model_name + "_" + additional_training_parameters_name + ".yaml")

            print("Save final model")
            model.save(save_model_path)
            print("Save final replay buffer")
            model.save_replay_buffer(save_replay_buffer_path)
            if "curriculum_cb" in config["sb_callbacks"]:
                print("Save final additional training parameters")
                current_curriculum_step = env.unwrapped.env_method("get_variable", "current_curriculum_step")[0]
                common.save_yaml(path=save_additional_training_parameters_path, data={"current_curriculum_step": current_curriculum_step})

            # Delete inter model replay buffer
            if "saving_cb" in config["sb_callbacks"]:
                if config["sb_callbacks"]["saving_cb"]["save_inter_replay_buffer"] == False:
                    old_buffer_list = []
                    old_buffer_list.extend(glob.glob(rl_models_path+'inter_model_*.pkl'))
                    if len(old_buffer_list) > 0:
                        path_of_replay_buffer_to_delete = old_buffer_list[0]
                        print("Deleting intermediate replay buffer")
                        # Delete replay buffer
                        if os.path.isfile(path_of_replay_buffer_to_delete):
                            print("Deleting",path_of_replay_buffer_to_delete)
                            os.remove(path_of_replay_buffer_to_delete)
                        else:
                            print("Error: %s file was not found and could not be deleted." % path_of_replay_buffer_to_delete)

            # Upload best model to Hugging Face
            if params["online_model"]:
                model_path_list = glob.glob(rl_models_path+'best_model_*.zip')
                for model_path in model_path_list:
                    hugging_face.push_file_to_hub(
                        repo_id=huggingface_conf["repo_id"],
                        filename=model_path,
                        commit_message=huggingface_conf["commit_msg"],
                        )
        
        if mode == "test" or mode == "eval":
            # RL policy
            load_model_path = os.path.join(rl_models_path, model_name+ ".zip")
            # Create test environment
            env = DummyVecEnv([self.make_robosuite_env(env_id, robosuite_conf, seed=seed)])
            env = self.add_vec_wrappers(env, robosuite_conf)

            # Load model
            if rl_algo == "TQC":
                model = TQC.load(load_model_path, env)
            elif rl_algo == "TD3":
                model = TD3.load(load_model_path, env)
            elif rl_algo == "SAC":
                model = SAC.load(load_model_path, env)
            elif rl_algo == "RANDOM":
                lr_schedule = lambda _: 0.0  # dummy learning rate schedule
                model = RandomPolicy(env.observation_space, env.action_space, lr_schedule, seed=seed)
            elif rl_algo == "CPP":
                lr_schedule = lambda _: 0.0  # dummy learning rate schedule
                model = CPPPolicy(env.observation_space, env.action_space, lr_schedule, seed=seed)

            # Simulate environment
            print(f"Start {mode} environment.")
           
            if mode == "test":
                obs = env.reset()
                eprew = 0
                while True:
                    action, _states = model.predict(obs, deterministic=True)
                    if verbose >= 1:
                        print(f"action: {action}")
                    obs, reward, done, info = env.step(action)
                    if verbose >= 1:
                        print(f'reward: {reward}')
                    eprew += reward
                    if done:
                        if verbose >= 1:
                            print(f'eprew: {eprew}')
                            print(f'success: {info[0]["success"]}')                      
                        obs = env.reset()
                        eprew = 0
                        print("\nReset, start again.\n")

            if mode == "eval":
                
                # Get evaluation config
                eval_config = common.get_config("eval")
                eval_episodes = eval_config["eval_episodes"]
                eval_keys = eval_config["eval_keys"]
                
                # Create evaluation helper callback
                eval_helper_cb_instance = HelperEvalCallback(eval_keys=eval_keys, verbose=2)

                # Start evaluation
                evaluate_policy(
                    model,
                    env,
                    n_eval_episodes=eval_episodes,
                    deterministic=True,
                    return_episode_rewards=False,
                    warn=False,
                    callback=eval_helper_cb_instance.callback,
                )

                # Get mean and std values
                print()
                print("Evaluation result:")
                buffer_means = eval_helper_cb_instance.get_buffer_means(precision=2)
                
                if save_eval_files:
                    print("Save evaluation results")
                    eval_save_folder = "./" + eval_config["save_eval_folder"] + "/" + params["config"] + "_" + params["agent"] + "_" + params["best_metric"] + "/"
                    eval_save_file = "eval"
                    common.create_folder(eval_save_folder)
                    
                    eval_save_path = eval_save_folder + eval_save_file + ".json"
                    common.save_json(path=eval_save_path, data=buffer_means)

        env.close()

if __name__ == "__main__":
    # INITIALIZE
    # ===========================================================  
    rl = RL()
    
    # GET TERMINAL PARAMETERS AND CONFIG
    # ===========================================================
    params = rl.get_terminal_params()
    config = common.get_config(params["config"])

    # Ensure thread safe start method for SB3 SubprocVecEnv
    # See: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    # See: https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods
    if params["num_cpu"] > 1 or params["mode"] == "train":
        params["multi_proc_start_method"] = "forkserver"
        freeze_support()
        set_start_method(params["multi_proc_start_method"])
        p = Process(target=rl.main(params, config))
        p.start()
    else:
        rl.main(params, config)