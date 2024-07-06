import sys
import re
import inspect
import os
import json
from typing import Optional, Any

import cv2
from InquirerPy import prompt
from gymnasium import Env

import common
from common.algorithms import Evaluation, Algorithm
from common.params import TaxiDriverParams, FrozenLakeParams
from common.policies import Policy
from common.rewards import Rewards


class UserMode:

    CONFIGURATION_PATH = os.path.join(os.path.dirname(__file__), "configuration.json")

    params: dict = {}

    def build_param_question(self, params: dict, p_name: str, p_def: dict) -> Optional[dict]:
        """
        Build a question for a parameter based on its metadata.
        :param params: Current parameters values
        :param p_name: Parameter name
        :param p_def: Parameter definition
        :return: Question dictionary
        """

        p_metadata = p_def.metadata

        if (prerequisite := p_metadata.get("prerequisite")) and not self.check_prerequisite(params, prerequisite):
            return

        question_type: str = self.get_question_type(p_metadata.get("type"))

        question = {
            'type': question_type,
            'name': p_name,
            'message': f"{p_metadata.get("description")} ({p_metadata.get("type", str).__name__})",
        }

        if question_type == 'input':
            question["default"] = str(p_def.default or "")

        if question_type == 'number':
            if p_metadata.get("type") in [int, float] and p_metadata.get("optional", False):
                question["type"] = 'input'
                question["validate"] = lambda x: self.validate_number_or_none(x, p_metadata.get("type"), p_metadata.get("min", None), p_metadata.get("max", None))
            else:
                question["min_allowed"] = p_metadata.get("min", None)
                question["max_allowed"] = p_metadata.get("max", None)
                question["float_allowed"] = p_metadata.get("type") == float and not p_metadata.get("optional", False)

            if default := p_def.default:
                question["default"] = default

        return question

    @staticmethod
    def check_prerequisite(params: dict, prerequisite: tuple) -> bool:
        """
        Check if a prerequisite is met.
        :param params: parameters
        :param prerequisite: prerequisite
        :return: True if the prerequisite is met, False otherwise
        """
        if isinstance(prerequisite[1], tuple):
            return params.get(prerequisite[0]) in set(prerequisite[1])
        return params.get(prerequisite[0]) == prerequisite[1]

    @staticmethod
    def validate_number_or_none(value: str, param_type: type, min_value: Optional[float], max_value: Optional[float]) -> bool:
        """
        Validate if a value is a float or None. Used since we can't validate floats or optional with InquirerPy.
        It also checks if the value is within the specified range.
        :param value: value to validate
        :param param_type: parameter type
        :param min_value: minimum value
        :param max_value: maximum value
        :return: True if the value is a float or None, False otherwise
        """
        if value == '':
            return True
        try:
            param_type(value)
            return ((min_value is not None and param_type(value) >= min_value and max_value is not None and float(value) <= max_value)
                    or (min_value is None and max_value is None))
        except ValueError:
            return False

    @staticmethod
    def get_question_type(param_type: type) -> str:
        """
        Get the question type based on the parameter type.
        :param param_type: parameter type
        :return: question type
        """
        if param_type == bool:
            return 'confirm'
        if param_type in [int, float]:
            return 'number'
        return 'input'

    @staticmethod
    def get_classes(module_name: str, parent_class: type = None) -> list:
        """
        Get all classes from a module that are subclasses of a parent class.
        :param module_name: name of the module to get classes from
        :param parent_class: parent class to filter classes by
        :return: list of classes that are subclasses of the parent class
        """
        return [
            {"name": cls_name, "value": cls_name} for cls_name, cls_obj in inspect.getmembers(sys.modules[module_name])
            if inspect.isclass(cls_obj) and issubclass(cls_obj, parent_class) and cls_obj is not parent_class
        ]

    @staticmethod
    def get_class_from_module(module_name: str, class_name: str) -> type:
        """
        Get a class from a module by its name.
        :param module_name: name of the module to get the class from
        :param class_name: name of the class to get
        :return: class object
        """
        return getattr(sys.modules[module_name], class_name)

    @staticmethod
    def get_constructor_parameters(cls) -> dict:
        """
        Get constructor parameters of a class with expected types.
        :param cls: class to get constructor parameters from
        :return: dictionary with parameter names as keys and expected types as values
        """
        return {
            name: parameter.annotation
            for name, parameter in inspect.signature(cls.__init__).parameters.items()
            if parameter.annotation != inspect._empty
        }

    def display_menu(self) -> None:
        """
        Display the main menu.
        :return: None
        """
        questions: list = [
            {
                'type': 'list',
                'name': 'mode',
                'message': 'Select an option',
                'choices': [
                    {"name": "Configure", "value": "configure"},
                    {"name": "Load configuration", "value": "load"},
                    {"name": "Save configuration", "value": "save"},
                    {"name": "Execute", "value": "execute"},
                    {"name": "Watch episode from last run", "value": "watch"},
                    {"name": "Load and play with a trained model", "value": "load_play_model"},
                    {"name": "Exit", "value": "exit"}
                ]
            }
        ]

        getattr(self, prompt(questions)["mode"])()

    def configure(self) -> None:
        """
        Configure all the parameters to run an experiment.
        :return: None
        """
        questions: list = [
            {
                'type': 'list',
                'name': 'environment',
                'message': 'Environment',
                'choices': self.get_classes(common.environments.__name__, common.environments.Game)
            },
            {
                'type': 'list',
                'name': 'algorithm',
                'message': 'Algorithm',
                'choices': self.get_classes(common.algorithms.__name__, common.algorithms.Algorithm)
            },
            {
                'type': 'list',
                'name': 'policy',
                'message': 'Policy',
                'choices': self.get_classes(common.policies.__name__, common.policies.Policy)
            }
        ]

        self.params = self.params | prompt(questions)

        params_cls = self.get_class_from_module(common.params.__name__, f"{self.params['environment']}Params")

        for f_name, f_def in params_cls.__dataclass_fields__.items():
            if f_def.metadata.get("configurable", True):
                question = self.build_param_question(self.params, f_name, f_def)

                if question:
                    self.params = self.params | prompt(question)

        print("Configuration done.")

        self.ask_action("save")

    def load(self, initialize: bool = False) -> None:
        """
        Load a configuration from a file.
        :param initialize: indicated if load is called when initializing the program
        :return: None
        """
        try:
            with open(self.CONFIGURATION_PATH, "r") as f:
                self.params = json.load(f)

            print("Saved configuration loaded.")

            if not initialize:
                self.ask_action("execute")
            else:
                print(f"Configured for {self.params['environment']}, {self.params['algorithm']} and {self.params['policy']}.")
                self.display_menu()
        except FileNotFoundError:
            print("No configuration file found.")
            self.display_menu()

    def save(self) -> None:
        """
        Save the current configuration to a file.
        :return: None
        """
        if not self.params:
            print("No configuration to save.")
            self.display_menu()

        with open(self.CONFIGURATION_PATH, "w") as f:
            f.write(json.dumps(self.params))

        print("Configuration saved.")

        self.ask_action("execute")

    def ask_action(self, action: str) -> None:
        """
        Ask the user if they want to execute the current configuration.
        :return: None
        """
        question = [
            {
                'type': 'confirm',
                'name': action,
                'message': f'Do you want to {action} this configuration ?'
            }
        ]

        if prompt(question)[action]:
            getattr(self, action)()
        getattr(self, "display_menu")()

    def cast_params(self) -> None:
        """
        Since inquirerpy only returns strings, we need to cast the parameters to the correct type.
        Types are describe in the metadata of the parameters from the dataclass.
        :return: None
        """
        params_cls = self.get_class_from_module(common.params.__name__, f"{self.params['environment']}Params")

        # Cast parameters to the correct type
        for f_name, f_def in params_cls.__dataclass_fields__.items():
            if f_def.metadata.get("configurable", True) and f_name in self.params:
                if f_def.metadata.get("optional", False) and not self.params[f_name]:
                    self.params[f_name] = None
                else:
                    if (target_type := f_def.metadata.get("type")) == tuple:
                        self.params[f_name] = eval(self.params[f_name])
                    else:
                        self.params[f_name] = target_type(self.params[f_name])

    def check_params(self) -> None:
        """
        Check if the current configuration is valid and can be executed.
        :return: None
        """
        if not self.params:
            print("No configuration to execute.")
            self.display_menu()

        self.cast_params()

    def execute(self) -> None:
        """
        Execute the current configuration. It will create an environment, a policy and an algorithm based on the
        configuration and run the algorithm. It will then plot the results.
        An example of the workflow can be found in the gymm.py file of taxi_driver or frozen_lake directories.
        :return:
        """
        self.check_params()

        params_cls = self.get_class_from_module(common.params.__name__, f"{self.params['environment']}Params")
        params = params_cls(**{k: v for k, v in self.params.items() if k in self.get_constructor_parameters(params_cls)})

        # Create environment with recording
        env_cls = self.get_class_from_module(common.environments.__name__, self.params["environment"])
        env: Env = env_cls(params).env

        # Create policy
        policy_cls = self.get_class_from_module(common.policies.__name__, self.params["policy"])
        policy: Policy = policy_cls(**{k: v for k, v in self.params.items() if k in self.get_constructor_parameters(policy_cls)})

        # Create algorithm
        algorithm_cls = self.get_class_from_module(common.algorithms.__name__, self.params["algorithm"])
        algorithm: Algorithm = algorithm_cls(env=env, params=params, policy=policy)

        algorithm.run()

        # Create environment without recording
        env: Env = env_cls(params, should_record=False).env
        algorithm.env = env

        plots_cls = self.get_class_from_module(common.plots.__name__, f"{self.params['environment']}Plots")
        plots_cls.plot(policy=algorithm.computed_policy, algorithm=algorithm, env=env, params=params)

        print("Execution done.")

        self.display_menu()

    @staticmethod
    def read_video(filename: str, params: Any):
        """
        Read a video file with OpenCV.
        :param filename: filename of the video
        :param params: parameters
        :return: None
        """
        cap = cv2.VideoCapture(os.path.join(params.saveepisode_folder, filename))

        cv2.namedWindow('Episode', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Episode', cv2.WND_PROP_TOPMOST, 1)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow("Episode", frame)

            if cv2.waitKey(60) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def transform_filename(pattern: str, replacement: str, filename: str):
        """
        Transform a filename using a pattern and a replacement.
        :param pattern: pattern to match
        :param replacement: replacement
        :param filename: filename to transform
        :return: transformed filename
        """
        return re.sub(pattern, replacement, filename)

    @staticmethod
    def extract_episode_number(file_name) -> int:
        """
        Extract the episode number from a filename.
        :param file_name: filename
        :return: episode number
        """
        return int(file_name.split('-')[-1].split('.')[0])

    def watch(self, last_episode_watched: Optional[str] = None) -> None:
        """
        Watch an episode save during the last training.
        :return: None
        """
        self.check_params()

        params_cls = self.get_class_from_module(common.params.__name__, f"{self.params['environment']}Params")
        params = params_cls(**{k: v for k, v in self.params.items() if k in self.get_constructor_parameters(params_cls)})

        pattern = r'rl-video-episode-(\d+)\.mp4'
        replacement = r'Episode \1'

        try:
            files = [f for f in os.listdir(params.saveepisode_folder) if re.match(r'rl-video-episode-(\d+)\.mp4', f)]
        except FileNotFoundError:
            print("No episodes to watch.")
            print(f"Episodes should be located in {os.path.abspath(params.saveepisode_folder)}")
            self.display_menu()

        filenames = [self.transform_filename(pattern, replacement, filename) for filename in files]

        questions = [
            {
                "type": "list",
                "name": "episode",
                "message": "Select an episode",
                "default": last_episode_watched,
                "choices": [
                    {
                        "name": "Back",
                        "value": None
                    }
                ] + [
                    {
                        "name": name,
                        "value": value
                    } for name, value in sorted(zip(filenames, files), key=lambda x: self.extract_episode_number(x[1]))
                ]
            }
        ]

        selected_episode = prompt(questions)["episode"]
        if selected_episode:
            print("Press Q to quit the video.")
            self.read_video(selected_episode, params)
            self.watch(selected_episode)
        else:
            self.display_menu()

    @staticmethod
    def ask_number_of_games() -> int:
        """
        Ask the user the number of games to play after loading a model.
        :return: number of games
        """
        return int(prompt([
            {
                "type": "number",
                "name": "n_games",
                "message": "Number of games to play",
                "default": 100,
                "min_allowed": 1,
            }
        ])["n_games"])

    def load_play_model(self) -> None:
        """
        Load a trained model and play with it.
        :return: None
        """
        self.check_params()

        params_cls = self.get_class_from_module(common.params.__name__, f"{self.params['environment']}Params")
        params = params_cls(**{k: v for k, v in self.params.items() if k in self.get_constructor_parameters(params_cls)})

        try:
            filenames = os.listdir(params.savemodel_folder)
        except FileNotFoundError:
            print("No models to play with.")
            print(f"Models should be loaded from {os.path.abspath(params.savemodel_folder)}")
            self.display_menu()

        questions = [
            {
                "type": "list",
                "name": "model",
                "message": "Select a model to load and play with",
                "choices": [
                       {
                           "name": "Back",
                           "value": None
                       }
                   ] + [
                       {
                           "name": name,
                           "value": name
                       } for name in filenames
                   ]
            }
        ]

        selected_model = prompt(questions)["model"]
        if selected_model:
            # Switch render mode to human only when evaluating
            render_mode, params.render_mode = params.render_mode, "human"

            # Create environment without recording and with human render mode
            env_cls = self.get_class_from_module(common.environments.__name__, self.params["environment"])
            env: Env = env_cls(params, should_record=False).env

            # Change the render fps to 20 to fasten the evaluation
            env.metadata["render_fps"] = 20

            # Switch back to the original render mode
            params.render_mode = render_mode

            evaluation = Evaluation(env=env, params=params, model_name=selected_model)

            rewards_cls = self.get_class_from_module(common.rewards.__name__, f"{self.params['environment']}Rewards")
            rewards: Rewards = rewards_cls()

            n_games = self.ask_number_of_games()

            print(f"Playing with the model for {n_games} games...")
            _, losses = evaluation.evaluate(rewards, n_runs=n_games)
            print(f"Won {n_games - sum(losses)} games out of {n_games}.")

            self.load_play_model()
        else:
            self.display_menu()

    @staticmethod
    def exit():
        """
        Method to exit the program.
        :return: None
        """
        sys.exit(0)
