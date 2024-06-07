import sys
import inspect
import os
import json
from typing import Optional

from InquirerPy import prompt

import common


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
            return min_value is not None and param_type(value) >= min_value and max_value is not None and float(value) <= max_value
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

    def display_menu(self):
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
                    {"name": "Exit", "value": "exit"}
                ]
            }
        ]

        getattr(self, prompt(questions)["mode"])()

    def configure(self):
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

        for f_name, f_def in common.params.TaxiDriverParams.__dataclass_fields__.items():
            if f_def.metadata.get("configurable", True):
                question = self.build_param_question(self.params, f_name, f_def)

                if question:
                    self.params = self.params | prompt(question)

        print("Configuration done.")

        self.ask_action("save")

    def load(self):
        """
        Load a configuration from a file.
        :return: None
        """
        try:
            with open(self.CONFIGURATION_PATH, "r") as f:
                self.params = json.load(f)

            print("Configuration loaded.")

            self.ask_action("execute")
        except FileNotFoundError:
            print("No configuration file found.")
            self.display_menu()

    def save(self):
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

    def ask_action(self, action: str):
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

    def cast_params(self):
        # Cast parameters to the correct type
        for f_name, f_def in common.params.TaxiDriverParams.__dataclass_fields__.items():
            if f_def.metadata.get("configurable", True) and f_name in self.params:
                if f_def.metadata.get("optional", False) and not self.params[f_name]:
                    self.params[f_name] = None
                else:
                    self.params[f_name] = f_def.metadata.get("type")(self.params[f_name])

    def execute(self):
        if not self.params:
            print("No configuration to execute.")
            self.display_menu()

        self.cast_params()

        params_cls = self.get_class_from_module(common.params.__name__, f"{self.params['environment']}Params")
        params = params_cls(**{k: v for k, v in self.params.items() if k in self.get_constructor_parameters(params_cls)})

        env_cls = self.get_class_from_module(common.environments.__name__, self.params["environment"])
        env = env_cls(params).env

        policy_cls = self.get_class_from_module(common.policies.__name__, self.params["policy"])
        policy = policy_cls(**{k: v for k, v in self.params.items() if k in self.get_constructor_parameters(policy_cls)})

        algorithm_cls = self.get_class_from_module(common.algorithms.__name__, self.params["algorithm"])
        algorithm = algorithm_cls(env=env, params=params, policy=policy)

        algorithm.run()

        plots_cls = self.get_class_from_module(common.plots.__name__, f"{self.params['environment']}Plots")
        plots_cls.plot(policy=algorithm.computed_policy, algorithm=algorithm, env=env, params=params)

        print("Execution done.")

        self.display_menu()

    @staticmethod
    def exit():
        """
        Method to exit the program.
        :return: None
        """
        sys.exit(0)
