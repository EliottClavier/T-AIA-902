import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from common.user_mode import UserMode


def main():
    UserMode()


if __name__ == "__main__":
    main()
