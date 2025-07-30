from src.interfaces import ActionType
from .type0_left import ActionType0Left
from .type0_right import ActionType0Right
from .type1 import ActionType1
from .action_commands import ActionType0LeftCommand, ActionType0RightCommand, ActionType1Command

__all__ = [
    'ActionType',
    'ActionType0Left',
    'ActionType0Right',
    'ActionType1',
    'ActionType0LeftCommand',
    'ActionType0RightCommand',
    'ActionType1Command',

    'ACTION_COMMAND_MAPPING'
]

__version__ = '1.4.0'

__author__ = 'ZhuoQiuMcgill'

ACTION_TYPE_MAPPING = {
    0: ActionType0Left,
    1: ActionType0Right,
    2: ActionType1,
}

ACTION_TYPE_NAMES = {
    0: "ActionType0Left",
    1: "ActionType0Right",
    2: "ActionType1",
}

# Command mappings for prediction pipeline
ACTION_COMMAND_MAPPING = {
    0: ActionType0LeftCommand,
    1: ActionType0RightCommand,
    2: ActionType1Command,
}

ACTION_COMMAND_NAMES = {
    0: "ActionType0LeftCommand",
    1: "ActionType0RightCommand", 
    2: "ActionType1Command",
}
