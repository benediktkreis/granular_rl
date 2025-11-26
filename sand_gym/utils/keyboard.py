import numpy as np
from pynput import keyboard
from pynput.keyboard import Key

ACTION_MAPPING = {
    'a': np.array([-1., 0., 0.]),
    'd': np.array([1., 0., 0.]),
    's': np.array([0., -1., 0.]),
    'w': np.array([0., 1., 0.]),
    Key.space: np.array([0., 0., 1.]),
    Key.shift: np.array([0., 0., -1.]),
    'r': True
}

class Keyboard:
    def __init__(self) -> None:
        self._init_keyboard_listener()

    def _init_keyboard_listener(self) -> None:
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self._keys_pressed = []

    def _on_press(self, key) -> None:
        if key not in self._keys_pressed:
            self._keys_pressed.append(key)

    def _on_release(self, key) -> None:
        if key in self._keys_pressed:
            self._keys_pressed.remove(key)

    def get_mapping(self, key):
        mapping = None
        try:
            mapping = ACTION_MAPPING[key.char]
        except:
            try:
                mapping =  ACTION_MAPPING[key]
            except:
                pass
        return mapping
    
    def get_action(self, action_scale: float = 1.0) -> np.array:
        action = np.zeros(3)
        reset = False
        for key in self._keys_pressed:
            mapping = self.get_mapping(key)
            if type(mapping) == np.ndarray:
                action += action_scale * mapping
            if type(mapping) == bool and key.char=="r":
                reset = mapping

        return np.array(action), reset