
from pynput import keyboard

keyboard_status = {}


def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
        
    except AttributeError:
        print('special key {0} pressed'.format(
            key))
    if key.char in ['w','a','s','d']:
        keyboard_status[key.char] = 1
    print(keyboard_status)
    
def on_release(key):
    print('{0} released'.format(
        key))
    if key.char in ['w','a','s','d']:
        keyboard_status[key.char] = 0
    if key == keyboard.Key.esc:
        # Stop listener
        return False
    print(keyboard_status)


# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

# # ...or, in a non-blocking fashion:
# listener = keyboard.Listener(
#     on_press=on_press,
#     on_release=on_release)
# listener.start()