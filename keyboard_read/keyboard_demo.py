import keyboard
import time
while 1:
    if keyboard.is_pressed('a'): 
        print(1)
    else:
        print(0)
    time.sleep(0.001)