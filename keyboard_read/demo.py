
from pynput import keyboard
import time
import numpy as np
keyboard_status = {}


class KeyBoardStatus:
    def __init__(self, result):
        self.mKeyboardStatus = {}
        self.mKeyboardStatus['w'] =0
        self.mKeyboardStatus['a'] =0
        self.mKeyboardStatus['s'] =0
        self.mKeyboardStatus['d'] =0
        self.mIsPlaying = False
        self.mResult = result
    
    def updateResult(self):
        self.mResult[0] = self.mKeyboardStatus['w']
        self.mResult[1] = self.mKeyboardStatus['a']
        self.mResult[2] = self.mKeyboardStatus['s']
        self.mResult[3] = self.mKeyboardStatus['d']
    def on_press(self,key):
        try:
            # print('{0} pressed'.format(
            #     key.char))
            key.char
        except AttributeError:
            print('special key {0} pressed'.format(
                key))
        if key.char in ['w','a','s','d']:
            self.mKeyboardStatus[key.char] = 1
            self.updateResult()
        #print("jb1",self.mKeyboardStatus)
    
    def on_release(self,key):
        # print('{0} released'.format(
        #     key))
        try:
            # print('{0} pressed'.format(
            #     key.char))
            key.char
        except AttributeError:
            print("attrib error")
            return
        if key.char in ['w','a','s','d']:
            self.mKeyboardStatus[key.char] = 0
            #print("jb2",self.mKeyboardStatus)
            self.updateResult()
        if key == keyboard.Key.esc:
            self.mIsPlaying = False
            return False
        
    def start(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()
        self.mIsPlaying = True
    

# result = np.array([0,0,0,0])
# kb_status = KeyBoardStatus(result)
# kb_status.start()

# while 1:
#     print(result)
#     time.sleep(0.001)
