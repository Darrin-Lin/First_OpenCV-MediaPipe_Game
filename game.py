#pygame too slow
import cv2
import mediapipe as mp
import numpy as np
import random
import pygame as pg

window_width = 500
window_height = 500
mp_draw = mp.solutions.drawing_utils
VdoCap = cv2.VideoCapture(0)
screen = pg.display.set_mode((window_width, window_height))
back = pg.Surface(screen.get_size())
back = back.convert()
back.fill((127, 127, 127))
screen.blit(back, (0, 0))
msdu = mp.solutions.drawing_utils          # mediapipe 繪圖方法
msds = mp.solutions.drawing_styles         # mediapipe 繪圖樣式
msh = mp.solutions.hands                     # mediapipe 偵測手掌方法
hand_x = list()
hand_y = list()
for i in range(0, 21):
    hand_x.append(-1)
    hand_y.append(-1)
spriteGroup = pg.sprite.Group()
pg.init()

with msh.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    if not VdoCap.isOpened():
        print("Cannot open camera")
        exit()
    run = 1
    while run:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = 0
                break
            ret, img = VdoCap.read()
            if not ret:
                print("Cannot receive frame")
                run = 0
                break
            img = cv2.resize(img, (window_width, window_height))
            img_BGRtoRGB = cv2.cvtColor(
                img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB，因BGR不易辨別
            handResults = hands.process(img_BGRtoRGB)
            if handResults.multi_hand_landmarks:
                for hand_landmarks in handResults.multi_hand_landmarks:
                    # 將節點和骨架繪製到影像中
                    for i in range(4, 21, 4):
                        hand_x[i] = int(
                            hand_landmarks.landmark[i].x * window_width)
                        hand_y[i] = int(
                            hand_landmarks.landmark[i].y * window_height)
                        print(
                            i, hand_landmarks.landmark[i].x*window_width, hand_landmarks.landmark[i].y*window_height)
                        pg.draw.circle(back, (0, 0, 255),
                                       (hand_x[i], hand_y[i]), 5, 2)
                    screen.blit(back, (0, 0))
                    pg.display.flip()
                    # back.fill((127, 127, 127))
pg.quit()
