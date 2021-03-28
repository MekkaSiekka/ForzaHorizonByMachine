import sys
import pygame
#https://code.visualstudio.com/docs/python/python-tutorial#_create-a-python-hello-world-source-code-file
pygame.init()

size = width, height = 640, 480
dx = 1
dy = 1
x= 163
y = 120
black = (0,0,0)
white = (255,255,255)

screen = pygame.display.set_mode(size)

while 1:

    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    x += dx
    y += dy

    input = ''
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                print("W")
            if event.key == pygame.K_a:
                print("A")   
            if event.key == pygame.K_s:
                print("S")
            if event.key == pygame.K_d:
                print("D")         
    # if x < 0 or x > width:   
    #     dx = -dx

    # if y < 0 or y > height:
    #     dy = -dy

    # screen.fill(black)

    # pygame.draw.circle(screen, white, (x,y), 8)

    # pygame.display.flip()