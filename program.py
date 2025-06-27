import pygame
import threading
from math import fabs
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

# settings

SIZE = 588
running = True

prediction = ''

pixels = [255 for _ in range(SIZE * SIZE)]

def get_converted_pixels():
    converted = [0 for _ in range(784)]

    for x in range(28):
        for y in range(28):
            x_mid = x * 21 + 10
            y_mid = y * 21 + 10

            sum = 0

            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    tx = x_mid + dx
                    ty = y_mid + dy

                    sum += getPixel(tx, ty)
            
            avg = sum / 441
            avg /= 255
            avg = 1 - avg

            converted[y * 28 + x] = avg
    
    return [converted]

show_plot = False

with open("data/neural_data.dat", "rb") as f:
    w_i_h = pickle.load(f)
    b_i_h = pickle.load(f)
    w_h_o = pickle.load(f)
    b_h_o = pickle.load(f)

def drawPixel(WINDOW, x, y, color):
    pygame.draw.rect(WINDOW, (color, color, color), (x, y, 1, 1), 1)

def getPixel(x, y):
    return pixels[y * SIZE + x]

def setPixel(x, y, color):
    pixels[y * SIZE + x] = color

def main():
    global running, pixels, show_plot, prediction

    pygame.init()
    pygame.font.init()

    FONT = pygame.font.SysFont('comicsans', 15)
    WINDOW = pygame.display.set_mode((SIZE, SIZE))
    pygame.display.set_caption("Digit Recognizer")
    clock = pygame.time.Clock()

    mouse_down = False
    mouse_x = 0
    mouse_y = 0
    WINDOW.fill((255, 255, 255))
    while running:
        clock.tick(100)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == pygame.BUTTON_LEFT:
                    mouse_down = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == pygame.BUTTON_LEFT:
                    mouse_down = False
            elif event.type == pygame.MOUSEMOTION:
                mouse_x, mouse_y = pygame.mouse.get_pos()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    WINDOW.fill((255, 255, 255))
                    pixels = [255 for _ in range(SIZE * SIZE)]
                elif event.key == pygame.K_p:
                    show_plot = True
        'print(mouse_x, mouse_y)'
        
        # update
        if mouse_down:
            if mouse_x >= 0 and mouse_x < SIZE and mouse_y >= 0 and mouse_y < SIZE:
                for dx in range(-10, 11):
                    for dy in range(-10, 11):
                        if mouse_x + dx >= 0 and mouse_x + dx < SIZE and mouse_y + dy >= 0 and mouse_y + dy < SIZE:
                            setPixel(mouse_x + dx, mouse_y + dy, (fabs(dx) + fabs(dy)) * 0.1)
                            drawPixel(WINDOW, mouse_x + dx, mouse_y + dy, (fabs(dx) + fabs(dy)) * 0.1)

        # render
        '''for y in range(SIZE):
            for x in range(SIZE):
                drawPixel(x, y, getPixel(x, y))'''
        
        WINDOW.fill((255, 255, 255), (0, 0, 100, 20))
        WINDOW.blit(FONT.render(f"Prediction : {prediction}", True, (0, 0, 0)), (0, 0, 50, 10))

        pygame.display.update()
    pygame.quit()

def predict():
    global running, show_plot, prediction
    counter = 1
    while running:
        if counter % 200 == 0:
            img = np.array(get_converted_pixels()).T

            # forward propagation : input -> hidden1
            h_pre = b_i_h + w_i_h @ img
            h = 1 / (1 + np.exp(-h_pre))

            # forward propagation : hidden2 -> output
            o_pre = b_h_o + w_h_o @ h
            o = 1 / (1 + np.exp(-o_pre))

            prediction = o.argmax()
            '''os.system('cls')'''
            counter = 1

            if show_plot:
                show_plot = False
                plt.imshow(img.reshape((28, 28)), cmap="Greys")
                plt.show()
        counter += 1


thread1 = threading.Thread(target=main)
thread2 = threading.Thread(target=predict)

if __name__ == "__main__":
    thread1.start()
    thread2.start()

    thread2.join()
    thread1.join()
