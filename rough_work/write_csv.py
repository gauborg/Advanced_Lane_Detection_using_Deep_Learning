# code for extracting images from a video stream

'''
Description: This code writes .csv file
Author: Gaurav Borgaonkar
Date: 15 June 2020

'''

import os
import cv2
import csv
import random

x_list = []
y_list = []
z_list = []

print()

with open('test.csv', 'w', newline='') as csvfile:
    
    for i in range(100):
        num1 = random.uniform(3.5, 4.6)
        num2 = random.random()
        num3 = random.uniform(5, 2)
        x_list.append(num1)
        y_list.append(num2)
        z_list.append(num3)

        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([num1] + [num2] + [num3])

print(len(x_list))
print(len(y_list))
print(len(z_list))


