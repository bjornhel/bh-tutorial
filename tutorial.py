# Dette tegnet brukes til å kommentere koden

import pydicom as dcm   # Importer modulen som heter pydicom og kall den dcm i koden slik at vi kan skrive dcm.read_file(.....) f.eks.
import numpy as np   # Numpy er den viktigste modulen for vitenskaplig python. Da får man tilgang til en numpy array som brukes til bilder, matriser ++
import pandas as pd  # Her får man tilgang til en pd.DaataFrame som brukes til å holde på tabeller med data av forskjellig type. F.eks data importert fra excel.
import matplotlib.pyplot as plt  # Dette brukes mye til å plotte data.
# import matplotlib as mpl  # alternativ måte å importere matplotlib på. Her importerer man flere biblioteker enn pyplot. Man kan nå samme funksjoner som over ved å skrive mål.pyplot osv...
# from matplotlib import pyplot as plt  # Alternativ måte å importere pyplotmodulen og kalle den plt.
import cv2  # Dette er et bibliotek for computer vision


def read_img_np():
    "This function reads a 16 bit uint 512, 512 image"
    pixels = np.fromfile('Data\\NE_1.raw', dtype='uint16')
    im = pixels.reshape(512,512)
    return im

def conv_16_8bit(im16):
    im8 = im16/65353*255
    im8 = np.uint8(im8)
    return im8


def draw_lines(lines, img, show=False):
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    dim = max(img.shape)
    line_coords = None
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + dim * (-b))
        y1 = int(y0 + dim * (a))
        x2 = int(x0 - dim * (-b))
        y2 = int(y0 - dim * (a))

        if line_coords is None:
            line_coords = [x1, y1, x2, y2]
        else:
            line_coords.append([x1, y1, x2, y2])

        if show:
            x1d = int(x0 + dim * (-b))
            y1d = int(y0 + dim * (a))
            x2d = int(x0 - dim * (-b))
            y2d = int(y0 - dim * (a))
            lines_img = cv2.line(color_img, (x1d, y1d), (x2d, y2d), (0, 0, 65535), 2)

    if show:
        cv2.imshow('Edge detection', color_img)
        cv2.waitKey()

    return line_coords

def main2():
    im = read_img_np()
    im8 = conv_16_8bit(im) # Convert images to 8 bit for edge detection.
    edges = cv2.Canny(im8, im8.min(), im8.max(), apertureSize=3)  # Find the edges in the image.
    lines = cv2.HoughLines(edges, 1, np.pi/1800, 100)
    line_coords = draw_lines(lines[0], im, True)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.imshow(im, cmap='gray')
    ax2.imshow(edges, cmap='gray')
    plt.show()

def main():
    met = dcm.read_file('Data\\IM32', stop_before_pixels=True) # Les DICOM Metadata
    dicom_file = dcm.dcmread('Data\\IM32')
    im = dicom_file.pixel_array + met.RescaleIntercept
    plt.imshow(im, cmap='gray', vmin=-30, vmax=30)  # Lag plottet (men ikke vis det enda).
    plt.show()  # Vis plottet





# Python begynner fra toppen og leser nedover. Den definerer funksjoner etterhvert som den går.
# Derfor må man ikke begynne programmet på toppen.
# Når man kjører python som et script settes __name__ variabelen til strengen '__main__'
# Denne brukes til å fortelle hvor python skal starte å kjøre koden på denne måten.
# Her kaller vi funksjonen main
if __name__ == '__main__':
    main()
    #main2()