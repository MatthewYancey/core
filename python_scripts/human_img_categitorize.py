# script for quickly categorizing images
import glob
import matplotlib.pyplot as plt
import matplotlib.image as image
from shutil import copyfile

path_img = '/mnt/data/Projects/up_image/data_raw/crop_springs/'
img_count = 0

for f in glob.glob(path_img + '*.jpg'):
    print(f)

    img = image.imread(f)
    plt.imshow(img, cmap='gray')
    plt.ion()
    plt.show()
    plt.pause(0.0001)

    # gets the user input
    cat = input('Category: ')

    if cat != '0':
        path_save = path_img + cat + '/' + f.replace(path_img, '')
        copyfile(f, path_save)

        img_count += 1
        print(img_count)
