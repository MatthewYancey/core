from PIL import Image
import os
import glob

pth_data_in = os.getcwd() + '/../data_raw/jpg/'
pth_data_out = os.getcwd() + '/../data_raw/jpg_resize/'

size_reduction = 0.05

for img_file in glob.glob(pth_data_in + '*.jpg'):
    print(img_file)
    img = Image.open(img_file)
    s = img.size
    new_img = img.resize((int(s[0]*size_reduction), int(s[1]*size_reduction)), Image.ANTIALIAS)
    new_img.save(pth_data_out + img_file.replace(pth_data_in, ''), 'JPEG')
