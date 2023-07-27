import os
import numpy as np
import pydicom
import png
from skimage import exposure
import cv2
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
import med2image

def convert_to_png(file):

    image_dcm = pydicom.read_file(file)
    image = image_dcm.pixel_array

    cv2.imshow("dicom", image)
    cv2.waitKey(0)


    cv2.imwrite(file.replace('.dcm', '.png'), image)

    """
    ds = pydicom.dcmread(file)

    image = ds.pixel_array
    image = exposure.equalize_adapthist(image)

    cv2.imshow("dicom", image)
    cv2.waitKey(0)





#    if 'WindowWidth' in ds:
#        print('Dataset has windowing')
#    windowed = apply_voi_lut(ds.pixel_array, ds)

    shape = ds.pixel_array.shape

    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    # Write the PNG file
    with open(f'{file.strip(".dcm")}.png', 'wb') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, image_2d_scaled)
    """

home = 'C:/Users/user/Desktop'

os.chdir(home)

file = home + '/1-001.dcm'

convert_to_png(file)
