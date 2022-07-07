from PIL import Image
import os
import sys

# Remove the spaces from images names
def remove_spaces(images_input_dir, all_files):
    for filename in all_files:
        os.rename(os.path.join(images_input_dir, filename), os.path.join(images_input_dir, filename.replace(' ', '')))


# Convert all images to '.png'
def covert_to_png(images_input_dir, images_output_dir, all_files, filetype):
    for filename in all_files:
        prefix = filename.split('.')[0]
        png_filename = prefix + '.png'
        if filetype != '*.png':
            im1 = Image.open(os.path.join(images_input_dir, filename))
            im1.save(os.path.join(images_output_dir, png_filename))

# So far, you read the images from 'images_data_set/' 
# and you add the no spaces png images to the folder 'images_data_set/data_set/'

images_input_dir = sys.argv[1] # dir_images2
images_output_dir = sys.argv[2] # dir_images
data_set_input_dir = sys.argv[3] # input directory
filetype = sys.argv[4] # image_type

all_files = [f for f in os.listdir(images_input_dir) if f.endswith(filetype[1:])]

remove_spaces(images_input_dir, all_files)

all_files2 = [f for f in os.listdir(images_input_dir) if f.endswith(filetype[1:])]
#print(all_files2)
#print(images_input_dir)
covert_to_png(images_input_dir, images_output_dir, all_files2, filetype)
