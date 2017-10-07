import cv2
import scipy
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dense, Dropout
import matplotlib.cm as cm



print('LETS LOAD THE IMAGE\n\n')
file_path = 'images/udacity_sdc.png'
load_img = cv2.imread(file_path)
gray_img = cv2.cvtColor(load_img, cv2.COLOR_BGR2GRAY)
resize_img = scipy.misc.imresize(gray_img, 0.3)
resize_img = resize_img.astype('float32')/255
plt.imshow(resize_img, cmap = 'gray')
plt.show()

print('\n\n\_______________________________________________________\n\n')
print('NOW LETs SPECIFY THE FILTERS\n\n')

filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])


filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = [filter_1, filter_2, filter_3, filter_4]

#visualize all the filters
fig = plt.figure(figsize=(10,5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks = [])
    ax.imshow(filters[i], cmap = 'gray')
    ax.set_title('Filters %s' %str(i+1))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if filters[i][x][y]<0 else 'black')
  


print('\n______________________________________________________________\n\n')
          
            
#Visualize the Map for each filter
print('NOW VISUALIZE THE MAPS FOR EACH FILTER\n'  )

print("now lets design a model\n")
plt.imshow(resize_img, cmap = 'gray')

model = Sequential()
model.add(Convolution2D(1,(4,4), activation = 'relu', input_shape = (resize_img.shape[0], resize_img.shape[1],1)))
#apply convolutional filters and produce output
# apply convolutional filter and return output
def apply_filter(img, index, filter_list, ax):
    # set the weights of the filter in the convolutional layer to filter_list[i]
    model.layers[0].set_weights([np.reshape(filter_list[i], (4,4,1,1)), np.array([0])])
    # plot the corresponding activation map
    ax.imshow(np.squeeze(model.predict(np.reshape(img, (1, img.shape[0], img.shape[1], 1)))), cmap='gray')

# visualize all filters
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))

# visualize all activation maps
fig = plt.figure(figsize=(20, 20))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    apply_filter(resize_img, i, filters, ax)
    ax.set_title('Activation Map for Filter %s' % str(i+1))


