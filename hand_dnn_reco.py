import numpy as np
import scipy
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from tf_utils import predict

## PUT IMAGE NAME ## 
my_image = "Lin_five.jpg"
## END IMAGE NAME ##

# Open trained model
parameters = pickle.load(open('handpk','rb'))

# We preprocess your image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
plt.show()