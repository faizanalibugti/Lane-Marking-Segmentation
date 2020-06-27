import os
import sys
import cv2
from utils.preprocessing import *
from network.lane_segmentator import Segmentator
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from utils.base_util import Timer, file_list
import scipy.misc
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_gpu', 1,
                     'Number of GPUs to use.')

model_dir = './model'
image_size = (1080, 720)
data_dir = './images'

sess = tf.Session()
input_image = tf.placeholder(dtype=tf.float32, shape=[None, image_size[1], image_size[0], 3])

segmentation = Segmentator(
    params={
        'base_architecture': 'resnet_v2_50',
        'batch_size': 1,
        'fine_tune_batch_norm': False,
        'num_classes': 2,
        'weight_decay': 0.0001,
        'output_stride': 16,
        'batch_norm_decay': 0.9997
    }
)

logits = segmentation.network(inputs=input_image, is_training=False)

predict_classes = tf.expand_dims(
    tf.argmax(logits, axis=3, output_type=tf.int32),
    axis=3
)

variables_to_restore = tf.contrib.slim.get_variables_to_restore()
get_ckpt = tf.train.init_from_checkpoint(
    ckpt_dir_or_file='./model',
    assignment_map={v.name.split(':')[0]: v for v in variables_to_restore}
)

print('Model restored successfully!')
sess.run(tf.global_variables_initializer())


print('[Inferencing on image...]')

image = cv2.imread('./images/image.png')
#image_data = np.array(Image.open(image).resize(image_size, Image.ANTIALIAS)).astype(np.float32)
image_data = scipy.misc.imresize(image, [720, 1080])
image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
start = time.time()
predictions = sess.run(
    predict_classes,
    feed_dict={
        input_image: np.expand_dims(image_data, 0)
    }
)
red_image = np.transpose(np.tile(predictions[0], [1, 1, 3]), [2, 0, 1])
red_image[0] = red_image[0] * 255
red_image[1] = red_image[1] * 0
red_image[2] = red_image[2] * 0
red_image = np.transpose(red_image, [1, 2, 0])
# overlay = (red_image * 0.4 + i * 0.6).astype(np.uint8)
overlay = (red_image * 0.4 + image_data * 0.6).astype(np.uint8)
end = time.time() - start
print('Infernce time: {} seconds'.format(end))

plt.imshow(overlay)
plt.show()