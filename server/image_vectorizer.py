################################################################################################################################
# This file is used to extract features from dataset and save it on disc
# inputs: 
# outputs: 
################################################################################################################################

import random
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
import os
from scipy import ndimage
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pickle
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def create_inception_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Session() as sess:
    model_filename = os.path.join(
        'imagenet', 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))
  return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    print(image_data_tensor)
    bottleneck_values = sess.run(
      		bottleneck_tensor,
      		{image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

# Get outputs from second-to-last layer in pre-built model


boots_files = [
    'uploads/Boots/' + f for  f in os.listdir('uploads/Boots')
]
sandals_files = [ 
    'uploads/Sandals/' + f for f in os.listdir('uploads/Sandals')
]
shoes_files = [
    'uploads/Shoes/' + f for f in os.listdir('uploads/Shoes')
]
slippers_files = [
    'uploads/Slippers/' + f for f in os.listdir('uploads/Slippers')
]
# apparel_files = [
#     'uploads/apparel/' + f for f in os.listdir('uploads/apparel')
# ]

all_files = boots_files + shoes_files + slippers_files + sandals_files

# 获取所有文件
# shirt_files = [
#     'uploads/shirt/' + f for f in os.listdir('uploads/shirt')
# ]
# all_files = shirt_files

# 随机改变文件位置
random.shuffle(all_files)

num_images = 10000
neighbor_list = all_files[:num_images]
# 将文件序列写入到pickle文件中
with open('neighbor_list_recom.pickle','wb') as f:
        pickle.dump(neighbor_list,f)
print("saved neighbour list")

# 创建num_images个2048维向量
extracted_features = np.ndarray((num_images, 2048))
# Session 是 Tensorflow 为了控制,和输出文件的执行的语句. 运行 session.run() 可以获得你要得知的运算结果, 或者是你所要运算的部分.
sess = tf.Session()

## 通过模型获取graph
graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())	


# 对要处理的每一项进行操作
for i, filename in enumerate(neighbor_list):

    (shotname,extension) = os.path.splitext(filename);

    if extension == '.jpg' or extension == '.png' or extension == '.jpeg':
        # 读取图片文件
        image_data = gfile.FastGFile(filename, 'rb').read()
        # 获取图片特征
        features = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

        #添加到抽取特征列表中
        extracted_features[i:i+1] = features   
        
        if i % 250 == 0:
            print(i)
       
np.savetxt("../lib/saved_features_recom.txt", extracted_features)
print("saved exttracted features")






