from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import keras
import tensorflow as tf

# 模型手动下载然后放到目录C:\Users\用户名\.keras\models下，下载地址如下：
# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

# 载入模型并打印
model = VGG16()
print(model.summary())

# 加载一副测试图片
image = load_img("dog.jpg", target_size=(224, 224))

# 转为数组
image = img_to_array(image)

# 重塑成4D
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# 预处理图片
image = preprocess_input(image)

# 预测
predict_result = model.predict(image)

# 解析预测结果
label = decode_predictions(predict_result)

# 打印出三个概率最大的分类
for idx in range(0, 3):
    print("类别：%s        概率：%0.4f"%(label[0][idx][1], label[0][idx][2]))

# 清理
keras.backend.clear_session()
tf.reset_default_graph()
