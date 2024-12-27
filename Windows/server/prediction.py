import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import MODEL_PATH_1, MODEL_PATH_2

# 加载模型
def load_models():
    my_model = keras.models.load_model(MODEL_PATH_1)
    my_model2 = keras.models.load_model(MODEL_PATH_2)
    return my_model, my_model2

# 进行预测
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    class_names = ['Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Leaf_Blast', 'Rice___Neck_Blast']
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# 双模型预测
def bi_model_predict(img):
    my_model, my_model2 = load_models()
    predicted_class, confidence = predict(my_model, img)
    if confidence >= 60:
        return predicted_class, confidence
    else:
        return predict(my_model2, img)
