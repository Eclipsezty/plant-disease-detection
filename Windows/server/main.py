import socket
import base64
import threading

import cv2

from utils import decode_image, parse_data, save_image, translate_CN
from prediction import bi_model_predict
import tensorflow as tf
import os

# 定义服务器地址和端口
address = ('192.168.0.102', 6666)

# 创建全局锁对象，用于保护对共享资源（如文件）的访问
file_lock = threading.Lock()


def handle_client(tcpClient, addr):
    """处理单个客户端请求"""
    print(f"客户端IP地址: {addr}")
    base64_data = ''

    while True:
        data = tcpClient.recv(1024)
        base64_data += str(data, 'utf-8').strip()
        if not data:
            break

    # 解析接收到的数据
    image_data_str, gps_data_str, phone_number = parse_data(base64_data)

    # 解码图像并保存到本地
    img = decode_image(image_data_str)
    image_path = save_image(img)

    # 调整图像大小并运行模型预测
    TFimg = tf.image.resize(cv2.imread(image_path), [200, 200])
    predicted_class, confidence = bi_model_predict(TFimg)

    # 根据预测结果生成响应信息
    if confidence < 60:
        result = "无法识别，请重新拍照"
    else:
        result = f"检测结果：{translate_CN(predicted_class)}  可信度：{confidence}%"

    print(result)

    # 将结果发送回客户端
    try:
        message = result
    except:
        message = "0"

    tcpClient.send((message + "\n").encode())
    print("P Complete")

    # 使用锁保护写入文件的操作
    with file_lock:
        with open("detection_results.csv", "a", encoding="utf-8") as f:
            f.write(f"{image_path},{translate_CN(predicted_class)},{confidence},{gps_data_str},{phone_number}\n")

    # 关闭与客户端的连接
    tcpClient.close()


def start_server():
    """启动服务器并监听客户端连接"""
    tcpServe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpServe.bind(address)
    tcpServe.listen(5)
    print("服务器启动，等待连接...")

    while True:
        # 接收客户端连接
        tcpClient, addr = tcpServe.accept()
        # 每个客户端连接启动一个新线程来处理
        client_thread = threading.Thread(target=handle_client, args=(tcpClient, addr))
        client_thread.start()

    tcpServe.close()


if __name__ == "__main__":
    start_server()
