import base64
from PIL import Image
from io import BytesIO
import os
from config import IMAGE_STORE_PATH

# 解码图像
def decode_image(image_data_str):
    img = base64.b64decode(image_data_str)
    image_data = BytesIO(img)
    im = Image.open(image_data)
    return im

# 解析数据
def parse_data(base64_data):
    try:
        image_data_str, gps_data_str, phone_number = base64_data.split('|')
        latitude, longitude = gps_data_str.split(',')
        latitude = float(latitude)
        longitude = float(longitude)
        print(f"Received GPS location - Latitude: {latitude}, Longitude: {longitude}")
        print(f"Received Phone Number: {phone_number}")
        return image_data_str, gps_data_str, phone_number
    except ValueError:
        print("Error parsing data")
        return None, None, "Unknown"

# 保存图像
def save_image(im):
    # 确保保存路径存在，如果不存在则创建
    if not os.path.exists(IMAGE_STORE_PATH):
        os.makedirs(IMAGE_STORE_PATH)

    # 根据已有文件生成新的文件名
    existing_files = [f for f in os.listdir(IMAGE_STORE_PATH) if f.startswith("ImageData") and f.endswith(".jpg")]
    imageNum = (
            max([int(f.split("ImageData")[1].split(".")[0]) for f in existing_files], default=0) + 1
    )
    img_path = os.path.join(IMAGE_STORE_PATH, f"ImageData{imageNum}.jpg")

    # 保存图像
    im.save(img_path)
    return img_path

# 英文到中文翻译
def translate_CN(eng):
    if eng == "Rice___Brown_Spot":
        return "褐斑病"
    elif eng == "Rice___Healthy":
        return "健康"
    elif eng == "Rice___Leaf_Blast":
        return "叶瘟"
    elif eng == "Rice___Neck_Blast":
        return "穗颈瘟"
    else:
        return "未知疾病"
