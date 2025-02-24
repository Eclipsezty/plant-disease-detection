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
    """解析数据：图片数据|GPS数据|电话号码|面积"""
    try:
        parts = base64_data.split('|')
        if len(parts) != 4:
            print("Error: Invalid data format")
            return None, None, "Unknown", 0
            
        image_data_str, gps_data_str, phone_number, area_str = parts
        
        # 解析GPS数据
        try:
            latitude, longitude = gps_data_str.split(',')
            latitude = float(latitude)
            longitude = float(longitude)
            print(f"Received GPS location - Latitude: {latitude}, Longitude: {longitude}")
        except ValueError:
            print("Error parsing GPS data")
            gps_data_str = "0.0,0.0"
            
        # 解析面积数据
        try:
            area = float(area_str)
            if area < 0.1 or area > 100:
                print("Warning: Area out of valid range")
                area = max(min(area, 10000), 0.0)
        except ValueError:
            print("Error parsing area data")
            area = 0
            
        print(f"Received Phone Number: {phone_number}")
        print(f"Received Area: {area}")
        
        return image_data_str, gps_data_str, phone_number, area
    except Exception as e:
        print(f"Error parsing data: {str(e)}")
        return None, None, "Unknown", 0

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
