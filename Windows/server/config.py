# 配置文件，存储常量、路径等
MODEL_PATH_1 = "F:/Study/FYP/training/models/Rice-InceptionResNetV2_NoPool_e30.h5"
MODEL_PATH_2 = "F:/Study/FYP/training/models/Rice-InceptionResNetV2_NoPool_e30.h5"
IMAGE_PATH = "F:/Study/FYP/training/Image/"
RESULTS_CSV = "detection_results.csv"
IMAGE_STORE_PATH = "server/image_store/"

# 不同病害对应的农药推荐
MEDICINE_RECOMMENDATIONS = {
    'Rice___Brown_Spot': {
        'medicine': '井冈霉素',  # 推荐农药
        'base_dosage': 25,      # 每亩基础用量(ml)
        'max_area': 1000,        # 最大处理面积(亩)
        'min_area': 0         # 最小处理面积(亩)
    },
    'Rice___Leaf_Blast': {
        'medicine': '稻瘟灵',
        'base_dosage': 30,
        'max_area': 1000,
        'min_area': 0
    },
    'Rice___Neck_Blast': {
        'medicine': '稻瘟灵',
        'base_dosage': 35,
        'max_area': 1000,
        'min_area': 0
    },
    'Rice___Healthy': {
        'medicine': '无需用药',
        'base_dosage': 0,
        'max_area': 1000,
        'min_area': 0
    }
}