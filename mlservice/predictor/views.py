from django.shortcuts import render
import joblib
import tensorflow as tf
from django.core.files.storage import default_storage
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from django.conf import settings
import os

# 모델 로드 (서버 시작 시 1번만)
model = load_model("mlservice/fruits_classifier_model.keras")

# 클래스 이름 매핑 (train_gen.class_indices 기반)
class_labels = {0: 'Apple 10', 1: 'Apple 11', 2: 'Apple 12', 3: 'Apple 13', 4: 'Apple 14', 5: 'Apple 17', 6: 'Apple 18', 7: 'Apple 19', 8: 'Apple 5', 9: 'Apple 6', 10: 'Apple 7', 11: 'Apple 8', 12: 'Apple 9', 13: 'Apple Braeburn 1', 14: 'Apple Core 1', 15: 'Apple Crimson Snow 1', 16: 'Apple Golden 1', 17: 'Apple Golden 2', 18: 'Apple Golden 3', 19: 'Apple Granny Smith 1', 20: 'Apple Pink Lady 1', 21: 'Apple Red 1', 22: 'Apple Red 2', 23: 'Apple Red 3', 24: 'Apple Red Delicious 1', 25: 'Apple Red Yellow 1', 26: 'Apple Red Yellow 2', 27: 'Apple Rotten 1', 28: 'Apple hit 1', 29: 'Apple worm 1', 30: 'Apricot 1', 31: 'Avocado 1', 32: 'Avocado Black 1', 33: 'Avocado Green 1', 34: 'Avocado ripe 1', 35: 'Banana 1', 36: 'Banana 3', 37: 'Banana 4', 38: 'Banana Lady Finger 1', 39: 'Banana Red 1', 40: 'Beans 1', 41: 'Beetroot 1', 42: 'Blackberrie 1', 43: 'Blackberrie 2', 44: 'Blackberrie half rippen 1', 45: 'Blackberrie not rippen 1', 46: 'Blueberry 1', 47: 'Cabbage red 1', 48: 'Cabbage white 1', 49: 'Cactus fruit 1', 50: 'Cactus fruit green 1', 51: 'Cactus fruit red 1', 52: 'Caju seed 1', 53: 'Cantaloupe 1', 54: 'Cantaloupe 2', 55: 'Carambula 1', 56: 'Carrot 1', 57: 'Cauliflower 1', 58: 'Cherimoya 1', 59: 'Cherry 1', 60: 'Cherry 2', 61: 'Cherry 3', 62: 'Cherry 4', 63: 'Cherry 5', 64: 'Cherry Rainier 1', 65: 'Cherry Rainier 2', 66: 'Cherry Rainier 3', 67: 'Cherry Sour 1', 68: 'Cherry Wax Black 1', 69: 'Cherry Wax Red 1', 70: 'Cherry Wax Red 2', 71: 'Cherry Wax Red 3', 72: 'Cherry Wax Yellow 1', 73: 'Cherry Wax not ripen 1', 74: 'Cherry Wax not ripen 2', 75: 'Chestnut 1', 76: 'Clementine 1', 77: 'Cocos 1', 78: 'Corn 1', 79: 'Corn Husk 1', 80: 'Cucumber 1', 81: 'Cucumber 10', 82: 'Cucumber 11', 83: 'Cucumber 3', 84: 'Cucumber 4', 85: 'Cucumber 5', 86: 'Cucumber 7', 87: 'Cucumber 9', 88: 'Cucumber Ripe 1', 89: 'Cucumber Ripe 2', 90: 'Dates 1', 91: 'Eggplant 1', 92: 'Eggplant long 1', 93: 'Fig 1', 94: 'Ginger Root 1', 95: 'Gooseberry 1', 96: 'Granadilla 1', 97: 'Grape Blue 1', 98: 'Grape Pink 1', 99: 'Grape White 1', 100: 'Grape White 2', 101: 'Grape White 3', 102: 'Grape White 4', 103: 'Grapefruit Pink 1', 104: 'Grapefruit White 1', 105: 'Guava 1', 106: 'Hazelnut 1', 107: 'Huckleberry 1', 108: 'Kaki 1', 109: 'Kiwi 1', 110: 'Kohlrabi 1', 111: 'Kumquats 1', 112: 'Lemon 1', 113: 'Lemon Meyer 1', 114: 'Limes 1', 115: 'Lychee 1', 116: 'Mandarine 1', 117: 'Mango 1', 118: 'Mango Red 1', 119: 'Mangostan 1', 120: 'Maracuja 1', 121: 'Melon Piel de Sapo 1', 122: 'Mulberry 1', 123: 'Nectarine 1', 124: 'Nectarine Flat 1', 125: 'Nut 1', 126: 'Nut 2', 127: 'Nut 3', 128: 'Nut 4', 129: 'Nut 5', 130: 'Nut Forest 1', 131: 'Nut Pecan 1', 132: 'Onion Red 1', 133: 'Onion Red Peeled 1', 134: 'Onion White 1', 135: 'Orange 1', 136: 'Papaya 1', 137: 'Passion Fruit 1', 138: 'Peach 1', 139: 'Peach 2', 140: 'Peach Flat 1', 141: 'Pear 1', 142: 'Pear 2', 143: 'Pear 3', 144: 'Pear Abate 1', 145: 'Pear Forelle 1', 146: 'Pear Kaiser 1', 147: 'Pear Monster 1', 148: 'Pear Red 1', 149: 'Pear Stone 1', 150: 'Pear Williams 1', 151: 'Pepino 1', 152: 'Pepper Green 1', 153: 'Pepper Orange 1', 154: 'Pepper Red 1', 155: 'Pepper Yellow 1', 156: 'Physalis 1', 157: 'Physalis with Husk 1', 158: 'Pineapple 1', 159: 'Pineapple Mini 1', 160: 'Pistachio 1', 161: 'Pitahaya Red 1', 162: 'Pomegranate 1', 163: 'Pomelo Sweetie 1', 164: 'Potato Red 1', 165: 'Potato Red Washed 1', 166: 'Potato Sweet 1', 167: 'Potato White 1', 168: 'Quince 1', 169: 'Quince 2', 170: 'Quince 3', 171: 'Quince 4', 172: 'Rambutan 1', 173: 'Raspberry 1', 174: 'Redcurrant 1', 175: 'Salak 1', 176: 'Strawberry 1', 177: 'Strawberry Wedge 1', 178: 'Tamarillo 1', 179: 'Tangelo 1', 180: 'Tomato 1', 181: 'Tomato 10', 182: 'Tomato 2', 183: 'Tomato 3', 184: 'Tomato 4', 185: 'Tomato 5', 186: 'Tomato 7', 187: 'Tomato 8', 188: 'Tomato 9', 189: 'Tomato Cherry Maroon 1', 190: 'Tomato Cherry Orange 1', 191: 'Tomato Cherry Red 1', 192: 'Tomato Cherry Red 2', 193: 'Tomato Cherry Yellow 1', 194: 'Tomato Heart 1', 195: 'Tomato Maroon 1', 196: 'Tomato Maroon 2', 197: 'Tomato Yellow 1', 198: 'Tomato not Ripen 1', 199: 'Walnut 1', 200: 'Watermelon 1', 201: 'Zucchini 1', 202: 'Zucchini dark 1'}


def predict_view(request):
    if request.method == "POST" and request.FILES.get("image"):
        # 업로드된 이미지 저장
        img_file = request.FILES["image"]
        file_path = default_storage.save("tmp/" + img_file.name, img_file)

        try:
            # 이미지 불러오기
            img_path = os.path.join(settings.MEDIA_ROOT, file_path)
            img = image.load_img(img_path, target_size=(100, 100))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # 예측
            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]

            return render(request, "result.html", {"prediction": predicted_class})
        
        finally:
            # 예측 완료 후 임시 파일 삭제
            if os.path.exists(img_path):
                os.remove(img_path)

    return render(request, "predict.html")


def home(request):
    return render(request, 'home.html')
