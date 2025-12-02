import cv2 as cv
import os
import time
from tqdm import tqdm
from glob import glob
from character_segmentation import segment
from segmentation import extract_words
from train import prepare_char, featurizer
import pickle
import multiprocessing as mp

model_name = '2L_NN.sav'
def load_model():
    location = 'models'
    if os.path.exists(location):
        model = pickle.load(open(f'models/{model_name}', 'rb'))
        return model
        
def run2(obj):
    word, line = obj
    model = load_model()
    # For each word in the image
    char_imgs = segment(line, word)
    txt_word = ''
    # For each character in the word
    for char_img in char_imgs:
        try:
            ready_char = prepare_char(char_img)
        except:
            # breakpoint()
            continue
        feature_vector = featurizer(ready_char)
        predicted_char = model.predict([feature_vector])[0]
        txt_word += predicted_char
    return txt_word


def run(image_path):
    # Read test image
    full_image = cv.imread(image_path)
    predicted_text = ''

    # Start Timer
    before = time.time()
    words = extract_words(full_image)       # [ (word, its line),(word, its line),..  ]
    pool = mp.Pool(mp.cpu_count())
    predicted_words = pool.map(run2, words)
    pool.close()
    pool.join()
    # Stop Timer
    after = time.time()

    # append in the total string.
    for word in predicted_words:
        predicted_text += word
        predicted_text += ' '

    exc_time = after-before
    # Create file with the same name of the image
    img_name = os.path.basename(image_path).split('.')[0]

    with open(f'output/text/{img_name}.txt', 'w', encoding='utf8') as fo:
        fo.writelines(predicted_text)

    return (img_name, exc_time)


if __name__ == "__main__":
    types = ['png', 'jpg', 'bmp']
    images_paths = []

    for t in types:
        images_paths.extend(glob(f'test/*.{t}'))

    before = time.time()

    for image_path in tqdm(images_paths, total=len(images_paths)):
        # نفذ الدالة اللي تستخرج النص
        result = run(image_path)   # نتيجتها (id, text)

        # اعرض فقط النص المستخرج بدون حفظه
        print("\n==============================")
        print(f"📌 Image: {image_path}")
        print("==============================")
        print(result[1])  # عرض النص فقط
        print("==============================\n")

    after = time.time()
    print(f"Total time to finish {len(images_paths)} images: {after - before}")
