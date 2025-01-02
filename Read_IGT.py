import cv2, os, shutil, pyautogui, glob, time
from PIL import Image, ImageGrab
import pyperclip as pc
import numpy as np

# 画像白黒変換時の閾値を設定
Threshold      = 220

# https://ja.stackoverflow.com/questions/31588/opencv-%E3%81%A7%E7%94%BB%E5%83%8F%E3%82%92%E7%AD%89%E5%88%86%E3%81%97%E3%81%9F%E3%81%84
def Picture_Split(Picture_Name, X_Num, Y_Num):
    #ライブラリのインポート
    #import cv2

    img = cv2.imread(Picture_Name + ".png")
    height, width, channels = img.shape

    width_split = X_Num
    height_split = Y_Num
    new_img_height = int(height / height_split)
    new_img_width = int(width / width_split)

    for h in range(height_split):
        height_start = h * new_img_height
        height_end = height_start + new_img_height

        for w in range(width_split):
            width_start = w * new_img_width
            width_end = width_start + new_img_width

            file_name = Picture_Name+ "_" + str(h) + str(w) + ".png"
            clp = img[height_start:height_end, width_start:width_end]
            cv2.imwrite(file_name, clp)


# generateimage array
# https://python.joho.info/opencv/opencv-svm-digits-python/
def create_images_array(load_img_paths):
    #import glob
    #import cv2
    #import numpy as np

    imgs = []
    # 画像群の配列を生成
    for load_img_path in load_img_paths:
        # 画像をロード, グレースケール変換
        # 色反転, 32*32にリサイズ, 1次元配列に変換
        img = cv2.imread(load_img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.bitwise_not(img)
        img = cv2.resize(img, (32, 32))
        img = img.flatten()
        imgs.append(img)
    return np.array(imgs, np.float32)

# support vector machine
# https://python.joho.info/opencv/opencv-svm-digits-python/
def SVM():
    # 学習用の画像ファイルの格納先（0～9）
    LOAD_TRAIN_IMG0S_PATH = './Model/0/*'
    LOAD_TRAIN_IMG1S_PATH = './Model/1/*'
    LOAD_TRAIN_IMG2S_PATH = './Model/2/*'
    LOAD_TRAIN_IMG3S_PATH = './Model/3/*'
    LOAD_TRAIN_IMG4S_PATH = './Model/4/*'
    LOAD_TRAIN_IMG5S_PATH = './Model/5/*'
    LOAD_TRAIN_IMG6S_PATH = './Model/6/*'
    LOAD_TRAIN_IMG7S_PATH = './Model/7/*'
    LOAD_TRAIN_IMG8S_PATH = './Model/8/*'
    LOAD_TRAIN_IMG9S_PATH = './Model/9/*'

    # 作成した学習モデルの保存先
    SAVE_TRAINED_DATA_PATH = './Model/svm_trained_data.xml'
    
    # 検証用の画像ファイルの格納先（分割後IGTスクリーンショット）
    LOAD_TEST_IMGS_PATH = './Data/*'

    # 学習用の画像ファイルのパスを取得
    load_img0_paths = glob.glob(LOAD_TRAIN_IMG0S_PATH)
    load_img1_paths = glob.glob(LOAD_TRAIN_IMG1S_PATH)
    load_img2_paths = glob.glob(LOAD_TRAIN_IMG2S_PATH)
    load_img3_paths = glob.glob(LOAD_TRAIN_IMG3S_PATH)
    load_img4_paths = glob.glob(LOAD_TRAIN_IMG4S_PATH)
    load_img5_paths = glob.glob(LOAD_TRAIN_IMG5S_PATH)
    load_img6_paths = glob.glob(LOAD_TRAIN_IMG6S_PATH)
    load_img7_paths = glob.glob(LOAD_TRAIN_IMG7S_PATH)
    load_img8_paths = glob.glob(LOAD_TRAIN_IMG8S_PATH)
    load_img9_paths = glob.glob(LOAD_TRAIN_IMG9S_PATH)

    # 学習用の画像ファイルをロード
    imgs0 = create_images_array(load_img0_paths)
    imgs1 = create_images_array(load_img1_paths)
    imgs2 = create_images_array(load_img2_paths)
    imgs3 = create_images_array(load_img3_paths)
    imgs4 = create_images_array(load_img4_paths)
    imgs5 = create_images_array(load_img5_paths)
    imgs6 = create_images_array(load_img6_paths)
    imgs7 = create_images_array(load_img7_paths)
    imgs8 = create_images_array(load_img8_paths)
    imgs9 = create_images_array(load_img9_paths)
    imgs = np.r_[imgs0, imgs1, imgs2, imgs3, imgs4, imgs5, imgs6, imgs7, imgs8, imgs9]

    # 正解ラベルを生成
    labels0 = np.full(len(load_img0_paths), 0, np.int32)
    labels1 = np.full(len(load_img1_paths), 1, np.int32)
    labels2 = np.full(len(load_img2_paths), 2, np.int32)
    labels3 = np.full(len(load_img3_paths), 3, np.int32)
    labels4 = np.full(len(load_img4_paths), 4, np.int32)
    labels5 = np.full(len(load_img5_paths), 5, np.int32)
    labels6 = np.full(len(load_img6_paths), 6, np.int32)
    labels7 = np.full(len(load_img7_paths), 7, np.int32)
    labels8 = np.full(len(load_img8_paths), 8, np.int32)
    labels9 = np.full(len(load_img9_paths), 9, np.int32)
    labels = np.array([np.r_[labels0, labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9]])

    # SVMで学習モデルの作成（カーネル:LINEAR 線形, gamma:1, C:1）
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setGamma(1)
    svm.setC(1)
    svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
    svm.train(imgs, cv2.ml.ROW_SAMPLE, labels)

    # 学習結果を保存
    svm.save(SAVE_TRAINED_DATA_PATH)

    # 分割後IGTスクリーンショットを入力し、画像に書かれた数字を予測
    test_img_paths = glob.glob(LOAD_TEST_IMGS_PATH)
    test_imgs_temp = create_images_array(test_img_paths)
    test_imgs = np.r_[test_imgs_temp]
    svm = cv2.ml.SVM_load(SAVE_TRAINED_DATA_PATH)
    predicted = svm.predict(test_imgs)

    # 予測結果を表示
    #print("predicted:", predicted[1].T)

    # float型で結果が出るので、文字列に変換して結合
    predicted_join = ""
    for i in range(6):
        predicted_join += str(int(float(predicted[1][i])))

    # 予測結果をreturn
    return(predicted_join)

def main():
    # カレントディレクトリを指定 -> AutoHotKey側で制御
    #os.chdir('/Users/Zer0/Desktop/Melee_All_Target_Tests')

    # IGTをスクリーンショット→白黒変換
    pyautogui.press('printscreen')
    time.sleep(1)
    clipboard            = ImageGrab.grabclipboard()

    Time_Min             = clipboard.crop((2178, 109, 2261, 147))
    Time_Min.save('./Workspace/1_min.png')
    Time_Min_Temp        = cv2.imread("./Workspace/1_min.png")
    Time_Min_Gray        = cv2.cvtColor(Time_Min_Temp, cv2.COLOR_BGR2GRAY)
    ret,Time_Min_Binary  = cv2.threshold(Time_Min_Gray, Threshold, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('./Workspace/1_min_cv2out.png',Time_Min_Binary)
    shutil.copyfile("./Workspace/1_min_cv2out.png", "./Data/1_min_cv2out.png")
    Picture_Split('./Data/1_min_cv2out', 2, 1)
    os.remove('./Data/1_min_cv2out.png')

    Time_Sec             = clipboard.crop((2286, 109, 2369, 147))
    Time_Sec.save('./Workspace/2_sec.png')
    Time_Sec_Temp        = cv2.imread("./Workspace/2_sec.png")
    Time_Sec_Gray        = cv2.cvtColor(Time_Sec_Temp, cv2.COLOR_BGR2GRAY)
    ret,Time_Sec_Binary  = cv2.threshold(Time_Sec_Gray, Threshold, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('./Workspace/2_sec_cv2out.png',Time_Sec_Binary)
    shutil.copyfile("./Workspace/2_sec_cv2out.png", "./Data/2_sec_cv2out.png")
    Picture_Split('./Data/2_sec_cv2out', 2, 1)
    os.remove('./Data/2_sec_cv2out.png')

    Time_Msec            = clipboard.crop((2380, 118, 2444, 147))
    Time_Msec.save('./Workspace/3_msec.png')
    Time_Msec_Temp       = cv2.imread("./Workspace/3_msec.png")
    Time_Msec_Gray       = cv2.cvtColor(Time_Msec_Temp, cv2.COLOR_BGR2GRAY)
    ret,Time_Msec_Binary = cv2.threshold(Time_Msec_Gray, Threshold, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('./Workspace/3_msec_cv2out.png',Time_Msec_Binary)
    shutil.copyfile("./Workspace/3_msec_cv2out.png", "./Data/3_msec_cv2out.png")
    Picture_Split('./Data/3_msec_cv2out', 2, 1)
    os.remove('./Data/3_msec_cv2out.png')

    # SVMのモデルを作成し、画像認識してその結果を取得
    IGT= SVM()
    
    #print(IGT)

    # IGTをクリップボードにコピー
    pc.copy(IGT)

    # Ctrl + V -> AutoHotKey側で制御
    #pyautogui.hotkey('ctrl', 'v')

if __name__ == '__main__':
    main()
