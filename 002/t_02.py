import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Reshape


def create_model():
    # MobileNetV2 のロード
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    
    # 追加レイヤーの定義
    x = base_model.layers[-1].output
    x = Conv2D(4, kernel_size=3, activation='sigmoid')(x)
    x = Reshape((4,))(x)
    
    # モデル構築
    model = Model(inputs=base_model.inputs, outputs=x)
    return model

### Function create_model 実行 
model = create_model()
# 学習済み重みロード
model.load_weights('path/to/your/weights.h5')


def detect_teeth(frame, model):
    # 画像をモデルに入力するために、前処理
    
    input_image = cv2.resize(frame, (224, 224))
    input_image = np.expand_dims(input_image, axis=0)
    # ピクセルを 0　～ 1　の範囲で正規化
    input_image = input_image / 255.0
    
    # 歯の検出
    predictions = model.predict(input_image)
    
    # バウンディングボックスを取得（モデルによって返された4つの値を解釈します）
    x, y, w, h = predictions[0] * frame.shape[1]
    # バウンディングボックスを描画
    cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)

    return frame


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("カメラを開けません。")
        return

    # ============= 
    while True:
        ret, frame = cap.read()

        if not ret:
            print("フレームを取得できません。")
            break

        # 歯の検出
        frame_with_detection = detect_teeth(frame, model)

        cv2.imshow("歯の検出", frame_with_detection)

        if cv2.waitKey(1) & 0xFF == ord('t'):
            break

    cap.release()
    cv2.destroyAllWindows()

# main 実行
if __name__ == "__main__":
    main()
    
    