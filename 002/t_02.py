import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

### =========================== 歯を検出する

def load_model():
    model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'
    model = hub.load(model_handle).signatures['serving_default']
    return model

def preprocess_frame(frame):
    # 画像をRGBからBGRに変換（OpenCVはBGR形式を使用）
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 画像をuint8に変換
    frame_uint8 = frame_rgb.astype(np.uint8)
    return frame_uint8

def detect_teeth(frame, model):
    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                   'train', 'tvmonitor']

    class_id = class_names.index('person')  # 人間のクラスIDは16（詳細なクラスIDはclass_namesを参照）

    converted_img = tf.convert_to_tensor(frame)
    converted_img = tf.expand_dims(converted_img, 0)
    result = model(converted_img)
    boxes = result['detection_boxes'][0].numpy()
    classes = result['detection_classes'][0].numpy().astype(int)
    scores = result['detection_scores'][0].numpy()

    for i in range(len(scores)):
        if scores[i] > 0.4 and classes[i] == class_id:
            h, w, _ = frame.shape
            box = boxes[i] * np.array([h, w, h, w])
            box = box.astype(np.int32)
            cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_names[class_id]}: {scores[i]:.2f}', (box[1], box[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    model = load_model()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("カメラを開けません。")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("フレームを取得できません。")
            break

        frame = preprocess_frame(frame)
        frame_with_detection = detect_teeth(frame, model)

        cv2.imshow("人間の歯の検出", frame_with_detection)

        if cv2.waitKey(1) & 0xFF == ord('t'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
