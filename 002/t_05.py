import cv2
import dlib


# ================================== OpenCV 口を検出

def detect_faces_and_mouth(frame, detector, predictor):
    dets = detector(frame, 1)
    for det in dets:
        # 顔の領域を取得
        x, y, w, h = det.left(), det.top(), det.width(), det.height()

        # 顔の領域を矩形で描画
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 顔のランドマーク（目、鼻、口などの特徴点）を検出
        landmarks = predictor(frame, det)

        # 口の領域を取得
        mouth_x, mouth_y = landmarks.part(48).x, landmarks.part(48).y
        mouth_w, mouth_h = landmarks.part(54).x - mouth_x, landmarks.part(57).y - mouth_y

        # 口の領域を矩形で描画
        cv2.rectangle(frame, (mouth_x, mouth_y), (mouth_x+mouth_w, mouth_y+mouth_h), (0, 0, 255), 2)

    return frame

def main():
    # 学習済み顔検出器をダウンロード
    detector = dlib.get_frontal_face_detector()

    # 学習済みランドマーク検出器をダウンロード
    #predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    
    predictor = dlib.shape_predictor("../../venv/Lib/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("カメラを開けません。")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("フレームを取得できませ。")
            break

        frame_with_faces_and_mouth = detect_faces_and_mouth(frame, detector, predictor)

        cv2.imshow("顔と口の検出", frame_with_faces_and_mouth)

        if cv2.waitKey(1) & 0xFF == ord('t'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
