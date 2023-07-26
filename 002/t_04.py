import cv2
import dlib

### ====================== とりあえず　顔検出

def detect_faces(frame, detector):
    dets = detector(frame, 1)
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

def main():
    # 学習済み顔検出器をダウンロード
    detector = dlib.get_frontal_face_detector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("カメラを開けません。")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("フレームを取得できません。")
            break

        frame_with_faces = detect_faces(frame, detector)

        cv2.imshow("顔の検出", frame_with_faces)

        if cv2.waitKey(1) & 0xFF == ord('t'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
