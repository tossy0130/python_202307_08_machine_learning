import cv2
from face_recognition.api import load_image_file, face_locations, face_encodings, compare_faces, face_distance
import os

def register_known_faces(known_faces_folder):
    # あらかじめ登録していた顔の画像を読み込んでデータベースを作成
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(known_faces_folder):
        image_path = os.path.join(known_faces_folder, file_name)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(file_name.split('.')[0])  # 画像のファイル名から拡張子を除いた名前を取得

    return known_face_encodings, known_face_names

def face_detection_and_recognition(camera_id=0, known_faces_folder='faces_dir'):
    # あらかじめ登録していた顔のデータベースを作成
    known_face_encodings, known_face_names = register_known_faces(known_faces_folder)

    # カメラキャプチャを開始
    cap = cv2.VideoCapture(camera_id)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # グレースケールに変換
        rgb_frame = frame[:, :, ::-1]

        # 顔の検出
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 顔の特徴量とデータベースを比較し、登録済みの顔かどうかを判定
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # 顔を囲む矩形を描画
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # 顔の名前を描画
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # 結果を表示
        cv2.imshow('Face Detection and Recognition', frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 後処理
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_detection_and_recognition()
