import cv2

# ======================== haarcascade_mcs_mouth.xml を使って、歯を抽出　（失敗）

def detect_teeth(camera_id=0):
    # Haar Cascade分類器を読み込む
   # cascade_file = './path_to_haarcascade_mcs_mouth.xml'  # ダウンロードしたファイルのパスを指定
   
    cascade_file = './haarcascade_mcs_mouth.xml'
    cascade = cv2.CascadeClassifier(cascade_file)

    # カメラキャプチャを開始
    cap = cv2.VideoCapture(camera_id)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 歯の物体検出を実行
        teeth = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # 歯を囲む矩形を描画
        for (x, y, w, h) in teeth:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 結果を表示
        cv2.imshow('Teeth Detection', frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('t'):
            break

    # 後処理
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_teeth()
