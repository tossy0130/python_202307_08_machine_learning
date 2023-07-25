import cv2

# =============== OpneCV テスト 01 外部カメラ（USBカメラ）起動

print("実行開始")

def main():
    
    # カメラの初期化
    cap = cv2.VideoCapture(0)
    
    # カメラが正しく初期化されたかチェック
    if not cap.isOpened():
        print("カメラが開きません。")
        return
    
    while True:
        
        # フレームをキャプチャします
        ret, frame = cap.read()
        
        # フレームの取得に成功したかチェック
        if not ret:
            print("フレームの取得に失敗")
            break
        
        # フレームを表示
        cv2.imshow("USBカメラ", frame)
        
        # 停止 tを、押すと停止
        if cv2.waitKey(1) & 0xFF == ord('t'):
            break
        
    # カメラのリソースを解放する
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()