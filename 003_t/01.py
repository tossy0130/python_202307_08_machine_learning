import sys
# import face_recognition
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import glob
# import dlib


# ============ 外部カメラ　取得して　表示
def main():
    
    video_capture = cv2.VideoCapture(0)
    
    
    if not video_capture.isOpened():
        print("カメラが開きません。")
        return
    
    while True:
        # ビデオの単一フレームを取得
        ret, frame = video_capture.read()

        # 結果をビデオに表示（エラー対応）
        if ret:
            cv2.imshow('Video', frame)
        else:
            print('retry')
            
        face_locations = face_recognition.face_locations(frame)
        # ===
        for(top, right, bottom, left) in face_locations:
            # 顔領域に枠を描画する
            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
        

        # t キーで終了
        if cv2.waitKey(1) & 0xFF == ord('t'):
            break
        
    # カメラのリソースを解放する
    video_capture.release()
    cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    main()