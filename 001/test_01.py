import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================  OpneCV テスト 02 

img = cv2.imread('./img/sample.png')                  # read the image
cv2.imshow("result", img)                       # create window
cv2.waitKey(0)                                  # wait the input
cv2.destroyAllWindows()


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


imgplot = plt.imshow(img)


average_color_per_row = np.average(img, axis=0)
average_color = np.average(average_color_per_row, axis=0)
average_color = np.uint8(average_color)
print("RGBの平均値は", end="")
print(average_color)


maxcolor = max(average_color)
if (maxcolor == average_color[0]):
    maxcolorText = "赤"
elif (maxcolor == average_color[1]):
    maxcolorText = "緑"
else:
    maxcolorText = "青"

print("この写真は「" + maxcolorText + "系」の色が多い")