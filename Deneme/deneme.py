
import cv2
import numpy as np
import numpy as np
import cv2

img = cv2.imread(r'C:\Users\BULUT\Desktop\indir.jpg', 0)
img = cv2.medianBlur(img, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 3, 100, param1=100, param2=60, minRadius=10, maxRadius=100)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 2)

cv2.imshow('detected circles', cimg)
cv2.imshow('res', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

# construct the argument parse and parse the arguments
# loop over the images
image = cv2.imread(r'‪C:\Users\Durkan\Desktop\durkan.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
tight = cv2.Canny(blurred, 300, 400)
auto = auto_canny(tight)

# show the images
cv2.imshow("Original", image)
cv2.imshow("Edges", np.hstack([tight, auto]))
cv2.waitKey(0)

