import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img=cv2.imread("news.jpg")
gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray_img,
scaleFactor=1.05,
minNeighbors=5)
#scaleFactor means, loop search 5% down scaled image, bigger face
#small value, higher accuracy
#minNeighbors, how many neighbors to search around the window

for x, y, w, h in faces:
    img=cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

resized=cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
#print(faces)
cv2.imshow("Gray", resized)
cv2.waitKey(10000)
cv2.destroyAllWindows()
