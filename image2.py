import cv2

file_path = './haarcascade_frontalface_default.xml'
classifier = cv2.CascadeClassifier(file_path)
lena_path = 'lena.png'
lena_read = cv2.imread(lena_path)
cam = cv2.VideoCapture(0)
while True:
    succes,img = cam.read()
    
    detect = classifier.detectMultiScale(
    img,
    minSize = (200,200), 
    maxSize = (1000,1000)       
    )

    for x, y, w, h in detect:
            cv2.rectangle(
            img,
            (x,y),
            (x+w, y+h),     
            (0, 255, 0),   
            3,              
)

    cv2.imshow('Label Window', img)
    if cv2.waitKey(1) and 0xFF == ord('O'):
        break