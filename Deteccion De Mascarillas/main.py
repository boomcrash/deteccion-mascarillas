import cv2
import os
import mediapipe as mp

mp_caras=mp.solutions.face_detection
etiquetas=["Con_mascarilla","Sin_mascarilla"]

mascarilla=cv2.face.LBPHFaceRecognizer_create()
mascarilla.read("mascarilla_model.xml")
captura=cv2.VideoCapture(0)

with mp_caras.FaceDetection(min_detection_confidence=0.5) as caretas:
    while True:
        ret,frame=captura.read()
        if ret ==False:break
        frame=cv2.flip(frame,1)

        #detectar rostro
        height,width,_=frame.shape
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=caretas.process(frame_rgb)

        if results.detections is not None:
            for detection in results.detections:
                xmin = int(detection.location_data.relative_bounding_box.xmin*width)
                ymin = int(detection.location_data.relative_bounding_box.ymin*height)
                w = int(detection.location_data.relative_bounding_box.width*width)
                h = int(detection.location_data.relative_bounding_box.height*height)
                if xmin <0 and ymin<0:
                    continue
                try:
                    face_image = frame[ymin:ymin+h,xmin:xmin+w]
                    face_image = cv2.cvtColor(face_image,cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image,(72,72),interpolation=cv2.INTER_CUBIC)
                    cv2.imshow("face_image",face_image)
                    result = mascarilla.predict(face_image)
                        #cv2.putText(frame, "{}".format(result), (xmin, ymin - 5), 1, 1.3, (210, 124, 176), 1, cv2.LINE_AA)
                    if result[1] < 150:
                            color = (0, 255, 0) if etiquetas[result[0]] == "Con_mascarilla" else (0, 0, 255)
                            cv2.putText(frame, "{}".format(etiquetas[result[0]]), (xmin, ymin - 15), 2, 1, color, 1, cv2.LINE_AA)
                            cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2)
                except:
                    pass
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

        cv2.imshow("Frame",frame)
        tecla=cv2.waitKey(1)
        if(tecla==27):
            break




captura.release()
cv2.destroyAllWindows()