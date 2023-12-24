import numpy as np
import torch
from torchvision import transforms
from model.CNN import CNN
import cv2
import time
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('video', None, 'path to input video')

def main():
    label =['man', 'woman']
    print(label)
    label = np.array(label)

    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((54,54)),transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    model = CNN(2)
    model.load_state_dict(torch.load('./weight/CNN_54lite_2.pt', map_location = 'cpu'))
    model.eval()
    # atch. I installed OpenCV3.0 and everything is running as per expec

    #cv2.data.haarcascades +
    face_classifier = cv2.CascadeClassifier("./model/haarcascade_frontalface_default.xml")

    video_path = FLAGS.video
    print('hello')
    if(video_path == '0'):
        vid = cv2.VideoCapture(0)
        print("Video from Webcam", video_path )
    else:
        vid = cv2.VideoCapture(video_path)
        print("Video from: ", video_path )
        
    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        #print(frame.shape)
        #frame = cv2.flip(frame,1) #480*640 -> 1280 x 720
        frame = cv2.resize(frame, (640,480), interpolation=cv2.INTER_NEAREST)
    #     gray =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = do_GaussianBlur(gray)
        #print(frame.shape)
        
        
        faces = face_classifier.detectMultiScale(frame,scaleFactor=2,minNeighbors=3)
        
        for (x, y, w, h) in faces:
            #print(f"{x},{y},{w},{h}----")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            x1=x
            y1=y
            x = int(x-0.3*w)
            y = int(y-0.3*h)
            w = int(1.4*w)
            h = int(1.4*h)
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #print(f"{x},{y},{w},{h}\n")
            #face_picture.append({int(abs(x)),int(abs(y)),w,h})
            #cv2.imshow("face", frame[int(abs(y)):int(y+h), int(abs(x)):int(x+w)])
            data = transform(frame[int(abs(y)):int(y+h), int(abs(x)):int(x+w)])
            data = torch.unsqueeze(data, dim = 0)
            pred = model(data)
            _, y = torch.max(pred, 1)
    #         print(f"{label[int(y)]} , {int(y)}")
            if int(y)==0: #boy
                cv2.putText(frame, label[int(y)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 3)
            else:         #girl    
                cv2.putText(frame, label[int(y)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 5)    

        #calculate fps
        new_frame_time = time.time()
        fps = 1/(new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"fps: {round(fps,1)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        
        #frame = cv2.resize(frame, (1920,1080), interpolation=cv2.INTER_NEAREST)
        #show vedio
        cv2.imshow("window",frame)
        #break condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass