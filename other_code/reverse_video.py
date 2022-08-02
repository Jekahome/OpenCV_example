import cv2
cap = cv2.VideoCapture("/container_data/source/video/Terminator.mp4")
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter("/container_data/source/video/Terminator_reversed2.avi",fourcc,fps,(int(width*0.5),int(height*0.5)))
print("No. of frames are :{}".format(frames))
print("FPS is :{}".format(fps))
frame_index = frames-1
if(cap.isOpened()):
    while(frame_index!=0):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
        ret,frame = cap.read()
        frame = cv2.resize(frame,(int(width*0.5),int(height*0.5)))
        out.write(frame)
        frame_index = frame_index-1
        if(frame_index%100==0):
            print(frame_index)
            
out.release()            
cap.release() 
cv2.destroyAllWindows()