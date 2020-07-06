# Lane Marking Segmentation

1. Download or clone this repo

2. Download the model (250 MB) from the following link https://drive.google.com/drive/folders/12WuBTK_ZOr3QpAOHDMXsGTkudBr59PXO?usp=sharing and extract it to the repo's directory on your hard disk

3. Images input to the neural network are resized to 500x300  


# To inference on:

* Screen Capture:
Run **python screen.py** to inference on screen capture images

Line 65 can be modified to desired width and height of the screen to capture

* Images:
Run **python image.py** to inference on provided sample image

To inference on custom images, place the image file in the './images' folder of the repo with the filename as image.png

* Videos:
Run **python video.py** to inference on the provided sample video (project_video.mp4)

The results will be saved in out.mp4 which will be written to the same directory as project_video.mp4

