# drawEmoji
This program supports drawing sth in webcam and change the drawing to emoji.
The functions implemented by opencv, mediapipe, torch.

## 1. Overview
<img width="1392" alt="스크린샷 2023-05-27 오후 4 10 45" src="https://github.com/dripdropdr/drawEmoji/assets/81093298/2b26451e-f073-4326-805f-682a6f4e7e7e">
<img width="1392" alt="스크린샷 2023-05-27 오후 4 11 08" src="https://github.com/dripdropdr/drawEmoji/assets/81093298/9bca9aad-9849-4cb9-a705-7cc994c53441">
<img width="1348" alt="스크린샷 2023-05-27 오후 4 15 28" src="https://github.com/dripdropdr/drawEmoji/assets/81093298/c7177131-2931-488f-9c0c-f36ae61e913f">

- Webcam drawing
- doodle recognition
- Image on frame

## 2. Method

In Webcam Drawing, I referred the [Air-Canvas-Projects](https://github.com/infoaryan/Air-Canvas-project).    
After perceiving the position of index finger, stacked it and visualize on each frame.

The program shows 2 windown, pained window and webcam window. If user draw sth, the virtual drawing is shown on painted window, too.   
I exploited this input objects of classifier after cropping and resizing.   

Drawing classifier is ResNet18. I tunned this network with CrossEntrophy loss, Adamw optimizer, CosineAnnealing scheduler, and some data transform on 22epochs. The detailed information of this is in model.py.  

I trained the classifier using [quick-draw](https://github.com/googlecreativelab/quickdraw-dataset/tree/master) dataset. Quick-Draw dataset is offered by Google, composed to 384 drawing classes.   
Only 45 classes and 3000 data point by class of this dataset is used for drawEmoji. You can see exploited classes and their number in dataset.py label2id.   
The datasets is offered by csv. So I pre-processed this. The detail of eda is in eda.py.   

In programm running, I got prediction result from the classifier and call corresponding emoji image with random position.   
I collected the matched emoji image with drawing from [emojipedia](https://emojipedia.org/).   
Finally, the program put these emojis on frame and visualize it.   

Under is overall steps:   

**Steps**
1. Set painted window and webcam both.
2. Hand Tracking to get point of index finger.
3. Stack points and draw them on iterative webcam frame.
4. If user directs clear button, all stacked points and drawing are cleared.
5. If user directs emoji button, we can get virtual drawing objects from the painted window. It is processed and inputted to classifier.
6. When classifier omitted the output class, corresponding emoji is shown on frame with random position.
7. If user directs emoji clear button, all emoji is disappeared.


## 3. Installation
  ```
$ conda create -n emoji python=3.7
$ pip install -r requirements.txt
  ```

Download [this trained weights](https://drive.google.com/file/d/1Xwkc76gcH05-ojBetPWYWcdOhtITtf2_/view?usp=sharing), and put this file at output/emoji folder.


```
$ python main.py
```
Run the program

## 4. Limitations
- Accuracy of classifier -> need more training
- Emoji image background -> transperancy reflection
   
## 5. Reference
https://github.com/infoaryan/Air-Canvas-project/tree/master
https://github.com/googlecreativelab/quickdraw-dataset/tree/master
https://emojipedia.org/
