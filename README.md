# SocialDistancingAI
Using python, deep learning and computer vision to monitor social distancing.
Idea Credits: LandingAI

[![Watch the video](/images/cover.png)](https://www.youtube.com/watch?v=kxFwbn7Tap0&feature=emb_title)

# 
In a world recovering from the shock of COVID-19, it has become increasingly important to practice social distancing. A deep understanding of the pandemic shows how one person's negligence can cause widespread harm which will be hard to negate. So, using Computer Vision and Deep Learning might help monitor the practice of social distancing. 

Nowadays, social distancing along with other basic sanitary mesures are very important to keep the spread of the Covid-19 as slow as possible. But this project is only a proof of concept and was not made to be use to monitor social distancing in public or private areas because of ethical and privacy issues

Using a faster model in order to perform real-time social distancing analysis.
Use a model more robust to occlusions.
Automatic calibration is a very well known problem in Computer vision and could improve a lot the bird eye view transformation on different scenes.

The application detects people who are close by and not adhering to the distancing norms and marks them in RED bounding boxes, signifying risk. Others, are in GREEN.

The Dashboard at the right, gives a visual representation of the data.
The number next to the GREEN and RED icons are the number of SAFE and RISK people. Whereas, the BLACK is the total number of people in the frame. 


# How to install?
It’s advisable to [make a new virtual environment](https://towardsdatascience.com/setting-up-python-platform-for-machine-learning-projects-cfd85682c54b) for this project and install the dependencies. Following steps can be taken to download get started with the project

## Clone the repository
```
git clone https://github.com/aqeelanwar/SocialDistancingAI.git
```
## Install required packages
The provided requirements.txt file can be used to install all the required packages. Use the following command

```
cd SocialDistancingAI
pip install –r requirements.txt
```

### Pre-Requisitos
- Python3
- yolov3.cfg
- coco.names
- yolov3.weights


## Run the project
```
cd SocialDistancingAI
python main.py --videopath "vid_short.mp4"
```

Running main.py will open a window of the first frame in the video. At this point the code expects the user to mark 6 points by clicking appropriate positions on the frame.

#### First 4 points:
The first 4 among the 6 required points are used to mark the Region of Interest (ROI) where you want to monitor. Moreover, the lines marked by these points should be parallel lines in real world as seen from above. For example these lines could be the curbs of the road.
These 4 points need to be provided in a pre-defined order which is following.

* __Point1 (bl)__: Bottom left
* __Point2 (br)__: Bottom right
* __Point3 (tl)__: Top left
* __Point4 (tr)__: Top right



#### Last 2 points:
The last two points are used to mark two points 6 feet apart in the region of interest. For example this could be a person's height (easier to mark on the frame)

The gif below will help understand the points better
![mark4points](images/mark4points.gif)
![mark4points](images/pedestrian.gif)

## How does it work?
The complete block diagram of the algorithm can be seen below.
![Block Diagram](images/block_diagram.png)

A detailed article on the working of this project can be found [here](https://medium.com/@aqeel.anwar/monitoring-social-distancing-using-ai-c5b81da44c9f)

__Idea credits: LandingAI__
