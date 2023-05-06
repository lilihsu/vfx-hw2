# README
# Requirement

```
python== .8.10
numpy==1.24.2
opencv-python==4.7.0.72
Pillow==7.0.0
scipy==1.10.1
```
# Usage
Run following command to generate the panorama image:

```
python main.py -i PATH/TO/Image -o PATH/TO/OUTPUT
```
Arguments:
* -i [input directory]: Specifies the directory containing input images
* -o [output directoy]: Specifies the destination of the result 
* -f [focal_length]: Specifies the focal length of the camera
* -s [scale]: Specifies the scale for resizing the input images to save the time
* -b [blending method]: Specifies the blending methods, "linear" for linear blending and "poisson" for poisson blending

