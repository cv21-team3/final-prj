# final-prj

## Removal of Background People from Crowded Scenery Image Using Target Detection and Refilling

We present an application of human detection and inpainting to automatically remove unwanted background person(s) in photographs of crowded scenery images. We experiment with various object detection and targeting algorithms to optimally detect and distinguish between photograph target person and removal target(person(s) who has to be removed), produce individual masks to minimize masked background regions, as well as selecting neighborhood pixels to produce realistic refilling to cover removed pixels. The final pipeline is to be able to transform the original crowded photograph to an alone photograph of the target person and distribute the application for ordinary usages. Furthermore, we expect an application to videos.

### Report
[Proposal](report/CV21_Team3_Proposal.pdf)
[Final Report](report/CV21_Team3_Final.pdf)
[Presentation](report/CV21_Team3_Presentation.pptx)

### Set up
First, you have to install Python3.6 on your computer. Since Python versions higher than 3.6 are not compatible with ```neuralgym```, you must have 3.6 installed. Then, install required libraries with the following command.
  
```pip3 install -r requirements.txt```
  
You could have problems with installing ```neuralgym```. Then, enter the following command.
  
```pip install git+https://github.com/JiahuiYu/neuralgym```
  
You are now ready to run our inpainting codes! Use the following command to run.  

```python main.py --video data/video/8.mp4 --output data/video/result.mp4 --checkpoint model_logs/release_places2_256_deepfill_v2 --method flow```  
  
The options are as follows:  
```--image```: Path to the input image.  
```--video```: Path to the input video. Either one of the ```--image``` or ```--video``` must be set.  
```--mask```: Path to the external mask. You don't have to modify this option.   
```--output```: Path at which you would like to put the results.   
```--checkpoint_dir```: Path to the checkpoint. You don't have to modify this option.  
```--method```: Video inpainting method; ```naive``` for the naive inpainting, ```classical``` for the improved video inpainting with classical frame inpainting, ```flow``` for the improved inpainting with deep learning inpainting.  
  
It takes about 90 minutes to inpaint an image with the classical method and up to 2 hours to inpaint a video. 



