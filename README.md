# final-prj

## Removal of Background People from Crowded Scenery Image Using Target Detection and Refilling

We present an application of human detection and inpainting to automatically remove unwanted background person(s) in photographs of crowded scenery images. We experiment with various object detection and targeting algorithms to optimally detect and distinguish between photograph target person and removal target(person(s) who has to be removed), produce individual masks to minimize masked background regions, as well as selecting neighborhood pixels to produce realistic refilling to cover removed pixels. The final pipeline is to be able to transform the original crowded photograph to an alone photograph of the target person and distribute the application for ordinary usages. Furthermore, we expect an application to videos.

### Set up
```pip3 install -r requirements.txt```