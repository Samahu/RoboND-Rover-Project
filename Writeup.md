## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./images/rock_thresh.png
[image2]: ./images/map_40_percent.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

In the notebook and the perception.py I have add a function called rock_thresh() to identify rocks, the implementation is as follows:
```python
def rock_thresh(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([90, 100, 100])
    upper_yellow = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask
```
Below I show the result of applying rock_thresh to generate the mask that is then passed to cv2.bitwise_and along with the warped image to extract the rock:

![alt text][image1]

I have also modified the perspect_transform() function such that it would return a mask that applies the same perspective transform used to generate the warped image on an array of ones. This mask will be used later in the `process_image` step to correctly extract obstacles based on the intensity invert of the navigable threshold mask image.
```python
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:, :, 0]), M, (img.shape[1], img.shape[0]))
    return warped, mask
```

#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.

The following code block, lists the steps added to the `process_image` method such that it would extract navigable, obstacle and rock details from the transformed warped image that is observed by the rover. This is done the `color_thresh` method and the `rock_thresh` method. After that I tranform pixel coordinates into rover centered coordinates using the `rover_coords` method. Then I pass rover coords the `pix_to_world` method to get coords in the world/global space. These coords are written into the `data.worldmap` field and each info is mapped into a separate channel.

```python    
        # Let's create more images to add to the mosaic, first a warped image
    warped, mask = perspect_transform(img, source, destination)
        # Add the warped image in the upper right hand corner
    output_image[0:img.shape[0], img.shape[1]:] = warped
    
    idx = data.count
    xpos = data.xpos[idx]
    ypos = data.ypos[idx]
    yaw = data.yaw[idx]
    scale = 2 * dst_size
    world_size = data.worldmap.shape[0]
        
    navigable_thresh = color_thresh(warped)
    navigable_xpix, navigable_ypix  = rover_coords(navigable_thresh)
    navigable_x_world, navigable_y_world = pix_to_world(navigable_xpix, navigable_ypix,
                                                        xpos, ypos, yaw, world_size, scale)
    data.worldmap[navigable_y_world, navigable_x_world, 2] = 255
    
    
    
    obstacle_thresh = np.absolute(np.float32(navigable_thresh) - 1) * mask
    obstacle_xpix, obstacle_ypix  = rover_coords(obstacle_thresh)
    obstacle_x_world, obstacle_y_world = pix_to_world(obstacle_xpix, obstacle_ypix,
                                                      xpos, ypos, yaw, world_size, scale)
    data.worldmap[obstacle_y_world, obstacle_x_world, 0] = 255
    
    nav_pix = data.worldmap[:, :, 2] > 0
    data.worldmap[nav_pix, 0] = 0

    rock_threshed = rock_thresh(warped)
    rock_xpix, rock_ypix = rover_coords(rock_threshed)
    rock_x_world, rock_y_world = pix_to_world(rock_xpix, rock_ypix,xpos, ypos, yaw, world_size, scale)
    data.worldmap[rock_y_world, rock_x_world, :] = 255
```

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

Before I describe the `perception_step` I have updated the default color threshold cut off value to be as follows:
```python
def color_thresh(img, rgb_thresh=(160, 140, 130)):
```
This is to allow identifiying ground that has shadow cast upon as navigable terrain.


The following block contains the full for the perception step:
```python
# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # 1) Define source and destination points for perspective transform
    image = Rover.img
    bottom_offset = 6
    dst_size = 5
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                    [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                    [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                    [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                    ])
    # 2) Apply perspective transform
    warped, mask = perspect_transform(image, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable_threshed = color_thresh(warped)
    obstacle_threshed = np.absolute(np.float32(navigable_threshed) - 1) * mask
    rock_threshed = rock_thresh(warped)
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
    Rover.vision_image[: , :, 0] = 255 * obstacle_threshed
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
    Rover.vision_image[:, :, 1] = rock_threshed
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[: , :, 2] = 255 * navigable_threshed

    # 5) Convert map image pixel values to rover-centric coords
    obstacle_xpix, obstacle_ypix  = rover_coords(obstacle_threshed)
    navigable_xpix, navigable_ypix  = rover_coords(navigable_threshed)    
    rock_xpix, rock_ypix = rover_coords(rock_threshed)
    # 6) Convert rover-centric pixel values to world coordinates

    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw
    scale = 2 * dst_size
    world_size = Rover.worldmap.shape[0]

    obstacle_x_world, obstacle_y_world = pix_to_world(obstacle_xpix, obstacle_ypix,
                                                      xpos, ypos, yaw, world_size, scale)
    navigable_x_world, navigable_y_world = pix_to_world(navigable_xpix, navigable_ypix,
                                                        xpos, ypos, yaw, world_size, scale)
    rock_x_world, rock_y_world = pix_to_world(rock_xpix, rock_ypix,xpos, ypos, yaw, world_size, scale)

    # only accomdate mapping to when pitch and roll are small enough
    if angle_close_to_zero(Rover.roll) and angle_close_to_zero(Rover.pitch):
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] = 255
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        Rover.worldmap[rock_y_world, rock_x_world, :] = 255
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] = 255
    else:
        print("ignoring mapping")

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    dist, angles = to_polar_coords(navigable_xpix, navigable_ypix)
        # Rover.nav_dists = rover_centric_pixel_distances
    Rover.nav_dists = dist
        # Rover.nav_angles = rover_centric_angles
    Rover.nav_angles = angles
    
    return Rover
```
The code does pretty much the same as `process_image` in the notebook with one exception, which is it writes the resulting world coordinates to `Rover.worldmap` ONLY when both `pitch` and `roll` angles are close to zero. Angles are identified as close to zero using this method:
```python
def angle_close_to_zero(angle, epsilon = 0.5):
    return abs(angle) < epsilon or abs(360 - angle) < epsilon
```

At the end of the `perception` method I transform rover centered coordinates into rover centered polar coordinates and  then store the generated values within the `Rover` variable such that these values are accessible to the next step `decision`.

The following block contains the complete code for the modified `decision_step`:
The changes are as follows:
- reduce max_steer from 15 to 12
- introduce few helper fields `steer_bias`, `hold_throttle`, `average_vel`, `stuck_detected` and `stuck_timer_started`
- To make sure I traverese the map, I try to make the Rover stay close to the left side/wall. This is achieved by introducing a `steer_bias` value that helps to maintain `mean(Rover.nav_angles) < -12`. When this condition is maintained the `steer_bias` is gradually relaxed.
- Whenever `mean(Rover.nav_angles)` starts to go beyond [-12, -15] I release the throttle such that the Rover would slow a bit allowing it to better control the steering.
- Additionally I have defined a stuck detection mechansim to help getting the vehicle back on track by halting any control until the conditon of being stuck is cleared.

```python
# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    max_steer = 12    

    # introduce steer bias
    if not hasattr(Rover, 'steer_bias'):
        Rover.steer_bias = 0

    if not hasattr(Rover, 'hold_throttle'):
        Rover.hold_throttle = False

    if not hasattr(Rover, 'average_vel'):
        Rover.average_vel = 0

    if not hasattr(Rover, 'stuck_detected'):
        Rover.stuck_detected = False

    if not hasattr(Rover, 'stuck_timer_started'):
        Rover.stuck_timer_started = False

    # Detect if we are stuck
    a = 0.1
    Rover.average_vel = a * Rover.average_vel + (1 - a) * Rover.vel

    stuck_speed_threshold = 0.15 # m/s
    stuck_time_thresh = 5 # seconds

    if not Rover.stuck_detected:
        if abs(Rover.average_vel) < stuck_speed_threshold and not Rover.stuck_timer_started:
            Rover.stuck_timer_started = True
            Rover.stuck_timer = Rover.total_time

        if Rover.stuck_timer_started and (Rover.total_time - Rover.stuck_timer > stuck_time_thresh):
            Rover.stuck_detected = True

    if abs(Rover.average_vel) > stuck_speed_threshold:
            Rover.stuck_timer_started = False
            Rover.stuck_detected = False


    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:

        the_mean = np.mean(Rover.nav_angles * 180/np.pi)

        if the_mean > -12 and not Rover.stuck_detected:    # add bias once such that we keep to the left
            Rover.steer_bias = max_steer
            Rover.hold_throttle = True  # slow down until we get back on track
        elif the_mean < -15 and not Rover.stuck_detected:
            Rover.hold_throttle = True  # slow down until we get back on track
        else:
            Rover.steer_bias *= 0.99    # relax bias gradually
            Rover.hold_throttle = False # Release throttle

        print("the_mean = ", the_mean, ", the_bias = ", Rover.steer_bias)

        # Check for Rover.mode status
        if Rover.mode == 'forward':
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:
                # If mode is forward, navigable terrain looks good
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel and not Rover.hold_throttle:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- max_steer
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi) + Rover.steer_bias, -max_steer, max_steer)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- max_steer degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -max_steer # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi) + Rover.steer_bias, -max_steer, max_steer)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover
```

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: I run the simulator at a resolution of 1600x1200 using `Good` profile for the graphics.**

On most of the runs, the Rover is able to achieve 40% coverage of the map with +60% fidelity within 4-5 minutes as shown below:

![alt text][image2]

Here are some observations:
- The Rover is very conservative when appling the throttle. Due to this it takes much longer to get a full coverage of the map.
- The stuck detection mechanism is not always effective and can be improved.