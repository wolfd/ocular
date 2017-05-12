---
layout: post
title: What's the deal?
---

You may have noticed that this repo doesn't have much in terms of "good" "structured" "working" "code", which are terms that I think should describe most code the people rely on.
Rather, this has a bunch of experiments that I've used to learn how to implement visual odometry. Here's the breakdown:

### Original ROS pipeline 
There are a few files (in `scripts`) that are for the original ROS implementation, and despite how well the code is broken up into managable bits, the implementation doesn't work.
The point of the ROS parts was to learn all of the details of how monocular visual odometry would work, doing a number of the parts by hand.

### Newer, shorter, "working" let's-just-use-OpenCV monocular version
The `scripts/mono.py` file and the files it references are the newer experiment. In this one I use `findEssentialMat` and `recoverPose` to do the heavy lifting.

### Newer, shorter, "not-working" stereo version
The `scripts/stereo.py` file shows how I was trying to implement stereo visual odometry before I gave up and used the monocular version.
I used `StereoSGBM` to create a disparity map, `reprojectImageTo3D` to create a 3D point map, and SIFT + `BFMatcher`'s `knnMatch` + Lowe Ratio Test to find correspondences frame to frame with the left camera.
Then it uses those matched keypoints to get corresponding 3D coordinates, which it then passes to a rigid transform solver...
The rigid transform solver tries its best, but fails. See the [project stories post]({{ "/2017/05/12/stories" | prepend: site.baseurl }}) for more info on that.
