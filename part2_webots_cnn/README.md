# Applied Robotics and AI Project – Part 2 (Assignments 3, 4, 5)
CIFAR-10 CNN + colour-object detection and classification in Webots (e-puck).

## 1. Folder layout
```
cnn/                       # trains and evaluates the network (run OUTSIDE Webots)
  common.py                #   class names, preprocessing shared by every script
  train_cnn.py             #   builds + trains the CNN, prints ALL layers, accuracy, confusion matrix
  predict_images.py        #   predictions on YOUR images (put >=3 photos in cnn/my_images/)
  optimise_model.py        #   TFLite quantisation + accuracy/size/latency benchmark
  inference_server.py      #   optional TCP server for offloading inference out of Webots
  my_images/               #   put your own test photos here
  results/                 #   generated: model, summary, plots, json, .tflite files
controllers/cnn_detector/  # the Webots controller (copy this folder into <your project>/controllers/)
  cnn_detector.py          #   state machine: SEARCH > ALIGN > APPROACH > CLASSIFY > RECOVER
  colour_utils.py          #   HSV colour blob detection (all colour ranges in ONE table)
  cnn_inference.py         #   Local / Remote / Null classifier back-ends
figures/make_figs.py       # regenerates the architecture / pipeline / state-machine diagrams
```

## 2. Setup
```
pip install -r requirements.txt
```
Webots runs controllers with the Python set in *Tools > Preferences > Python command*. Point it at the
interpreter that has `numpy`, `opencv-python` (and `tensorflow` or `tflite-runtime`) installed.
If you cannot install TensorFlow for Webots, use the **offload** mode in section 5.

## 3. Train and evaluate the CNN (brief outputs 1-3)
```
cd cnn
python train_cnn.py --epochs 60      # layers -> results/model_summary.txt, accuracy -> results/results.json
python predict_images.py             # your 3+ images -> results/own_image_predictions.png
python optimise_model.py             # -> results/optimisation.json + optimisation_chart.png + *.tflite
```
Copy the numbers into the highlighted `[ ... ]` fields of the report and slides.

## 4. Run the Webots demo (brief output 4)
1. Open your world with an **e-puck** (Part 1 world). Add the controller: set the robot's `controller` field to `cnn_detector`.
2. Add coloured objects (e.g. Solid + Box, 0.3 x 0.3 x 0.1 m blocks, `physics` on, random positions). For the
   CNN-classification demo, give one object a **texture** showing a CIFAR-10 class (car, ship, horse...) -
   plain coloured boxes are not CIFAR-10 objects, so their CNN label will be arbitrary (explained in the report).
3. Optional: increase the camera resolution in the e-puck's `camera_width` / `camera_height` fields
   (default 52 x 39; check the field names in your Webots version).
4. Press Play. Console prints `[detect] ...` lines and a final **DETECTION SUMMARY**.
5. Keys (click the 3D view first): `C` classify now, `P` save frame, `S` print summary.

Environment variables: `CNN_MODEL` (model path), `CNN_REMOTE` (`host:port`), `RUN_SECONDS` (default 180).

## 5. Offloading (optimisation option)
```
cd cnn && python inference_server.py --model results/tflite_dynamic.tflite     # terminal 1
CNN_REMOTE=127.0.0.1:5005   # set as environment variable for the Webots controller, then press Play
```

## 6. Hardware / simulation notes
* Robot: e-puck - two wheel motors (`left wheel motor`, `right wheel motor`, max 6.28 rad/s), 8 IR proximity sensors
  (`ps0`-`ps7`), forward camera `camera` (BGRA frames).
* Colour ranges (OpenCV HSV, H 0-179): red 0-10 and 170-179, green 40-85, blue 95-130, yellow 20-35. Tune in `colour_utils.py`
  for your lighting.
* CNN input: 32 x 32 RGB, raw 0-255 (the model contains its own `Rescaling(1/255)` layer).

## 7. Known limitations
* CIFAR-10 has 10 classes only; objects outside them are forced into the nearest class (use the confidence value).
* Colour cooldown identifies "already reported" by colour, not by position (no odometry/map).
* Low-resolution e-puck camera limits distance at which classification is reliable.
