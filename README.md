# SBEP

This repository constains the code for Muñoz Sánchez et al. "Scenario-based Evaluation of Prediction Models for
Automated Vehicles" accepted for publication in ITSC 2022. At the moment the tags are derived using individual trajectories only. In the future this will be extended to more complex scenarios (e.g. pedestrian crossing at non-designated crossing).

## Installation ##

### Cloning ###

When cloning this repository, clone the submodules as well, which can be done with:

```
git clone --recurse-submodules <repository cloning URL>
```

### Environment Setup ###

It is advisable to create a virtual environment. You may use your preferred environment (e.g. conda, virtualenv...). The
code in this repository was created using Python 3.8.10. Once that is created and activated, install requirements with

```
pip install -r requirements.txt
```

Next, make sure to modify ```WaymoClassificaiton/config.py``` to include your desired paths and other configurations (
e.g. batch size when reading data).

## Tagging ##

If everything went well, you are ready to produce tags for each trajectory simply running

```
python WaymoClassification/scenario_classifier.py
```

**NOTE:** This will create a separate file for each scene. If running on the entire dataset, it will take up a lot of
space.
Don't forget to update your desired paths in ```config.py```.

Now one can see how many tags are assigned to different trajectories. To do so, run
```python WaymoClassification/tag_counter```, which will print the counts both as output, and saved to file wherever it
was specified in the configuration.

## Predictions ##

Now we can make predictions with different models.

- To make simple constant velocity predictions, run ```python WaymoClassification/prediction/predictor_cv.py```.
- To make LSTM predictions, run ```python WaymoClassification/prediction/predictor_lstm_per_agent.py```.
- To make predictions with MotionCNN, first follow instructions to prerender the data. We are not re-training the model,
  so it suffices to only pre-render the test split using the ```prerender.py``` script found within the MotionCNN
  submodule: <br> ```python prerender.py ``` situated in the folder `MotionCNN/waymo_motion_prediction_orig` (which is an older verison, adapted for our purposes). 
<br>Then, run ```python predictor_motionCNN.py``` found in `WaymoClassification/prediction/`. Don't forget to add whatever `OUT_FOLDER` you used during the prerender in `config.py`. 
<br>For the predictions, we used a pre-trained xception71 model that can be downloaded from the original repository: https://github.com/kbrodt/waymo-motion-prediction-2021/releases/tag/0.1
<br>Finally, to make the predictions with this model, run `python predictor_motionCNN.py` in `WaymoClassification/prediction/` .

## Ground Truth ##

At the moment predictions are only made if the road user is visible in the scene at prediction time. In the future, we
might one to assess a model's capabilities to predict the trajectory of momentarily disappearing objects. To that end,
it will be convenient to have the ground truth states saved separately instead of saving them with the predictions, to
easily load them per scene. You can do so running 
```python gt_extractor.py``` in `WaymoClassification`

## Evaluation ##
Finally, you are ready to check the overall accuracies, and additionally bucket them per tag. To do so, run ```python evaluate.py``` in `WaymoClassification`, and the accuracies will be saved in whatever path you specified for this purpose in `config.py`. Now you can use your preferred plotting or visualization tool to inspect the results. 