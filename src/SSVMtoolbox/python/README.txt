SSVM-Python is a Python version of DSMI Lab's Smooth Support Vector Machine Toolbox
See: http://dmlab8.csie.ntust.edu.tw/download.html#toolbox

References:
Smooth Support Vector Machine is Prof.Lee's paper.
See: http://research.cs.wisc.edu/dmi/svm/ssvm/

Model Selection for Support Vector Machines via Uniform Design
See: http://140.118.155.137/%E8%80%81%E5%B8%AB%E8%AB%96%E6%96%87/%E6%9C%9F%E5%88%8A/J8_Model%20Selection%20for%20Support%20Vector%20Machines%20via%20Uniform%20Design.pdf

Uniform Design Tables
See: http://uic.edu.hk/isci/UniformDesign/UD%20Tables.html

Usage:
Put SSVM-Python on the PYTHON-PATH and install "Numpy"

=======================================================================================================
Training:
Trainer( data, label_pos )                      Instantiate the Trainer object with a data set.
Trainer.make( r = 1 , v = 1 )                   Setup the partitions to use for cross-validation.
Trainer.tune( c = 100, g = 0.1, k = 1, s = 0 )  Set parameters for model training.
Trainer.train()                                 Train
Trainer.save( model_name = model)               Save the trained model.
=======================================================================================================


#Training
1)Import the module
----------------------------------
from ssvm.trainer import Trainer

2)Create a Trainer object
----------------------------------
A Trainer object expects to be initialized with a 2D Numpy array representing 
containing the data. label_pos indicates which column is the label.

trainer = Trainer( data, label_pos )

3)Setup the Trainer
----------------------------------
r -> How much of the data to use for the reduced set
v -> How many folds to use for cross-validation

trainer.make( r = 1 , v = 1 )

4)Set Training parameters
----------------------------------
c = SVM penality
g = gamma parameter of the RBF kernel
k = linear vs nonlinear, 0 -> linear, 1 -> nonlinear
s = How to deal with multi-class classification, 0 -> One-Against-One, 1 -> One-Against-Rest

trainer.tune( c = 100, g = 0.1, k = 1, s = 0 )

5)Begin Training
----------------------------------
Use the train() method to begin training.
Which parameters are used during training depends on what was passed to the
tune() method earlier.

trainer.train()

6)Save the model
----------------------------------
Save the trained model to use for prediction later on.
trainer.save( model_name = model )



=======================================================================================================
Prediction:
Predictor( model )                              Instantiate Predictor object with a saved trained model
Predictor.predict( data )                       Make prediction on data with the Predictor object
=======================================================================================================



#Prediction
1)Indicated the saved model to use
----------------------------------
from ssvm.predictor import Predictor
predictor = Predictor( model.pkl )

2)Make predictions
----------------------------------
Use the Predictor to make predictions on the data, assumed to be 2D numpy array
with the same number of columns (minus the label) as that used in the Training 
object.

predictor.predict( data )



=======================================================================================================
Tuning:
GridSearch(trainer, C_start=-6, C_end=10, G_start=-12, r=1, v=10, k=1, s=0) 
                                                Hyperparameter optimization
Hibiscus(trainer, fround=13, sround=9, tround=5, C_start=-6, C_end=10, G_start=-12, r=1, v=10, k=1, s=0)
                                                Fast model selection via uniform design
=======================================================================================================



#Tuning
1)GridSearch
----------------------------------
First, we instantiate the Trainer object with a data set and import the GirdSearch function from tuning.
Return is list [Best_C, Best_G].
There also print others information.

from ssvm.tuning import GridSearch
[Best_C, Best_G] = GridSearch( trainer, -3, 12, -6 )

"""
Best C value: XXX
Best Gamma value: XXX
Training accuracy: XXX
Validation accuracy: XXX
During time: XXX
"""

2)Hibiscus
----------------------------------
First, we instantiate the Trainer object with a data set and import the Hibiscus function from tuning.
We need to assign how much point want to sprinkle in three round.
Uniform table range: 3 ~ 30
Return is list [Best_C, Best_G]
There also print others information.

from ssvm.tuning import Hibiscus
[Best_C, Best_G] = Hibiscus( trainer, 30, 20, 10 )

"""
Best C value: XXX
Best Gamma value: XXX
Training accuracy: XXX
Validation accuracy: XXX
During time: XXX
"""




=======================================================================================================

If you have any questions, please contact any of the following:
Author:
    Prof.Lee(yuh-jye@mail.ntust.edu.tw)
Maintainer:
    Evan(evan176@hotmail.com)
    
=======================================================================================================
