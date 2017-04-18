SSVM-Python是修改自LDSMI Lab的Smooth Support Vector Machine Toolbox
http://140.118.155.137/download.html#toolbox

使用說明：
使用時請先將模組SSVM-Python放置於目錄底下

=======================================================================================================
Training:
Trainer( data, label_pos )                             創建訓練物件，並匯入資料
Trainer.make( r = 1 , v = 1 )                          初始化資料，將資料切成不同partition來作CrossValidation
Trainer.tune( c = 100, g = 0.1, k = 1, s = 0 )         設定訓練參數
Trainer.train()                                        啟動資料訓練
Trainer.save( model_name = model)                      儲存訓練完成的模型
=======================================================================================================


#Training
1)匯入模組

from trainer import Trainer

2)產生Trainer object
先將要作訓練的資料集轉換成numpy 2d-array，這邊假設data 是 mxn array
而且包含label的資料在其中，而label_pos則是指名是用哪一欄的資料當作label

trainer = Trainer( data, label_pos )

3)初始化Trainer
將資料集作處理以便開始訓練
r -> 想要取多少比例的資料作reduced set
v -> 要分成幾份來作CrossValidation

trainer.make( r = 1 , v = 1 )

4)設定訓練時的參數
c = 在SVM中控制penalty的參數
g = RBF kernel中的gamma值
k = 選擇要用linear或者nonlinear, 0 -> linear, 1 -> nonlinear
s = 在多個類別時決定要怎樣去做SVM training, 0 -> One-Again-One, 1 -> One-Again-Rest

trainer.tune( c = 100, g = 0.1, k = 1, s = 0 )

5)開始訓練
只要使用train() 這個function就可以開始訓練資料
而所使用的參數則是根據使用者調整tune()

trainer.train()

6)儲存模型
將訓練好的模型儲存下來，之後再來做predict
Trainer.save( model_name = model )



=======================================================================================================
Prediction:
Predictor( model )                                     創建預測物件，並匯入模型，指定模型名稱
Predictor.predict( data )                              用已經創業的預測物件來預測新的資料
=======================================================================================================


#Prediction
1)指定模型名稱
創建Predictor的時候要指定model name
Predictor( model.pkl )

2)預測新資料
用已經創見好了predictor來預測新資料，新的資料維度必須跟training data一樣，且為numpy mxn array
Predictor.predict( data )
