I couldnt get any of the stacklight dataset but I went with the traffic lights, Steps for the Industrial Stacklight Identification with alerts.And also I have included person dataset.

Step1: Collect the dataset of the traffic lights and download it in YOLOV11 format(in this I have used Yolo version you can use what ever u want)

Step2: Now write the training code .

Step3: For running the codes I have created a Virtual Environment using anaconda prompt in that download all the neccesary libraries that required  for this.

Step4: Run the train.py file, after that check the runs file, in that weights file would be present check whether best.ot is present or not.

Step5: Now run the inference script which I have provided  the alerts that for based on lights, if its green the machinery is in working mode and if there is any person is present in the picture then the alert goes "PERSON IS PRESENT, ALERT!!", if its orange then it goes "FAULT PRESENT IN MACHINERY, NEED TECHNINICIANS", if it red no alert would be there.

Step6: Now run thhe decision tree, why wwe use this? - The decision tree code replicates the decision-making logic of the inference script using a decision tree classifier from scikit-learn. It takes the same input detected classes from images and produces similar outputs machinery status and alerts by modeling the decision rules explicitly. 
