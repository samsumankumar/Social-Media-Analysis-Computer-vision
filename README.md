# Social-Media-Analysis-Computer-vision
This project was developed as a part of Course work by a team of 5 students of George Mason University in association with Accenture Federal Services.The primary objective of the model is 
development of a Machine Learning Model to classify the images posted on Instagram platform with respect to the country. We considered images with hashtags corresponding to United States of America, China and India
The total project consists of three parts
1. Data Collection
2. Data Modeling
3. Gradcam visualizations<br />
In the Data Collection phase, we collected 30,000 images from Instagram, 10,000 images per country based on hypernational hashtags.
In the Modeling phase, we used pretrained models like VGG16, ResNet50 to build binary and tertiary classifiers. We also developed a CNN model from scratch. However, the accuracy of VGG16 model is better compared to the other two models
In the Visualizations phase, we used gradcam and guided gradcam for visualizing the features used by model for classification. We implemented gradcam on the misclassified images.
# Model Deployment:
In order to run the model on local machine, use the compatible version of tensorflow (version 2.2 or higher) and python3 and run python3 application.py in the command prompt.<br />
If you are planning to host the flask web application on Virtual Cloud Instance, replace the localhost with 0.0.0.0 and enable port 80 on the instance to serve the application. <br />
# Future Work:
1.Accuracy can be improved by increasing the amount of data per each class <br />
2.Images from more number of countries can be included
# Team:
1. Una Suman Kumar Patro (Product Owner)(https://www.linkedin.com/in/samsumankumar/)
2. Chandana Narla (Scrum Master)(https://www.linkedin.com/in/chandana-narla-990285115/)
3. Anitha Mutyala (Developer)(https://www.linkedin.com/in/anitha-mutyala-a46605112/)
4. Suhasini Konimeti Naresh Kumar (Developer)(https://www.linkedin.com/in/suhasini-konimeti-naresh-kumar-8ab072192/)
5. Sai Vikas Devisetty (Developer)(https://www.linkedin.com/in/saivikas-devisetty/)
