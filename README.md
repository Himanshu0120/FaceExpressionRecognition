# FaceExpressionRecognition
 Recognizing different facial expressions using image classification and OpenCv.
 
 ### Process involved here is -
 * Training the model - The first step is to train an image classication model and saving it. Here keras framework is used to do the same.
 * Detecting face - For recognising the expressions from live camera the first process is to detect the face. Here frontal face haarcascade is used to detect the face.
 * Croping and resizing face image - Next step is to crop the face portion from the image and process it to convert it in format acceptable by the m0del.
 * Input and Display - The image is then given to the model which returns the result. The result is the displayed with the image.

The model gives accurate results for expressions like Happy , Normal and Surprised but the accuracy is not so good for other expressions like Sad or Angry.
This is beacuse the quality of the data set. This accuracy can be increased in future using high quality datset.
