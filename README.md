# Image_segmentation
Data Collection:
Data can be found from link below.
https://www.kaggle.com/datasets/jesperdramsch/siim-acr-pneumothorax-segmentation-data



Data Pre-processing:
A custom data generator was designed that will generate the data in batches and also resize the image to required dimensions. For the model used in this case we have resized the image to 256*256 pixel size It's used to load, preprocess, and generate batches of X-ray images and their corresponding masks. A detailed explanation about the data generator class is shared below.
Parameters:
•	file_path_list (list): A list of file paths to DICOM format X-ray images.
•	labels (dict): A dictionary mapping image IDs to Run-Length Encoding (RLE) encoded masks.
•	batch_size (int, optional): The batch size for data generation. Default is 32.
•	img_size (int, optional): The desired image size (width and height). Default is 256.
•	channels (int, optional): The number of color channels in the image (e.g., 1 for grayscale). Default is 1.
•	shuffle (bool, optional): Whether to shuffle the dataset after each epoch. Default is True.
Methods:
__len__(self)
•	This method returns the number of batches per epoch. It's used for specifying how many batches to generate in one epoch.
__getitem__(self, index)
•	This method generates one batch of data. It retrieves the file paths and corresponding masks for the specified batch index.
on_epoch_end(self)
•	This method is called at the end of each epoch and is responsible for updating the order in which the data will be processed in the next epoch. If shuffle is True, it shuffles the indexes.
__data_generation(self, file_path_list_temp)
•	This method generates data for the batch and returns input images X and corresponding masks y.
•	It loops through the file_path_list_temp to load and preprocess images.
•	It converts the image to the specified img_size, and normalizes the pixel values.
•	If no mask is available for an image (based on the labels), it creates an empty mask.
•	If multiple RLE-encoded masks are available, it aggregates them to create a single mask.
•	The masks are resized to match the specified img_size.
•	The X and y arrays are filled with the processed images and masks.
•	The pixel values are normalized by dividing by 255 to bring them into the range [0, 1].
 
Data Visualization:
You can use the plot_with_mask_and_bbox function to visually inspect X-ray images along with their masks and bounding boxes. This can be helpful for understanding and verifying image segmentation results in medical imaging tasks.
Parameters:
•	file_path (str): The file path to the DICOM format X-ray image.
•	mask_encoded_list (list): A list of mask encodings in Run-Length Encoding (RLE) format.
•	figsize (tuple, optional): A tuple specifying the size of the output figure. Default is (20, 10).


 
 
Model Architecture.

Evaluation metric:
The evaluation metric used for the model was dice coefficient. Dice similarity coefficient is a spatial overlap index and a reproducibility validation metric. It was also called the proportion of specific agreement by Fleiss (14). The value of a DSC ranges from 0, indicating no spatial overlap between two sets of binary segmentation results, to 1, indicating complete overlap.

 

 
Model Architecture.

We first tried with the Unet model , which is one the models created for medical image segmentation domain. Model architectures is explained below.
he U-Net architecture is a convolutional neural network (CNN) designed for semantic image segmentation, which is the process of classifying each pixel in an image into a specific category or object class. U-Net is known for its effectiveness in biomedical image segmentation tasks, but it's also used in various other domains. Here's an explanation of the U-Net model architecture:
1. Contracting Path (Encoder):
•	The architecture of U-Net starts with a contracting path that resembles a typical CNN's convolutional and pooling layers. This path downsamples the input image to capture hierarchical features.
•	The contracting path consists of several repeated blocks, each composed of two 3x3 convolutional layers (often followed by batch normalization and a rectified linear unit, ReLU) and a 2x2 max-pooling layer.
•	The number of feature channels increases as you go deeper into the network.
2. Bottleneck:
•	The contracting path eventually narrows down to a bottleneck layer where the spatial information is compressed and the network captures high-level features.
•	The bottleneck is formed by the same repeated blocks of convolutional layers.
3. Expansive Path (Decoder):
•	The expansive path, also called the decoder, takes the high-level features and gradually upscales them back to the original image resolution.
•	Each block in the expansive path typically consists of two 3x3 transposed convolutional layers (often followed by batch normalization and ReLU) and is used to increase the spatial resolution.
•	Skip connections are crucial in U-Net and are used to concatenate feature maps from the contracting path to the corresponding layers in the expansive path. These skip connections enable the network to retain detailed spatial information during the upsampling process.
•	The final layer of the expansive path usually uses a 1x1 convolution to produce the pixel-wise segmentation mask.
4. Output Layer:
•	The output layer produces a pixel-wise segmentation mask that represents the predicted class labels for each pixel in the input image.
•	The number of output channels in the mask typically corresponds to the number of classes or categories the network is trained to segment.
Advantages of U-Net:
•	Effective Semantic Segmentation: U-Net is highly effective in segmenting objects and regions of interest in images.
•	Feature Preservation: The skip connections allow U-Net to preserve fine details and spatial information during the upsampling process.
•	Versatility: U-Net can be adapted for various image segmentation tasks in different domains, including medical image analysis, satellite image segmentation, and more.
•	Relatively Compact: It provides good results with a relatively small number of parameters compared to some other deep learning models.

with 20 epoch we are able to get dice coefficient of 0.994.

For Hyperparameter tunning , tried with couple of activation functions like relu ,elu ,selu. Out of these Selu was the one with the best dice coefficient. 
I also tried to use unet++  as an alternative model but it overfitting.
SELU activation function is defined mathematically as follows:
For a given input x:
•	If x > 0, SELU(x) = scale * x, where "scale" is a constant greater than 1, typically around 1.0507.
•	If x <= 0, SELU(x) = scale * alpha * (exp(x) - 1), where "alpha" is a constant, typically around 1.67326.
The SELU activation function introduces self-normalizing properties, which means that if you apply it to the weights of a neural network and certain conditions are met, the output of each layer will have a mean close to 0 and a standard deviation close to 1. This helps address the vanishing and exploding gradient problems during training.


Apart from this other hyperparameters can be also adjusted but due to time constraints , was not able to apply them.



