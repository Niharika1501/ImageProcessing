# ImageProcessing
The project can process the image by performing following actions :
1. Rescale: It rescales the batch of input images according to a given scale . Where s is the scaling factor. This function has to return a dictionary of rescaled images where the key is the name of the image and the value is corresponding rescaled output in accordance to the random indexes picked up.
2. Resize : It resizes the batch of input images to the given dimensions. Where h is the number of rows and w is the number of columns. This function has to return a dictionary of resized images where the key is the name of the image and the value is corresponding resized output in accordance to the random indexes picked up. The datatype of output has to be a numpy array
3. Translate the image: It translates the batch of input images. Where tx is the translation in x direction and ty is the translation in the y direction. This function has to return a dictionary of translated images where the key is the name of the image and the value is corresponding translated output in accordance to the random indexes picked up.
4. Crop : It crops the batch of input images to the given dimensions. Where id1,id2,id3,id4 are tuples of the indexes where cropping has to be performed
#   id1 .----------------------------------------.(x,y) = id2
    #   |                                        |
    #   |                                        |
    #   |                                        |
# id4   .----------------------------------------. id3
This function has to return a dictionary of cropped images where the key is the name of the image and the value is corresponding cropped output in accordance to the random indexes picked up.

5. Blur: Blurring the batch of input images in accordance to the filter specified. This function has to return a dictionary of blurred images where the key is the name of the image and the value is corresponding blurred output in accordance to the random indexes picked up.
6. GrayScale: Converts a batch of rgb image to a gray image. This function has to return a dictionary of gray images where the key is the name of the image and the value is corresponding gray image in accordance to the random indexes picked up.
7. Rotate the image: Rotates the batch of input images to the given dimensions. Where theta is the angle of rotation. This function has to return a dictionary of gray images where the key is the name of the image and the value is corresponding gray image in accordance to the random indexes picked up.
