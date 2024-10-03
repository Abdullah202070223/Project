
---

# Report on Solar Panel Data Analysis Code

## Introduction

This project aims to analyze solar panel data using TensorFlow and Keras in Python. The code includes steps for importing data, inspecting the data structure, and visualizing images from different classes of solar panel conditions.

## Code Explanation

### Importing Libraries

```python
# import necessary libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import os, random
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, InputLayer, Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
```

- **TensorFlow and Keras**: Used for building and training machine learning models.
- **NumPy**: Provides support for large, multi-dimensional arrays and matrices.
- **Pandas**: Useful for data manipulation and analysis.
- **Matplotlib**: Used for plotting graphs and visualizations.
- **OS and Random**: Helps in file manipulation and generating random numbers.
- **Keras Layers**: Importing neural network layers to build models.

### Data Acquisition Methods

```markdown
There are 3 methods of acquiring the dataset:

1st Method: Mounting my Google Drive into Google Colab and unzipping the dataset.
2nd Method: Using the API command from the Roboflow project site.
3rd Method: Copying the dataset URL from the GitHub repository, downloading, and extracting the contents into Colab.
```

This section explains three different methods to acquire the dataset.

### Listing Files

```python
# list the files of solar panels classes
!dir C:\\Users\\Emad\\Desktop\\the_last_pro\\solar_pv_dataset
```

Uses a Windows shell command to list directories and files in the specified path.

### Inspecting the Data Structure

```python
import os

# walk through the solar thermal directory and list the number of images
for dirpath, dirnames, filenames in os.walk( solar_pv_dataset ):
  print(f There are: {len(dirnames)} directories and {len(filenames)} images in {dirpath} )
```

- **os.walk**: Traverses directories to count and list the number of subdirectories and files.

### Directory Listing

```python
# The file root directory of the solar panel classes
!dir /a C:\\Users\\Emad\\Desktop\\the_last_pro\\solar_pv_dataset
```

Lists directory contents with attributes using a Windows-specific command.

### Getting Class Names

```python
# get the class names programmatically
import pathlib
import numpy as np

data_dir = pathlib.Path( C:/Users/Emad/Desktop/the_last_pro/solar_pv_dataset )   # set it as a path object
class_names = np.array(sorted([item.name for item in data_dir.glob( * )]))  # created a list of class names from the train sub directories
print(class_names)
```

- **pathlib.Path**: Used for filesystem path operations.
- **glob**: Retrieves directories representing class names.

### Counting Total Images

```python
# data path
data_path =  solar_pv_dataset 

# set the total number of images
total_images = 0

# go through the class name
for class_name in class_names:
  target_folder = data_path +  /  + class_name
  total_images += len((os.listdir(target_folder)))

print(f There are total number of {total_images} images in our solar pv dataset. )
```

Calculates the total number of images across all classes.

### Viewing Random Images

```python
# create a function to view random images
def view_random_image(target_dir, target_class = random.choice(class_names)):
     
  Views a random image from the target directory of the target class (Clean, Crack, Dirty, Bird dropping)
     

  # Visualize images from the train data
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg    # import matplotlib module to work with images
  import random, os

  # Setup target directory
  target_folder = target_dir + target_class

  # get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)
  print(random_image)

  # Read in the image and plot it using matplotlib
  image = mpimg.imread(target_folder +  /  + random_image[0])
  plt.imshow(image)
  plt.title(target_class)
  plt.axis( Off );

  print(f Image shape: {image.shape} )

  return image
```

- **Function Definition**: Displays a random image from a specified class.
- **matplotlib.image**: Reads and displays the image.

### Displaying a Random Image

```python
view_random_image( C:/Users/Emad/Desktop/the_last_pro/solar_pv_dataset/ )
```

Calls the function to display a random image from the dataset.

## Conclusion

This code provides a comprehensive approach to analyzing solar panel images, including data acquisition, inspection, and visualization. The use of libraries like TensorFlow and Keras facilitates the development of machine learning models for further analysis.

---
