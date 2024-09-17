# -*- coding: utf-8 -*-
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.image as mpimg

# Validation data is updated to use to save the best model among epochs
class PowerModeAutopilot(nn.Module):
    # write your model here
    def __init__(self, keep_prob=0.5):
        super(PowerModeAutopilot, self).__init__()
        #############################################
        """
        VGG style model:
        2 convs (3x3, depth 64)
        pool
        2 convs (3x3, depth 128)
        pool 
        3 convs (^^, depth 256)
        pool
        3 convs (^^, depth 512)
        pool
        3 convs (^^, depth 512)
        pool
        fully connected layer 1, fully connected 2, softmax

        
        Other things we can try:
        dropout layers?
        try vgg 19? that would turn the 3 convs into 4, may be too many params though!
        """
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 1
        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Block 2
        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Block 3
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # Block 4
        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        # Block 5: probably wont need
        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 12, 4096) # this also needs to be the same input at line 98
        # was 512 *3
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1)  # 1000 = num classes = ??
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=keep_prob)
        #############################################
        
        
    def forward(self, x):
        # Block 1
        x = nn.functional.relu(self.conv11(x))
        x = nn.functional.relu(self.conv12(x))
        x = self.pool(x)
        
        # Block 2
        x = nn.functional.relu(self.conv21(x))
        x = nn.functional.relu(self.conv22(x))
        x = self.pool(x)
        
        # Block 3
        x = nn.functional.relu(self.conv31(x))
        x = nn.functional.relu(self.conv32(x))
        x = nn.functional.relu(self.conv33(x))
        x = self.pool(x)
        
        # Block 4
        x = nn.functional.relu(self.conv41(x))
        x = nn.functional.relu(self.conv42(x))
        x = nn.functional.relu(self.conv43(x))
        x = self.pool(x)
        
        # print(x.size)
        # print(x.shape) # [batch size, channel num, h, w]
        # print(f"FULLY CONNECTED LAYER INPUT: {x.size(0)*x.size(1)*x.size(2)}")

        # Flatten for fully connected layers
        # was  x = x.view(-1, x.size(0)*x.size(1)*x.size(2))  # -1 automatically considers batch size
        x = x.reshape(x.size(0), -1) 
        # Fully connected layers with dropout
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Softmax for classification
        #x = nn.functional.softmax(x, dim=1)
        
        return x
        #############################################
        

class PowerMode_autopilot:
    def __init__(self, data_path='data', learning_rate=1.0e-4, keep_prob=0.5, batch_size=40,
                 save_best_only=True, test_size=0.2, steps_per_epoch=20000, epochs=10):
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS = 66, 200, 3
        self.data_path = data_path
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.save_best_only = save_best_only
        self.batch_size = batch_size
        self.test_size = test_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_data(self): # ADC
        """
        Read data from driving_log.csv file, then split the data using train_test_split function into training set and validation set,
        For every piece of data,
        X represents images of left, center, right cameras
        y represents steering value, as reference
        :return: training sets and validation sets of images and steering value (format: X_train, X_valid, y_train, y_valid)
        """
        #############################################
        """
                Notes 
                X is in form: [[center, left, right],  [center, left, right], [center, left, right] ....]
      
        """
        #"Your code here"
        X_array = []
        Y_array = []
        data_frame = pd.read_csv('driving_log.csv', header=None)

        # loop through each row in CSV
        for index, row in data_frame.iterrows():
            direction_set = []
            direction_set.append(row[0])
            direction_set.append(row[1])
            direction_set.append(row[2])
            X_array.append(direction_set)

            Y_array.append(row[3])

        # create numpy arrays
        X = np.array(X_array)
        Y = np.array(Y_array)

        # 20% used for testing
        X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size= 0.2, random_state= 42)
        return X_train, X_valid, y_train, y_valid
        #############################################    

    """
    augment image
    """

    def augment(self, center, left, right, steering_angle, range_x=100, range_y=10):
        """
        image augmentation using following functions
        :param center:
        :param left:
        :param right:
        :param steering_angle:
        :param range_x:
        :param range_y:
        :return:
        """
        image, steering_angle = self.choose_image(center, left, right, steering_angle)
        image, steering_angle = self.random_flip(image, steering_angle)
        image, steering_angle = self.random_translate(image, steering_angle, range_x, range_y)
        image = self.random_shadow(image)
        image = self.random_brightness(image)

        return image, steering_angle

    def choose_image(self, center, left, right, steering_angle):
        """
        randomly choose an image: center, left, or right
        in case of left image or right image, steering angle needs to be adjusted accordingly
        :param center:
        :param left:
        :param right: images to choose from
        :param steering_angle:
        :return: result of choosing
        """
        choice = np.random.choice(3)
        if choice == 0:
            return self.load_image(left), steering_angle + 0.2
        elif choice == 1:
            return self.load_image(right), steering_angle - 0.2
        return self.load_image(center), steering_angle

    def random_flip(self, image, steering_angle):
        """
        randomly flip the image and invert steering angle accordingly or not,
        in case one direction steering happens way more than the other direction,
        which could lead to lack of training data diversity
        :param image:
        :param steering_angle:
        :return: image after flip
        """
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle
        return image, steering_angle

    def random_translate(self, image, steering_angle, range_x, range_y):
        """
        translate the image horizontally and vertically, randomly.
        modify the steering angle accordingly
        :param image:
        :param steering_angle:
        :param range_x:
        :param range_y:
        :return:
        """
        # getting a transformation matrix, trans_x and trans_y denote horizontal and vertical shift value
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        steering_angle += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])

        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image, steering_angle

    def random_shadow(self, image):
        """
        add some random shadow because there are different light conditions in different situations
        :param image:
        :return:
        """
        # get the range of the shadow
        x1, y1 = self.IMAGE_WIDTH * np.random.rand(), 0
        x2, y2 = self.IMAGE_WIDTH * np.random.rand(), self.IMAGE_HEIGHT

        xm, ym = np.mgrid[0:image.shape[0], 0:image.shape[1]]

        mask = np.zeros_like(image[:, :, 1])
        mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    def random_brightness(self, image):
        """
        adjust image brightness randomly
        convert the image to HSV model to adjust brightness more easily
        :param image:
        :return:
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = hsv[:, :, 2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    """
    preprocess image
    """

    def preprocess(self, image):
        """
        preprocess the image, using the following functions
        :param image:
        :return:
        """
        image = self.crop(image)
        image = self.resize(image)
        image = self.rgb2yuv(image)
        return image

    def crop(self, image):
        """
        crop the image, keep the pixels between 60 from top and 25 from bottom,
        to get rid of the sky and the car itself, focus on the road
        :param image:
        :return:
        """
        return image[60:-25, :, :]

    def resize(self, image):
        """
        resize the image, using cv2.INTER_AREA as interpolation since we are shrinking the image
        :param image:
        :return:
        """
        return cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), cv2.INTER_AREA)

    def rgb2yuv(self, image):
        """
        convert the image from RGB format to YUV format
        :param image:
        :return:
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    def build_model(self):
        """
        build a model using PyTorch nn.Module
        :return: the model
        """
        model = PowerModeAutopilot(keep_prob=self.keep_prob).to(self.device)
        return model

    def load_image(self, image_file):
        """
        read an image file
        :param image_file: image file name
        :return: the array representing the image
        """
        return mpimg.imread(os.path.join(self.data_path, image_file.strip()))

    # ... (keep the augmentation and preprocessing functions the same)

    def batch_generator(self, image_paths, steering_angles, is_training):
        """
        generate a batch of training images and corresponding steering angles
        :param image_paths:
        :param steering_angles:
        :param is_training:
        :return:
        """
        
        while True:
            images = np.empty([self.batch_size, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS])
            steers = np.empty(self.batch_size)

            i = 0
            for index in np.random.permutation(image_paths.shape[0]):
                center, left, right = image_paths[index]
                steering_angle = steering_angles[index]
                if is_training and np.random.rand() < 0.6:
                    image, steering_angle = self.augment(center, left, right, steering_angle)
                else:
                    image = self.load_image(center)
               # print(f"Batch Generator, Image: {type(image)}, shape: {image.shape}")
                images[i] = self.preprocess(image)

                steers[i] = steering_angle
                i += 1
                if i == self.batch_size:
                    break


            images = images.transpose((0, 3, 1, 2))
            images = torch.from_numpy(images).float().to(self.device)
            steers = torch.from_numpy(steers).float().unsqueeze(1).to(self.device)
            yield images, steers

    def train_model(self, model, X_train, X_valid, y_train, y_valid): #ADC
        """
        train the model
        :param model:
        :param X_train:
        :param X_valid:
        :param y_train:
        :param y_valid:
        :return:
        """
        #############################################    
        # Loss Function
        criterion = nn.MSELoss()
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
     
        # generate 
       
        train_batches = self.batch_generator(X_train, y_train, True)
        test_batches = self.batch_generator(X_valid, y_valid, False)

        top_loss =  float('inf')
        train_batch_count = 0

        for epoch in range(self.epochs):
            model.train() # set model to training mode
            running_train_loss = 0.0
            for i in range(self.steps_per_epoch): # 100 steps per epoch
            #for image, steering_angles in train_loader:
                
                image, steering_angles = next(train_batches)

                image = image.to(self.device)
                steering_angles = steering_angles.to(self.device)

                # Forward pass
                outputs = model(image) 
                loss = criterion(outputs, steering_angles)
               

                # Backward and optimize
                optimizer.zero_grad()  # Reset the gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model parameters

                running_train_loss += loss.item()
                train_batch_count += 1
                torch.cuda.empty_cache()

            train_loss_avg = running_train_loss / train_batch_count
            print("Epoch: ", epoch, "Training Loss: ", train_loss_avg)

        model.eval() # Set the model to evaluation mode
        running_test_loss = 0.0
        test_batch_count = 0
        with torch.no_grad():
            for i in range(self.steps_per_epoch): # 100 steps per epoch
            #for image, steering_angles in train_loader:
                
                image, steering_angles = next(test_batches)

                image = image.to(self.device)
                steering_angles = steering_angles.to(self.device)
 
                # Forward pass
                outputs = model(image)
                loss = criterion(outputs, steering_angles)
 
                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1) # first return value is the max value
 
                running_test_loss += loss.item()
                test_batch_count += 1

            # Calculate average loss and accuracy
            test_loss_avg = running_test_loss / test_batch_count
            print("Running Validation Loss: ", test_loss_avg)

            if (self.save_best_only) and (test_loss_avg < top_loss):
                top_loss = test_loss_avg
                print("saving model")
                torch.save(model.state_dict(), '1024_vgg_3epochs.pth')
 


        #############################################

def main():
    autopilot = PowerMode_autopilot(data_path='your_data_path', learning_rate=1.0e-3, keep_prob=0.5, batch_size=40,
                                    save_best_only=True, test_size=0.2, steps_per_epoch=2000, epochs=3)

    data = autopilot.load_data()

    model = autopilot.build_model()

    autopilot.train_model(model, *data)


if __name__ == '__main__':
    main()