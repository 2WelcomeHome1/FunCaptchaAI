
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, \
                        BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam as adam
from keras import initializers
import math

class Model (Sequential):
    def __init__(self, X_train, num_classes, lr_rate):
        super().__init__()
        self.initializer = initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512), seed=42)
        self.X_train, self.num_classes, self.lr_rate = X_train, num_classes, lr_rate
        
    
    def create_model(self):#Building the model
        
        self.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=self.X_train.shape[1:]))
        self.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
        self.add(Dropout(rate=0.15))
        
        self.add(MaxPooling2D(pool_size=2))

        self.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
        self.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
        self.add(Dropout(rate=0.25))

        self.add(MaxPooling2D(pool_size=2))

        self.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
        self.add(BatchNormalization (momentum=0.9, epsilon=1e-5))
        self.add(Dropout(rate=0.5))
        

        self.add(GlobalAveragePooling2D()) 
        self.add(Dense(50, activation='relu'))
        self.add(Dense(self.num_classes,kernel_initializer=self.initializer, bias_initializer=self.initializer, activation='softmax')) 

        self.compile(optimizer = adam(self.lr_rate), loss = 'categorical_crossentropy', metrics= ['accuracy'])

        return self
