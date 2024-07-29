# This code is the main structure of the model. a and b are the definitions of the discard rates
# of different modules, and the cross-validation of the discard rates can be realized
# by modifying the values of a and b.
from tensorflow.keras.layers import GaussianDropout
class DropConnect(layers.Layer):
    def __init__(self, drop_connect_rate=0.5, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_connect_rate = drop_connect_rate



        def call(self, inputs, training=None):
            if training:
                # Generate weight mask
                weight_mask = tf.random.uniform(tf.shape(inputs)[:2], dtype=tf.float32) < self.drop_connect_rate
                weight_mask = tf.cast(weight_mask, dtype=tf.float32)

                # Expand dimensions of weight_mask if necessary
                weight_mask = tf.expand_dims(weight_mask, axis=-1)
                inputs = inputs * weight_mask / (1 - self.drop_connect_rate)
            return inputs

    def get_config(self):
        config = super(DropConnect, self).get_config()
        config.update({'drop_connect_rate': self.drop_connect_rate})
        return config



# Build the model
inputs = Input(shape=(30, GloUse['SL'][n - 1], 1))  # Two channels
input1 = layers.Lambda(lambda x: x)(inputs)
input2 = layers.Lambda(lambda x: x)(inputs)

# Define dropout rates
a = 0.6
b = 0.6

# First channel convolution layers
x1 = Conv2D(filters=64, kernel_size=(40, 1), strides=(1, 1), padding='same', activation='tanh')(input1)
x1 = GaussianDropout(a)(x1)
x1 = Conv2D(filters=32, kernel_size=(25, 1), strides=(1, 1), padding='same', activation='tanh')(x1)
x1 = GaussianDropout(a)(x1)
x1 = Conv2D(filters=16, kernel_size=(20, 1), strides=(1, 1), padding='same', activation='tanh')(x1)
x1 = GaussianDropout(a)(x1)
x1 = Conv2D(filters=8, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh')(x1)
x1 = GaussianDropout(a)(x1)
x1 = Conv2D(filters=1, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='tanh')(x1)
x1 = GaussianDropout(a)(x1)
x1 = Flatten()(x1)

# Second channel convolution layers
x2 = Conv2D(filters=128, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh')(input2)
x2 = GaussianDropout(a)(x2)
x2 = Conv2D(filters=64, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh')(x2)
x2 = GaussianDropout(a)(x2)
x2 = Conv2D(filters=32, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh')(x2)
x2 = GaussianDropout(a)(x2)
x2 = Conv2D(filters=32, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh')(x2)
x2 = GaussianDropout(a)(x2)
x2 = Conv2D(filters=1, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='tanh')(x2)
x2 = GaussianDropout(a)(x2)
x2 = Flatten()(x2)

# Concatenate outputs from both channels
x = Concatenate()([x1, x2])

# Add an LSTM layer
x = Reshape((1, -1))(x)  # Flatten features into one dimension
x = LSTM(50, activation='tanh', return_sequences=False)(x)

# Fully connected layers
x = Dropout(b)(x)
x = Dense(100, activation='tanh')(x)
x = Dense(1, name='out')(x)

# Build the model
model = Model(inputs=inputs, outputs=x)

# Compile the model
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse', optimizer=adam)

# Add checkpoint callback
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              monitor='val_loss',  # Monitor validation loss
                              save_weights_only=True,
                              verbose=1)

# Train the model
history1 = model.fit(train_input, train_output, batch_size=512, epochs=200, shuffle=True)  # Training phase

# Adjust learning rate and recompile model
adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mse', optimizer=adam)

# Continue training with validation split and callbacks
history2 = model.fit(train_input, train_output, validation_split=0.33, batch_size=512, epochs=50, shuffle=True,
                     callbacks=[cp_callback])  # Testing phase
