img_w = 56
img_h = 56
ysize=36
n_labels = 4
channels = 3
kernel = 3
crop_size = 10

autoencoder = models.Sequential()
#autoencoder.add(ZeroPadding2D((1,1), input_shape=(3, img_h, img_w), dim_ordering='th'))

encoding_layers = [
    Convolution2D(16, kernel, kernel, border_mode='same', input_shape=(img_h, img_w, channels)),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(16, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Convolution2D(32, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(32, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
]

autoencoder.encoding_layers = encoding_layers

for l in autoencoder.encoding_layers:
    autoencoder.add(l)

decoding_layers = [

    UpSampling2D(),
    Convolution2D(32, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(32, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Convolution2D(16, kernel, kernel, border_mode='same'),
    BatchNormalization(),
    Activation('relu'),
    Convolution2D(n_labels, 1, 1, border_mode='valid'),
    BatchNormalization(),
]
autoencoder.decoding_layers = decoding_layers
for l in autoencoder.decoding_layers:
    autoencoder.add(l)

autoencoder.add(Cropping2D(cropping=((crop_size, crop_size), (crop_size, crop_size))))
autoencoder.add(Reshape((ysize*ysize, n_labels)))
#autoencoder.add(Reshape((n_labels,ysize * ysize)))
#autoencoder.add(Permute((2, 1)))
autoencoder.add(Activation('softmax'))
#autoencoder.add(Permute((1, 2)))
autoencoder.add(Reshape((ysize,ysize,n_labels)))

#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#rmsprop = keras.optimizers.RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08, decay=0.0)
autoencoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#autoencoder.compile(loss=your_loss, optimizer='adam', metrics=['accuracy'])
autoencoder.summary()
