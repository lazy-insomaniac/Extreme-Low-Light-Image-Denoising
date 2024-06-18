import os
import numpy as np 
import random
from PIL import Image , ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf 
from glob import glob
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanAbsoluteError,MeanSquaredError
import cv2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.losses import MeanSquaredError

# Constants
IMAGE_SIZE = 256
BATCH_SIZE = 16
MAX_TRAIN_IMAGES = 400

# Preprocessing functions
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    image = image / 255.0
    return image

def load_and_preprocess_data(low_light_paths, high_light_paths):
    low_light_images = [preprocess_image(path) for path in low_light_paths]
    high_light_images = [preprocess_image(path) for path in high_light_paths]
    return tf.data.Dataset.from_tensor_slices((low_light_images, high_light_images))

def paired_data_generator(low_light_paths, high_light_paths):
    dataset = load_and_preprocess_data(low_light_paths, high_light_paths)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

# Build Zero-DCE model
def build_dce_net():
    input_img = keras.Input(shape=[None, None, 3])
    conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(input_img)
    conv2 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv1)
    conv3 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv2)
    conv4 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv3)
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(int_con2)
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(int_con3)
    return keras.Model(inputs=input_img, outputs=x_r)

# Loss functions
def color_constancy_loss(x):
    mean_rgb = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
    mr, mg, mb = mean_rgb[:, :, :, 0], mean_rgb[:, :, :, 1], mean_rgb[:, :, :, 2]
    d_rg = tf.square(mr - mg)
    d_rb = tf.square(mr - mb)
    d_gb = tf.square(mb - mg)
    return tf.sqrt(tf.square(d_rg) + tf.square(d_rb) + tf.square(d_gb))

def exposure_loss(x, mean_val=0.6):
    x = tf.reduce_mean(x, axis=3, keepdims=True)
    mean = tf.nn.avg_pool2d(x, ksize=16, strides=16, padding="VALID")
    return tf.reduce_mean(tf.square(mean - mean_val))

def illumination_smoothness_loss(x):
    batch_size = tf.shape(x)[0]
    h_x = tf.shape(x)[1]
    w_x = tf.shape(x)[2]
    count_h = (tf.shape(x)[2] - 1) * tf.shape(x)[3]
    count_w = tf.shape(x)[2] * (tf.shape(x)[3] - 1)
    h_tv = tf.reduce_sum(tf.square((x[:, 1:, :, :] - x[:, : h_x - 1, :, :])))
    w_tv = tf.reduce_sum(tf.square((x[:, :, 1:, :] - x[:, :, : w_x - 1, :])))
    batch_size = tf.cast(batch_size, dtype=tf.float32)
    count_h = tf.cast(count_h, dtype=tf.float32)
    count_w = tf.cast(count_w, dtype=tf.float32)
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

class SpatialConsistencyLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super(SpatialConsistencyLoss, self).__init__(reduction="none")
        self.left_kernel = tf.constant([[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32)
        self.right_kernel = tf.constant([[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=tf.float32)
        self.up_kernel = tf.constant([[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32)
        self.down_kernel = tf.constant([[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=tf.float32)

    def call(self, y_true, y_pred):
        original_mean = tf.reduce_mean(y_true, 3, keepdims=True)
        enhanced_mean = tf.reduce_mean(y_pred, 3, keepdims=True)
        original_pool = tf.nn.avg_pool2d(original_mean, ksize=4, strides=4, padding="VALID")
        enhanced_pool = tf.nn.avg_pool2d(enhanced_mean, ksize=4, strides=4, padding="VALID")

        d_original_left = tf.nn.conv2d(original_pool, self.left_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_original_right = tf.nn.conv2d(original_pool, self.right_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_original_up = tf.nn.conv2d(original_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_original_down = tf.nn.conv2d(original_pool, self.down_kernel, strides=[1, 1, 1, 1], padding="SAME")

        d_enhanced_left = tf.nn.conv2d(enhanced_pool, self.left_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_enhanced_right = tf.nn.conv2d(enhanced_pool, self.right_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_enhanced_up = tf.nn.conv2d(enhanced_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME")
        d_enhanced_down = tf.nn.conv2d(enhanced_pool, self.down_kernel, strides=[1, 1, 1, 1], padding="SAME")

        d_left = tf.square(d_original_left - d_enhanced_left)
        d_right = tf.square(d_original_right - d_enhanced_right)
        d_up = tf.square(d_original_up - d_enhanced_up)
        d_down = tf.square(d_original_down - d_enhanced_down)
        return d_left + d_right + d_up + d_down

def ssim(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

mse = MeanSquaredError()
mae = MeanAbsoluteError()
def supervised_loss(y_true,y_pred):
    return mse(y_true,y_pred) + 0.1 * mae(y_true,y_pred)

vggg = VGG19(include_top=False, weights='imagenet')
def perceptual_loss(y_true, y_pred):
    vgg = vggg
    vgg.trainable = False
    content_layers = ['block5_conv2']  
    outputs = [vgg.get_layer(name).output for name in content_layers]
    model = keras.Model([vgg.input], outputs)
    true_features = model(y_true)
    pred_features = model(y_pred)
    return mse(true_features, pred_features)

@tf.keras.utils.register_keras_serializable()
class ZeroDCE(keras.Model):
    def __init__(self, **kwargs):
        super(ZeroDCE, self).__init__(**kwargs)
        self.dce_model = build_dce_net()

    def compile(self, learning_rate, **kwargs):
        super(ZeroDCE, self).compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.spatial_constancy_loss = SpatialConsistencyLoss(reduction="none")

    def get_enhanced_image(self, low_light, output):
        r1 = output[:, :, :, :3]
        r2 = output[:, :, :, 3:6]
        r3 = output[:, :, :, 6:9]
        r4 = output[:, :, :, 9:12]
        r5 = output[:, :, :, 12:15]
        r6 = output[:, :, :, 15:18]
        r7 = output[:, :, :, 18:21]
        r8 = output[:, :, :, 21:24]
        x = low_light + r1 * (tf.square(low_light) - low_light)
        x = x + r2 * (tf.square(x) - x)
        x = x + r3 * (tf.square(x) - x)
        enhanced_image = x + r4 * (tf.square(x) - x)
        x = enhanced_image + r5 * (tf.square(enhanced_image) - enhanced_image)
        x = x + r6 * (tf.square(x) - x)
        x = x + r7 * (tf.square(x) - x)
        enhanced_image = x + r8 * (tf.square(x) - x)
        return enhanced_image

    def call(self, low_light):
        dce_net_output = self.dce_model(low_light)
        return self.get_enhanced_image(low_light, dce_net_output)

    def compute_losses(self, low_light, output, high_light):
        enhanced_image = self.get_enhanced_image(low_light, output)
        loss_illumination = 200 * illumination_smoothness_loss(output)
        loss_spatial_constancy = tf.reduce_mean(self.spatial_constancy_loss(enhanced_image, low_light))
        loss_color_constancy = 5 * tf.reduce_mean(color_constancy_loss(enhanced_image))
        loss_exposure = 10 * tf.reduce_mean(exposure_loss(enhanced_image))
        loss_supervised = 26 * supervised_loss(high_light, enhanced_image)
        loss_perceptual = 7 * tf.reduce_mean(perceptual_loss(high_light, enhanced_image))
        loss_ssim = 9 * ssim(high_light, enhanced_image)
        total_loss = (loss_illumination 
         + loss_spatial_constancy 
         + loss_color_constancy 
         + loss_exposure 
         + loss_supervised 
         + loss_perceptual
         + loss_ssim)
        return {"total_loss": total_loss, "illumination_smoothness_loss": loss_illumination,
                "spatial_constancy_loss": loss_spatial_constancy, "color_constancy_loss": loss_color_constancy,
                "exposure_loss": loss_exposure, "supervised_loss": loss_supervised, "perceptual_loss": loss_perceptual,
                "ssim_loss": loss_ssim}

    def train_step(self, data):
        low_light, high_light = data  # Unpack the input data
        with tf.GradientTape() as tape:
            output = self.dce_model(low_light)
            losses = self.compute_losses(low_light, output, high_light)
        gradients = tape.gradient(losses["total_loss"], self.dce_model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.dce_model.trainable_weights))
        return losses

    def test_step(self, data):
        low_light, high_light = data  # Unpack the input data
        output = self.dce_model(low_light)
        return self.compute_losses(low_light, output, high_light)

# Initialize and load the model
zero_dce_model = ZeroDCE()
zero_dce_model.dce_model.load_weights('zero_dce_model_full.weights.h5')

def apply_gaussian_blur(image, kernel_size=(3, 3), sigma=0.6):
    image_np = np.array(image)
    blurred_image_np = cv2.GaussianBlur(image_np, kernel_size, sigma)
    return blurred_image_np

def post_process(image):
    image_np = np.array(image)
    denoised_image = cv2.fastNlMeansDenoisingColored(image_np, None, 7, 7, 5, 5)
    denoised_image = apply_gaussian_blur(denoised_image)
    return Image.fromarray(denoised_image)    

def infer(original_image):
    original_image = original_image.convert('RGB')
    original_image = original_image.resize((600, 400))
    image = tf.keras.preprocessing.image.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = zero_dce_model(image)
    output_image_np = np.uint8(output_image[0] * 255)
    output_image_pil = Image.fromarray(output_image_np)
    output_image_pil = post_process(output_image_pil) 
    return output_image_pil


low_light_dir = 'test/low/'
high_light_dir = 'test/high/'
    
os.makedirs(high_light_dir, exist_ok=True)
    
low_light_images = sorted(glob(os.path.join(low_light_dir, '*.png')))
    
for low_light_path in low_light_images:
    original_image = Image.open(low_light_path)
    enhanced_image = infer(original_image)
    base_name = os.path.basename(low_light_path)
    high_light_path = os.path.join(high_light_dir, base_name)
    enhanced_image.save(high_light_path)
    print(f"Processed {base_name} and saved to {high_light_path}")
