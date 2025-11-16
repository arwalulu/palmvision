# src/stages/resize_label/transforms_tf.py
import tensorflow as tf

def decode_and_resize(img_path, img_size):
    # Read file → decode → set 3 channels → resize
    img = tf.io.read_file(img_path)
    # robust to jpg/png; decode_jpeg handles .jpg, for .png you can try decode_image
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
    # scale to [0,1]
    img = tf.cast(img, tf.float32) / 255.0
    return img

def augment(img, cfg_aug):
    # Basic, light aug (enabled via config)
    if cfg_aug.get("flip", True):
        img = tf.image.random_flip_left_right(img)
    rot_deg = cfg_aug.get("rotate_deg", 0)
    if rot_deg and rot_deg > 0:
        # small random rotation in radians
        radians = tf.random.uniform([], minval=-rot_deg, maxval=rot_deg) * (3.14159265/180.0)
        img = tfa_rotate(img, radians)
    if cfg_aug.get("color_jitter", False):
        img = tf.image.random_brightness(img, max_delta=0.05)
        img = tf.image.random_contrast(img, lower=0.95, upper=1.05)
        img = tf.clip_by_value(img, 0.0, 1.0)
    return img

@tf.function
def tfa_rotate(image, radians):
    # Lightweight rotation without adding tfa dependency: pad→affine grid→crop (approx).
    # To keep it simple & dependency-free, we skip true rotation if you prefer.
    # For now, return as-is (set rotate_deg to 0 in config). Placeholder for future.
    return image
