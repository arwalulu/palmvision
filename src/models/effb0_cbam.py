# src/models/effb0_cbam.py
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from src.configs.loader import load_config
from src.models.cbam import cbam_block

def build_effb0_cbam(cfg=None):
    if cfg is None:
        cfg = load_config()

    num_classes = len(cfg.classes)
    img_size    = int(cfg.training["img_size"])
    l2w         = 1e-5
    drop        = 0.2

    inputs = layers.Input(shape=(img_size, img_size, 3), name="input")

    # (optional) use preprocess_input — our pipeline scaled to [0,1],
    # EfficientNet expects [0,255] with its own normalization. So rescale back:
    x = inputs * 255.0
    x = preprocess_input(x)

    # Pretrained EfficientNetB0
    base = EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x, pooling=None
    )
    # Freeze early blocks (optional, good for small data)
    for layer in base.layers[:200]:
        layer.trainable = False

    feat = base.output                 # shape ~ (7, 7, 1280) at 224x224

    # CBAM on top of backbone features
    attn = cbam_block(feat, reduction=16, name="cbam")

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(attn)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(drop, name="drop_head")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(l2w),
        name="pred"
    )(x)

    model = Model(inputs, outputs, name="EffB0_CBAM")

    # Compile (no training yet)
    opt = tf.keras.optimizers.Adam(learning_rate=float(cfg.training["lr"]))
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        ],
    )
    return model
