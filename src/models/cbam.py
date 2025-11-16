# src/models/cbam.py

from tensorflow import keras
from tensorflow.keras import layers

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(
            1,
            kernel_size,
            padding="same",
            use_bias=False
        )
        self.sigmoid = layers.Activation("sigmoid")

    def call(self, x):
        K = keras.ops
        avg_pool = K.mean(x, axis=-1, keepdims=True)
        max_pool = K.max(x, axis=-1, keepdims=True)
        concat = keras.ops.concatenate([avg_pool, max_pool], axis=-1)

        attn = self.conv(concat)
        attn = self.sigmoid(attn)
        return x * attn


# ---- Channel Attention (no Lambda) ----

def channel_attention(x, reduction=16, name="ca"):
    ch = x.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)

    shared = keras.Sequential([
        layers.Dense(ch // reduction, activation="relu", use_bias=False),
        layers.Dense(ch, activation="sigmoid", use_bias=False),
    ])

    avg_out = shared(avg_pool)
    max_out = shared(max_pool)
    scale = layers.Add()([avg_out, max_out])
    scale = layers.Reshape((1, 1, ch))(scale)
    return layers.Multiply()([x, scale])


# ---- CBAM Block ----

def cbam_block(x, reduction=16, name="cbam"):
    x = channel_attention(x, reduction=reduction, name=f"{name}_ca")
    x = SpatialAttention(kernel_size=7, name=f"{name}_sa")(x)
    return x
