import os

import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
# tf.config.run_functions_eagerly(True)

BATCH_SIZE   = 32
IMG_HEIGHT   = 224
IMG_WIDTH    = 224
NUM_CLASSES  = 10
AUTOTUNE     = tf.data.experimental.AUTOTUNE
IMG_SIZE = (224, 224)
EPOCHS = 50
initial_epoch  = 0


df = pd.read_csv('data/train.csv')
labels = df['label'].values.astype(np.int32)
pixels = df.drop('label', axis=1).values.astype(np.float32)

pixels /= 255.0

dataset = tf.data.Dataset.from_tensor_slices((pixels, labels))

def preprocess(flat_pixels, label):
    # flat_pixels: [784] -> [28,28,1]
    img = tf.reshape(flat_pixels, [28, 28, 1])
    # 复制到 3 通道 -> [28,28,3]
    img = tf.image.grayscale_to_rgb(img)
    # 调整到目标尺寸 -> [224,224,3]
    img = tf.image.resize(img, [224,224])
    return img, label

dataset = (dataset
           .shuffle(buffer_size=10000)
           .map(preprocess, num_parallel_calls=AUTOTUNE)
           .batch(BATCH_SIZE)
           .prefetch(AUTOTUNE)
)

base_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet',
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = models.Model(inputs=inputs, outputs=outputs)

for layer in model.layers:
    if hasattr(layer, 'trainable'):
        print(f"{layer.name:30s} trainable={layer.trainable}")

model.compile(
    optimizer=optimizers.SGD(learning_rate=1e-3, momentum=0.9),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

ckpt_cb = ModelCheckpoint(
    filepath='checkpoints/resnet50_finetune.{epoch:02d}.h5',
    save_weights_only=False,       # 如果想连 optimizer 状态一起保存，就 False
    save_freq='epoch'              # 每个 epoch 都保存
)

model = load_model('checkpoints/resnet50_finetune.20.h5')

history = model.fit(
    dataset,
    epochs=EPOCHS,                 # 比如要跑 50 轮
    callbacks=[ckpt_cb],
    initial_epoch=initial_epoch    # 你想从第几轮开始，这里默认 0
)

print("训练完成，只有最后一层 Dense 的权重被更新。")

history_df = pd.DataFrame(history.history)
os.makedirs('training_logs', exist_ok=True)
csv_path = os.path.join('training_logs', 'history.csv')
history_df.to_csv(csv_path, index=False)
print(f"训练历史已保存到 {csv_path}")

# 2. 绘制并保存 Loss 曲线
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
loss_png = os.path.join('training_logs', 'loss_curve.png')
plt.savefig(loss_png)
plt.close()
print(f"Loss 曲线已保存到 {loss_png}")

# 3. 绘制并保存 Accuracy 曲线
plt.figure()
# 注意 TensorFlow 1.x 的 History 可能叫 'acc' 而非 'accuracy'
acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
plt.plot(history.history[acc_key], label='train_acc')
if f'val_{acc_key}' in history.history:
    plt.plot(history.history[f'val_{acc_key}'], label='val_acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
acc_png = os.path.join('training_logs', 'accuracy_curve.png')
plt.savefig(acc_png)
plt.close()
print(f"Accuracy 曲线已保存到 {acc_png}")

df_test = pd.read_csv('data/test.csv')  # shape (28000, 784)
test_pixels = df_test.values.astype(np.float32) / 255.0  # 归一化

# ──── 步骤 2：构建 tf.data.Dataset ───────────────────────────
def preprocess_test(flat_pixels):
    img = tf.reshape(flat_pixels, [28, 28, 1])
    img = tf.image.grayscale_to_rgb(img)
    img = tf.image.resize(img, IMG_SIZE)
    return img

ds_test = tf.data.Dataset.from_tensor_slices(test_pixels)
ds_test = ds_test.map(preprocess_test, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ──── 步骤 3：模型预测 ─────────────────────────────────────────
# 输出是 shape (28000, 10) 的概率分布
pred_probs = model.predict(ds_test, verbose=1)
# 取最大值所在的索引作为预测标签
pred_labels = np.argmax(pred_probs, axis=1).astype(int)  # shape (28000,)

# ──── 步骤 4：构造提交 DataFrame ───────────────────────────────
submission = pd.DataFrame({
    'ImageId': np.arange(1, len(pred_labels) + 1),  # 1 到 28000
    'Label': pred_labels
})

# ──── 步骤 5：保存为 CSV ───────────────────────────────────────
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv, first few rows:")
print(submission.head(3))