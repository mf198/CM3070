import tensorflow as tf
import datetime
import os
import shutil

def train_model(df_train):
    """
    Train the model and log metrics to TensorBoard.
    
    Args:
        df_train: Processed training dataset.
    
    Returns:
        Trained model.
    """
    # Remove old TensorBoard logs (Prevents clutter)
    log_path = "logs/fit"
    if os.path.exists(log_path):
        shutil.rmtree(log_path)  # Clears old logs before training

    # Create a new log directory with timestamp
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Define a simple model (modify as needed)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # Train the model with TensorBoard logging
    model.fit(df_train, epochs=10, batch_size=64, callbacks=[tensorboard_callback])

    return model
