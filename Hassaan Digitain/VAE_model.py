import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf
#from tensorflow.keras import layers
#from tensorflow.keras.layers import Dense, Input
#from tensorflow.keras import Sequential
#from tensorflow.keras.activations import sigmoid

# Load the data
file_path = 'Claculated-Gc and MMR.xlsx'
df = pd.read_excel(file_path)

new_df = df.drop(columns=['Proben_name'])
# Normalize the data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(new_df.values)

# Define the latent space dimension
latent_dim = 2

# Define the encoder as a Sequential model
encoder = Sequential([
    layers.InputLayer(input_shape=(data_normalized.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(latent_dim * 2)  # Output both mean and log variance
])

# Define the sampling function
def sampling(z_mean, z_log_var):
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Get the z_mean and z_log_var from the encoder
inputs = layers.Input(shape=(data_normalized.shape[1],))
encoder_output = encoder(inputs)
z_mean, z_log_var = layers.Lambda(lambda x: x[:, :latent_dim])(encoder_output), layers.Lambda(lambda x: x[:, latent_dim:])(encoder_output)

# Sample from the latent space
z = sampling(z_mean, z_log_var)

# Define the decoder as a Sequential model
decoder = Sequential([
    layers.InputLayer(input_shape=(latent_dim,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(data_normalized.shape[1], activation='sigmoid')
])


# Define the VAE model
vae_outputs = decoder(z)
vae = Model(inputs, vae_outputs)

# Define the VAE loss
#lenght=len(inputs)
#reconstruction_loss =(inputs-vae_outputs)**2
#inputs=tf.cast(inputs,dtype=tf.float32)
#vae_outputs=tf.cast(vae_outputs,dtype=tf.float32)
#reconstruction_loss=MeanSquaredError(inputs,vae_outputs)
reconstruction_loss=1
reconstruction_loss *= data_normalized.shape[1]
kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
kl_loss = tf.reduce_sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Train the VAE
vae.fit(data_normalized, epochs=50, batch_size=32, validation_split=0.1)

# Generate 50 new samples
n_samples = 50
z_samples = np.random.normal(size=(n_samples, latent_dim))
generated_data = decoder.predict(z_samples)

# Inverse transform the generated data back to the original scale
generated_data_original_scale = scaler.inverse_transform(generated_data)

# Convert to DataFrame
generated_df = pd.DataFrame(generated_data_original_scale, columns=new_df.columns)

# Save the synthetic data to a new Excel file
output_file_path = 'synthetic_data.xlsx'
generated_df.to_excel(output_file_path, index=False)

# Display the first few rows of the generated synthetic data
print(generated_df.head())
