import numpy as np
import tensorflow as tf

# Загрузка тренировочных и тестовых данных
y_smp_train = np.load('/content/gdrive/MyDrive/20231013-SFO/y_smp_train.npy')
pars_smp_train = np.load('/content/gdrive/MyDrive/20231013-SFO/pars_smp_train.npy')
y_smp_test = np.load('/content/gdrive/MyDrive/20231013-SFO/y_smp_test.npy')

# Размеры данных
train_size = y_smp_train.shape[0]
test_size = y_smp_test.shape[0]
param_size = pars_smp_train.shape[1]

# Создание модели TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(200, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(1024, activation='relu'), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(param_size)
])

# Компиляция и обучение модели
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(y_smp_train, pars_smp_train, epochs=1, batch_size=32)

# Прогнозирование параметров для тестовых данных
predicted_params = model.predict(y_smp_test)

# Расчет характеристик апостериорного распределения
mean = np.mean(predicted_params, axis=0)
quantile_10 = np.percentile(predicted_params, 10, axis=0)
quantile_25 = np.percentile(predicted_params, 25, axis=0)
quantile_50 = np.percentile(predicted_params, 50, axis=0)
quantile_75 = np.percentile(predicted_params, 75, axis=0)
quantile_90 = np.percentile(predicted_params, 90, axis=0)

# Создание тензора с характеристиками апостериорного распределения
result = np.stack((mean, quantile_10, quantile_25, quantile_50, quantile_75, quantile_90), axis=-1)

# Сохранение результата
np.save('/content/gdrive/MyDrive/20231013-SFO/predicted_params.npy', result)
print('predicted_params')
