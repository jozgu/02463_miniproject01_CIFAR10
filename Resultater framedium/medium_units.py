#%%
# Install GPy, GPyOpt

import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import matplotlib.pyplot as plt
import GPy
from GPyOpt.methods import BayesianOptimization
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, cross_val_score


def preprocess_data(X, Y):
    X = K.applications.densenet.preprocess_input(X)
    Y = K.utils.to_categorical(Y)
    return X, Y

# load the Cifar10 dataset, 50,000 training images and 10,000 test images (here used as validation data)
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

x_train, y_train, x_test, y_test = x_train[:1000], y_train[:1000], x_test[:200], y_test[:200]

x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

input_tensor = K.Input(shape=(32, 32, 3))
# resize images to the image size upon which the network was pre-trained
resized_images = K.layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(input_tensor)
base_model = K.applications.DenseNet201(include_top=False,
                                        weights='imagenet',
                                        input_tensor=resized_images,
                                        input_shape=(224, 224, 3),
                                        pooling='max',
                                        classes=1000)
output = base_model.layers[-1].output
base_model = K.models.Model(inputs=input_tensor, outputs=output)


# using the training data
train_datagen = K.preprocessing.image.ImageDataGenerator()
train_generator = train_datagen.flow(x_train,
                                     y_train,
                                     batch_size=32,
                                     shuffle=False)
features_train = base_model.predict(train_generator)
# repeat the same operation with the test data (here used for validation)
val_datagen = K.preprocessing.image.ImageDataGenerator()
val_generator = val_datagen.flow(x_test,
                                 y_test,
                                 batch_size=32,
                                 shuffle=False)
features_valid = base_model.predict(val_generator)

def build_model(unit1, unit2, rate = 0.8, learning_rate = 0.005, l2=(0.1)*1e-3, activation=1):
  """function that builds a model for the head classifier"""
  # weights are initialized as per the he et al. method
  initializer = K.initializers.he_normal()
  input_tensor = K.Input(shape=features_train.shape[1])
  activation_dict = {1: 'relu', 2: 'elu', 3: 'tanh'}
  layer1 = K.layers.Dense(units=unit1,
                         activation=activation_dict[activation],
                         kernel_initializer=initializer,
                        kernel_regularizer=K.regularizers.l2(l2=l2))
  output = layer1(input_tensor)
  layer1 = K.layers.Dense(units=unit2,
                         activation=activation_dict[activation],
                         kernel_initializer=initializer,
                        kernel_regularizer=K.regularizers.l2(l2=l2))
  output = layer1(input_tensor)
  dropout = K.layers.Dropout(rate)
  output = dropout(output)
  softmax = K.layers.Dense(units=10,
                           activation='softmax',
                           kernel_initializer=initializer,
                        kernel_regularizer=K.regularizers.l2(l2=l2))
  output = softmax(output)
  model = K.models.Model(inputs=input_tensor, outputs=output)
  # compile the densely-connected head classifier (here, "model")
  model.compile(
           optimizer=K.optimizers.Adam(learning_rate=learning_rate),
           loss='categorical_crossentropy',
           metrics=['accuracy'])
  # Define some callback functions to be used by the model during training
  # reduce learning rate when val_accuracy has stopped improving
  lr_reduce = K.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                            factor=0.6,
                                            patience=2,
                                            verbose=1,
                                            mode='max',
                                            min_lr=1e-7)
  # stop training when val_accuracy has stopped improving
  early_stop = K.callbacks.EarlyStopping(monitor='val_accuracy',
                                         patience=3,
                                         verbose=1,
                                         mode='max')
  # callback to save the Keras model and (best) weights obtained on an epoch basis
  checkpoint = K.callbacks.ModelCheckpoint('cifar10.h5',
                                           monitor='val_accuracy',
                                           verbose=1,
                                           save_weights_only=False,
                                           save_best_only=True,
                                           mode='max',
                                           save_freq='epoch')
  return model, lr_reduce, early_stop, checkpoint


def fit_model(model, lr_reduce, early_stop, checkpoint):
  """function that trains the head classifier"""
  history = model.fit(features_train, y_train,
                      batch_size=32,
                      epochs=30,
                      verbose=1,
                      callbacks=[lr_reduce, early_stop, checkpoint],
                      validation_data=(features_valid, y_test),
                      shuffle=True)
  return history


def evaluate_model(model):
  """function that evaluates the head classifier"""
  evaluation = model.evaluate(features_valid, y_test)
  return evaluation

# define the kernel for the Bayesian surrogate model using the Matern kernel
kernel = GPy.kern.src.stationary.Matern52(input_dim=50, variance=10.0, lengthscale=2.0)
# hyperparameter bounds
bounds =  [
          {'name': 'unit1', 'type': 'discrete', 'domain': (4, 8, 16, 32, 64, 128, 256, 512, 1024)},
          {'name': 'unit2', 'type': 'discrete', 'domain': (4, 8, 16, 32, 64, 128, 256, 512, 1024)}
          ]
# Note: 'activation' domain parameters (1, 2, 3) correspond to strings ('relu', 'elu', 'tanh'); dictionary defined in build_model()
# objective function for the model optimization:
def f(x):
  """objective function of the Bayesian surrogate model"""
  print()
  print("Hyperparameters:", x)
  # Retrieve 'accuracy' from the previously saved model
  try:
    previous_best_model = K.models.load_model('cifar10_best.h5')
    previous_evaluation = evaluate_model(previous_best_model)
  except Exception:
    previous_best_model = None
  model, lr_reduce, early_stop, checkpoint = build_model(
                                        unit1 = int(x[:,0]),
                                        unit2 = int(x[:,1])
                                        )
  history = fit_model(model, lr_reduce, early_stop, checkpoint)
  evaluation = evaluate_model(model)
  print()
  print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0],
  evaluation[1]))
  print(evaluation)
  # compare previous and current validation accuracies
  if not previous_best_model:
    K.models.save_model(model, 'cifar10_best.h5', overwrite=False,
    include_optimizer=True)
  if previous_best_model and evaluation[1] > previous_evaluation[1]:
    K.models.save_model(model, 'cifar10_best.h5', overwrite=True,
    include_optimizer=True)
  del model
  del previous_best_model
  K.backend.clear_session()
  return evaluation[1]
# Initializing X and Y, and adding noise (if need be)
#X_init = np.array([[int(16)]])
#Y_init = f(X_init)
noise = 0.2
opt = BayesianOptimization(f=f,
                                 domain=bounds,
                                 model_type='GP',
                                 kernel=kernel,
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.01,
                                 # X=X_init,
                                 # Y=-Y_init,
                                 noise_var = noise**2,
                                 exact_feval=False,
                                 normalize_Y=False,
                                 maximize=True,
                                 verbosity=False)

                              
#%% OPRINDELIG efter "opt ="

print()
print("=====================")
print("=====================")
print()
opt.run_optimization(max_iter=15, verbosity=False)
opt.plot_acquisition()
opt.plot_convergence()
opt.save_report('bayes_opt.txt')


# print optimized model
activation_dict = {1: 'relu', 2: 'elu', 3: 'tanh'}
print("optimized accuracy: {0}".format(abs(opt.fx_opt)))


# reinstantiate the best model from saved file
best_model = K.models.load_model('cifar10_best.h5')
best_model.summary()
loss, acc = best_model.evaluate(features_valid, y_test)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

best_model.evaluate(features_valid, y_test)


data_path = 'bayes_opt.txt'
with open(data_path, 'r') as f:
  lines = f.read().split('\n')
for line in lines:
  print(line)


# %% FRA EX 4 efter "opt ="
opt.acquisition.exploration_weight=0.5

#%%
opt.run_optimization(max_iter = 25, verbosity = False)
opt.plot_acquisition()
opt.plot_convergence()
# %%
def f2(x):
  """objective function of the Bayesian surrogate model"""
  print()
  print("Hyperparameters:", x)
  # Retrieve 'accuracy' from the previously saved model
  try:
    previous_best_model = K.models.load_model('cifar10_best.h5')
    previous_evaluation = evaluate_model(previous_best_model)
  except Exception:
    previous_best_model = None
  model, lr_reduce, early_stop, checkpoint = build_model(
                                        unit1 = int(x['unit1']),
                                        unit2 = int(x['unit2'])
                                        )
  history = fit_model(model, lr_reduce, early_stop, checkpoint)
  evaluation = evaluate_model(model)
  print()
  print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0],
  evaluation[1]))
  print(evaluation)
  # compare previous and current validation accuracies
  if not previous_best_model:
    K.models.save_model(model, 'cifar10_best.h5', overwrite=False,
    include_optimizer=True)
  if previous_best_model and evaluation[1] > previous_evaluation[1]:
    K.models.save_model(model, 'cifar10_best.h5', overwrite=True,
    include_optimizer=True)
  del model
  del previous_best_model
  K.backend.clear_session()
  return evaluation[1]
# define the dictionary for GPyOpt
domain_random_sample = {
        # "n_epochs": range(1, 11,2), 
        # "n_epochs": range(10, 1,10), 
        "n_units": (32, 64, 128, 256, 512),
        "lr": (0.5e-3, 1e-3, 1.5e-3, 2e-3),
        }

domain =  [
          {'name': 'unit1', 'type': 'discrete', 'domain': (4, 8, 16, 32, 64, 128, 256, 512, 1024)},
          {'name': 'unit2', 'type': 'discrete', 'domain': (4, 8, 16, 32, 64, 128, 256, 512, 1024)}
          ]
domain_random_sample = {
        'unit1': (4, 8, 16, 32, 64, 128, 256, 512, 1024),
        'unit2': (4, 8, 16, 32, 64, 128, 256, 512, 1024)}


obj_list = []
param_list = list(ParameterSampler(domain_random_sample, n_iter=3, random_state=32))
print("param_list: ", param_list)
for params in param_list:
    accuracy = f2(params)
    obj_list.append(accuracy)
    print(f"Accuracy with params {params} was {accuracy:.2f}")
print("Accuracies for Random Search: ", obj_list)
