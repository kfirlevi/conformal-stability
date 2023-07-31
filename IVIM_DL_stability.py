# import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import scipy.special as ss

# define ivim and RMSE functions
def ivim(b, Dp, Dt, Fp):
    return Fp*np.exp(-b*Dp) + (1-Fp)*np.exp(-b*Dt)
def RMSE(a, b):
    #size_vec = np.shape(a)
    RMSE = np.sum((a-b)**2)/(100*100)
    RMSE = np.sqrt(RMSE)
    return RMSE
def binomial_prob(B, N, delta):
    return ss.binom(N, B) * (delta ** B) * ((1-delta) ** (N-B))
def getQunatile(net, calibration_dataset, alpha):
    net.eval()
    with torch.no_grad():
        S_calib_pred, Dp_calib_pred, Dt_calib_pred, Fp_calib_pred = net(torch.from_numpy(calibration_dataset.astype(np.float32)))

    Dp_scores = np.sort(np.abs(Dp_calib_pred - Dp_calib), axis=None)
    Dt_scores = np.sort(np.abs(Dt_calib_pred - Dt_calib), axis=None)
    Fp_scores = np.sort(np.abs(Fp_calib_pred - Fp_calib), axis=None)

    ###################################################################
    # 2 extract the 1-alpha quantile from the non-conformity scores   #
    ###################################################################
    # alpha = 0.05
    k = int(np.ceil((len(X_calib) + 1) * (1 - alpha)))
    Fp_quantile = Fp_scores[k]
    Dp_quantile = Dp_scores[k]
    Dt_quantile = Dt_scores[k]

    return Dp_quantile, Dt_quantile, Fp_quantile


# define b values
b_values = np.array([0, 10, 20, 60, 150, 300, 500, 1000])
b_values_no0 = torch.FloatTensor(b_values[1:])

# training data
num_samples = 2000000
X_train = np.zeros((num_samples, len(b_values)))
Dp_train = np.random.uniform(0.01, 0.1, size=(num_samples, ))
Dt_train = np.random.uniform(0.0005, 0.002, size=(num_samples, ))
Fp_train = np.random.uniform(0.1, 0.4, size=(num_samples, ))
for index, b_value in enumerate(b_values):
    X_train[:, index] = ivim(b_value, Dp_train, Dt_train, Fp_train)
# add some noise
X_train_real = X_train + np.random.normal(scale=0.01, size=(num_samples, len(b_values)))
X_train_imag = np.random.normal(scale=0.01, size=(num_samples, len(b_values)))
X_train = np.sqrt(X_train_real**2 + X_train_imag**2)

# calibration data
num_samples_calib = 25000
X_calib = np.zeros((num_samples_calib, len(b_values)))
Dp_calib = np.random.uniform(0.01, 0.1, size=(num_samples_calib, ))
Dt_calib = np.random.uniform(0.0005, 0.002, size=(num_samples_calib, ))
Fp_calib = np.random.uniform(0.1, 0.4, size=(num_samples_calib, ))
for index, b_value in enumerate(b_values):
    X_calib[:, index] = ivim(b_value, Dp_calib, Dt_calib, Fp_calib)
X_calib = X_calib[:, 1:]

# test data
num_samples_test = 100
X_test = np.zeros((num_samples_test, len(b_values)))
Dp_test = np.random.uniform(0.01, 0.1, size=(num_samples_test, ))
Dt_test = np.random.uniform(0.0005, 0.002, size=(num_samples_test, ))
Fp_test = np.random.uniform(0.1, 0.4, size=(num_samples_test, ))
for index, b_value in enumerate(b_values):
    X_test[:, index] = ivim(b_value, Dp_test, Dt_test, Fp_test)

# Create the neural network and instantiate it
class Net(nn.Module):
    def __init__(self, b_values_no0):
        super(Net, self).__init__()

        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        for i in range(2): # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), 3))

    def forward(self, X):
        params = torch.abs(self.encoder(X))  # Dp, Dt, Fp
        Dp = params[:, 0].unsqueeze(1)
        Dt = params[:, 1].unsqueeze(1)
        Fp = params[:, 2].unsqueeze(1)

        X = Fp*torch.exp(-self.b_values_no0*Dp) + (1-Fp)*torch.exp(-self.b_values_no0*Dt)

        return X, Dp, Dt, Fp

# Loss function
criterion = nn.MSELoss()

# training hyperparameters initialization
batch_size = 128
num_batches = len(X_train) // batch_size
X_train = X_train[:, 1:]  # exclude the b=0 value as signals are normalized
best_loss = 1e16
num_bad_epochs = 0
patience = 10  # 20  # 50
loss_vector = np.zeros((1, 1000))
Dp_vector = np.zeros((1, 1000))
Dt_vector = np.zeros((1, 1000))
Fp_vector = np.zeros((1, 1000))

# stability test hyperparameters initialization
n = 100000  # 200000
kappa = int(np.floor(num_samples / n))
X_train_partitioned = np.zeros(shape=(kappa, n, X_train.shape[1]))
X_train_partitioned_n_minus_1 = np.zeros(shape=(kappa, n - 1, X_train.shape[1]))

# splitting the data
for i in range(kappa):
    partition_start, partition_end = i*n, (i+1)*n
    X_train_partitioned[i, :, :] = X_train[partition_start:partition_end, :]
    X_train_partitioned_n_minus_1[i, :, :] = X_train[partition_start:partition_end-1, :]

random_seeds = np.random.randint(low=100, high=900, size=kappa)
print('random_seeds = ' + str(random_seeds))
iteration_delta = np.zeros(shape=(kappa, 3))

# net initialization
net_stability = np.empty(shape=(kappa, 2), dtype=object)
for k in range(kappa):
    for i in range(2):
        torch.manual_seed(random_seeds[k])
        net_stability[k, i] = Net(b_values_no0)

quantile_stability = np.zeros(shape=(kappa, 2, 3))

# training for stability inference
for k in range(kappa):
    X_train_stability = np.array([X_train_partitioned[k, :, :], X_train_partitioned_n_minus_1[k, :, :]], dtype=object)
    stability_sample = X_test[k, 1:].reshape((1, 7))
    stability_sample_pred = np.zeros([X_train_stability.shape[0], 3])

    trainloader_stability = np.empty(X_train_stability.shape, dtype=object)
    for index, dataset in enumerate(X_train_stability):
        trainloader_stability[index] = utils.DataLoader(torch.from_numpy(dataset.astype(np.float32)),
                                                        batch_size=batch_size, shuffle=True, drop_last=True)

    # Train
    for index, dataset in enumerate(X_train_stability):
        print('iteration ' + str(k) + ', start training model ' + str(index))
        trainloader = trainloader_stability[index]
        best_loss = 1e16
        num_bad_epochs = 0
        net = net_stability[k, index]
        optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 0.0001
        for epoch in range(1000):
            print("-----------------------------------------------------------------")
            print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
            net.train()
            running_loss = 0

            for i, X_batch in enumerate(tqdm(trainloader), 0):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                X_pred, Dp_pred, Dt_pred, Fp_pred = net(X_batch)
                loss = criterion(X_pred, X_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            loss_vector[0,epoch] = running_loss
            Dp_vector[0,epoch] = Dp_pred[0]
            Dt_vector[0,epoch] = Dt_pred[0]
            Fp_vector[0,epoch] = Fp_pred[0]

            print("Loss: {}".format(running_loss))
            # early stopping
            if running_loss < best_loss:
                print("############### Saving good model ###############################")
                final_model = net.state_dict()
                best_loss = running_loss
                num_bad_epochs = 0
            else:
                num_bad_epochs = num_bad_epochs + 1
                if num_bad_epochs == patience:
                    print("-----------------------------------------------------------------")
                    print("Done due to patience, best loss: {}".format(best_loss))
                    print("Stopped at epoch: {}".format(epoch))
                    break

        print("-----------------------------------------------------------------")
        print("Done")
        print("best loss: {}".format(best_loss))

        # Restore best model
        net.load_state_dict(final_model)

        net.eval()
        with torch.no_grad():
            _, Dp, Dt, Fp = net(torch.from_numpy(stability_sample.astype(np.float32)))

        Dp = Dp.numpy()
        Dt = Dt.numpy()
        Fp = Fp.numpy()

        # make sure Dp is the larger value between Dp and Dt
        if np.mean(Dp) < np.mean(Dt):
            Dp, Dt = Dt, Dp
            Fp = 1 - Fp

        stability_sample_pred[index, :] = [Dp, Dt, Fp]

        # quantile_stability[k, index] = getQunatile(net, X_calib, alpha=0.05)
        # @TODO: add qunatile calculations and save to an array

    iteration_delta[k, :] = np.abs(stability_sample_pred[0, :] - stability_sample_pred[1, :])
    # @TODO: calculate coverage delta using quantile (euclidean distance between lower and upper)

    print('iteration ' + str(k) + ', prediction for n samples: ' + str(stability_sample_pred[0, :]))
    print('iteration ' + str(k) + ', prediction for n-1 samples: ' + str(stability_sample_pred[1, :]))
    print('iteration ' + str(k) + ', abs delta: ' + str(iteration_delta[k, :]))

# Binomial test for stability
epsilon_values_to_test = np.logspace(-5, 0, 6)
lambda_values_to_test = np.linspace(0.1, 0.5, 5)  # np.linspace(0.05, 0.5, 10)

B_scanned = np.zeros((len(epsilon_values_to_test), 3), dtype=int)
binomial_stats = np.zeros((len(epsilon_values_to_test), len(lambda_values_to_test), 3), dtype=float)
T_hat = np.zeros((len(epsilon_values_to_test), len(lambda_values_to_test), 3), dtype=bool)

for e_index, eps in enumerate(epsilon_values_to_test):
    delta_compare_eps = iteration_delta > np.repeat(eps, 3)
    B_scanned[e_index, :] = np.sum(delta_compare_eps, axis=0)
    for l_index, l in enumerate(lambda_values_to_test):
        binomial_stats[e_index, l_index, :] = binomial_prob(B_scanned[e_index, :], kappa, l)

T_hat = binomial_stats < 0.1
