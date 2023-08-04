# import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from sklearn.linear_model import LinearRegression
from mapie.regression import MapieRegressor
from tqdm import tqdm


# define ivim function
def ivim(b, Dp, Dt, Fp):
    return Fp * np.exp(-b * Dp) + (1 - Fp) * np.exp(-b * Dt)


# define b values
b_values = np.array([0, 10, 20, 60, 150, 300, 500, 1000])
# training data
num_samples = 100000
X_train = np.zeros((num_samples, len(b_values)))
X_calib = np.zeros((num_samples // 100, len(b_values)))

Dp_train = np.random.uniform(0.01, 0.1, size=(len(X_train), 1))
Dt_train = np.random.uniform(0.0005, 0.002, size=(len(X_train), 1))
Fp_train = np.random.uniform(0.1, 0.4, size=(len(X_train), 1))
for i in range(len(X_train)):
    X_train[i, :] = ivim(b_values, Dp_train[i], Dt_train[i], Fp_train[i])

Dp_calib = np.random.uniform(0.01, 0.1, size=(len(X_calib), 1))
Dt_calib = np.random.uniform(0.0005, 0.002, size=(len(X_calib), 1))
Fp_calib = np.random.uniform(0.1, 0.4, size=(len(X_calib), 1))
for i in range(len(X_calib)):
    X_calib[i, :] = ivim(b_values, Dp_calib[i], Dt_calib[i], Fp_calib[i])

# add some noise
X_train_real = X_train + np.random.normal(scale=0.01, size=(len(X_train), len(b_values)))
X_train_imag = np.random.normal(scale=0.01, size=(len(X_train), len(b_values)))
X_train = np.sqrt(X_train_real ** 2 + X_train_imag ** 2)

X_calib_real = X_calib + np.random.normal(scale=0.01, size=(len(X_calib), len(b_values)))
X_calib_imag = np.random.normal(scale=0.01, size=(len(X_calib), len(b_values)))
X_calib = np.sqrt(X_calib_real ** 2 + X_calib_imag ** 2)


class Net(nn.Module):
    def __init__(self, b_values_no0):
        super(Net, self).__init__()

        self.b_values_no0 = b_values_no0
        self.fc_layers = nn.ModuleList()
        for i in range(3):  # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values_no0), len(b_values_no0)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values_no0), 3))

    def forward(self, X):
        params = torch.abs(self.encoder(X))  # Dp, Dt, Fp
        Dp = params[:, 0].unsqueeze(1)
        Dt = params[:, 1].unsqueeze(1)
        Fp = params[:, 2].unsqueeze(1)

        X = Fp * torch.exp(-self.b_values_no0 * Dp) + (1 - Fp) * torch.exp(-self.b_values_no0 * Dt)

        return X, Dp, Dt, Fp


# Network
b_values_no0 = torch.FloatTensor(b_values[1:])
net = Net(b_values_no0)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

batch_size = 128
num_batches = len(X_train) // batch_size
X_train = X_train[:, 1:]  # exlude the b=0 value as signals are normalized
X_calib = X_calib[:, 1:]
trainloader = utils.DataLoader(torch.from_numpy(X_train.astype(np.float32)),
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=True)

# Best loss
best = 1e16
patience = 10
num_epochs = 200

# Train
num_bad_epochs = 0
losses = np.zeros(num_epochs)
fig, ax = plt.subplots()
plt.ion()

#%%
for epoch in range(num_epochs):
    print("-----------------------------------------------------------------")
    print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
    net.train()
    running_loss = 0.
    for i, X_batch in enumerate(tqdm(trainloader), 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        X_pred, Dp_pred, Dt_pred, Fp_pred = net(X_batch)
        loss = criterion(X_pred, X_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print("Loss: {}".format(running_loss))
    # early stopping
    if running_loss < best:
        print("############### Saving good model ###############################")
        final_model = net.state_dict()
        best = running_loss
        num_bad_epochs = 0
    else:
        num_bad_epochs = num_bad_epochs + 1
        if num_bad_epochs == patience:
            print("Done, best loss: {}".format(best))
            break

    losses[epoch] = running_loss
    plt.ylim((0, max(losses[:])))
    plt.xlim((0, num_epochs-1))
    plt.title('Losses per epochs:')
    plt.plot(list(range(epoch + 1)), losses[:epoch + 1], '-o')
    plt.pause(0.1)
plt.show(block=False)
# plt.waitforbuttonpress()


print("Done")
# Restore best model
net.load_state_dict(final_model)


#############################################################
# 1 computing non-conformity scores on the calibration set  #
#############################################################
net.eval()
with torch.no_grad():
    S_calib_pred, Dp_calib_pred, Dt_calib_pred, Fp_calib_pred = net(torch.from_numpy(X_calib.astype(np.float32)))

Dp_calib_pred = Dp_calib_pred.numpy()
Dt_calib_pred = Dt_calib_pred.numpy()
Fp_calib_pred = Fp_calib_pred.numpy()

if np.mean(Dp_calib_pred) < np.mean(Dt_calib_pred):
    Dp_calib_pred, Dt_calib_pred = Dt_calib_pred, Dp_calib_pred
    Fp_calib_pred = 1 - Fp_calib_pred

Dp_scores = np.sort(np.abs(Dp_calib_pred - Dp_calib), axis=None)
Dt_scores = np.sort(np.abs(Dt_calib_pred - Dt_calib), axis=None)
Fp_scores = np.sort(np.abs(Fp_calib_pred - Fp_calib), axis=None)

###################################################################
# 2 extract the 1-alpha quantile fron the non-conformity scores   #
###################################################################
alpha = 0.15
k = int(np.ceil((len(X_calib) + 1) * (1 - alpha)))
Dp_quantile = Dp_scores[k]
Dt_quantile = Dt_scores[k]
Fp_quantile = Fp_scores[k]


x = np.linspace(0, len(X_calib), len(X_calib))
figure, axis = plt.subplots(3, 2)
axis[0, 0].set_title('Dp Estimation with coverage:')
axis[0, 0].plot(x, Dp_calib, '-.')
axis[0, 0].plot(x, Dp_calib_pred, '-.', color='red')
axis[0, 0].plot(x, np.add(Dp_calib, Dp_quantile), '-o', color='black')
axis[0, 0].plot(x, np.add(Dp_calib, -Dp_quantile), '-o', color='black')
axis[1, 0].set_title('Dt Estimation with coverage:')
axis[1, 0].plot(x, Dt_calib, '-.')
axis[1, 0].plot(x, Dt_calib_pred, '-.', color='red')
axis[1, 0].plot(x, np.add(Dt_calib, Dt_quantile), '-o', color='black')
axis[1, 0].plot(x, np.add(Dt_calib, -Dt_quantile), '-o', color='black')
axis[2, 0].set_title('Fp Estimation with coverage:')
axis[2, 0].plot(x, Fp_calib, '-.')
axis[2, 0].plot(x, Fp_calib_pred, '-.', color='red')
axis[2, 0].plot(x, np.add(Fp_calib, Fp_quantile), '-o', color='black')
axis[2, 0].plot(x, np.add(Fp_calib, -Fp_quantile), '-o', color='black')
axis[0, 1].set_title(f'Dp Coverage size Box plot\ntop whisker is at 1 - $alpha$ = {1-alpha} signifance level')
axis[0, 1].boxplot(Dp_scores, whis=[25, 100 * (1 - alpha)])
axis[1, 1].set_title(f'Dt Coverage size Box plot\ntop whisker is at 1 - $alpha$ = {1-alpha} signifance level')
axis[1, 1].boxplot(Dt_scores, whis=[25, 100 * (1 - alpha)])
axis[2, 1].set_title(f'Fp Coverage size Box plot\ntop whisker is at 1 - $alpha$ = {1-alpha} signifance level')
axis[2, 1].boxplot(Fp_scores, whis=[25, 100 * (1 - alpha)])
plt.show()

#
# # define parameter values in the three regions
# S0_region0, S0_region1, S0_region2 = 1500, 1400, 1600
# Dp_region0, Dp_region1, Dp_region2 = 0.02, 0.04, 0.06
# Dt_region0, Dt_region1, Dt_region2 = 0.0015, 0.0010, 0.0005
# Fp_region0, Fp_region1, Fp_region2 = 0.1, 0.2, 0.3
# # image size
# sx, sy, sb = 100, 100, len(b_values)
# # create image
# dwi_image = np.zeros((sx, sy, sb))
# Dp_truth = np.zeros((sx, sy))
# Dt_truth = np.zeros((sx, sy))
# Fp_truth = np.zeros((sx, sy))
#
# # fill image with simulated values
# for i in range(sx):
#     for j in range(sy):
#         if (40 < i < 60) and (40 < j < 60):
#             # region 0
#             dwi_image[i, j, :] = S0_region0 * ivim(b_values, Dp_region0, Dt_region0, Fp_region0)
#             Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region0, Dt_region0, Fp_region0
#         elif (20 < i < 80) and (20 < j < 80):
#             # region 1
#             dwi_image[i, j, :] = S0_region1 * ivim(b_values, Dp_region1, Dt_region1, Fp_region1)
#             Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region1, Dt_region1, Fp_region1
#         else:
#             # region 2
#             dwi_image[i, j, :] = S0_region2 * ivim(b_values, Dp_region2, Dt_region2, Fp_region2)
#             Dp_truth[i, j], Dt_truth[i, j], Fp_truth[i, j] = Dp_region2, Dt_region2, Fp_region2
# # add some noise
# dwi_image_real = dwi_image + np.random.normal(scale=15, size=(sx, sy, sb))
# dwi_image_imag = np.random.normal(scale=15, size=(sx, sy, sb))
# dwi_image = np.sqrt(dwi_image_real ** 2 + dwi_image_imag ** 2)
# # plot simulated diffusion weighted image
# fig, ax = plt.subplots(2, 4)  # , figsize=(10, 10))
# b_id = 0
# for i in range(2):
#     for j in range(4):
#         ax[i, j].imshow(dwi_image[:, :, b_id], cmap='gray', clim=(0, 1600))
#         ax[i, j].set_title('b = ' + str(b_values[b_id]))
#         ax[i, j].set_xticks([])
#         ax[i, j].set_yticks([])
#         b_id += 1
# # plt.subplots_adjust(hspace=-0.6)
# plt.show()
#
# # normalize signal
# dwi_image_long = np.reshape(dwi_image, (sx * sy, sb))
# S0 = np.expand_dims(dwi_image_long[:, 0], axis=-1)
# dwi_image_long = dwi_image_long[:, 1:] / S0
#
# net.eval()
# with torch.no_grad():
#     _, Dp, Dt, Fp = net(torch.from_numpy(dwi_image_long.astype(np.float32)))
#
# Dp = Dp.numpy()
# Dt = Dt.numpy()
# Fp = Fp.numpy()
#
# # make sure Dp is the larger value between Dp and Dt
# if np.mean(Dp) < np.mean(Dt):
#     Dp, Dt = Dt, Dp
#     Fp = 1 - Fp
#
# fig, ax = plt.subplots(2, 3)
#
# Dp_plot = ax[0, 0].imshow(np.reshape(Dp, (sx, sy)), cmap='gray', clim=(0.01, 0.07))
# ax[0, 0].set_title('Dp, estimated')
# ax[0, 0].set_xticks([])
# ax[0, 0].set_yticks([])
# fig.colorbar(Dp_plot, ax=ax[0, 0], fraction=0.046, pad=0.04)
#
# Dp_t_plot = ax[1, 0].imshow(Dp_truth, cmap='gray', clim=(0.01, 0.07))
# ax[1, 0].set_title('Dp, ground truth')
# ax[1, 0].set_xticks([])
# ax[1, 0].set_yticks([])
# fig.colorbar(Dp_t_plot, ax=ax[1, 0], fraction=0.046, pad=0.04)
#
# Dt_plot = ax[0, 1].imshow(np.reshape(Dt, (sx, sy)), cmap='gray', clim=(0, 0.002))
# ax[0, 1].set_title('Dt, estimated')
# ax[0, 1].set_xticks([])
# ax[0, 1].set_yticks([])
# fig.colorbar(Dt_plot, ax=ax[0, 1], fraction=0.046, pad=0.04)
#
# Dt_t_plot = ax[1, 1].imshow(Dt_truth, cmap='gray', clim=(0, 0.002))
# ax[1, 1].set_title('Dt, ground truth')
# ax[1, 1].set_xticks([])
# ax[1, 1].set_yticks([])
# fig.colorbar(Dt_t_plot, ax=ax[1, 1], fraction=0.046, pad=0.04)
#
# Fp_plot = ax[0, 2].imshow(np.reshape(Fp, (sx, sy)), cmap='gray', clim=(0, 0.4))
# ax[0, 2].set_title('Fp, estimated')
# ax[0, 2].set_xticks([])
# ax[0, 2].set_yticks([])
# fig.colorbar(Fp_plot, ax=ax[0, 2], fraction=0.046, pad=0.04)
#
# Fp_t_plot = ax[1, 2].imshow(Fp_truth, cmap='gray', clim=(0, 0.4))
# ax[1, 2].set_title('Fp, ground truth')
# ax[1, 2].set_xticks([])
# ax[1, 2].set_yticks([])
# fig.colorbar(Fp_t_plot, ax=ax[1, 2], fraction=0.046, pad=0.04)
#
# # plt.subplots_adjust(hspace=-0.5)
# plt.show()
