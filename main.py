import os
import argparse
import numpy as np
import logging
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from sr2_optim import *
from autoaugment import *
from resnet import *
import torchvision.models as models
from pyhessian import hessian

Fs = []
stop_criterion = []


def get_sigma0(model):
    criterion = nn.CrossEntropyLoss()
    for batch_id, (inputs, target) in enumerate(train_data):

        if batch_id == 0:
            inputs, target = inputs.to(device), target.to(device)
            # create the hessian computation module
            hessian_comp = hessian(model, criterion, data=(inputs, target), cuda=True)

            # Now let's compute the top eigenvalue. This only takes a few seconds.
            top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
            print("The top Hessian eigenvalue of this model is %.4f" % top_eigenvalues[-1])
            return top_eigenvalues[-1]


def train(epoch):
    model.train()
    lossvals = []
    l1loss = []
    criterion = nn.CrossEntropyLoss()

    for batch_id, (inputs, target) in enumerate(train_data):
        inputs, target = inputs.to(device), target.to(device)

        def closure():
            optimizer.zero_grad()
            op = model(inputs)
            loss_f = criterion(op, target)

            # Accumulate the L1/L0 loss across all layers and save it into the l1loss array
            laccum = 0
            for m in model.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    if args.reg == 'l1':
                        l1 = torch.sum(torch.abs(m.weight))
                        laccum += l1.item()
                    elif args.reg == 'l0':
                        l0 = torch.count_nonzero(m.weight)
                        laccum += l0.item()
                    elif args.reg == 'l12':
                        l12 = torch.sum((torch.abs(m.weight) + 1e-6).sqrt())
                        laccum += l12.item()
                    elif args.reg == 'l23':
                        l23 = torch.sum((torch.abs(m.weight) + 1e-6).pow(2 / 3))
                        laccum += l23.item()
            return loss_f, laccum

        loss, reg, norm_s, sigma, rho, stop = optimizer.step(closure=closure)

        if stop:
            print(' >> Stopping criteria activated')

        stp_c = np.sqrt(sigma) * norm_s
        stop_criterion.append(stp_c)

        l1loss.append(reg)
        current_obj = loss.item() + reg
        Fs.append(current_obj)

        if batch_id % 50 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t f: {:e}\t h: {:e}\t '
                'f+h: {:e}\t ||s||: {:e}'
                '\t sigma: {:e} \t rho:  {:e}'.format(
                    epoch + 1, batch_id * len(inputs), len(train_data.dataset),
                    100. * batch_id / len(train_data), loss.item(), reg, current_obj, norm_s, sigma,
                    rho))


        if stop:
            print(' >> Stopping criteria activated')
            break

    return lossvals, l1loss, stop


def test():
    model.eval()
    test_loss = 0
    total_test = 0
    correct_test = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_test += targets.size(0)
            correct_test += predicted.eq(targets).sum().item()
            test_acc = 100. * correct_test / total_test

    print("Top 1 accuracy: {:.0f}%".format(test_acc))
    return test_acc


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--reg', default='l1', help='which optimizer to use')
parser.add_argument('--lam', type=float, default=0.001, help="reg. param")
parser.add_argument('--max_epochs', type=int, default=300, help="maximum epochs")
parser.add_argument('--eta1', type=float, default=0.00075, help="eta 1 in SR2")
parser.add_argument('--eta2', type=float, default=0.999, help="eta 2 in SR2")
parser.add_argument('--g1', type=float, default=5.57, help="gamma 1 in SR2")
parser.add_argument('--g3', type=float, default=0.79, help="gamma 3 in SR2")
parser.add_argument('--wd', type=float, default=0.02, help="weight decay")
parser.add_argument('--beta', type=float, default=0.9, help="momentum coeff")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--precond', default='andrei', help="Preconditioner: none, Adam, Andrei")

args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the dataset
transform_train = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),  # fill parameter needs torchvision installed from source
        transforms.RandomHorizontalFlip(),
        # CIFAR10Policy(),
        transforms.ToTensor(),
        # Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = torch.utils.data.DataLoader(datasets.CIFAR10(root='dataset', train=True, transform=transform_train,
                                                          download=True),
                                         batch_size=128, num_workers=6, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='dataset', train=False, transform=transform_test),
                                          batch_size=100, num_workers=4, shuffle=False)

print('==> Building model..')
model = models.densenet121(num_classes=10)          # for CIFAR-10

# model = ResNet34()

# model = models.densenet201(n_classes=100)         # for CIFAR-100

model.to(device)

print('==> Computing sigma_0..')
sigma = get_sigma0(model)
# sigma = 41          # works better with R34


# Initialize the optimizer with the given parameters optimizer
if args.precond == 'andrei':
    if args.reg == 'l0':
        optimizer = SR2optimAndreil0(model.parameters(), eta1=args.eta1, eta2=args.eta2, g1=args.g1, g3=args.g3,
                           lmbda=args.lam, sigma=sigma, weight_decay=args.wd, beta=args.beta)
    elif args.reg == 'l1':
        optimizer = SR2optimAndreil1(model.parameters(), eta1=args.eta1, eta2=args.eta2, g1=args.g1, g3=args.g3,
                           lmbda=args.lam, sigma=sigma, weight_decay=args.wd, beta=args.beta)
    elif args.reg == 'l12':
        optimizer = SR2optimAndreil12(model.parameters(), eta1=args.eta1, eta2=args.eta2, g1=args.g1, g3=args.g3,
                           lmbda=args.lam, sigma=sigma, weight_decay=args.wd, beta=args.beta)
    elif args.reg == 'l23':
        optimizer = SR2optimAndreil23(model.parameters(), eta1=args.eta1, eta2=args.eta2, g1=args.g1, g3=args.g3,
                           lmbda=args.lam, sigma=sigma, weight_decay=args.wd, beta=args.beta)
    else:
        print('>> Regularization term not supported')
        
elif args.precond == 'adam':      
    if args.reg == 'l0':
        optimizer = SR2optimAdaml0(model.parameters(), eta1=args.eta1, eta2=args.eta2, g1=args.g1, g3=args.g3,
                           lmbda=args.lam, sigma=sigma, weight_decay=args.wd, beta=args.beta)
    elif args.reg == 'l1':
        optimizer = SR2optimAdaml1(model.parameters(), eta1=args.eta1, eta2=args.eta2, g1=args.g1, g3=args.g3,
                           lmbda=args.lam, sigma=sigma, weight_decay=args.wd, beta=args.beta)
    elif args.reg == 'l12':
        optimizer = SR2optimAdaml12(model.parameters(), eta1=args.eta1, eta2=args.eta2, g1=args.g1, g3=args.g3,
                           lmbda=args.lam, sigma=sigma, weight_decay=args.wd, beta=args.beta)
    elif args.reg == 'l23':
        optimizer = SR2optimAdaml23(model.parameters(), eta1=args.eta1, eta2=args.eta2, g1=args.g1, g3=args.g3,
                           lmbda=args.lam, sigma=sigma, weight_decay=args.wd, beta=args.beta)
    else:
        print('>> Regularization term not supported')
        
elif args.precond == 'none':
    if args.reg == 'l1':
        optimizer = SR2optiml1(model.parameters(), eta1=args.eta1, eta2=args.eta2, g1=args.g1, g3=args.g3,
                           lmbda=args.lam, sigma=sigma, weight_decay=args.wd, beta=args.beta)
    elif args.reg == 'l0':
        optimizer = SR2optiml0(model.parameters(), eta1=args.eta1, eta2=args.eta2, g1=args.g1, g3=args.g3,
                           lmbda=args.lam, sigma=sigma, weight_decay=args.wd, beta=args.beta)
    elif args.reg == 'l12':
        optimizer = SR2optiml12(model.parameters(), eta1=args.eta1, eta2=args.eta2, g1=args.g1, g3=args.g3,
                           lmbda=args.lam, sigma=sigma, weight_decay=args.wd, beta=args.beta)
    elif args.reg == 'l23':
        optimizer = SR2optiml23(model.parameters(), eta1=args.eta1, eta2=args.eta2, g1=args.g1, g3=args.g3,
                           lmbda=args.lam, sigma=sigma, weight_decay=args.wd, beta=args.beta)
    elif args.reg == 'none':
        optimizer = SR2optim(model.parameters(), eta1=args.eta1, eta2=args.eta2, g1=args.g1, g3=args.g3,
                           lmbda=args.lam, sigma=sigma, weight_decay=args.wd, beta=args.beta)
    else:
        print('>> Regularization term not supported')
else:
    print('>> Preconditionner term not supported')

# test_accs = []
# training_losses = []
# l_loss = []
run_id = 'sr2_r34'

# training
for epoch in range(args.max_epochs):
    # train network
    loss, reg, stop = train(epoch)
    # training_losses.append(loss)
    # l_loss.append(reg)

    # test network
    acc_test = test()
    # test_accs.append(acc_test)

    if stop:
        break

print('Successful steps: ', optimizer.successful_steps)
print('Failed steps: ', optimizer.failed_steps)

torch.save(model.state_dict(), "data/weight_final_" + run_id)
# np.save("data/loss_" + run_id, training_losses)
# np.save("data/L__" + run_id, l_loss)
# np.save("data/acc_" + run_id, test_accs)

