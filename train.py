"""Trains models with the Blundell et al. Variational Inference approach."""
import argparse
import os
import torch
from torch import nn
from torch.utils import data
from tqdm.auto import tqdm
import math
import torch.nn.functional as F
from train_utils import load_datasets

LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
DATA_DIR = os.path.join(os.getcwd(), 'data', 'processed')
MODEL_DIR = 'models/bnn'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)]).to(DEVICE)
SIGMA_2 = torch.FloatTensor([math.exp(-6)]).to(DEVICE)


class ReparameterizedGaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = ReparameterizedGaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = ReparameterizedGaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=True):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)

class BBBBNN(nn.Module):
    def __init__(self, in_dim, n_classes, pos_weight):
        super().__init__()
        self.l1 = BayesianLinear(in_dim, 128)
        self.l2 = BayesianLinear(128, 128)
        self.l3 = BayesianLinear(128, n_classes)
        self.n_classes = n_classes

        self.nll = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='sum')

    def forward(self, x, sample=False):
        x = x.view(len(x), -1)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = self.l3(x, sample)
        return x

    def log_prior(self):
        return self.l1.log_prior \
               + self.l2.log_prior \
               + self.l3.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior \
               + self.l2.log_variational_posterior \
               + self.l3.log_variational_posterior

    def sample_elbo(self, input, target, samples, num_batches):
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        nlls = torch.zeros(samples).to(DEVICE)

        for i in range(samples):
            output = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
            nlls[i] = self.nll(output.squeeze(-1), target)
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = nlls.mean()

        loss = (log_variational_posterior - log_prior) / num_batches + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood


def train(net, optimizer, epoch, train_loader):
    net.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(
            data, target, samples=4, num_batches=len(train_loader))
        loss.backward()
        optimizer.step()

        performance_dict = {
            'loss': loss.item(),
            'log_prior': log_prior.item(),
            'log_variational_posterior': log_variational_posterior.item(),
            'negative_log_likelihood': negative_log_likelihood.item()
        }

        if batch_idx % 100 == 0:
            print(f'Training loss: {round(performance_dict["loss"], 3)}')

def validate(net, epoch, train_loader, test_loader, samples=128):
    net.eval()

    outputs = torch.zeros(samples, test_loader.dataset.tensors[0].shape[0], net.n_classes).to(
        DEVICE)

    log_priors = torch.zeros(samples, len(test_loader)).to(DEVICE)
    targets = torch.zeros(test_loader.dataset.tensors[0].shape[0])
    log_variational_posteriors = torch.zeros(samples, len(test_loader)).to(DEVICE)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(DEVICE), target.to(DEVICE)
            for i in range(samples):
                loc = batch_idx * test_loader.batch_size
                outputs[i, loc: loc + len(data)] = net(data, sample=True)
                targets[loc: loc + len(data)] = target
                log_priors[i, batch_idx] = net.log_prior()
                log_variational_posteriors[i, batch_idx] = net.log_variational_posterior()

    log_prior = log_priors.mean()
    log_variational_posterior = log_variational_posteriors.mean()

    negative_log_likelihood = torch.stack([net.nll(outputs[i].squeeze(-1), targets) for i in range(
        outputs.shape[0])]).mean() / len(test_loader)

    loss = (log_variational_posterior - log_prior) / len(train_loader) + negative_log_likelihood

    print(f"Loss: \t {loss}")

    if not os.path.exists(MODEL_DIR):

        os.makedirs(MODEL_DIR, exist_ok=True)

    save_path = os.path.join(MODEL_DIR, f'{epoch}.pth')

    torch.save(net.state_dict(), save_path)


def main(train_batch_size: int, test_batch_size: int, lr: float, epochs: int):
    """prepare and run training and testing.

    parameters
    ----------
    train_batch_size: int
        train batch size
    test_batch_size: int
        test batch size
    lr: float
        learning rate
    epochs: int
        number of epochs to run.
    """

    train_dataset, test_dataset = load_datasets(data_dir=DATA_DIR)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               **LOADER_KWARGS)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              **LOADER_KWARGS)

    pos_weight = torch.Tensor([1 / 0.0845])
    model = BBBBNN(in_dim=44, n_classes=1, pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs), position=0):
        train(model, optimizer, epoch, train_loader)
        validate(model, epoch, train_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=256)
    args = parser.parse_args()

    main(args.batch_size, args.test_batch_size, args.lr, args.epochs)