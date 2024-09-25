import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
import torchmetrics

class EarlyStop():
    def __init__(self, max_patience, maximize=False):
        self.maximize=maximize
        self.max_patience = max_patience
        self.best_loss = None
        self.patience = max_patience + 0
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            self.patience = self.max_patience + 0
        elif loss < self.best_loss:
            self.best_loss = loss
            self.patience = self.max_patience + 0
        else:
            self.patience -= 1
        return not bool(self.patience)
    
class GroupwiseMetric(Metric):
    def __init__(self, metric,
                 grouping = "cell_lines",
                 average = "macro",
                 nan_ignore=False,
                 alpha=0.00001,
                 residualize = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.grouping = grouping
        self.metric = metric
        self.average = average
        self.nan_ignore = nan_ignore
        self.residualize = residualize
        self.alpha = alpha
        self.add_state("target", default=torch.tensor([]))
        self.add_state("pred", default=torch.tensor([]))
        self.add_state("drugs", default=torch.tensor([]))
        self.add_state("cell_lines", default=torch.tensor([]))
    def get_residual(self, X, y):
        w = self.get_linear_weights(X, y)
        r = y-(X@w)
        return r
    def get_linear_weights(self, X, y):
        A = X.T@X
        Xy = X.T@y
        n_features = X.size(1)
        A.flatten()[:: n_features + 1] += self.alpha
        return torch.linalg.solve(A, Xy).T
    def get_residual_ind(self, y, drug_id, cell_id, alpha=0.001):
        X = torch.cat([y.new_ones(y.size(0), 1),
                       torch.nn.functional.one_hot(drug_id),
                       torch.nn.functional.one_hot(cell_id)], 1).float()
        return self.get_residual(X, y)

    def compute(self) -> Tensor:
        if self.grouping == "cell_lines":
            grouping = self.cell_lines
        elif self.grouping == "drugs":
            grouping = self.drugs
        metric = self.metric
        if not self.residualize:
            y_obs = self.target
            y_pred = self.pred
        else:
            y_obs = self.get_residual_ind(self.target, self.drugs, self.cell_lines)
            y_pred = self.get_residual_ind(self.pred, self.drugs, self.cell_lines)
        average = self.average
        nan_ignore = self.nan_ignore
        unique = grouping.unique()
        proportions = []
        metrics = []
        for g in unique:
            is_group = grouping == g
            metrics += [metric(y_obs[grouping == g], y_pred[grouping == g])]
            proportions += [is_group.sum()/len(is_group)]
        if average is None:
            return torch.stack(metrics)
        if (average == "macro") & (nan_ignore):
            return torch.nanmean(y_pred.new_tensor([metrics]))
        if (average == "macro") & (not nan_ignore):
            return torch.mean(y_pred.new_tensor([metrics]))
        if (average == "micro") & (not nan_ignore):
            return (y_pred.new_tensor([proportions])*y_pred.new_tensor([metrics])).sum()
        else:
            raise NotImplementedError
    
    def update(self, preds: Tensor, target: Tensor,  drugs: Tensor,  cell_lines: Tensor) -> None:
        self.target = torch.cat([self.target, target])
        self.pred = torch.cat([self.pred, preds])
        self.drugs = torch.cat([self.drugs, drugs]).long()
        self.cell_lines = torch.cat([self.cell_lines, cell_lines]).long()
        
def get_residual(X, y, alpha=0.001):
    w = get_linear_weights(X, y, alpha=alpha)
    r = y-(X@w)
    return r
def get_linear_weights(X, y, alpha=0.01):
    A = X.T@X
    Xy = X.T@y
    n_features = X.size(1)
    A.flatten()[:: n_features + 1] += alpha
    return torch.linalg.solve(A, Xy).T
def residual_correlation(y_pred, y_obs, drug_id, cell_id):
    X = torch.cat([y_pred.new_ones(y_pred.size(0), 1),
                   torch.nn.functional.one_hot(drug_id),
                   torch.nn.functional.one_hot(cell_id)], 1).float()
    r_pred = get_residual(X, y_pred)
    r_obs = get_residual(X, y_obs)
    return torchmetrics.functional.pearson_corrcoef(r_pred, r_obs)

def get_residual_ind(y, drug_id, cell_id, alpha=0.001):
    X = torch.cat([y.new_tensor.ones(y.size(0), 1), torch.nn.functional.one_hot(drug_id), torch.nn.functional.one_hot(cell_id)], 1).float()
    return get_residual(X, y, alpha=alpha)

def average_over_group(y_obs, y_pred, metric, grouping, average="macro", nan_ignore = False):
    unique = grouping.unique()
    proportions = []
    metrics = []
    for g in unique:
        is_group = grouping == g
        metrics += [metric(y_obs[grouping == g], y_pred[grouping == g])]
        proportions += [is_group.sum()/len(is_group)]
    if average is None:
        return torch.stack(metrics)
    if (average == "macro") & (nan_ignore):
        return torch.nanmean(y_pred.new_tensor([metrics]))
    if (average == "macro") & (not nan_ignore):
        return torch.mean(y_pred.new_tensor([metrics]))
    if (average == "micro") & (not nan_ignore):
        return (y_pred.new_tensor([proportions])*y_pred.new_tensor([metrics])).sum()
    else:
        raise NotImplementedError
        
class ResNet(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=1024, dropout=0.1, n_layers = 6, norm = "layernorm"):
        super().__init__()
        self.mlps = nn.ModuleList()
        if norm == "layernorm":
            norm = nn.LayerNorm
        elif norm == "batchnorm":
            norm = nn.BatchNorm1d
        else:
            norm = nn.Identity
        for l in range(n_layers):
            self.mlps.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                           norm(hidden_dim),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, embed_dim)))
        self.lin = nn.Linear(embed_dim, 1)
    def forward(self, x):
        for l in self.mlps:
            x = (l(x) + x)/2
        return self.lin(x)
    
class Model(nn.Module):
    def __init__(self, embed_dim=256,
                 hidden_dim=1024,
                 dropout=0.1,
                 n_layers = 6,
                 norm = "layernorm"):
        super().__init__()
        self.resnet = ResNet(embed_dim, hidden_dim, dropout, n_layers, norm)
        self.embed_d = nn.Sequential(nn.LazyLinear(embed_dim), nn.ReLU())
        self.embed_c = nn.Sequential(nn.LazyLinear(embed_dim), nn.ReLU())
    def forward(self, c, d):
        return self.resnet(self.embed_d(d) + self.embed_c(c))
    
def evaluate_step(model, loader, metrics, device):
    metrics.increment()
    model.eval()
    for x in loader:
        with torch.no_grad():
            out = model(x[0].to(device), x[1].to(device))
            metrics.update(out.squeeze(),
                           x[2].to(device).squeeze(),
                           cell_lines = x[3].to(device).squeeze().to(device),
                           drugs = x[4].to(device).squeeze().to(device))
    return {it[0]:it[1].item() for it in metrics.compute().items()}

def train_step(model, optimizer, loader, config, device):
    loss = nn.MSELoss()
    ls = []
    model.train()
    for x in loader:
        optimizer.zero_grad()
        out = model(x[0].to(device), x[1].to(device))
        l = loss(out.squeeze(), x[2].to(device).squeeze())
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["clip_norm"])
        ls += [l.item()]
        optimizer.step()
    return np.mean(ls)

def train_model(config, train_dataset, validation_dataset=None, use_momentum=True, callback_epoch = None):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=config["optimizer"]["batch_size"],
                                           drop_last=True,
                                          shuffle=True)
    if validation_dataset is not None:
        val_loader = torch.utils.data.DataLoader(validation_dataset,
                                               batch_size=config["optimizer"]["batch_size"],
                                               drop_last=False,
                                              shuffle=False)
    model = Model(**config["model"])
    optimizer = torch.optim.Adam(model.parameters(), config["optimizer"]["learning_rate"])
    device = torch.device(config["env"]["device"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    early_stop = EarlyStop(config["optimizer"]["stopping_patience"])
    model.to(device)
    metrics = torchmetrics.MetricTracker(torchmetrics.MetricCollection(
    {"R_cellwise_residuals":GroupwiseMetric(metric=torchmetrics.functional.pearson_corrcoef,
                          grouping="drugs",
                          average="macro",
                          residualize=True),
    "R_cellwise":GroupwiseMetric(metric=torchmetrics.functional.pearson_corrcoef,
                          grouping="cell_lines",
                          average="macro",
                          residualize=False),
    "MSE":torchmetrics.MeanSquaredError()}))
    metrics.to(device)
    use_momentum = True    
    for epoch in range(config["env"]["max_epochs"]):
        train_loss = train_step(model, optimizer, train_loader, config, device)
        lr_scheduler.step(train_loss)
        if validation_dataset is not None:
            validation_metrics = evaluate_step(model,val_loader, metrics, device)
            if epoch > 0 & use_momentum:
                val_target = 0.2*val_target + 0.8*validation_metrics['R_cellwise_residuals']
            else:
                val_target = validation_metrics['R_cellwise_residuals']
        else:
            val_target = None
        if callback_epoch is None:
            print(f"epoch : {epoch}: train loss: {train_loss} Smoothed R interaction (validation) {val_target}")
        else:
            callback_epoch(epoch, val_target)
        if early_stop(train_loss):
            break
    return val_target, model