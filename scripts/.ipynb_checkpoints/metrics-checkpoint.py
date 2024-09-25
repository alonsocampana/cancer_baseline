import torch
from torch import Tensor
from torchmetrics import Metric
import torchmetrics

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