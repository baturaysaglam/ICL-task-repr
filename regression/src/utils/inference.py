import os
import yaml

from munch import Munch
import numpy as np
import torch

from ..models import build_model
from ..samplers import get_data_sampler
from ..tasks import get_task_sampler

from .function_vector import add_function_vector, translate_transformer_output
from .learnable_task_vector import add_learnable_task_vector
from .in_context_vector import add_icv


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def prepare_model(run_path, batch_size, device):
    model, conf = get_model_from_run(run_path)
    model.to(device)
    model.eval()

    n_dims = conf.model.n_dims

    data_sampler = get_data_sampler(conf.training.data, n_dims)
    task_sampler = get_task_sampler(
        conf.training.task,
        n_dims,
        batch_size,
        **conf.training.task_kwargs
    )

    model._backbone.config.output_attentions = True
    model._backbone.config.output_hidden_states = True

    resid_dim = conf.model['n_embd']
    n_layers = conf.model.n_layer
    n_heads = conf.model['n_head']
    head_dim = resid_dim // n_heads

    return model, conf, task_sampler, data_sampler, (n_dims, resid_dim, n_layers, n_heads, head_dim)


def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model, conf


def prepare_save_dirs(task_name):
    results_path = os.path.join('results', task_name)
    figures_path = os.path.join(results_path, 'figures')
    preds_path = os.path.join(results_path, 'predictions')

    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    if not os.path.exists(preds_path):
        os.makedirs(preds_path)

    return figures_path, preds_path


def gen_orthogonal_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)

    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_post_i = xs[:, i : i + 1, :]
        xs_train_pre_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
            xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )
        xs_test_post_i_normalized = (
            xs_test_post_i_orthogonalized
            * xs_test_post_i.norm(dim=2).unsqueeze(2)
            / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i : i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def evaluate_model(model, true_model, seq_xs, seq_ys, L, FV=None, ICV=None, dummy=None, scale=1):
    assert FV is None or ICV is None, "FV and ICV cannot be used together."

    batch_size, seq_len, dim = seq_xs.shape[0], seq_xs.shape[1], seq_xs.shape[2]
    pred = torch.zeros((batch_size, seq_len)).to(seq_xs.device)

    with torch.no_grad():
        for t in np.arange(1, seq_len + 1):
            xs = seq_xs[:, :t, :]
            ys = seq_ys[:, :t]

            if dummy is not None:
                if isinstance(dummy, (float, np.floating)):
                    dummy = [dummy]

                dummy_indices = [int(m * t) for m in dummy]
                dummy_indices = sorted(dummy_indices, reverse=True)

                for dummy_idx in dummy_indices:
                    # Dummy query-response is chosen to be zeros
                    dummy_inp_xs = torch.zeros(batch_size, 1, dim).to(seq_xs.device)
                    dummy_inp_ys = torch.zeros(batch_size, 1).to(seq_xs.device)

                    xs = torch.cat((xs[:, :dummy_idx, :], dummy_inp_xs, xs[:, dummy_idx:, :]), dim=1)
                    ys = torch.cat((ys[:, :dummy_idx], dummy_inp_ys, ys[:, dummy_idx:]), dim=1)
            else:
                dummy_indices = None

            out_dim = ys.shape[1]

            if FV is not None:
                _, model_out = add_function_vector(model, xs, ys, L, FV, dummy_indices, scale, output_hidden_states=True)
            elif ICV is not None:
                model_out = add_icv(model, xs, ys, ICV, lambda_=scale)
            else:
                _, model_out = model.get_transformer_out(xs, ys)

            ys_hat = translate_transformer_output(true_model, model_out, out_dim)
            ys_hat = ys_hat[:, -1]

            pred[:, t - 1] = ys_hat

    return pred


def evaluate_model_on_LTV(model, true_model, seq_xs, seq_ys, LTV=None, dummy=None, scale=1):
    batch_size, seq_len, dim = seq_xs.shape[0], seq_xs.shape[1], seq_xs.shape[2]
    pred = torch.zeros((batch_size, seq_len)).to(seq_xs.device)

    with torch.no_grad():
        for t in np.arange(1, seq_len + 1):
            xs = seq_xs[:, :t, :]
            ys = seq_ys[:, :t]

            if dummy is not None:
                if isinstance(dummy, (float, np.floating)):
                    dummy = [dummy]

                dummy_indices = [int(m * t) for m in dummy]
                dummy_indices = sorted(dummy_indices, reverse=True)

                for dummy_idx in dummy_indices:
                    # Dummy query-response is chosen to be zeros
                    dummy_inp_xs = torch.zeros(batch_size, 1, dim).to(seq_xs.device)
                    dummy_inp_ys = torch.zeros(batch_size, 1).to(seq_xs.device)

                    xs = torch.cat((xs[:, :dummy_idx, :], dummy_inp_xs, xs[:, dummy_idx+1:, :]), dim=1)
                    ys = torch.cat((ys[:, :dummy_idx], dummy_inp_ys, ys[:, dummy_idx+1:]), dim=1)
            else:
                dummy_indices = None

            out_dim = ys.shape[1]

            if LTV is not None:
                ys_hat = add_learnable_task_vector(model, xs, ys, LTV, dummy_indices, scale)
            else:
                _, model_out = model.get_transformer_out(xs, ys)
                ys_hat = translate_transformer_output(true_model, model_out, out_dim)

            ys_hat = ys_hat[:, -1]
            pred[:, t - 1] = ys_hat

    return pred


# Special function for ICV
def evaluate_on_sequence(model, true_model, seq_xs, seq_ys, icv=None, lambda_=100):
    batch_size, seq_len, dim = seq_xs.shape[0], seq_xs.shape[1], seq_xs.shape[2]
    pred = np.zeros((batch_size, seq_len))

    with torch.no_grad():
        for t in np.arange(1, seq_len + 1):
            xs = seq_xs[:, :t, :]
            ys = seq_ys[:, :t]

            out_dim = ys.shape[1]

            if icv is not None:
                model_out = add_icv(model, xs, ys, icv, lambda_)
            else:
                _, model_out = model.get_transformer_out(xs, ys)

            ys_hat = translate_transformer_output(true_model, model_out, out_dim)
            ys_hat = ys_hat[:, -1]

            pred[:, t - 1] = ys_hat

    return pred


# Special function for ICV
def evaluate_on_query(model, true_model, xs, ys, icv=None, lambda_=100):
    with torch.no_grad():
        out_dim = ys.shape[1]

        if icv is not None:
            model_out = add_icv(model, xs, ys, icv, lambda_)
        else:
            _, model_out = model.get_transformer_out(xs, ys)

        ys_hat = translate_transformer_output(true_model, model_out, out_dim)
        ys_hat = ys_hat[:, -1]

    return ys_hat


def save_pred_loss(tensor_list, keys, save_path):
    numpy_arrays = {key: tensor.cpu().data.numpy() if tensor is not None else None for tensor, key in
                    zip(tensor_list, keys)}
    np.savez(save_path, **numpy_arrays)
