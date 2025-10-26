
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from CAT.dataset.dataset import Dataset
from CAT.model.utils import StraightThrough

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_seeds(seedNum):
    np.random.seed(seedNum)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seedNum)
    np.random.seed(seedNum)
    random.seed(seedNum)

def open_json(path_):
    with open(path_) as fh:
        data = json.load(fh)
    return data

def data_split(datapath, fold, seed):
    data = open_json(datapath)
    random.Random(seed).shuffle(data)
    fields = ['q_ids',  'labels']  # 'ans', 'correct_ans',
    del_fields = []
    for f in data[0]:
        if f not in fields:
            del_fields.append(f)
    for d in data:
        for f in fields:
            d[f] = np.array(d[f])
        for f in del_fields:
            if f not in fields:
                del d[f]
    N = len(data)//5
    test_fold, valid_fold = fold-1, fold % 5
    test_data = data[test_fold*N: (test_fold+1)*N]
    valid_data = data[valid_fold*N: (valid_fold+1)*N]
    train_indices = [idx for idx in range(len(data))]
    train_indices = [idx for idx in train_indices if idx //
                     N != test_fold and idx//N != valid_fold]
    train_data = [data[idx] for idx in train_indices]

    if params.smoke:
        # build a tiny synthetic train_loader for smoke testing
        B = params.train_batch_size
        nq = params.n_question
        input_labels = torch.zeros(B, nq).long()
        output_labels = torch.zeros(B, nq).long()
        input_mask = torch.zeros(B, nq).long()
        output_mask = torch.zeros(B, nq).long()
        # set a few observed entries
        k = min(5, nq)
        # create simple pattern for train batch
        for i in range(B):
            idxs = list(range(i, i + k))
            idxs = [j % nq for j in idxs]
            input_labels[i, idxs] = 1
            input_mask[i, idxs] = 1
            # random outputs
            output_labels[i, idxs] = torch.randint(0, 2, (len(idxs),))
            output_mask[i, idxs] = 1
        batch = {'input_labels': input_labels, 'input_mask': input_mask,
                 'output_labels': output_labels, 'output_mask': output_mask}
        train_loader = [batch]

        # build synthetic validation and test lists (list of dicts with 'q_ids' and 'labels')
        import random as _random
        valid_data = []
        test_data = []
        n_valid = 20
        n_test = 20
        for i in range(n_valid):
            qcount = _random.randint(1, min(10, nq))
            q_ids = _random.sample(range(nq), qcount)
            labels = _random.choices([0, 1], k=qcount)
            valid_data.append({'q_ids': q_ids, 'labels': labels})
        for i in range(n_test):
            qcount = _random.randint(1, min(10, nq))
            q_ids = _random.sample(range(nq), qcount)
            labels = _random.choices([0, 1], k=qcount)
            test_data.append({'q_ids': q_ids, 'labels': labels})

        # Training loop
        for epoch in range(params.n_epoch):
            train_metrics = train_model()

            # Evaluate on validation set
            total_v = 0
            total_corr_weighted = 0.0
            config['mode'] = 'eval'
            for d in valid_data:
                nq = params.n_question
                input_labels_v = torch.zeros(1, nq).long()
                output_labels_v = torch.zeros(1, nq).long()
                input_mask_v = torch.zeros(1, nq).long()
                output_mask_v = torch.zeros(1, nq).long()
                q_ids = list(map(int, d['q_ids']))
                labels = list(map(int, d['labels']))
                if len(q_ids) > 0:
                    output_labels_v[0, q_ids] = torch.LongTensor(labels)
                    output_mask_v[0, q_ids] = 1
                batch_v = {'input_labels': input_labels_v, 'input_mask': input_mask_v,
                           'output_labels': output_labels_v, 'output_mask': output_mask_v}
                with torch.no_grad():
                    out_eval, _ = run_biased(batch_v, config)
                acc = compute_accuracy(out_eval, output_labels_v, output_mask_v)
                valid_count = int(output_mask_v.sum().item())
                total_corr_weighted += acc * valid_count
                total_v += valid_count
            config['mode'] = 'train'
            val_acc = (total_corr_weighted / total_v) if total_v > 0 else 0.0
            print(f"Epoch {epoch} VALID acc: {val_acc:.4f}")
            metrics_file = os.path.join(os.getcwd(), 'metrics.csv')
            with open(metrics_file, 'a', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow([epoch, 'valid', f"{train_metrics['loss']:.6f}", f"{val_acc:.6f}"])

            # Evaluate on test set
            total_v = 0
            total_corr_weighted = 0.0
            config['mode'] = 'eval'
            for d in test_data:
                nq = params.n_question
                input_labels_t = torch.zeros(1, nq).long()
                output_labels_t = torch.zeros(1, nq).long()
                input_mask_t = torch.zeros(1, nq).long()
                output_mask_t = torch.zeros(1, nq).long()
                q_ids = list(map(int, d['q_ids']))
                labels = list(map(int, d['labels']))
                if len(q_ids) > 0:
                    output_labels_t[0, q_ids] = torch.LongTensor(labels)
                    output_mask_t[0, q_ids] = 1
                batch_t = {'input_labels': input_labels_t, 'input_mask': input_mask_t,
                           'output_labels': output_labels_t, 'output_mask': output_mask_t}
                with torch.no_grad():
                    out_eval, _ = run_biased(batch_t, config)
                acc = compute_accuracy(out_eval, output_labels_t, output_mask_t)
                test_count = int(output_mask_t.sum().item())
                total_corr_weighted += acc * test_count
                total_v += test_count
            config['mode'] = 'train'
            test_acc = (total_corr_weighted / total_v) if total_v > 0 else 0.0
            print(f"Epoch {epoch} TEST acc: {test_acc:.4f}")
            with open(metrics_file, 'a', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow([epoch, 'test', f"{train_metrics['loss']:.6f}", f"{test_acc:.6f}"])
    def reset(self, batch):
        input_labels, _, input_mask = get_inputs(batch)
        obs_state = ((input_labels-0.5)*2.)  # B, 948
        train_mask = torch.zeros(
            input_mask.shape[0], self.n_question).long().to(device)
        env_states = {'obs_state': obs_state, 'train_mask': train_mask,
                      'action_mask': input_mask.clone()}
        return env_states
    

    def step(self, env_states):
        obs_state,  train_mask = env_states[
            'obs_state'], env_states['train_mask']
        state = obs_state*train_mask  # B, 948
        return state

    def pick_sample(self,sampling, config):
        student_embed = config['meta_param']
        n_student = len(config['meta_param'])
        action = self.pick_uncertain_sample(student_embed, config['available_mask'])
        config['train_mask'][range(n_student), action], config['available_mask'][range(n_student), action] = 1, 0
        return action
        

    def forward(self, batch, config):
        #get inputs
        input_labels = batch['input_labels'].to(device).float()
        student_embed = config['meta_param']#
        output = self.compute_output(student_embed)
        train_mask = config['train_mask']
        #compute loss
        if config['mode'] == 'train':
            output_labels, output_mask = get_outputs(batch)
            #meta model parameters 
            output_loss = compute_loss(output, output_labels, output_mask, reduction=False)/len(train_mask)
            #for adapting meta model parameters
            if self.n_query!=-1:
                input_loss = compute_loss(output, input_labels, train_mask, reduction=False)
            else:
                input_loss = normalize_loss(output, input_labels, train_mask)
            #loss = input_loss*self.alpha + output_loss
            return {'loss': output_loss, 'train_loss': input_loss, 'output': self.sigmoid(output).detach().cpu().numpy()}
        else:
            input_loss = compute_loss(output, input_labels, train_mask,reduction=False)
            return {'output': self.sigmoid(output).detach().cpu().numpy(), 'train_loss': input_loss}

    def pick_uncertain_sample(self, student_embed, available_mask):
        with torch.no_grad():
            output = self.compute_output(student_embed)
            output = self.sigmoid(output)
            inf_mask = torch.clamp(
                torch.log(available_mask.float()), min=torch.finfo(torch.float32).min)
            scores = torch.min(1-output, output)+inf_mask
            actions = torch.argmax(scores, dim=-1)
            return actions

    def compute_output(self, student_embed):
        if self.tp=='irt':
            # embedded_question_difficulty = self.question_difficulty.weight
            # embedded_question_dq = self.question_dq.weight
            # output = embedded_question_difficulty * (student_embed - embedded_question_dq)
            output = (student_embed - self.question_difficulty)
            #output = self.tmp*(student_embed - self.question_difficulty)
        else:
            #output = self.output_layer(self.layers(student_embed))
            #stu_emb = torch.sigmoid(self.student_emb(stu_id))
            k_difficulty = self.k_difficulty
            e_discrimination = self.e_discrimination
            kn_emb = self.kn_emb
            #e_discrimination = torch.sigmoid(self.e_discrimination) * 10
            # prednet
            student_embed = student_embed.unsqueeze(1)
            input_x = e_discrimination * (student_embed - k_difficulty) *kn_emb.to(device)
            input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
            input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
            output = self.prednet_full3(input_x)
            output = output.squeeze()
        return output
        
def clone_meta_params(batch):
    # expand meta params to batch size and make sure the cloned tensor requires grad
    p = meta_params[0].expand(len(batch['input_labels']), -1).clone()
    p.requires_grad_(True)
    return [p]


def get_inputs(batch):
    # return input_labels, placeholder, input_mask
    return batch['input_labels'].to(device), None, batch['input_mask'].to(device)


def get_outputs(batch):
    return batch['output_labels'].to(device), batch['output_mask'].to(device)


def compute_loss(output, labels, mask, reduction=True):
    """
    Compute BCEWithLogits loss over masked positions.
    output: tensor logits of shape (B, Q) or (B,) ; labels, mask: same shape
    If reduction=True return mean over masked elements, else return sum over masked elements.
    """
    # Ensure tensors
    if not torch.is_tensor(output):
        output = torch.tensor(output, device=device)
    labels = labels.to(device).float()
    mask = mask.to(device).float()
    # make shapes compatible
    if output.dim() == 1:
        output = output.unsqueeze(1)
    loss = F.binary_cross_entropy_with_logits(output, labels.float(), reduction='none')
    masked = loss * mask.float()
    total = masked.sum()
    if reduction:
        denom = mask.sum()
        return (total / denom) if denom.item() > 0 else total
    else:
        return total


def normalize_loss(output, labels, mask):
    # normalized mean loss over masked entries
    return compute_loss(output, labels, mask, reduction=True)


def compute_accuracy(outputs, labels, mask):
    # outputs may be numpy array or torch tensor (probabilities)
    if outputs is None:
        return 0.0
    if isinstance(outputs, list):
        out = np.array(outputs)
    else:
        out = outputs
    # convert labels and mask to numpy
    if torch.is_tensor(labels):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.array(labels)
    if torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.array(mask)
    # ensure shapes align
    try:
        preds = (out >= 0.5).astype(int)
    except Exception:
        preds = (np.array(out) >= 0.5).astype(int)
    correct = (preds == labels_np) * (mask_np == 1)
    denom = mask_np.sum()
    if denom == 0:
        return 0.0
    return float(correct.sum()) / float(denom)


class MAMLModel(nn.Module):
    """Minimal stand-in MAML-like model used for smoke runs.

    It implements the small API used by the training script: reset, step,
    and callable (batch, config) -> dict with 'loss','train_loss','output'.
    For `tp=='irt'` it implements a simple item-difficulty model: output = student_embed - question_difficulty
    """

    def __init__(self, sampling, n_query, n_question, question_dim=1, tp='irt', emb=None):
        super().__init__()
        self.sampling = sampling
        self.n_query = n_query
        self.n_question = n_question
        self.tp = tp
        self.question_dim = question_dim
        # question difficulty parameter (Q, 1)
        self.question_difficulty = nn.Parameter(torch.zeros(n_question, 1))
        # for non-irt approximate params
        self.k_difficulty = nn.Parameter(torch.zeros(n_question, max(1, question_dim)))
        self.e_discrimination = nn.Parameter(torch.zeros(n_question, 1))
        self.sigmoid = torch.sigmoid

    def reset(self, batch):
        input_labels, _, input_mask = get_inputs(batch)
        obs_state = ((input_labels - 0.5) * 2.)
        train_mask = torch.zeros(input_mask.shape[0], self.n_question).long().to(device)
        env_states = {'obs_state': obs_state, 'train_mask': train_mask,
                      'action_mask': input_mask.clone().to(device)}
        return env_states

    def step(self, env_states):
        obs_state, train_mask = env_states['obs_state'], env_states['train_mask']
        state = obs_state * train_mask
        return state

    def compute_output(self, student_embed):
        # student_embed: tensor shape (B, D) or (B,1)
        if student_embed is None:
            B = 1
            student_embed = torch.zeros((B, 1), device=device)
        if self.tp == 'irt':
            # question_difficulty: (Q,1) -> expand to (B,Q)
            # student_embed expected shape (B,1)
            if student_embed.dim() == 1:
                student_embed = student_embed.unsqueeze(1)
            qdiff = self.question_difficulty.to(device).t()  # (1, Q)
            out = student_embed - qdiff
            return out.squeeze(-1)
        else:
            # simple sum-difference approximation for NCD
            stu = student_embed.float()
            if stu.dim() == 1:
                stu = stu.unsqueeze(0)
            # stu: (B, D) ; k_difficulty: (Q, D) -> compute (B, Q)
            out = torch.matmul(stu, self.k_difficulty.t().to(device))
            return out

    def forward(self, batch, config):
        # keep compatibility: call via model(batch, config)
        return self.__call__(batch, config)

    def __call__(self, batch, config):
        # get tensors
        input_labels = batch['input_labels'].to(device).float()
        output_labels = batch['output_labels'].to(device).float()
        input_mask = batch['input_mask'].to(device)
        output_mask = batch['output_mask'].to(device)
        # meta_param provided by script (B, D)
        student_embed = config.get('meta_param', None)
        if student_embed is None:
            # default zero embedding
            student_embed = torch.zeros(input_labels.shape[0], 1, device=device)
        # ensure on device
        if torch.is_tensor(student_embed):
            student_embed = student_embed.to(device)
        # compute raw outputs (logits)
        output = self.compute_output(student_embed)
        # compute losses
        output_loss = compute_loss(output, output_labels, output_mask, reduction=False) / max(1, len(input_labels))
        input_loss = compute_loss(output, input_labels, input_mask, reduction=False)
        if config.get('mode', 'train') == 'train':
            return {'loss': output_loss, 'train_loss': input_loss, 'output': torch.sigmoid(output).detach().cpu().numpy()}
        else:
            return {'output': torch.sigmoid(output).detach().cpu().numpy(), 'train_loss': input_loss}


def collate_fn(n_question):
    """Return a simple collate function that stacks dict-of-tensors batches.

    This minimal collate function is sufficient for smoke and small datasets.
    """
    def _collate(batch_list):
        import torch
        if len(batch_list) == 0:
            return {}
        if isinstance(batch_list[0], dict):
            out = {}
            for k in batch_list[0].keys():
                vals = [b[k] for b in batch_list]
                try:
                    out[k] = torch.stack(vals, dim=0)
                except Exception:
                    out[k] = vals
            return out
        return batch_list

    return _collate

def inner_algo(batch, config, new_params, create_graph=False):

    for _ in range(params.inner_loop):
        config['meta_param'] = new_params[0]
        res = model(batch, config)
        loss = res['train_loss']
        grads = torch.autograd.grad(
            loss, new_params, create_graph=create_graph)
        new_params = [(new_params[i] - params.inner_lr*grads[i])
                      for i in range(len(new_params))]
        del grads
    config['meta_param'] = new_params[0]
    return

def run_biased(batch, config):
    new_params = clone_meta_params(batch)
    if config['mode'] == 'train':
        model.eval()
    pick_biased_samples(batch, config)
    optimizer.zero_grad()
    meta_params_optimizer.zero_grad()
    # Only run inner adaptation in training mode (avoid autograd during eval)
    if config.get('mode', 'train') == 'train':
        inner_algo(batch, config, new_params)
    if config['mode'] == 'train':
        model.train()
        optimizer.zero_grad()
        res = model(batch, config)
        loss = res['loss']
        loss.backward()
        optimizer.step()
        meta_params_optimizer.step()
        ####
    else:
        with torch.no_grad():
            res = model(batch, config)

    # return both outputs (numpy array) and loss (tensor or None)
    out = res.get('output', None)
    l = res.get('loss', None)
    return out, (l.item() if l is not None else None)
def pick_biased_samples(batch, config):
    new_params = clone_meta_params(batch)
    env_states = model.reset(batch)
    action_mask, train_mask = env_states['action_mask'], env_states['train_mask']
    for i in range(params.n_query):
        with torch.no_grad():
            state = model.step(env_states)
            train_mask = env_states['train_mask']
        if config['mode'] == 'train':
            train_mask_sample, actions = st_policy.policy(state, action_mask)
        else:
            with torch.no_grad():
                train_mask_sample, actions = st_policy.policy(
                    state, action_mask)
        action_mask[range(len(action_mask)), actions] = 0
        # env state train mask should be detached
        env_states['train_mask'], env_states['action_mask'] = train_mask + \
            train_mask_sample.data, action_mask
        if config['mode'] == 'train':
            # loss computation train mask should flow gradient
            config['train_mask'] = train_mask_sample+train_mask
            inner_algo(batch, config, new_params, create_graph=True)
            res = model(batch, config)
            loss = res['loss']
            st_policy.update(loss)

    config['train_mask'] = env_states['train_mask']
    return 

def create_parser():
    parser = argparse.ArgumentParser(description='ML')
    parser.add_argument('--model', type=str,
                        default='biirt-biased', help='type')
    parser.add_argument('--name', type=str, default='demo', help='type')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='type')
    parser.add_argument('--question_dim', type=int, default=4, help='type')
    parser.add_argument('--lr', type=float, default=1e-4, help='type') #
    parser.add_argument('--meta_lr', type=float, default=1e-4, help='type')
    parser.add_argument('--inner_lr', type=float, default=1e-1, help='type') #
    parser.add_argument('--inner_loop', type=int, default=5, help='type') #
    parser.add_argument('--policy_lr', type=float, default=2e-3, help='type') #
    parser.add_argument('--dropout', type=float, default=0.6, help='type')
    parser.add_argument('--dataset', type=str,
                        default='exam', help='eedi-1 or eedi-3')
    parser.add_argument('--fold', type=int, default=1, help='type')
    parser.add_argument('--n_query', type=int, default=20, help='type')
    parser.add_argument('--seed', type=int, default=221, help='type')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--n_question', type=int, default=50, help='number of questions')
    parser.add_argument('--n_epoch', type=int, default=1, help='number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=2, help='train batch size')
    parser.add_argument('--concept_num', type=int, default=5, help='number of concepts')
    parser.add_argument('--smoke', action='store_true', help='run a small smoke test with synthetic data')
    return parser.parse_args()


def train_model():
    config['mode'] = 'train'
    config['epoch'] = epoch
    model.train()

    total_loss = 0.0
    total_valid = 0
    total_correct_weighted = 0.0

    for batch in train_loader:
        # Select RL Actions, save in config and get outputs+loss
        outputs, loss_val = run_biased(batch, config)

        # compute batch accuracy and accumulate
        output_labels = batch['output_labels']
        output_mask = batch['output_mask']
        batch_acc = compute_accuracy(outputs, output_labels, output_mask)
        batch_valid = int(output_mask.sum().item()) if isinstance(output_mask, torch.Tensor) else int(np.array(output_mask).sum())
        total_correct_weighted += batch_acc * batch_valid
        total_valid += batch_valid
        if loss_val is not None:
            total_loss += float(loss_val) * (1 if batch_valid == 0 else batch_valid)

    epoch_loss = (total_loss / total_valid) if total_valid > 0 else 0.0
    epoch_acc = (total_correct_weighted / total_valid) if total_valid > 0 else 0.0
    print(f"Epoch {epoch} TRAIN loss: {epoch_loss:.6f} acc: {epoch_acc:.4f}")

    # save metrics to CSV
    metrics_file = os.path.join(os.getcwd(), 'metrics.csv')
    write_header = not os.path.exists(metrics_file)
    with open(metrics_file, 'a', newline='') as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(['epoch', 'split', 'loss', 'accuracy'])
        writer.writerow([epoch, 'train', f"{epoch_loss:.6f}", f"{epoch_acc:.6f}"])

    return {'loss': epoch_loss, 'acc': epoch_acc}

    #   
if __name__ == "__main__":
    params = create_parser()
    print(params)

    config = {
        'policy_path': 'policy.pt',
    }
    initialize_seeds(params.seed)

    #
    base, sampling = params.model.split('-')[0], params.model.split('-')[-1]
    if base == 'biirt':
        print('DEBUG: MAMLModel present before instantiation?', 'MAMLModel' in globals())
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=1,tp = 'irt').to(device)
        meta_params = [torch.zeros(1, 1, device=device, requires_grad=True)]
        # meta_params = [torch.Tensor(
        #    1, 1).normal_(-1., 1.).to(device).requires_grad_()]
    if base == 'binn':
        concept_name = params.dataset +'_concept_map.json'
        with open(concept_name, 'r') as file:
            concepts = json.load(file)
        num_concepts = params.concept_num
        concepts_emb = [[0.] * num_concepts for i in range(params.n_question)]
        if params.dataset=='exam':
            for i in range(1,params.n_question):
                for concept in concepts[str(i)]:
                    concepts_emb[i][concept] = 1.0   
        else:
            for i in range(params.n_question):
                for concept in concepts[str(i)]:
                    concepts_emb[i][concept] = 1.0
        concepts_emb = torch.tensor(concepts_emb, dtype=torch.float32).to(device)
        model = MAMLModel(sampling=sampling, n_query=params.n_query,
                          n_question=params.n_question, question_dim=params.question_dim,tp ='ncd',emb=concepts_emb).to(device)
        meta_params = [torch.zeros((1, num_concepts), device=device, requires_grad=True)]
        # meta_params = [torch.Tensor(
        #     1,num_concepts).normal_(-1., 1.).to(device).requires_grad_()]
        # meta_params = [torch.Tensor(
        #     1, 1).normal_(-1., 1.).to(device).requires_grad_()]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=1e-8)

    meta_params_optimizer = torch.optim.SGD(
        meta_params, lr=params.meta_lr, weight_decay=2e-6, momentum=0.9)
        # neptune_exp.log_text(
        #     'model_summary', repr(model))
    #
            # neptune_exp.log_text(
            #     'ppo_model_summary', repr(ppo_policy.policy))
    betas = (0.9, 0.999)
    # StraightThrough expects a config dict containing 'device' and 'betas'
    st_policy = StraightThrough(params.n_question, params.n_question,
                                params.policy_lr, {'device': device, 'betas': betas})
            # neptune_exp.log_text(
            #     'biased_model_summary', repr(st_policy.policy))
    #
    if params.smoke:
        # build a tiny synthetic train_loader for smoke testing
        B = params.train_batch_size
        nq = params.n_question
        input_labels = torch.zeros(B, nq).long()
        output_labels = torch.zeros(B, nq).long()
        input_mask = torch.zeros(B, nq).long()
        output_mask = torch.zeros(B, nq).long()
        # set a few observed entries
        k = min(2, nq)
        input_labels[:, :k] = 1
        input_mask[:, :k] = 1
        output_labels[:, :k] = 0
        output_mask[:, :k] = 1
        batch = {'input_labels': input_labels, 'input_mask': input_mask,
                 'output_labels': output_labels, 'output_mask': output_mask}
        train_loader = [batch]
        for epoch in range(params.n_epoch):
            train_metrics = train_model()
            # For smoke mode, use same synthetic batch as validation/test
            # compute and print validation/test accuracy using the same batch
            # reuse batch variable defined above
            config['mode'] = 'eval'
            with torch.no_grad():
                out_eval, _ = run_biased(batch, config)
            val_acc = compute_accuracy(out_eval, batch['output_labels'], batch['output_mask'])
            print(f"Epoch {epoch} VALID acc: {val_acc:.4f}")
            # append validation to CSV
            metrics_file = os.path.join(os.getcwd(), 'metrics.csv')
            with open(metrics_file, 'a', newline='') as fh:
                writer = csv.writer(fh)
                writer.writerow([epoch, 'valid', f"{train_metrics['loss']:.6f}", f"{val_acc:.6f}"])
            config['mode'] = 'train'
    else:
        data_path = os.path.normpath('data/train_task_' + params.dataset + '.json')
        train_data, valid_data, test_data = data_split(
            data_path, params.fold, params.seed)
        train_dataset, valid_dataset, test_dataset = Dataset(
            train_data), Dataset(valid_data), Dataset(test_data)
        #
        num_workers = 3
        collate_fn = collate_fn(params.n_question)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, collate_fn=collate_fn, batch_size=params.train_batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
        for epoch in range(params.n_epoch):
            train_metrics = train_model()
            # Evaluate on valid and test sets if available (data_split returns lists)
            def eval_list(data_list, split_name):
                # data_list: list of dicts with 'q_ids' and 'labels' arrays
                total_v = 0
                total_corr_weighted = 0.0
                config['mode'] = 'eval'
                for d in data_list:
                    # build single-sample batch
                    nq = params.n_question
                    input_labels = torch.zeros(1, nq).long()
                    output_labels = torch.zeros(1, nq).long()
                    input_mask = torch.zeros(1, nq).long()
                    output_mask = torch.zeros(1, nq).long()
                    q_ids = np.array(d['q_ids'], dtype=int)
                    labels = np.array(d['labels'], dtype=int)
                    if len(q_ids) > 0:
                        output_labels[0, q_ids] = torch.LongTensor(labels)
                        output_mask[0, q_ids] = 1
                    batch_d = {'input_labels': input_labels, 'input_mask': input_mask,
                               'output_labels': output_labels, 'output_mask': output_mask}
                    with torch.no_grad():
                        out_eval, _ = run_biased(batch_d, config)
                    acc = compute_accuracy(out_eval, output_labels, output_mask)
                    valid_count = int(output_mask.sum().item())
                    total_corr_weighted += acc * valid_count
                    total_v += valid_count
                config['mode'] = 'train'
                avg_acc = (total_corr_weighted / total_v) if total_v > 0 else 0.0
                print(f"Epoch {epoch} {split_name.upper()} acc: {avg_acc:.4f}")
                # append to CSV
                metrics_file = os.path.join(os.getcwd(), 'metrics.csv')
                with open(metrics_file, 'a', newline='') as fh:
                    writer = csv.writer(fh)
                    writer.writerow([epoch, split_name, f"{train_metrics['loss']:.6f}", f"{avg_acc:.6f}"])

            if valid_data:
                eval_list(valid_data, 'valid')
            if test_data:
                eval_list(test_data, 'test')
    torch.save(st_policy.policy.state_dict(),config['policy_path'])
