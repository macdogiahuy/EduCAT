import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from CAT.dataset.dataset import Dataset as BaseDataset
from CAT.dataset.train_dataset import TrainDataset
from CAT.model.IRT import IRTModel


def read_csv(path):
    rows = []
    with open(path, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def build_tuples_and_concept_map(csv_rows_list):
    # csv_rows_list: list of rows where each row has user_id,question_id,concept_id,answer
    tuples = []
    concept_map = {}
    user_map = {}
    max_user = 0
    max_q = 0
    max_concept = 0
    for r in csv_rows_list:
        try:
            uid = int(r['user_id'])
            qid = int(r['question_id'])
            cid = int(r.get('concept_id', 0))
            ans = int(r['answer'])
        except Exception:
            # try alternative names
            uid = int(r.get('student_id', r.get('user', 0)))
            qid = int(r.get('question_id', r.get('q_id', 0)))
            cid = int(r.get('concept_id', 0))
            ans = int(r.get('answer', r.get('correct', 0)))
        tuples.append((uid, qid, ans))
        concept_map.setdefault(qid, [])
        if cid not in concept_map[qid]:
            concept_map[qid].append(cid)
        max_user = max(max_user, uid)
        max_q = max(max_q, qid)
        max_concept = max(max_concept, cid)
    return tuples, concept_map, max_user, max_q, max_concept


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--checkpoint', type=str, default=os.path.join(os.getcwd(), 'best_irt.pt'))
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base, 'data_samples')

    train_csv = os.path.join(data_dir, 'train.csv')
    valid_csv = os.path.join(data_dir, 'valid.csv')
    test_csv = os.path.join(data_dir, 'test.csv')

    if not os.path.exists(train_csv):
        print('No train.csv found in data_samples; please add train.csv')
        raise SystemExit(1)

    train_rows = read_csv(train_csv)
    # read valid/test rows if available (use them to compute global vocab sizes)
    valid_rows = read_csv(valid_csv) if os.path.exists(valid_csv) else []
    test_rows = read_csv(test_csv) if os.path.exists(test_csv) else []

    # build tuples and concept map from train only (training interactions)
    tuples, concept_map, max_user_train, max_q_train, max_concept_train = build_tuples_and_concept_map(train_rows)

    # compute global max ids from all splits to avoid embedding index issues at eval time
    all_rows = train_rows + valid_rows + test_rows
    if all_rows:
        _, _, max_user_all, max_q_all, max_concept_all = build_tuples_and_concept_map(all_rows)
    else:
        max_user_all, max_q_all, max_concept_all = max_user_train, max_q_train, max_concept_train

    num_students = max_user_all + 1
    num_questions = max_q_all + 1
    num_concepts = max_concept_all + 1

    print(f'Built dataset: students={num_students}, questions={num_questions}, concepts={num_concepts}, interactions={len(tuples)}')

    # construct base dataset expected by IRTModel.init_model
    base_dataset = BaseDataset(tuples, concept_map, num_students, num_questions, num_concepts)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # config for IRTModel
    config = {
        'num_dim': 1,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'device': device,
        'policy': 'bobcat',
        'policy_path': os.path.join(base, 'policy.pt')
    }
    # StraightThrough expects 'betas' in config; provide defaults
    config['betas'] = (0.9, 0.999)

    model = IRTModel(**config)
    model.init_model(base_dataset)

    train_dataset = TrainDataset(tuples, concept_map, num_students, num_questions, num_concepts)

    # Build DataLoader for training
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=config['learning_rate'], weight_decay=args.weight_decay)

    # valid_rows and test_rows were read earlier for sizing

    metrics_file = os.path.join(os.getcwd(), 'metrics.csv')
    write_header = not os.path.exists(metrics_file)

    num_epochs = config['num_epochs']
    best_val_auc = None
    best_epoch = -1
    patience = args.patience
    no_improve = 0
    checkpoint_path = args.checkpoint

    for epoch in range(num_epochs):
        model.model.train()
        total_loss = 0.0
        batches = 0
        for sid, qid, concepts_emb, labels in train_loader:
            sid = sid.to(device).long()
            qid = qid.to(device).long()
            labels = labels.to(device).float()
            preds = model.model(sid, qid).view(-1)
            loss = model._loss_function(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        train_loss = (total_loss / batches) if batches > 0 else 0.0

        # evaluation helper
        def eval_rows(rows, batch_size=1024):
            # Vectorized/batched evaluation to avoid per-sample model calls
            if not rows:
                return {'acc': None, 'auc': None}
            uids = []
            qids = []
            y_true = []
            for r in rows:
                uids.append(int(r.get('user_id', r.get('student_id', 0))))
                qids.append(int(r.get('question_id', r.get('q_id', 0))))
                y_true.append(int(r.get('answer', r.get('correct', 0))))

            y_true = np.array(y_true)
            y_score = []
            model.model.eval()
            with torch.no_grad():
                for i in range(0, len(uids), batch_size):
                    batch_u = torch.LongTensor(uids[i:i+batch_size]).to(device)
                    batch_q = torch.LongTensor(qids[i:i+batch_size]).to(device)
                    preds = model.model(batch_u, batch_q).view(-1).cpu().numpy()
                    y_score.extend(preds.tolist())

            y_score = np.array(y_score)
            y_pred = (y_score >= 0.5).astype(int)
            acc = float((y_pred == y_true).sum()) / max(1, len(y_true))
            try:
                auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None
            except Exception:
                auc = None
            return {'acc': acc, 'auc': auc}

        valid_metrics = eval_rows(valid_rows)
        test_metrics = eval_rows(test_rows)

        # early stopping based on validation AUC if available, else validation acc
        current_val_score = None
        if valid_metrics.get('auc') is not None:
            current_val_score = valid_metrics['auc']
        elif valid_metrics.get('acc') is not None:
            current_val_score = valid_metrics['acc']

        if current_val_score is not None:
            if best_val_auc is None or current_val_score > best_val_auc:
                best_val_auc = current_val_score
                best_epoch = epoch
                no_improve = 0
                # save checkpoint
                try:
                    torch.save(model.model.state_dict(), checkpoint_path)
                except Exception:
                    pass
            else:
                no_improve += 1
        # if patience exceeded, stop
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch} (no improvement for {patience} epochs)')
            break

        # print and append to CSV
        print(f'Epoch {epoch} TRAIN loss: {train_loss:.6f} valid_acc: {valid_metrics.get("acc")} test_acc: {test_metrics.get("acc")}')
        with open(metrics_file, 'a', newline='') as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(['epoch', 'split', 'loss', 'accuracy', 'auc'])
                write_header = False
            writer.writerow([epoch, 'train', f'{train_loss:.6f}', '', ''])
            if valid_metrics.get('acc') is not None:
                writer.writerow([epoch, 'valid', f'{train_loss:.6f}', f"{valid_metrics['acc']:.6f}", f"{valid_metrics['auc'] if valid_metrics['auc'] is not None else ''}"])
            if test_metrics.get('acc') is not None:
                writer.writerow([epoch, 'test', f'{train_loss:.6f}', f"{test_metrics['acc']:.6f}", f"{test_metrics['auc'] if test_metrics['auc'] is not None else ''}"])

    # load best checkpoint if available
    try:
        if os.path.exists(checkpoint_path):
            model.model.load_state_dict(torch.load(checkpoint_path))
            model.model.to(device)
    except Exception:
        pass

    print('Training finished')
