import csv
import os
import random
from collections import Counter, defaultdict


def read_all_rows(paths):
    rows = []
    for p in paths:
        with open(p, 'r', newline='') as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                # normalize column names to expected
                row = {
                    'student_id': r.get('student_id', r.get('user_id', '')),
                    'question_id': r.get('question_id', r.get('q_id', '')),
                    'concept_id': r.get('concept_id', ''),
                    'is_correct': r.get('is_correct', r.get('answer', r.get('correct', '')))
                }
                rows.append(row)
    return rows


def deduplicate(rows):
    # Group by (student,question) and pick majority label; if tie pick last
    groups = defaultdict(list)
    for r in rows:
        key = (r['student_id'], r['question_id'])
        groups[key].append(r['is_correct'])

    cleaned = []
    for (s, q), labels in groups.items():
        c = Counter(labels)
        if len(c) == 1:
            chosen = labels[-1]
        else:
            # majority label (as string)
            chosen = c.most_common(1)[0][0]
        # find a concept_id for this pair from original rows
        concept = None
        for r in rows:
            if r['student_id'] == s and r['question_id'] == q and r['is_correct'] in labels:
                concept = r.get('concept_id', '')
                break
        cleaned.append({'student_id': s, 'question_id': q, 'concept_id': concept, 'is_correct': chosen})
    return cleaned


def split_by_student(rows, train_frac=0.7, valid_frac=0.15, seed=42):
    students = defaultdict(list)
    for r in rows:
        students[r['student_id']].append(r)

    student_ids = list(students.keys())
    random.Random(seed).shuffle(student_ids)
    n = len(student_ids)
    n_train = int(n * train_frac)
    n_valid = int(n * valid_frac)

    train_ids = set(student_ids[:n_train])
    valid_ids = set(student_ids[n_train:n_train + n_valid])
    test_ids = set(student_ids[n_train + n_valid:])

    train_rows = []
    valid_rows = []
    test_rows = []
    for sid, rs in students.items():
        if sid in train_ids:
            train_rows.extend(rs)
        elif sid in valid_ids:
            valid_rows.extend(rs)
        else:
            test_rows.extend(rs)

    return train_rows, valid_rows, test_rows


def write_csv(path, rows):
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['student_id', 'question_id', 'concept_id', 'is_correct'])
        for r in rows:
            writer.writerow([r['student_id'], r['question_id'], r.get('concept_id', ''), r['is_correct']])


def main():
    base = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base, 'data_samples')
    paths = [os.path.join(data_dir, p) for p in ('train.csv', 'valid.csv', 'test.csv')]
    for p in paths:
        if not os.path.exists(p):
            print('Missing', p, '— aborting')
            return

    rows = read_all_rows(paths)
    print('Combined rows:', len(rows))
    cleaned = deduplicate(rows)
    print('After dedup:', len(cleaned))
    train_rows, valid_rows, test_rows = split_by_student(cleaned)
    print('Split sizes (students-based):', len(train_rows), len(valid_rows), len(test_rows))

    write_csv(os.path.join(data_dir, 'train.csv'), train_rows)
    write_csv(os.path.join(data_dir, 'valid.csv'), valid_rows)
    write_csv(os.path.join(data_dir, 'test.csv'), test_rows)
    print('Wrote cleaned train/valid/test to', data_dir)


if __name__ == '__main__':
    main()
