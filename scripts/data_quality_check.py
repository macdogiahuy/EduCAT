import csv
import os
from collections import Counter


def read_rows(path):
    rows = []
    with open(path, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def to_pair(r):
    return (int(r['student_id']), int(r['question_id']), int(r.get('is_correct', r.get('answer', r.get('correct', 0)))))


def summarize(rows):
    n = len(rows)
    dup = n - len({(r['student_id'], r['question_id'], r['is_correct']) for r in rows})
    label_counts = Counter([r['is_correct'] for r in rows])
    students = set([r['student_id'] for r in rows])
    questions = set([r['question_id'] for r in rows])
    return dict(n=n, duplicates=dup, labels=label_counts, students=len(students), questions=len(questions))


def main():
    base = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base, 'data_samples')
    train_p = os.path.join(data_dir, 'train.csv')
    valid_p = os.path.join(data_dir, 'valid.csv')
    test_p = os.path.join(data_dir, 'test.csv')

    for p in [train_p, valid_p, test_p]:
        if not os.path.exists(p):
            print('Missing', p)
            return

    train = read_rows(train_p)
    valid = read_rows(valid_p)
    test = read_rows(test_p)

    s_train = summarize(train)
    s_valid = summarize(valid)
    s_test = summarize(test)

    print('SUMMARY')
    print('train:', s_train)
    print('valid:', s_valid)
    print('test :', s_test)

    # overlap checks
    train_pairs = set((r['student_id'], r['question_id']) for r in train)
    valid_pairs = set((r['student_id'], r['question_id']) for r in valid)
    test_pairs = set((r['student_id'], r['question_id']) for r in test)

    print('\nOverlap (student,question) sizes:')
    print('train ∩ valid:', len(train_pairs & valid_pairs), 'of', len(valid_pairs))
    print('train ∩ test :', len(train_pairs & test_pairs), 'of', len(test_pairs))
    print('valid ∩ test :', len(valid_pairs & test_pairs), 'of', len(test_pairs))

    students_train = set(r['student_id'] for r in train)
    students_valid = set(r['student_id'] for r in valid)
    students_test = set(r['student_id'] for r in test)

    print('\nStudent overlap sizes:')
    print('train ∩ valid students:', len(students_train & students_valid), 'of', len(students_valid))
    print('train ∩ test  students:', len(students_train & students_test), 'of', len(students_test))

    # check if valid/test rows are subsets of train rows
    train_rows_set = set((r['student_id'], r['question_id'], r['is_correct']) for r in train)
    valid_rows_set = set((r['student_id'], r['question_id'], r['is_correct']) for r in valid)
    test_rows_set = set((r['student_id'], r['question_id'], r['is_correct']) for r in test)

    print('\nRow-level leakage:')
    print('valid subset of train:', valid_rows_set <= train_rows_set)
    print('test subset of train :', test_rows_set <= train_rows_set)

    # per-question label balance
    q_counts = {}
    for r in train + valid + test:
        q = r['question_id']
        q_counts.setdefault(q, []).append(r['is_correct'])

    extreme_q = 0
    for q, labs in q_counts.items():
        c = Counter(labs)
        total = sum(c.values())
        top = c.most_common(1)[0][1]
        if top / total >= 0.95:
            extreme_q += 1

    print(f"\nQuestions with >=95% identical labels: {extreme_q} of {len(q_counts)}")


if __name__ == '__main__':
    main()
