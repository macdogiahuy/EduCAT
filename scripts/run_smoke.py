# Minimal smoke test for EduCAT IRT model
import torch

from CAT.dataset.dataset import Dataset
from CAT.model.IRT import IRTModel

# tiny synthetic data: (student_id, question_id, correct)
data = [
    (0, 0, 1),
    (0, 1, 0),
    (1, 0, 1),
    (1, 1, 1),
]
concept_map = {0: [0], 1: [0]}
num_students = 2
num_questions = 2
num_concepts = 1

# create Dataset
data_obj = Dataset(data, concept_map, num_students, num_questions, num_concepts)

# model config
config = {
    'num_dim': 1,
    'learning_rate': 0.01,
    'batch_size': 2,
    'num_epochs': 1,
    'device': 'cpu',
    'betas': (0.9, 0.999),
}

irt = IRTModel(**config)
irt.init_model(data_obj)

# Create sample inputs
student_ids = torch.LongTensor([0, 1])
question_ids = torch.LongTensor([0, 1])

with torch.no_grad():
    preds = irt.model(student_ids, question_ids).view(-1)

print('Predictions:', preds.tolist())
print('Alpha[0]:', irt.get_alpha(0))
print('Beta[0]:', irt.get_beta(0))
print('Theta[0]:', irt.get_theta(0))
