import os

def bind_subjects(train_test: str):
    if train_test not in ['train', 'test']:
        raise ValueError("train_test must one of 'train', 'test'. current value: {}".format(train_test))

    base_path = os.path.join('.','per_subject')
    subjects_paths = ['Subject{}'.format(x) for x in range(1,8)]
    activities = ['boxingmoving', 'boxingstill', 'crawling', 'running', 'still', 'walking', 'walkinglow']

    for subject_path in subjects_paths:
        # TODO: remake dataloader
        pass


def load_datasets(args):
    pass