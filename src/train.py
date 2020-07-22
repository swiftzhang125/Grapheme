import os
import ast
import torch
import torch.nn as nn
from tqdm import tqdm
from model_dispatcher import MODEL_DISPATCHER
from dataset import BengaliDatasetTrain

DEVICE = 'cuda'
TRAINING_FOLDS_CSV = os.environ.get('TRAINING_FOLDS_CSV')

IMG_HEIGHT = int(os.environ.get('IMG_HEIGHT'))
IMG_WIDTH = int(os.environ.get('IMG_WIDTH'))

EPOCHS = int(os.environ.get('EPOCHS'))

TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE'))
VALIDATION_BATCH_SIZE = int(os.environ.get('VALIDATION_BATCH_SIZE'))

MODEL_MEAN = ast.literal_eval(os.environ.get('MODEL_MEAN'))
MODEL_STD = ast.literal_eval(os.environ.get('MODEL_STD'))

TRAIN_FOLDS = ast.literal_eval(os.environ.get('TRAIN_FOLDS'))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get('VALIDATION_FOLDS'))

BASE_MODEL = os.environ.get('BASE_MODEL')

def train_loop_fn(dataset, dataloader, model, optimizer):
    model.train()
    for step, d in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        image = d['image']
        grapheme_root = d['grapheme_root']
        vowel_diacritic = d['vowel_diacritic']
        consonant_diacritic = d['consonant_diacritic']

        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()


def eval_loop_fn(dataset, dataloader, model):
    model.eval()
    final_loss = 0
    counter = 0
    for step, d in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        counter += 1
        image = d['image']
        grapheme_root = d['grapheme_root']
        vowel_diacritic = d['vowel_diacritic']
        consonant_diacritic = d['consonant_diacritic']

        image = image.to(DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype=torch.long)

        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        final_loss += loss
    return final_loss / counter



def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    return (l1 + l2 + l3) / 3

def run():
    model = MODEL_DISPATCHER[BASE_MODEL](pretrained=True)
    model.to(DEVICE)

    train_dataset = BengaliDatasetTrain(
        folds=TRAIN_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=1
    )

    valid_dataset = BengaliDatasetTrain(
        folds=VALIDATION_FOLDS,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        mean=MODEL_MEAN,
        std=MODEL_STD
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           patience=5,
                                                           factor=0.3,
                                                           verbose=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(EPOCHS):
        train_loop_fn(train_dataset, train_loader, model, optimizer)
        val_score = eval_loop_fn(valid_dataset, valid_loader, model)
        scheduler.step(val_score)
        torch.save(model.state_dict(), f'{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin')


if __name__ == '__main__':
    run()