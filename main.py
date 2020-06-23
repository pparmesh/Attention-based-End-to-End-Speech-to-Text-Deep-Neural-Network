import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import Seq2Seq
from train_test import train, test, val
from dataloader import load_data, collate_train, collate_test, transform_letter_to_index, Speech2TextDataset, create_dictionaries
import csv


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

letter2index, index2letter = create_dictionaries(LETTER_LIST)

def vec_idx_char(idx):
        return index2letter[idx]
def main():
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=256, isAttended=True)
#     print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    criterion = nn.CrossEntropyLoss(reduce=False,reduction=None)
    nepochs = 18
    batch_size = 64 if DEVICE == 'cuda' else 1

    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
    character_text_train = transform_letter_to_index(transcript_train, LETTER_LIST)
    character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)

    train_dataset = Speech2TextDataset(speech_train, character_text_train)
    val_dataset = Speech2TextDataset(speech_valid, character_text_valid)

    test_dataset = Speech2TextDataset(speech_test, None, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, collate_fn=collate_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)#, collate_fn=collate_train)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)#, collate_fn=collate_test)

    for epoch in range(nepochs):
        train(model, train_loader, criterion, optimizer,epoch)
        # scheduler.step()
        val(model, val_loader, criterion, epoch)
        # Test and Save results
        test_preds = test(model, test_loader)
        test_preds = test_preds.cpu().numpy()
        results = []
        for i in range(test_preds.shape[0]):
            result = ""
            for j in range(test_preds.shape[1]):
                if (test_preds[i,j] == 0 or (test_preds[i,j] == 33)):
                    continue
                if (test_preds[i,j] == 34):
                    break
                result = result + index2letter[test_preds[i,j]]
            results.append(result)
        name = "Epoch_"+str(epoch) + "_LAS_submission.csv"
        ids = list(range(len(test_dataset)))
        ids.insert(0,'Id')
        results.insert(0,'Predicted')
        with open(name, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(ids, results))



if __name__ == '__main__':
    main()