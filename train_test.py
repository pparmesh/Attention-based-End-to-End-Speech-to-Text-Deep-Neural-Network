import time
import torch
### Add Your Other Necessary Imports Here! ###

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    model = model.to(DEVICE)
    start = time.time()
    print("In Train..\n")
    # 1) Iterate through your loader
    avg_perplexity = []
    for batch_idx, (X, X_lens, Y, Y_lens) in enumerate(train_loader): 
        # 2) Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion
        # with torch.autograd.set_detect_anomaly(True): # Remove when no longer debugging
        
        # 3) Set the inputs to the device.
        X = X.to(DEVICE)
        X_lens = X_lens.to(DEVICE) # all data & model on same DEVICE
        Y = Y.to(DEVICE)
        Y_lens = Y_lens.to(DEVICE)

        # 4) Pass your inputs, and length of speech into the model.
        predictions = model.forward(X, X_lens, Y, isTrain=True)

        # 5) Generate a mask based on the lengths of the text to create a masked loss. 
        mask = torch.arange(Y.shape[1]).unsqueeze(1).to(DEVICE) < (Y_lens-1).unsqueeze(0)
        mask = mask.T
        # 5.1) Ensure the mask is on the device and is the correct shape.
        mask = mask.to(DEVICE)
        # 6) If necessary, reshape your predictions and origianl text input 
        # 6.1) Use .contiguous() if you need to. 
        predictions = predictions.contiguous()
        predictions = torch.transpose(predictions,1,2)
        # 7) Use the criterion to get the loss.
        # Offset labels
        offset_pad = torch.zeros(Y.shape[0]).unsqueeze(1)
        offset_pad = offset_pad.type(torch.LongTensor).to(DEVICE)
        Y = torch.cat([Y[:,1:], offset_pad], dim=1)
        loss = criterion(predictions, Y)
        # 8) Use the mask to calculate a masked loss. 
        masked_loss = loss*mask
        loss = torch.sum(masked_loss) / Y_lens.shape[0]
        # 9) Run the backward pass on the masked loss. 
        optimizer.zero_grad()
        loss.backward()
        # 10) Use torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        
        # 11) Take a step with your optimizer
        optimizer.step()

        # 12) Normalize the masked loss
        norm_loss = torch.sum(masked_loss,1) / Y_lens
        # 13) Optionally print the training loss after every N batches
        perplexity = torch.mean(torch.exp(norm_loss))
        avg_perplexity.append(perplexity)
        print("Train Epoch: ",epoch, "Batch: ",batch_idx," Perplexity: ",perplexity)
    end = time.time()
    avg_perplexity = sum(avg_perplexity) / len(avg_perplexity)
    print("Avg Perplexity: ",avg_perplexity, "Time Elapsed: ",end-start)

def val(model, val_loader, criterion, epoch):
    model.eval()
    model = model.to(DEVICE)
    start = time.time()
    print("In Val..\n")
    avg_perplexity = []

    # 1) Iterate through your loader
    for batch_idx, (X, X_lens, Y, Y_lens) in enumerate(val_loader): 
        # 2) Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion
        # with torch.autograd.set_detect_anomaly(True): # Remove when no longer debugging
        
        # 3) Set the inputs to the device.
        X = X.to(DEVICE)
        X_lens = X_lens.to(DEVICE) # all data & model on same DEVICE
        Y = Y.to(DEVICE)
        Y_lens = Y_lens.to(DEVICE)

        # 4) Pass your inputs, and length of speech into the model.
        predictions = model.forward(X, X_lens, Y, isTrain=True)

        # 5) Generate a mask based on the lengths of the text to create a masked loss. 
        mask = torch.arange(Y.shape[1]).unsqueeze(1).to(DEVICE) < (Y_lens-1).unsqueeze(0)
        mask = mask.T
        # 5.1) Ensure the mask is on the device and is the correct shape.
        mask = mask.to(DEVICE)
        # 6) If necessary, reshape your predictions and origianl text input 
        # 6.1) Use .contiguous() if you need to. 
        predictions = predictions.contiguous()
        predictions = torch.transpose(predictions,1,2)
        # 7) Use the criterion to get the loss.
        offset_pad = torch.zeros(Y.shape[0]).unsqueeze(1)
        offset_pad = offset_pad.type(torch.LongTensor).to(DEVICE)
    
        Y = torch.cat([Y[:,1:], offset_pad], dim=1)
        loss = criterion(predictions, Y)
        # 8) Use the mask to calculate a masked loss. 
        masked_loss = loss*mask
        loss = torch.sum(masked_loss) / Y_lens.shape[0]
        # 12) Normalize the masked loss
        # print("Masked Loss ",masked_loss.shape)
        norm_loss = torch.sum(masked_loss,1) / Y_lens
        # 13) Optionally print the training loss after every N batches
        perplexity = torch.mean(torch.exp(norm_loss))
        avg_perplexity.append(perplexity)
        print("Val Batch: ",batch_idx," Perplexity: ",perplexity)
    end = time.time()
    avg_perplexity = sum(avg_perplexity) / len(avg_perplexity)
    print("Val Epoch: ",epoch, "Avg Perplexity: ",avg_perplexity, "Time Elapsed: ",end-start)

def test(model, test_loader):
    ### Write your test code here! ###
    model.eval()
    model.to(DEVICE)
    predictions = []
    # 1) Iterate through your loader
    for batch_idx, (X, X_lens) in enumerate(test_loader): 
        X = X.to(DEVICE)
        X_lens = X_lens.to(DEVICE) # all data & model on same device
        prediction_prob = model.forward(X, X_lens, isTrain=False)
        prediction = torch.argmax(prediction_prob, dim=2)
        predictions.append(prediction)

    return torch.cat(predictions, dim=0)