
from sklearn.metrics import confusion_matrix
import torch
import numpy

def evaluator(model, criterion, device, test_loader, display = False):
    model.eval()
    test_loss = 0
    correct   = 0
    num_test  = 0
    
    target_list = []
    output_list = []
    
    with torch.no_grad():
        for testdata in test_loader:
            data, target = testdata
            data, target = data.to(device), target.to(device)
            _, output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_test += data.size(0)
            
            target_list += target.cpu().tolist()
            output_list += output.argmax(dim=1, keepdim=True).cpu().numpy().ravel().tolist()

    
    tn, fp, fn, tp = confusion_matrix( target_list, output_list ).ravel()

    if display == True:
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}({}/{}), Pos: {}/{}, Neg: {}/{}\n'.format(
        test_loss, correct/num_test, correct, num_test, tp, (tp+fn), tn, (tn+fp)))
        
    return test_loss / num_test, correct / num_test