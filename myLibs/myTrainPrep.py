import torch
import numpy as np
from sklearn.metrics import confusion_matrix



def labeled_batch_creator(data, target, batch_size = 20, pos_data_ratio = -1, pos_label = 1, neg_label = 0):
    
    
    
    if pos_data_ratio == -1:
        batch_real = []
        for i in range(0, len(data), batch_size):
            batch_real.append({'data': torch.from_numpy(data[i:(i+batch_size)]).type(torch.float), 
                               'target': torch.from_numpy(target[i:(i+batch_size)]).type(torch.int64)})

    else:
        pos_data   = list(data[target == pos_label])
        pos_target = list(target[target == pos_label])
        neg_data   = list(data[target == neg_label])
        neg_target = list(target[target == neg_label])

        pos_batch_size = int(batch_size * pos_data_ratio)
        neg_batch_size = batch_size - pos_batch_size
        
        num_of_total_batch = max( int(len(pos_data)/pos_batch_size) + 1, int(len(neg_data)/pos_batch_size) + 1 )
        total_loop         = max( pos_batch_size, neg_batch_size ) * num_of_total_batch

        batch_real = []
        _data   = []
        _target = []
        for i in range(total_loop):
            if len(_data) == batch_size:
                
                batch_real.append({'data':   torch.from_numpy(np.array(_data)).type(torch.float), 
                                   'target': torch.from_numpy(np.array(_target)).type(torch.int64)})
                _data   = []
                _target = []
                
            _data.append( pos_data[ i % len(pos_data) ] )
            _data.append( neg_data[ i % len(neg_data) ] )
            _target.append( pos_target[ i % len(pos_data) ] )
            _target.append( neg_target[ i % len(neg_data) ] )
    
    return batch_real


def unlabeled_batch_creator(data, batch_size = 10):
    
    batch_real = []
    
    for i in range(0, len(data), batch_size):
        batch_real.append({'data': torch.from_numpy(data[i:(i+batch_size)]).type(torch.float), 
                           'target': torch.from_numpy(np.ones(batch_size)).type(torch.int64)})
    return batch_real




def myPerform(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct   = 0
    num_test  = 0
    
    target_list = []
    output_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_labeled = batch['data'][:, np.newaxis, :].to(device)
            y_labeled = batch['target'].to(device)
            
            y_pred_labeled   = model(x_labeled)  
            pdist   = y_pred_labeled.gather(1, y_labeled.view(-1, 1)).squeeze(1)
            labeled = torch.logsumexp(y_pred_labeled, dim=1)
            lossS   = -torch.mean(pdist) + torch.mean(labeled)
            
            test_loss += lossS
            num_test  += y_labeled.size(0)
            
            target_list += y_labeled.cpu().tolist()
            output_list += y_pred_labeled.argmax(dim=1, keepdim=True).cpu().numpy().ravel().tolist()

    
    tn, fp, fn, tp = confusion_matrix( target_list, output_list ).ravel()
        
    return test_loss / num_test, (tp+tn)/(tp+fn+tn+fp)