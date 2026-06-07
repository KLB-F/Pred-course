import torch

def fn_perte_ListMLE_LK_S(preds:torch.tensor, target:torch.tensor):
    
    S = []

    for k in range(preds.shape[0]):
        ttarget = torch.tensor([int(e)-1 for e in target[k] if e != -1], dtype=torch.long)# torch.argsort(torch.tensor([int(e)-1 for e in target[k] if e != -1], dtype=torch.long))
        P = torch.zeros(1)
        for i in range(ttarget.shape[0]):
            #print(target[i:])
            #print(preds[:])
            #print(torch.exp(preds[target[i:]]))
            #print(preds[target[i:]])
            P = P + preds[k][ttarget[i]] - torch.logsumexp(preds[k][ttarget[i:]], dim=0)
        
        S.append(-P)
    
    return torch.stack(S, dim=0).sum()/preds.shape[0]

def fn_perte_ListMLE_LK_R(preds:torch.tensor, target:torch.tensor):
    
    S = []

    for k in range(preds.shape[0]):
        ttarget = torch.argsort(torch.tensor([int(e)-1 for e in target[k] if e != -1], dtype=torch.long))
        P = torch.zeros(1)
        for i in range(ttarget.shape[0]):
            #print(target[i:])
            #print(preds[:])
            #print(torch.exp(preds[target[i:]]))
            #print(preds[target[i:]])
            P = P + preds[k][ttarget[i]] - torch.logsumexp(preds[k][ttarget[i:]], dim=0)
        
        S.append(-P)
    
    return torch.stack(S, dim=0).sum()/preds.shape[0]