import torch
from torch.utils.data import random_split, DataLoader
from torch import nn

import json

import metrique
from fonction_perte import fn_perte_ListMLE_LK_S

from hp_set_v1 import hippique_set_v1
from hp_sel_v1 import hippique_selecteur_v1_B

from hp_mod_v1 import hippique_modele_v1_C

import matplotlib.pyplot as plt

#Donnes

hp_set = hippique_set_v1("../Donnes")
hp_sel = hippique_selecteur_v1_B(hp_set)

entr_data, valid_data = random_split(hp_sel, [0.8, 0.2])

paris = [valid_data[i][-1] for i in range(len(valid_data))]
valid_data = [valid_data[i][:-1] for i in range(len(valid_data))]

paris_ent = [entr_data[i][-1] for i in range(len(entr_data))]
entr_data = [entr_data[i][:-1] for i in range(len(entr_data))]

entr_loader = DataLoader(entr_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

# %% Partie ML
print(" -- Entrainenement --")

modele = hippique_modele_v1_C()

modele.train()

epoch = 17

fn_perte = fn_perte_ListMLE_LK_S
optimiseur = torch.optim.Adam(modele.parameters(), lr=0.55e-4)

compteur_general = 0
err_liste = []
err_liste_train = []

err_train_tmp = [-1]*len(entr_loader)

for epoque in range(epoch):
    print("Epoch : ",epoque+1, " / ", epoch)
    if epoque == 6:
        for group in optimiseur.param_groups:
            group['lr'] /= 2
    elif epoque == 15:
        for group in optimiseur.param_groups:
            group['lr'] /= 1.3
    for x_batch, y_batch in entr_loader:
        
        #xt_batch, yt_batch = modele.inLAR_DATA(x_batch, y_batch)
        y = fn_perte(modele(x_batch), y_batch)
        y.backward()
        
        err_train_tmp[compteur_general%len(entr_loader)] = (float(y.detach().mean()))
        
        #for name, p in modele.named_parameters():
        #    if p.grad is not None:
        #        print(f"{name:40s} {p.grad.norm():.2e}")
        
        optimiseur.step()
        
        optimiseur.zero_grad()
        
        compteur_general += 1
        if compteur_general % 80 == 0 and compteur_general > 0:
            with torch.no_grad():
                m_perte = []
                for xv_batch, yv_batch in valid_loader:
    
                    #xvt_batch, yvt_batch = modele.inLAR_DATA(xv_batch, yv_batch)
                    m_perte.append(float(fn_perte(modele(xv_batch), yv_batch).mean()))
                err_liste.append(torch.tensor(m_perte).mean())
                err_liste_train.append(torch.tensor([e for e in err_train_tmp if e != -1]).mean())
                

print(" -- Fin de l'entrainement -- ")

modele.eval()

plt.plot([i*100 for i in range(len(err_liste))], err_liste, label="validation")
plt.plot([i*100 for i in range(len(err_liste))], err_liste_train, label="entrainement")
plt.title("Erreur moyenne")
plt.xlabel("step")
plt.ylabel("Erreur (MLE_LIST)")
plt.legend()
plt.show()

m = metrique.metrique(valid_data, paris, modele)
m.ESP_SetC_Aff()