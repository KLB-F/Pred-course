import torch
from torch import nn

class hippique_modele_v1(nn.Module):
    def __init__(self, taille:int=64):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.Linear(11, taille),
            nn.BatchNorm1d(taille),
            nn.ReLU(),
            nn.Linear(taille, taille//2),
            nn.ReLU(),
            nn.Linear(taille//2, taille//4),
            nn.ReLU(),
            nn.Linear(taille//4, 4),
            nn.BatchNorm1d(4),
            nn.Tanh(),
            nn.Linear(4, 1)
            )
    
    def forward(self, x):
        return self.c1(x)
        
class hippique_modele_v1_A(nn.Module):
    """
    hippique modele v1 rev A
    
    -- Fonctionne avec HP_SEL v1 --
    """
    def __init__(self):
        taille_conv = 256
        self.int_shape = 5
        super().__init__()
        self.ManConv = nn.Sequential(
            nn.Linear(11, taille_conv),
            nn.ReLU(),
            nn.Linear(taille_conv, taille_conv//2),
            nn.ReLU(),
            nn.Linear(taille_conv//2, taille_conv//3),
            nn.Tanh(),
            nn.Linear(taille_conv//3, taille_conv//3),
            nn.ReLU(),
            nn.Linear(taille_conv//3, taille_conv//4),
            nn.ReLU(),
            nn.Linear(taille_conv//4, self.int_shape),
            nn.Tanh()
            )
        self.Tete = nn.Sequential(
            nn.Linear(49*self.int_shape, 49*self.int_shape*2),
            nn.Tanh(),
            nn.Linear(49*self.int_shape*2, 49*self.int_shape*4),
            nn.ReLU(),
            nn.Linear(49*self.int_shape*4, 49*self.int_shape*2),
            nn.ReLU(),
            nn.Linear(49*self.int_shape*2, 49*self.int_shape//2),
            nn.ReLU(),
            nn.Linear(49*self.int_shape//2, 49*self.int_shape//2),
            nn.Tanh(),
            nn.Linear(49*self.int_shape//2, 49)
            )
    
    def forward(self, x):
        hippo_data = x[:8]
        chev_data = x[8:]
        int_data = torch.zeros((49,self.int_shape))
        for i in range(49):
            int_data[i] = self.ManConv(torch.cat([hippo_data, chev_data[i:i+3]]))
        return self.Tete(int_data.flatten())
    
    def inLAR_data(self, x_batch, y_batch):
        hippo_data = x_batch[:8]
        chev_data = x_batch[-1] + [[-1, -1, -1] for e in range(49-len(x_batch[-1]))]
        
        inLAR_x = torch.cat([torch.tensor(hippo_data), torch.tensor(chev_data).flatten()], dim = 0).flatten().to(dtype=torch.float)
        inLAR_y = y_batch + [-1 for e in range(49-len(y_batch))]
        
        return inLAR_x, torch.tensor(inLAR_y, dtype=torch.float)
    
class hippique_modele_v1_B(nn.Module):
    """
    hippique modele v1 rev B
    
    -- Fonctionne avec HP_SEL v1 rev B | fn_perte_ListMLE_LK_S --
    """
    def __init__(self):
        
        conv_terrain_taille = 32
        terrain_int_taille = 8
        
        conv_historique_taille = 80
        self.historique_int_taille = 8
        
        conv_cheval_taille = 32
        self.cheval_int_taille = 4
        
        super().__init__()
        
        self.Conv_T = nn.Sequential(
            nn.Linear(8, conv_terrain_taille),
            nn.ReLU(),
            nn.Linear(conv_terrain_taille, conv_terrain_taille//2),
            nn.BatchNorm1d(conv_terrain_taille//2),
            nn.ReLU(),
            nn.Linear(conv_terrain_taille//2, conv_terrain_taille//4),
            nn.Tanh(),
            nn.Linear(conv_terrain_taille//4, terrain_int_taille)
            )
        
        self.Conv_H = nn.Sequential(
            nn.Linear(10, conv_historique_taille),
            nn.ReLU(),
            nn.Linear(conv_historique_taille, int(conv_historique_taille//(2))),
            nn.LeakyReLU(),
            nn.Linear(conv_historique_taille//2, conv_historique_taille//4),
            nn.Tanh(),
            nn.Linear(conv_historique_taille//4, self.historique_int_taille)
            )
        
        self.Conv_C = nn.Sequential(
            nn.Linear(self.historique_int_taille+terrain_int_taille, conv_cheval_taille),
            nn.ReLU(),
            nn.Linear(conv_cheval_taille, self.cheval_int_taille)
            )
        
        self.Tete = nn.Sequential(
            nn.Linear(self.cheval_int_taille, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # score scalaire par cheval
            )

    def forward(self, x):
        hippo_data = self.Conv_T(x[:, :8])
        s = []
        
        for i in range(49):
            ic = torch.cat([hippo_data, self.Conv_H(x[:, i*10+8:i*10+18])], dim=1)
            s.append(self.Tete(self.Conv_C(ic)))
        
        return torch.stack(s, dim=1)
        
        
    def inLAR_DATA(self, xbatch, ybatch):
        
        hippo_data = torch.tensor(xbatch[:, :8], dtype=torch.float)
        chev_data = torch.full((xbatch.shape[0],49, 10), -1.0)
        
        for k in range(xbatch):
            for i in range(len(xbatch[-1])):
                for j in range(len(xbatch[-1][i])):
                    chev_data[i][j] = xbatch[-1][i][j]
        
        inLAR_y = ybatch + [-1 for e in range(49-len(ybatch))]
        return torch.cat([hippo_data,chev_data.flatten()]), torch.tensor(inLAR_y).to(dtype=torch.float)
    
class hippique_modele_v1_C(nn.Module):
    """
    hippique modele v1 rev C
    
    -- Fonctionne avec HP_SEL v1 rev B | fn_perte_ListMLE_LK_S --
    
    Meilleure score : 
       11.28 % de précision en simple gagnant | -0.48 de gain en simple gagnant
       0.89 % de précision en couple gagnant | -1.74 de gain en double gagnant
       
       Pour : batch = 32 | epoch = 15 | lr = 0.5e-4 puis 0.25e-4 pour epoque = 5 puis 0.25e-4/1.3 pour epoque = 13
    """
    def __init__(self):
        
        conv_terrain_taille = 32
        terrain_int_taille = 8
        
        conv_historique_taille = 156
        self.historique_int_taille = 8
        
        conv_cheval_taille = 32
        self.cheval_int_taille = 20
        
        super().__init__()
        
        self.Conv_T = nn.Sequential(
            nn.Linear(8, conv_terrain_taille),
            nn.ReLU(),
            nn.Linear(conv_terrain_taille, conv_terrain_taille//4),
            nn.BatchNorm1d(conv_terrain_taille//4),
            nn.ReLU(),
            nn.Linear(conv_terrain_taille//4, terrain_int_taille)
            )
        
        self.Conv_H = nn.Sequential(
            nn.Linear(10, conv_historique_taille),
            nn.ReLU(),
            nn.Linear(conv_historique_taille, int(conv_historique_taille//(1.5))),
            nn.BatchNorm1d(int(conv_historique_taille//(1.5))),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(int(conv_historique_taille//(1.5)), conv_historique_taille//2),
            nn.BatchNorm1d(conv_historique_taille//2),
            nn.ReLU(),
            nn.Linear(conv_historique_taille//2, conv_historique_taille//3),
            nn.BatchNorm1d(conv_historique_taille//3),
            nn.ReLU(),
            nn.Linear(conv_historique_taille//3, conv_historique_taille//4),
            nn.BatchNorm1d(conv_historique_taille//4),
            nn.ReLU(),
            nn.Linear(conv_historique_taille//4, conv_historique_taille//4),
            nn.BatchNorm1d(conv_historique_taille//4),
            nn.ReLU(),
            nn.Linear(conv_historique_taille//4, conv_historique_taille//5),
            nn.BatchNorm1d(conv_historique_taille//5),
            nn.ReLU(),
            nn.Linear(conv_historique_taille//5, conv_historique_taille//6),
            nn.BatchNorm1d(conv_historique_taille//6),
            nn.ReLU(),
            nn.Linear(conv_historique_taille//6, self.historique_int_taille)
            )
        
        self.Conv_C = nn.Sequential(
            nn.Linear(self.historique_int_taille+terrain_int_taille, conv_cheval_taille),
            nn.BatchNorm1d(conv_cheval_taille),
            nn.ReLU(),
            nn.Linear(conv_cheval_taille, self.cheval_int_taille)
            )
        
        self.Tete = nn.Sequential(
            nn.Linear(self.cheval_int_taille, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.Tanh(),
            nn.Linear(8, 1)
            )

    def forward(self, x):
        hippo_data = self.Conv_T(x[:, :8])
        s = []
        
        for i in range(49):
            ic = torch.cat([hippo_data, self.Conv_H(x[:, i*10+8:i*10+18])], dim=1)
            s.append(self.Tete(self.Conv_C(ic)))
        
        return torch.stack(s, dim=1)