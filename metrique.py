import torch
import random

import matplotlib.pyplot as plt

class metrique():
    def __init__(self, val_data, val_paris, modele):
        self.val_data = val_data
        self.modele = modele
        self.val_paris = val_paris
        
        
    def ESP_SimpleGagnant(self):
        gain = []
        succes_compt = 0
        compteur_tot = 0
        
        for i in range(len(self.val_data)):
            if "SIMPLE_GAGNANT" in self.val_paris[i][0].keys():
                compteur_tot += 1
                m_g = int(torch.argmax(self.modele(self.val_data[i][0].unsqueeze(0))))+1
                if m_g == int(list(self.val_paris[i][0]["SIMPLE_GAGNANT"].keys())[0]):
                    succes_compt += 1
                    #print(m_g, int(list(self.val_paris[i][0]["SIMPLE_GAGNANT"].keys())[0]))
                    gain.append(self.val_paris[i][0]["SIMPLE_GAGNANT"][str(m_g)]*2-2)
                else:
                    gain.append(-2)
        return float(torch.tensor(gain).to(dtype=torch.float).mean()), succes_compt/compteur_tot
    
    def ESP_CoupleGagnant(self):
        gain = []
        succes_compt = 0
        compteur_tot = 0
        
        for i in range(len(self.val_data)):
            if "COUPLE_GAGNANT" in self.val_paris[i][0].keys():
                compteur_tot += 1
                _, m_g = torch.sort(self.modele(self.val_data[i][0].unsqueeze(0)), descending=True, dim=1)
                m_g = [int(m_g[0][0])+1, int(m_g[0][1])+1]
                if str(m_g[0]) + "-" + str(m_g[1]) == (list(self.val_paris[i][0]["COUPLE_GAGNANT"].keys())[0]):
                    succes_compt += 1
                    #print(m_g, int(list(self.val_paris[i][0]["SIMPLE_GAGNANT"].keys())[0]))
                    gain.append(self.val_paris[i][0]["COUPLE_GAGNANT"][str(m_g[0]) + "-" + str(m_g[1])]*2-2)
                else:
                    gain.append(-2)
        return float(torch.tensor(gain).to(dtype=torch.float).mean()), succes_compt/compteur_tot
    
    def ESP_SimpleGagnant_ALEA(self):
        gain = []
        succes_compt = 0 
        compteur_tot = 0
        
        for i in range(len(self.val_data)):
            if "SIMPLE_GAGNANT" in self.val_paris[i][0].keys():
                compteur_tot += 1
                r_g = random.randint(1, len([e for e in self.val_data[i][1] if e != -1])+1)
                if r_g == int(list(self.val_paris[i][0]["SIMPLE_GAGNANT"].keys())[0]):
                    succes_compt += 1
                    #print(m_g, int(list(self.val_paris[i][0]["SIMPLE_GAGNANT"].keys())[0]))
                    gain.append(self.val_paris[i][0]["SIMPLE_GAGNANT"][str(r_g)]*2-2)
                else:
                    gain.append(-2)
        return float(torch.tensor(gain).to(dtype=torch.float).mean()), succes_compt/compteur_tot
    
    def ESP_CoupleGagnant_ALEA(self):
        gain = []
        succes_compt = 0 
        compteur_tot = 0
        
        for i in range(len(self.val_data)):
            if "COUPLE_GAGNANT" in self.val_paris[i][0].keys():
                compteur_tot += 1
                n_k = len([e for e in self.val_data[i][1] if e != -1])+1
                r_g = [random.randint(1, n_k), random.randint(1, n_k)]
                if str(r_g[0]) + "-" + str(r_g[1]) == (list(self.val_paris[i][0]["COUPLE_GAGNANT"].keys())[0]):
                    succes_compt += 1
                    #print(m_g, int(list(self.val_paris[i][0]["SIMPLE_GAGNANT"].keys())[0]))
                    gain.append(self.val_paris[i][0]["COUPLE_GAGNANT"][str(r_g[0]) + "-" + str(r_g[1]) ]*2-2)
                else:
                    gain.append(-2)
        return float(torch.tensor(gain).to(dtype=torch.float).mean()), succes_compt/compteur_tot
    
    def ESP_SetC_Aff(self):
        simple_g, a_simple_g = self.ESP_SimpleGagnant()   # ou [1] selon ce que tu veux afficher
        couple_g, a_couple_g = self.ESP_CoupleGagnant()
        
        simple_alea = [self.ESP_SimpleGagnant_ALEA() for i in range(150)]
        couple_alea = [self.ESP_CoupleGagnant_ALEA() for i in range(150)]
        
        a_simple_alea = [e[1] for e in simple_alea]
        simple_alea = [e[0] for e in simple_alea]
        
        a_couple_alea = [e[1] for e in couple_alea]
        couple_alea = [e[0] for e in couple_alea]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle("Gain - Paris", fontsize=14, fontweight='bold')
        
        # --- Graphe 1 : Simple ---
        ax1.hist(simple_alea, bins=30, edgecolor='white', alpha=0.85, label='Distribution aléatoire')
        ax1.axvline(x=simple_g, color='red', linewidth=2.5, linestyle='--', label=f'Modele = {simple_g:.4f}')
        ax1.axvline(x=sum(simple_alea)/len(simple_alea), color='orange', linewidth=1.5, linestyle=':', label=f'Moyenne MC = {sum(simple_alea)/len(simple_alea):.4f}')
        ax1.set_title("Simple gagnant - Gain")
        ax1.set_xlabel("Gain")
        ax1.set_ylabel("Fréquence")
        ax1.legend()
        
        # --- Graphe 2 : Couple ---
        ax2.hist(couple_alea, bins=30, edgecolor='white', alpha=0.85, label='Distribution aléatoire')
        ax2.axvline(x=couple_g, color='red', linewidth=2.5, linestyle='--', label=f'Modele = {couple_g:.4f}')
        ax2.axvline(x=sum(couple_alea)/len(couple_alea), color='orange', linewidth=1.5, linestyle=':', label=f'Moyenne MC = {sum(couple_alea)/len(couple_alea):.4f}')
        ax2.set_title("Couple gagnant - Gain")
        ax2.set_xlabel("Valeur")
        ax2.set_ylabel("Fréquence")
        ax2.legend()
        
        # --- Graphe 3 : Simple ---
        ax3.hist(a_simple_alea, bins=30, edgecolor='white', alpha=0.85, label='Distribution aléatoire')
        ax3.axvline(x=a_simple_g, color='red', linewidth=2.5, linestyle='--', label=f'Modele = {a_simple_g:.4f}')
        ax3.axvline(x=sum(a_simple_alea)/len(a_simple_alea), color='orange', linewidth=1.5, linestyle=':', label=f'Moyenne MC = {sum(a_simple_alea)/len(a_simple_alea):.4f}')
        ax3.set_title("Simple gagnant - Précison")
        ax3.set_xlabel("Précision")
        ax3.set_ylabel("Fréquence")
        ax3.legend()
        
        # --- Graphe 2 : Couple ---
        ax4.hist(a_couple_alea, bins=30, edgecolor='white', alpha=0.85, label='Distribution aléatoire')
        ax4.axvline(x=a_couple_g, color='red', linewidth=2.5, linestyle='--', label=f'Modele = {a_couple_g:.4f}')
        ax4.axvline(x=sum(a_couple_alea)/len(a_couple_alea), color='orange', linewidth=1.5, linestyle=':', label=f'Moyenne MC = {sum(a_couple_alea)/len(a_couple_alea):.4f}')
        ax4.set_title("Couple gagnant - Précision")
        ax4.set_xlabel("Précision")
        ax4.set_ylabel("Fréquence")
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        