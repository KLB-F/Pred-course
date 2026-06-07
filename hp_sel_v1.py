import torch
import json

import pandas as pd

import hp_sel_v1 as hippique_set

class hippique_selecteur_v1(torch.utils.data.Dataset):
    """
    Hippique selecteur v1.0
    
    Permet de sélectionner et associé les donnés à fournir au modèle
        Taux d'erreur lors de l'association  : ~ 2 % (9 réunions) 
                                               < 0.5 % (3 réunions)
    
    Variable principale : 
    
    self.train = liste contenant pour chaque course [allure, nature, hippodrome, discipline_mere, nebulosite, temperature, force_vent, date, info_cheval]
    avec info_cheval = [num_cheval, proportion_darrive_moyenne]
    
    self.target = liste conteannt pour chaque course l'ordre d'arrivee (en n° d'arrivé)
    
    self.paris = pour chaque course [paris, num_to_cheval] 
        paris : le dict contenant tout les paris 
        num_to_cheval : le dictionnaire permettant de passer du num° du cheval dans la course au n° du cheval en général
    
    Ne prend pas en compte les couples gagnants type "Autres cheveaux"
    
    """
    def __init__(self, hp_set:hippique_set):
        print("-- HP_Sel - Debut du chargement --")
        self.train = []
        self.target = []
        self.paris = []
        
        self.dict_cheval = hp_set.dict_cheval
        self.dict_jockey = hp_set.dict_jockey
        self.dict_hippodrome = hp_set.dict_hippodrome
        
        compteur_general = 0
        
        compteur_course = 0
        reunion_prec = None
        
        for course, reunion in hp_set:
            
            if compteur_course not in reunion.keys() or reunion_prec != reunion:
                compteur_course = 0
                reunion_prec = reunion
            
            allure = course["allure"]
            info_cheval = []
            
            temp_paris = {}
            
            num_a_cheval = {}
            
            for cle in course.keys():
                if type(cle) is not int and cle != "allure":
                    temp_paris[cle] = course[cle]
                elif type(cle) == int:
                    cheval = course[cle]["cheval"]
                    num_a_cheval[cle] = cheval
                    
                    h_rang = 0
                    l_rang = []
                    
                    for c_historique in course[cle]["courseCourues"]:
                        if "place" in c_historique and c_historique["place"] is not None and c_historique["nb_participants"] not in [None, 0] :
                            h_rang += c_historique["place"]/c_historique["nb_participants"]
                            l_rang.append(c_historique["place"]/c_historique["nb_participants"])
                    
                    h_rang = h_rang/len(course[cle]["courseCourues"]) if len(course[cle]["courseCourues"]) > 0 else -1
                    std = float(torch.tensor(l_rang).std())
                    
                    info_cheval.append([cheval, h_rang, std if std == std else -1])
            
            hippodrome = reunion["hippodrome"]
            nature = reunion["nature"]
            discipline_mere = reunion["discipline_mere"]
            nebulosite = reunion["nebulosite"]
            temperature = reunion["temperature"]
            force_vent = reunion["force_vent"]
            date = reunion["date"]
            
            liste_course = [cle for cle in reunion.keys() if type(cle) == int]
            for course_r in liste_course:
                try:
                    if "MULTI" in temp_paris.keys() and reunion[course_r] != -1 and list(temp_paris["MULTI"].keys())[0] != "Autres Chevaux" and [int(e) for e in list(temp_paris["MULTI"].keys())[0].split("-")] == [e[0] for e in reunion[course_r][:4]] and max(list(num_a_cheval.keys())) >= max([e[0] for e in reunion[course_r]]):
                        self.target.append([e[0] for e in reunion[course_r]])
                        self.train.append([allure, nature, hippodrome, discipline_mere, nebulosite, temperature, force_vent, date, info_cheval])
                        self.paris.append([temp_paris, num_a_cheval])
                    elif "MULTI" not in temp_paris.keys() and "MINI_MULTI" in temp_paris.keys() and reunion[course_r] != -1 and list(temp_paris["MINI_MULTI"].keys())[0] != "Autres Chevaux" and [int(e) for e in list(temp_paris["MINI_MULTI"].keys())[0].split("-")] == [e[0] for e in reunion[course_r][:4]] and max(list(num_a_cheval.keys())) >= max([e[0] for e in reunion[course_r]]):
                        self.target.append([e[0] for e in reunion[course_r]])
                        self.train.append([allure, nature, hippodrome, discipline_mere, nebulosite, temperature, force_vent, date, info_cheval])
                        self.paris.append([temp_paris, num_a_cheval]) 
                    elif "MULTI" not in temp_paris.keys() and "MINI_MULTI" not in temp_paris.keys() and "TRIO" in temp_paris.keys() and reunion[course_r] != -1 and list(temp_paris["TRIO"].keys())[0] != "Autres Chevaux" and [int(e) for e in list(temp_paris["TRIO"].keys())[0].split("-")] == [e[0] for e in reunion[course_r][:3]] and max(list(num_a_cheval.keys())) >= max([e[0] for e in reunion[course_r]]):
                        self.target.append([e[0] for e in reunion[course_r]])
                        self.train.append([allure, nature, hippodrome, discipline_mere, nebulosite, temperature, force_vent, date, info_cheval])
                        self.paris.append([temp_paris, num_a_cheval])
                    elif "MULTI" not in temp_paris.keys() and "MINI_MULTI" not in temp_paris.keys() and "COUPLE_GAGNANT" in temp_paris.keys() and reunion[course_r] != -1 and list(temp_paris["COUPLE_GAGNANT"].keys())[0] != "Autres Chevaux" and [int(e) for e in list(temp_paris["COUPLE_GAGNANT"].keys())[0].split("-")] == [e[0] for e in reunion[course_r][:2]] and max(list(num_a_cheval.keys())) >= max([e[0] for e in reunion[course_r]]):
                        #print([int(e) for e in list(temp_paris["COUPLE_GAGNANT"].keys())[0].split("-")])
                        #print([e[0] for e in reunion[course_r][:2]])
                        self.target.append([e[0] for e in reunion[course_r]])
                        self.train.append([allure, nature, hippodrome, discipline_mere, nebulosite, temperature, force_vent, date, info_cheval])
                        self.paris.append([temp_paris, num_a_cheval])
                except:
                    print(num_a_cheval)
                    print([e[0] for e in reunion[course_r]])
                    print(compteur_general)
                    print("Erreur | Hp_sel_v1 | Cle non valide -> COUPLE GAGNANT : ")
                    print(temp_paris["COUPLE_GAGNANT"].keys())
            compteur_course += 1
            compteur_general += 1
        print("-- HP_Sel - Chargement fini -- \nCourse après sélection : ", len(self.train))
        
    def __getitem__(self, index):
        return [self.train[index], self.target[index], self.paris[index]]
    
    def __len__(self):
        return len(self.train)
    
class hippique_selecteur_v1_B(torch.utils.data.Dataset):
    """
    Hippique selecteur v1.0 rev B
    
    Permet de sélectionner et associé les donnés à fournir au modèle
        Taux d'erreur lors de l'association  : ~ 2 % (9 réunions) 
                                               < 0.5 % (3 réunions)
    
    Variable principale : 
    
    self.train = liste contenant pour chaque course [allure, nature, hippodrome, discipline_mere, nebulosite, temperature, force_vent, date, info_cheval]
    avec info_cheval = [num_cheval, [liste des classements du cheval remené à 1]]
    
    self.target = liste contenant pour chaque course l'ordre d'arrivee (en n° d'arrivé)
    
    self.paris = pour chaque course [paris, num_to_cheval] 
        paris : le dict contenant tout les paris 
        num_to_cheval : le dictionnaire permettant de passer du num° du cheval dans la course au n° du cheval en général
    
    Ne prend pas en compte les couples gagnants type "Autres cheveaux"
    
    """
    def __init__(self, hp_set:hippique_set):
        print("-- HP_Sel - Debut du chargement --")
        self.train = []
        self.target = []
        self.paris = []
        
        self.dict_cheval = hp_set.dict_cheval
        self.dict_jockey = hp_set.dict_jockey
        self.dict_hippodrome = hp_set.dict_hippodrome
        
        compteur_general = 0
        
        compteur_course = 0
        reunion_prec = None
        
        for course, reunion in hp_set:
            
            if compteur_course not in reunion.keys() or reunion_prec != reunion:
                compteur_course = 0
                reunion_prec = reunion
            
            allure = course["allure"]
            info_cheval = []
            
            temp_paris = {}
            
            num_a_cheval = {}
            
            for cle in course.keys():
                if type(cle) is not int and cle != "allure":
                    temp_paris[cle] = course[cle]
                elif type(cle) == int:
                    cheval = course[cle]["cheval"]
                    num_a_cheval[cle] = cheval
    
    
                    l_rang = []
                    
                    for c_historique in course[cle]["courseCourues"]:
                        if "place" in c_historique and c_historique["place"] is not None and c_historique["nb_participants"] not in [None, 0] :

                            l_rang.append(c_historique["place"]/c_historique["nb_participants"])

                    info_cheval.append(l_rang)
            
            hippodrome = reunion["hippodrome"]
            nature = reunion["nature"]
            discipline_mere = reunion["discipline_mere"]
            nebulosite = reunion["nebulosite"]
            temperature = reunion["temperature"]
            force_vent = reunion["force_vent"]
            date = reunion["date"]
            
            def inter_hp_sel(allure, nature, hippodrome, discipline_mere, nebulosite, temperature, force_vent, date, info_cheval):
                chev_data = torch.full((49, 10), -1.0)
                
                for i in range(len(info_cheval)):
                    for j in range(len(info_cheval[i])):
                        chev_data[i][j] = info_cheval[i][j]
                return torch.cat([torch.tensor([allure, nature, hippodrome, discipline_mere, nebulosite, temperature, force_vent, date], dtype=torch.float), chev_data.flatten()], dim=0)
            
            liste_course = [cle for cle in reunion.keys() if type(cle) == int]
            for course_r in liste_course:
                try:
                    if "MULTI" in temp_paris.keys() and reunion[course_r] != -1 and list(temp_paris["MULTI"].keys())[0] != "Autres Chevaux" and [int(e) for e in list(temp_paris["MULTI"].keys())[0].split("-")] == [e[0] for e in reunion[course_r][:4]] and max(list(num_a_cheval.keys())) >= max([e[0] for e in reunion[course_r]]):
                        self.target.append(torch.tensor( ([e[0] for e in reunion[course_r]] + [-1 for e in range(49-len(reunion[course_r]))]), dtype=torch.int))
                        self.train.append(inter_hp_sel(allure, nature, hippodrome, discipline_mere, nebulosite, temperature, force_vent, date, info_cheval))
                        self.paris.append([temp_paris, num_a_cheval])
                    elif "MULTI" not in temp_paris.keys() and "MINI_MULTI" in temp_paris.keys() and reunion[course_r] != -1 and list(temp_paris["MINI_MULTI"].keys())[0] != "Autres Chevaux" and [int(e) for e in list(temp_paris["MINI_MULTI"].keys())[0].split("-")] == [e[0] for e in reunion[course_r][:4]] and max(list(num_a_cheval.keys())) >= max([e[0] for e in reunion[course_r]]):
                        self.target.append(torch.tensor( ([e[0] for e in reunion[course_r]] + [-1 for e in range(49-len(reunion[course_r]))]), dtype=torch.int))
                        self.train.append(inter_hp_sel(allure, nature, hippodrome, discipline_mere, nebulosite, temperature, force_vent, date, info_cheval))
                        self.paris.append([temp_paris, num_a_cheval]) 
                    elif "MULTI" not in temp_paris.keys() and "MINI_MULTI" not in temp_paris.keys() and "TRIO" in temp_paris.keys() and reunion[course_r] != -1 and list(temp_paris["TRIO"].keys())[0] != "Autres Chevaux" and [int(e) for e in list(temp_paris["TRIO"].keys())[0].split("-")] == [e[0] for e in reunion[course_r][:3]] and max(list(num_a_cheval.keys())) >= max([e[0] for e in reunion[course_r]]):
                        self.target.append(torch.tensor( ([e[0] for e in reunion[course_r]] + [-1 for e in range(49-len(reunion[course_r]))]), dtype=torch.int))
                        self.train.append(inter_hp_sel(allure, nature, hippodrome, discipline_mere, nebulosite, temperature, force_vent, date, info_cheval))
                        self.paris.append([temp_paris, num_a_cheval])
                    elif "MULTI" not in temp_paris.keys() and "MINI_MULTI" not in temp_paris.keys() and "COUPLE_GAGNANT" in temp_paris.keys() and reunion[course_r] != -1 and list(temp_paris["COUPLE_GAGNANT"].keys())[0] != "Autres Chevaux" and [int(e) for e in list(temp_paris["COUPLE_GAGNANT"].keys())[0].split("-")] == [e[0] for e in reunion[course_r][:2]] and max(list(num_a_cheval.keys())) >= max([e[0] for e in reunion[course_r]]):
                        #print([int(e) for e in list(temp_paris["COUPLE_GAGNANT"].keys())[0].split("-")])
                        #print([e[0] for e in reunion[course_r][:2]])
                        self.target.append(torch.tensor( ([e[0] for e in reunion[course_r]] + [-1 for e in range(49-len(reunion[course_r]))]), dtype=torch.int))
                        self.train.append(inter_hp_sel(allure, nature, hippodrome, discipline_mere, nebulosite, temperature, force_vent, date, info_cheval))
                        self.paris.append([temp_paris, num_a_cheval])
                except:
                    print(num_a_cheval)
                    print([e[0] for e in reunion[course_r]])
                    print(compteur_general)
                    print("Erreur | Hp_sel_v1 | Cle non valide -> COUPLE GAGNANT : ")
                    print(temp_paris["COUPLE_GAGNANT"].keys())
            compteur_course += 1
            compteur_general += 1
        print("-- HP_Sel - Chargement fini -- \nCourse après sélection : ", len(self.train))
        
    def __getitem__(self, index):
        return [self.train[index], self.target[index], self.paris[index]]
    
    def __len__(self):
        return len(self.train)