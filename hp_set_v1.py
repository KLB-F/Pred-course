import torch
import json

import os

import pandas as pd

from datetime import datetime
from datetime import date as ddate

class hippique_set_v1(torch.utils.data.Dataset):
    def __init__(self, fichier:str):
        print("-- HP_Set - Début du chargement --")
        
        super().__init__()
        
        #1er layer -> Créatiion de data semi-brut
        l_data = pd.read_csv(fichier+"/list_data.csv")
        
        self.data = []
        
        self.dict_hippodrome = {}
        compteur_dict_hippodrome = 0
        
        self.dict_nebulositeCode = {}
        compteur_dict_nebulositeCode = 0
        
        self.dict_cheval = {}
        compteur_dict_cheval = 0
        
        self.dict_jockey = {}
        compteur_dict_jockey = 0
        
        self.taille_nb_reunion = 0 #Le nombre de réunion présente dans le dataset
        self.taille_nb_courses = 0 #Le nombre de courses présentent dans le dataset
        self.nb_course_eng = 0
        
        date_ref = datetime(2023, 4, 23)
        
        for idx_data in range(len(l_data[l_data.columns[0]])):
            date = str(int(l_data[l_data.columns[0]][idx_data]))
            
            #Programme
            if len(date) < 8:
                date = "0" + date
            with open(fichier+"/"+date+"/programme.json", "r") as p:
                programme = json.load(p)
                p.close()
                
            p_ren = programme["programme"]["reunions"]
            
            self.p_t = programme
            
            Warning_ordre_arrive_pour_date = False
            
            reunion_data = []
            
            for reunion in p_ren:
                #Etude de la reunion
                reunion_data.append({})
                self.taille_nb_reunion += 1
                
                if reunion["hippodrome"]["libelleCourt"] not in self.dict_hippodrome.keys():
                    self.dict_hippodrome[reunion["hippodrome"]["libelleCourt"]] = compteur_dict_hippodrome
                    compteur_dict_hippodrome += 1
                reunion_data[-1]["hippodrome"] = self.dict_hippodrome[reunion["hippodrome"]["libelleCourt"]]
                
                if reunion["nature"] == "DIURNE":
                    reunion_data[-1]["nature"] = 1.
                elif reunion["nature"] == "NOCTURNE":
                    reunion_data[-1]["nature"] = 0.
                elif reunion["nature"] == "SEMINOCTURNE":
                    reunion_data[-1]["nature"] = 0.5
                else:
                    print("Erreur | HP_set_v1 | Nature inconnue :", reunion["nature"])
                
                if "PLAT" in reunion["disciplinesMere"]:
                    reunion_data[-1]["discipline_mere"] = 1.
                elif "TROT" in reunion["disciplinesMere"]:
                    reunion_data[-1]["discipline_mere"] = 2.
                elif "OBSTACLE" in reunion["disciplinesMere"] or "OBSTACLES" in reunion["disciplinesMere"]:
                    reunion_data[-1]["discipline_mere"] = 3.
                elif "MONTE" in reunion["disciplinesMere"]:
                    reunion_data[-1]["discipline_mere"] = 4.
                elif "ATTELE" in reunion["disciplinesMere"]:
                    reunion_data[-1]["discipline_mere"] = 5.
                elif "HAIE" in reunion["disciplinesMere"]:
                    reunion_data[-1]["discipline_mere"] = 6.
                elif "STEEPLECHASE" in reunion["disciplinesMere"]:
                    reunion_data[-1]["discipline_mere"] = 7.
                elif "CROSS" in reunion["disciplinesMere"]:
                    reunion_data[-1]["discipline_mere"] = 8.
                else:
                    print("Erreur | HP_set_v1 | Discipline mere inconnue :", reunion["disciplinesMere"])
                
                if "meteo" in reunion.keys():
                    if reunion["meteo"]["nebulositeCode"] not in self.dict_nebulositeCode.keys():
                        self.dict_nebulositeCode[reunion["meteo"]["nebulositeCode"]] = compteur_dict_nebulositeCode
                        compteur_dict_nebulositeCode += 1
                    reunion_data[-1]["nebulosite"] = self.dict_nebulositeCode[reunion["meteo"]["nebulositeCode"]]
                    
                    reunion_data[-1]["temperature"] = int(reunion["meteo"]["temperature"])
                    reunion_data[-1]["force_vent"] = int(reunion["meteo"]["forceVent"])
                else:
                    print("Warning | HP_set_v1 | Cle meteo non présente pour la réunion ", reunion["numOfficiel"], " datée du :", date)
                    reunion_data[-1]["nebulosite"] = -1
                    reunion_data[-1]["temperature"] = -1
                    reunion_data[-1]["farce_vent"] = -1
                
                reunion_data[-1]["nb_courses"] = len(reunion["courses"])
                reunion_data[-1]["date"] = (datetime.fromisoformat(date[4:]+date[2:4]+date[:2])-date_ref).days
                
                for i in range(len(reunion["courses"])):
                    courses = reunion["courses"][i]
                    self.nb_course_eng += 1
                    if "ordreArrivee" in courses.keys():
                        reunion_data[-1][i] = courses["ordreArrivee"]
                    else:
                        reunion_data[-1][i] = -1
                        if not Warning_ordre_arrive_pour_date:
                            print("Warning | HP_set_v1 | Ordre d'arrivée non présent pour une ou des réunions du " ,date)
                            Warning_ordre_arrive_pour_date = True
                
                self.reunion = reunion
                """
                reunion_data list[dict_keys(['hippodrome', 'nature', 'discipline_mere', 'nebulosite', 'temperature', 'force_vent', 'nb_courses', 'date', ordre_arrive])]
                """
                
            
            
                #Etude des résultats détaillés & performance détaillés
                for num_course in range(reunion_data[-1]["nb_courses"]):
                    if os.path.isfile(fichier+"/"+date+"/RD_"+str(len(reunion_data)-1)+"-"+str(num_course)+".json") and os.path.isfile(fichier+"/"+date+"/PD_"+str(len(reunion_data)-1)+"-"+str(num_course)+".json"):
                        
                        course_data = {}
                        
                        self.taille_nb_courses += 1
                        
                        with open(fichier+"/"+date+"/RD_"+str(len(reunion_data)-1)+"-"+str(num_course)+".json", "r") as f:
                            RD = json.load(f)
                            f.close()
                        with open(fichier+"/"+date+"/PD_"+str(len(reunion_data)-1)+"-"+str(num_course)+".json", "r") as f:
                            PD = json.load(f)
                            f.close()
                        
                        self.RD = RD
                        self.PD = PD
                        
                        for paris in RD:
                            course_data[paris["typePari"]] = {e["combinaison"]:e["dividendePourUnEuro"]/100 for e in paris["rapports"]}
                        
                        if PD["allure"] == "GALOP":
                            course_data["allure"] = 1.0
                        elif PD["allure"] == "TROT":
                            course_data["allure"] = 0.5
                        elif PD["allure"] == "None" or PD["allure"] == None:
                            course_data["allure"] = -1.
                        else:
                            print("Erreur | HP_set_v1 | Allure de course non connue : ", PD["allure"])
                        
                        for i in range(len(PD["participants"])):
                            cheval = PD["participants"][i]
                            num = cheval["numPmu"]
                            course_data[num] = {}
                            course_data[num]["index"] = i
                            
                            if not cheval["nomCheval"] in self.dict_cheval.keys():
                                self.dict_cheval[cheval["nomCheval"]] = compteur_dict_cheval
                                compteur_dict_cheval += 1
                            course_data[num]["cheval"] = self.dict_cheval[cheval["nomCheval"]]
                            
                            #Historique du cheval
                            course_data[num]["courseCourues"] = []
                            for c_couru in cheval["coursesCourues"]:
                                c_temp = {}
                                c_temp["date"] = (datetime.utcfromtimestamp(c_couru["date"]/1000)-date_ref).days
                                
                                if c_couru["hippodrome"] not in self.dict_hippodrome.keys():
                                    self.dict_hippodrome[c_couru["hippodrome"]] = compteur_dict_hippodrome
                                    compteur_dict_hippodrome += 1
                                c_temp["hippodrome"] = self.dict_hippodrome[c_couru["hippodrome"]]
                                
                                c_temp["distance"] = c_couru["distance"]
                                
                                if c_couru["discipline"] == "PLAT":
                                    c_temp["disciplines_mere"] = 1.
                                elif c_couru["discipline"] == "TROT":
                                    c_temp["disciplines_mere"] = 2.
                                elif c_couru["discipline"] == "OBSTACLE" or c_couru["discipline"] == "OBSTACLES":
                                    c_temp["disciplines_mere"] = 3.
                                elif c_couru["discipline"] == "MONTE":
                                    c_temp["disciplines_mere"] = 4.
                                elif c_couru["discipline"] == "ATTELE":
                                    c_temp["disciplines_mere"] = 5.
                                elif c_couru["discipline"] == "HAIE":
                                    c_temp["disciplines_mere"] = 6.
                                elif c_couru["discipline"] == "STEEPLECHASE":
                                    c_temp["disciplines_mere"] = 7.
                                elif c_couru["discipline"] == "CROSS":
                                    c_temp["disciplines_mere"] = 8.
                                else:
                                    print("Erreur | HP_set_v1 | {historique des courses} Discipline non connue : ", c_couru["discipline"])
                                
                                c_temp["nb_participants"] = c_couru["nbParticipants"]
                                
                                for e in c_couru["participants"]:
                                    if e["nomCheval"] == cheval["nomCheval"]:
                                        c_temp["place"] = e["place"]["place"]
                                        
                                        if e["nomJockey"] not in self.dict_jockey.keys():
                                            self.dict_jockey[e["nomJockey"]] = compteur_dict_jockey
                                            compteur_dict_jockey += 1
                                        c_temp["nom_jockey"] = self.dict_jockey[e["nomJockey"]]
                    
                                        if "poidsJockey" in c_couru.keys():
                                            c_temp["poids_jockey"] = c_couru["poidsJockey"]
                                        else:
                                            #print("Warning | HP_set_v1 | {historique des courses} Poids indéfinis")
                                            c_temp["poids_jockey"] = -1.
                                        c_temp["corde"] = e["corde"]
                                
                                course_data[num]["courseCourues"].append(c_temp)
                        
                        """
                        course_data : dict
                        
                        ["{type_paris}":dict("{condition_win_cheveaux}":gain_associé),
                         "allure":{allure},
                         int {numero} : dict("cheval":{num_cheval}, 
                                             "courseCourues": [dict( dict_keys(['date', 'hippodrome', 'distance', 'disciplines_mere', 'nb_participants', 'place', 'nom_jockey', 'poids_jockey', 'corde']) )])
                         ]
                        
                        """
                        
                        
                        self.data.append([course_data, reunion_data[-1]])
                    else:
                        pass
                        #print("Warning | HP_set_v1 | Pas de correspondance par apport au nombres de course pour la date ", date)
        print("-- HP_Set - Chargement fini  -- \nCourse chargés : ", self.taille_nb_courses, " pour ", self.taille_nb_reunion, " réunions")
            
    def __getitem__(self, index):
        return self.data[index]
        
    def __len__(self):
        return len(self.data)
    

                
                
                