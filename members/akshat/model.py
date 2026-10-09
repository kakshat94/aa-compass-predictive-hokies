import xpress as xp
import gurobipy as gp
import pandas as pd
from typing import Dict, Any
from members.akshat.config import ProjectConfig
from members.akshat.bidline import BidlineGenerator

class WorkforceOptimizer:
    def __init__(self, shift: str, demand: dict, coverage_days: dict, training_days: list, wc_day_sets: dict, max_bidlines: dict, coverage_vectors: dict, env: gp.Env):
        self.shift = shift
        self.demand = demand
        self.coverage_days = coverage_days
        self.training_days = training_days
        self.wc_day_sets = wc_day_sets
        self.max_bidlines = max_bidlines
        self.coverage_vectors = coverage_vectors
        self.env: gp.Env = env
        self.construct_var_sets()

    def construct_var_sets(self):
        self.X_set = [(wc, f"Number{i}", j) for wc in self.max_bidlines.keys() for i in range(self.max_bidlines[wc][self.shift]) for j in self.coverage_vectors.keys()]
        self.Y_set = [(wc, f"Number{i}", j, k) for wc in self.max_bidlines.keys() for i in range(self.max_bidlines[wc][self.shift]) for j in self.coverage_vectors.keys() for k in list(set(self.coverage_days[wc]) | set(self.training_days))]
    
    def build_model(self):
        """
        Formulates and solves the FICO Xpress optimization problem.
        Decision variables: Number of bidlines assigned to (workcenter, shift, offset).
        """
        self.model = gp.Model(f"Bidline_Model{self.shift}", env=self.env)

        self.X = self.model.addVars(
            self.X_set,  
            vtype=gp.GRB.BINARY, 
            name="bidlines"
        )
        self.Y = self.model.addVars(
                    self.Y_set,  
                    vtype=gp.GRB.BINARY, 
                    name="train"
                )
        # F = self.model.addVars(
        #                     self.F_set,  
        #                     vtype=gp.GRB.BINARY, 
        #                     name="flex"
        #                 )
        # Z = self.model.addVars(
        #                     self.Z_set,  
        #                     vtype=gp.GRB.BINARY, 
        #                     name="vacation"
        #                 )
        
        self.cov = self.model.addConstrs(
                    (gp.quicksum(self.X[wc, number, offset] * self.coverage_vectors[offset][day] 
                                - self.Y[wc, number, offset, day] * self.coverage_vectors[offset][day]
                                #+ F[wc, number, offset, day] * (1 - self.coverage_vectors[offset][day])
                                #- Z[wc, number, offset, day] * (1 - self.coverage_vectors[offset][day])
                                for wc, number, offset in self.X_set if wc == w)                          
                    >= self.demand[w][day] for w in self.demand.keys() for day in self.coverage_days[w]),
                    name="coverage"
                )

        self.alltrain = self.model.addConstrs(
                    (gp.quicksum(self.Y[wc, number, offset, day]
                                 #+ F[wc, number, offset] * (1 - self.coverage_vectors[offset][day])
                                 #- Z[wc, number, offset] * (1 - self.coverage_vectors[offset][day])
                                 for wc, number, offset in self.X_set)                          
                     <= 12 for day in self.training_days),
                    name="alltrain"
                )

        all_domestic_train_days = list({item for sublist in self.wc_day_sets['Domestic']['training_days'] for item in sublist})
        self.dom_trainbid_const = {}
        self.special_train = {}
        for w in self.wc_day_sets.keys():
            if w=='DOMESTIC':
                for i, per in enumerate(self.wc_day_sets[w]['training_days']):
                    self.dom_trainbid_const[f"Period{i}"] = self.model.addConstrs(
                            (gp.quicksum(self.Y[wc, number, offset, day]
                                            for day in per)                          
                                == self.X[wc, number, offset] for (wc, number, offset) in self.X_set),
                            name="domestic_train"
                        )
                self.model.addConstrs(
                            (gp.quicksum(self.Y[wc, number, offset, day]
                                            for day in self.coverage_days[wc] if day not in self.wc_day_sets[wc]['training_days'])                          
                                == 0 for (wc, number, offset) in self.X_set if wc==w),
                            name="domestic_train_0s"
                        )
                self.model.addConstrs(
                            (gp.quicksum(self.Y[wc, number, offset, day]
                                            for day in self.coverage_days[wc] if day in {i: e for i,e in self.coverage_vectors[offset].items() if e==0}.keys())                          
                                == 0 for (wc, number, offset) in self.X_set if wc==w),
                            name="domestic_train_0s_2"
                        )
                
            else:
                if len(self.wc_day_sets[w]['training_days'])==1:
                    self.special_train[w] = self.model.addConstrs(
                        (gp.quicksum(self.Y[wc, number, offset, day]
                                    for day in self.wc_day_sets[wc]['training_days'][0])
                        == self.X[wc, number, offset] for (wc, number, offset) in self.X_set if wc==w),
                        name="special_train"
                    )
                    self.model.addConstrs(
                                    (gp.quicksum(self.Y[wc, number, offset, day]
                                                    for day in self.coverage_days[wc] if day not in list(set(self.wc_day_sets[wc]['training_days'][0]) | set(all_domestic_train_days)))
                                        == 0 for (wc, number, offset) in self.X_set if wc==w),
                                    name="special_train_0s"
                                )
         
                else:
                    self.special_train[w] = {}
                    special_train_day = list({item for sublist in self.wc_day_sets[w]['training_days'] for item in sublist})
                    for i, per in enumerate(self.wc_day_sets[w]['training_days']):
                        self.special_train[w][f"Period{i}"] = self.model.addConstrs(
                                (gp.quicksum(self.Y[wc, number, offset, day]
                                                for day in per)                          
                                    == self.X[wc, number, offset] for (wc, number, offset) in self.X_set if wc==w),
                                name="special_train_0s"
                            )
                    self.model.addConstrs(
                                    (gp.quicksum(self.Y[wc, number, offset, day]
                                                    for day in self.coverage_days[wc] if day not in list(set(special_train_day) | set(all_domestic_train_days)))
                                        == 0 for (wc, number, offset) in self.X_set if wc==w),
                                    name="special_train_0s"
                    )
                self.model.addConstrs(
                                (gp.quicksum(self.Y[wc, number, offset, day]
                                                for day in self.coverage_days[wc] if day  in {i: e for i,e in self.coverage_vectors[offset].items() if e==0}.keys())
                                    == 0 for (wc, number, offset) in self.X_set if wc==w),
                                name="special_train_0s_2"
                            )

        self.model.setObjective(
            gp.quicksum(self.X[wc, number, offset] for (wc, number, offset) in self.X_set),
            gp.GRB.MINIMIZE
        )

        self.model.update()
        # return {
        #     'solution_df': pd.DataFrame(results),
        #     'obj_val': self.model.getObjVal(),
        #     'mip_status': self.model.getProbStatusString()
        # }


    