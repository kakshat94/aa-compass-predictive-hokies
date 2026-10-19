"""
Orchestrator class for managing workforce optimization models.
It coordinates demand data, work center day sets, and bidline generation.
"""

import xpress as xp
import gurobipy as gp
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from members.akshat.model import WorkforceOptimizer
from members.akshat.bidline import BidlineGenerator

class Orchestrator:
    """
    Orchestrator class for managing workforce optimization models.
    It coordinates demand data, work center day sets, and bidline generation.
    """
    def __init__(self, demand_df: pd.DataFrame, workcenter_day_sets: dict):
        self.demand_df = demand_df.copy()
        self.demand_df['SHIFT_DATE'] = pd.to_datetime(self.demand_df['SHIFT_DATE']).dt.strftime('%Y-%m-%d')
        self.wc_day_sets = workcenter_day_sets
        self.shifts = list(self.demand_df['SHIFT_TYPE'].unique())
        start_date = self.demand_df['SHIFT_DATE'].min()
        end_date = self.demand_df['SHIFT_DATE'].max()
        self.bidline_gen = BidlineGenerator(start_date, end_date)
        self.coverage_vectors = self.bidline_gen.get_all_coverage_vectors()
        self.max_bidlines = self.get_max_bidlines()


    def get_max_bidlines(self) -> Dict:
        max_bidlines = {}
        for wc in self.wc_day_sets.keys():
            max_bidlines[wc] = {}
            max_d = self.demand_df[self.demand_df['WORK_CENTER']==wc].groupby('SHIFT_TYPE')['DEMAND'].max().to_dict()
            for shift, value in max_d.items():
                q, r = divmod(value, 8)
                max_bidlines[wc][shift] = 15*q + 2*r
            
        
        return max_bidlines


    def initiate_models(self):
        """
        Constructs the workforce optimization model using Gurobi.
        """
        w_ctrs = self.wc_day_sets.keys()
        # Create Gurobi environment and model
        coverage_days = {}
        #coverage_days_midnight = {}
        all_training_days = {}
        #all_training_days_midnight = []
        for wc, d in self.wc_day_sets.items():
            training_days = [item for sublist in d['training_days'] for item in sublist]
            #training_days_midnight = [datetime.strptime(c, "%Y-%m-%d") - pd.Timedelta(days=1) for c in training_days]
            #training_days_midnight = [d.strftime("%Y-%m-%d") for d in training_days_midnight]
            coverage_days[wc] = d['demand_days'] + training_days
            if wc!='DOMESTIC':
                coverage_days[wc] = coverage_days[wc] + [item for sublist in self.wc_day_sets['Domestic']['training_days'] for item in sublist]
            #midnight_days = [datetime.strptime(c, "%Y-%m-%d") + pd.Timedelta(days=1) for c in coverage_days[wc]]
            #coverage_days_midnight[wc] = [d.strftime("%Y-%m-%d") for d in midnight_days]
            all_training_days[wc] = training_days
            #all_training_days_midnight.extend(training_days_midnight)
        self.models = {}
        total_bids = 0
        for j in self.shifts:
            demand = {}
            
            if j=='MIDNIGHT':
                for wc in w_ctrs:
                    demand[wc] = {d: self.demand_df[(self.demand_df['SHIFT_TYPE']==j) & (self.demand_df['WORK_CENTER']==wc) & (self.demand_df['SHIFT_DATE']==d)]['DEMAND'].values[0]
                                for d in coverage_days[wc]}
                self.models[j] = WorkforceOptimizer(j, demand, coverage_days, all_training_days, self.wc_day_sets, self.max_bidlines, self.coverage_vectors, gp.Env())
                self.models[j].build_model()
                self.models[j].model.optimize()
                total_bids += self.models[j].model.ObjVal
            else:
                for wc in w_ctrs:
                    demand[wc] = {d: self.demand_df[(self.demand_df['SHIFT_TYPE']==j) & (self.demand_df['WORK_CENTER']==wc) & (self.demand_df['SHIFT_DATE']==d)]['DEMAND'].values[0]
                                    for d in coverage_days[wc]}
                self.models[j] = WorkforceOptimizer(j, demand, coverage_days, all_training_days, self.wc_day_sets, self.max_bidlines, self.coverage_vectors, gp.Env())
                self.models[j].build_model()
                self.models[j].model.optimize()
                total_bids += self.models[j].model.ObjVal
        print('Total bids', total_bids)

