import xpress as xp
import pandas as pd
import yaml
import os
import logging
from typing import Dict, Any
from pipelines.phase1.config import ProjectConfig
from pipelines.phase1.bidline import BidlineGenerator

_PARAMS_PATH = os.path.join("params.yaml")
with open(_PARAMS_PATH) as _f:
    _PARAMS = yaml.safe_load(_f)
_FLEX_PARAMS: Dict[str, int] = {
    wc: cfg["max_flex_days"] for wc, cfg in _PARAMS.get("flex_days", {}).items()
}


class WorkforceOptimizer:
    def __init__(
        self,
        shift: str,
        demand: dict,
        coverage_days: dict,
        training_days: list,
        wc_day_sets: dict,
        max_bidlines: dict,
        coverage_vectors: dict,
    ):
        self.shift = shift
        self.demand = demand
        self.coverage_days = coverage_days
        self.training_days = training_days
        self.all_training_days = list(set().union(*self.training_days.values()))
        self.wc_day_sets = wc_day_sets
        self.max_bidlines = max_bidlines
        self.coverage_vectors = coverage_vectors
        self.construct_var_sets()

    def construct_var_sets(self):
        self.X_set = [
            (wc, f"Number{i}", j)
            for wc in self.max_bidlines.keys()
            for i in range(self.max_bidlines[wc][self.shift])
            for j in self.coverage_vectors.keys()
        ]
        self.Y_set = [
            (wc, f"Number{i}", j, k)
            for wc in self.max_bidlines.keys()
            for i in range(self.max_bidlines[wc][self.shift])
            for j in self.coverage_vectors.keys()
            for k in list(set(self.coverage_days[wc]) | set(self.all_training_days))
        ]

    def build_model(self):

        logging.info("Defining model for %s shift", self.shift)

        self.model = xp.problem(name=f"Bidline_Model{self.shift}")

        self.model.controls.outputlog = 0

        self.X = {
            idx: self.model.addVariable(vartype=xp.binary, name=f"X_{idx}")
            for idx in self.X_set
        }
        self.Y = {
            idx: self.model.addVariable(vartype=xp.binary, name=f"Y_{idx}")
            for idx in self.Y_set
        }
        self.F = {
            idx: self.model.addVariable(vartype=xp.binary, name=f"F_{idx}")
            for idx in self.Y_set
        }

        # self.Z = {idx: xp.var(vartype=xp.binary, name=f"Z_{idx}") for idx in self.Z_set}

        # self.model.addVariable(list(self.Z.values()))

        self.cov = [
            xp.Sum(
                self.X[wc, number, offset] * self.coverage_vectors[offset][day]
                - self.Y[wc, number, offset, day] * self.coverage_vectors[offset][day]
                + self.F[wc, number, offset, day]
                * (1 - self.coverage_vectors[offset][day])
                # - self.Z[wc, number, offset, day] * (1 - self.coverage_vectors[offset][day])
                for wc, number, offset in self.X_set
                if wc == w
            )
            >= self.demand[w][day]
            for w in self.demand.keys()
            for day in self.coverage_days[w]
        ]
        self.model.addConstraint(self.cov)

        self.alltrain = [
            xp.Sum(
                self.Y[w, number, offset, day]
                # + self.F[wc, number, offset] * (1 - self.coverage_vectors[offset][day])
                # - self.Z[wc, number, offset] * (1 - self.coverage_vectors[offset][day])
                for w, number, offset in self.X_set
                if w == wc
            )
            <= 12
            for wc in self.training_days.keys()
            for day in self.training_days[wc]
            if wc != "Domestic"
        ]
        self.model.addConstraint(self.alltrain)

        all_domestic_train_days = list(
            {
                item
                for sublist in self.wc_day_sets["Domestic"]["training_days"]
                for item in sublist
            }
        )

        self.domestictrain = [
            xp.Sum(
                self.Y[wc, number, offset, day]
                # + self.F[wc, number, offset] * (1 - self.coverage_vectors[offset][day])
                # - self.Z[wc, number, offset] * (1 - self.coverage_vectors[offset][day])
                for wc, number, offset in self.X_set
            )
            <= 12
            for day in all_domestic_train_days
        ]
        self.model.addConstraint(self.domestictrain)

        self.model.addConstraint(
            xp.Sum(
                self.F[wc, number, offset, day]
                for day in self.coverage_days[wc]
                if day
                in {
                    i: e for i, e in self.coverage_vectors[offset].items() if e == 1
                }.keys()
            )
            == 0
            for (wc, number, offset) in self.X_set
        )

        self.model.addConstraint(
            self.F[wc, number, offset, day] <= self.X[wc, number, offset]
            for (wc, number, offset) in self.X_set
            for day in self.coverage_days[wc]
            if day
            in {i: e for i, e in self.coverage_vectors[offset].items() if e == 0}.keys()
        )

        # self.model.addConstraint(
        #     self.F[wc, number, offset, day] <= (1 - self.Y[wc, number, offset, day])
        #     for (wc, number, offset, day) in self.Y_set
        # )

        self.dom_trainbid_const = {}
        self.special_train = {}
        self.flex_day = {}
        self.no_flex_day = {}

        for w in self.wc_day_sets.keys():
            if w in _FLEX_PARAMS:
                max_flex = _FLEX_PARAMS[w]
                self.flex_day[w] = [
                    xp.Sum(
                        self.F[wc, number, offset, day]
                        for wc, number, offset, day in self.Y_set
                        if wc == w
                        and number == n
                        and offset == off
                        and day
                        in {
                            i: e
                            for i, e in self.coverage_vectors[offset].items()
                            if e == 0
                        }.keys()
                    )
                    <= max_flex * self.X[w_c, n, off]
                    for (w_c, n, off) in self.X_set
                    if w_c == w
                ]
                self.model.addConstraint(self.flex_day[w])
            else:
                self.no_flex_day[w] = [
                    xp.Sum(
                        self.F[wc, number, offset, day]
                        for wc, number, offset, day in self.Y_set
                        if wc == w and number == n and offset == off
                    )
                    == 0
                    for (w_c, n, off) in self.X_set
                    if w_c == w
                ]
                self.model.addConstraint(self.no_flex_day[w])

            if w == "Domestic":
                for i, per in enumerate(self.wc_day_sets[w]["training_days"]):
                    self.dom_trainbid_const[f"Period{i}"] = [
                        xp.Sum(self.Y[wc, number, offset, day] for day in per)
                        == self.X[wc, number, offset]
                        for (wc, number, offset) in self.X_set
                    ]
                    self.model.addConstraint(self.dom_trainbid_const[f"Period{i}"])

                self.model.addConstraint(
                    xp.Sum(
                        self.Y[wc, number, offset, day]
                        for day in self.coverage_days[wc]
                        if day
                        not in list(
                            set(
                                list(
                                    {
                                        item
                                        for sublist in self.wc_day_sets[wc][
                                            "training_days"
                                        ]
                                        for item in sublist
                                    }
                                )
                            )
                            | set(all_domestic_train_days)
                        )
                    )
                    == 0
                    for (wc, number, offset) in self.X_set
                )
                self.model.addConstraint(
                    xp.Sum(
                        self.Y[wc, number, offset, day]
                        for day in self.coverage_days[wc]
                        if day
                        in {
                            i: e
                            for i, e in self.coverage_vectors[offset].items()
                            if e == 0.0
                        }.keys()
                    )
                    == 0
                    for (wc, number, offset) in self.X_set
                    if wc == w
                )
            else:
                if len(self.wc_day_sets[w]["training_days"]) == 1:
                    self.special_train[w] = [
                        xp.Sum(
                            self.Y[wc, number, offset, day]
                            for day in self.wc_day_sets[wc]["training_days"][0]
                        )
                        == self.X[wc, number, offset]
                        for (wc, number, offset) in self.X_set
                        if wc == w
                    ]
                    self.model.addConstraint(self.special_train[w])
                    self.model.addConstraint(
                        xp.Sum(
                            self.Y[wc, number, offset, day]
                            for day in self.coverage_days[wc]
                            if day
                            not in list(
                                set(self.wc_day_sets[wc]["training_days"][0])
                                | set(all_domestic_train_days)
                            )
                        )
                        == 0
                        for (wc, number, offset) in self.X_set
                        if wc == w
                    )
                else:
                    self.special_train[w] = {}
                    special_train_day = list(
                        {
                            item
                            for sublist in self.wc_day_sets[w]["training_days"]
                            for item in sublist
                        }
                    )
                    for i, per in enumerate(self.wc_day_sets[w]["training_days"]):
                        self.special_train[w][f"Period{i}"] = [
                            xp.Sum(self.Y[wc, number, offset, day] for day in per)
                            == self.X[wc, number, offset]
                            for (wc, number, offset) in self.X_set
                            if wc == w
                        ]
                        self.model.addConstraint(self.special_train[w][f"Period{i}"])
                    self.model.addConstraint(
                        xp.Sum(
                            self.Y[wc, number, offset, day]
                            for day in self.coverage_days[wc]
                            if day
                            not in list(
                                set(special_train_day) | set(all_domestic_train_days)
                            )
                        )
                        == 0
                        for (wc, number, offset) in self.X_set
                        if wc == w
                    )
                self.model.addConstraint(
                    xp.Sum(
                        self.Y[wc, number, offset, day]
                        for day in self.coverage_days[wc]
                        if day
                        in {
                            i: e
                            for i, e in self.coverage_vectors[offset].items()
                            if e == 0
                        }.keys()
                    )
                    == 0
                    for (wc, number, offset) in self.X_set
                    if wc == w
                )

        self.model.setObjective(
            xp.Sum(self.X[wc, number, offset] for (wc, number, offset) in self.X_set),
            sense=xp.minimize,
        )
        logging.info("Model definition for shift %s is completed", self.shift)
        # return {
        #     'solution_df': pd.DataFrame(results),
        #     'obj_val': self.model.attributes.objval,
        #     'mip_status': self.model.attributes.mipstatus
        # }
