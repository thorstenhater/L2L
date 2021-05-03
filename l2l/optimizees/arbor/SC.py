#!/usr/bin/env python3
from dataclasses import dataclass
from collections import defaultdict
import arbor as arb
from l2l.optimizees.optimizee import Optimizee
from random import randrange as rand
import pandas as pd
import numpy as np

@dataclass
class PhyPar:
    cm:    float = None
    tempK: float = None
    Vm:    float = None
    rL:    float = None


def load_allen_fit(fit):
    from collections import defaultdict
    import json

    with open(fit) as fd:
        fit = json.load(fd)

    param = defaultdict(PhyPar)
    mechs = defaultdict(dict)
    for block in fit['genome']:
        mech = block['mechanism'] or 'pas'
        region = block['section']
        name = block['name']
        value = float(block['value'])
        if name.endswith('_' + mech):
            name = name[:-(len(mech) + 1)]
        else:
            if mech == "pas":
                # transform names and values
                if name == 'cm':
                    param[region].cm = value/100.0
                elif name == 'Ra':
                    param[region].rL = value
                elif name == 'Vm':
                    param[region].Vm = value
                elif name == 'celsius':
                    param[region].tempK = value + 273.15
                else:
                    raise Exception(f"Unknown key: {name}")
                continue
            else:
                raise Exception(f"Illegal combination {mech} {name}")
        if mech == 'pas':
            mech = 'pas'
        mechs[(region, mech)][name] = value

    param = [(r, vs) for r, vs in param.items()]
    mechs = [(r, m, vs) for (r, m), vs in mechs.items()]

    default = PhyPar(None,  # not set in example file
                     float(fit['conditions'][0]['celsius']) + 273.15,
                     float(fit['conditions'][0]['v_init']),
                     float(fit['passive'][0]['ra']))

    erev = []
    for kv in fit['conditions'][0]['erev']:
        region = kv['section']
        for k, v in kv.items():
            if k == 'section':
                continue
            ion = k[1:]
            erev.append((region, ion, float(v)))

    return default, param, erev, mechs


class ArbSCOptimizee(Optimizee):
    def __init__(self, traj, fit, swc, ref):
        self.fns = (fit, swc, ref)

    def create_individual(self):
        fit, swc, ref = self.fns
        # randomise mechanisms
        self.defaults, self.regions, self.ions, mechs = load_allen_fit(fit)
        result = {}
        for r, m, vs in mechs:
            for k, _ in vs.items():
                key = f"{r}::{m}::{k}"
                result[key] = rand(-1.0, +1.0)
        return result

    def simulate(self, traj):
        fit, swc, ref = self.fns
        self.morphology = arb.load_swc_allen(swc, no_gaps=False)
        self.labels = arb.label_dict({'soma': '(tag 1)', 'axon': '(tag 2)',
                                      'dend': '(tag 3)', 'apic': '(tag 4)',
                                      'center': '(location 0 0.5)'})
        print("Reading ref {}", ref)
        self.reference = pd.read_csv(ref)['U/mV'].values[:-1]*1000.0
        decor = arb.decor()

        decor.discretization(arb.cv_policy_max_extent(20))
        decor.set_property(tempK=self.defaults.tempK, Vm=self.defaults.Vm, cm=self.defaults.cm, rL=self.defaults.rL)
        for region, vs in self.regions:
            decor.paint(f'"{region}"', tempK=vs.tempK, Vm=vs.Vm, cm=vs.cm, rL=vs.rL)
        for region, ion, e in self.ions:
            decor.paint(f'"{region}"', ion, rev_pot=e)
        decor.set_ion('ca', int_con=5e-5, ext_con=2.0, method=arb.mechanism('nernst/x=ca'))


        tmp = defaultdict(dict)
        print(traj)
        print(type(traj.individual))
        for key, val in traj.individual.params.items():
            region, mech, valuename = key.split('.')[-1].split('::')
            tmp[(region, mech)][valuename] = val

        for (region, mech), values in tmp.items():
            decor.paint(f'"{region}"', arb.mechanism(mech, values))


        decor.place('"center"', arb.iclamp(200, 1000, 0.15))
        cell = arb.cable_cell(self.morphology, self.labels, decor)
        model = arb.single_cell_model(cell)
        model.probe('voltage', '"center"', frequency=200000)
        model.catalogue = arb.allen_catalogue()
        model.catalogue.extend(arb.default_catalogue(), '')
        model.run(tfinal=1400, dt=0.005)
        voltages = np.array(model.traces[0].value[:])
        return (((voltages - self.reference)**2).sum(), )
