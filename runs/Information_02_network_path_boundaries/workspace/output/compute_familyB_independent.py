#!/usr/bin/env python3
"""Independent check of family B (DAG lower boundary) from raw observations only.

Checks:
  1. For action_id 'directed_disconnected_efficiency_lower_boundary', every case:
     |observed_global_efficiency - density| <= 1e-12.
     Reports max_abs_diff and any case beyond tolerance.
  2. Across ALL cases in BOTH actions (family A + family B):
     (E - density) >= -1e-12 universally. Reports any violation.

Reads ONLY data/source_data/computational_observations.json.
Family A cases carry no explicit 'density' field, so density is computed
as n_arcs / (n_nodes * (n_nodes - 1)).
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, '..', 'data', 'source_data',
                   'computational_observations.json')
OUT = os.path.join(BASE, 'evidence_familyB.json')

TOL = 1e-12
FAMILY_B_ID = 'directed_disconnected_efficiency_lower_boundary'


def density_of(c):
    if 'density' in c and c['density'] is not None:
        return c['density']
    n = c['n_nodes']
    L = c['n_arcs']
    return L / (n * (n - 1))


def main():
    with open(SRC) as f:
        obs = json.load(f)
    actions = obs['actions']

    family_b = None
    other_actions = []
    for a in actions:
        if a['action_id'] == FAMILY_B_ID:
            family_b = a
        else:
            other_actions.append(a)

    if family_b is None:
        raise SystemExit("action_id not found in source data")

    b_cases = family_b['cases']
    n_cases = len(b_cases)

    # --- Check 1: family B E == density within 1e-12 ---
    b_max_diff = 0.0
    b_violations = []
    for c in b_cases:
        e = c['observed_global_efficiency']
        rho = density_of(c)
        diff = abs(e - rho)
        if diff > b_max_diff:
            b_max_diff = diff
        if diff > TOL:
            b_violations.append({
                'n_nodes': c['n_nodes'],
                'n_arcs': c['n_arcs'],
                'construction': c.get('construction'),
                'observed_global_efficiency': e,
                'density': rho,
                'abs_diff': diff,
            })

    # --- Check 2: universal bound (E - density) >= -1e-12 for all cases in both actions ---
    all_cases = [c for a in other_actions for c in a['cases']] + b_cases
    lb_violations = []
    for c in all_cases:
        e = c['observed_global_efficiency']
        rho = density_of(c)
        if (e - rho) < -TOL:
            lb_violations.append({
                'n_nodes': c['n_nodes'],
                'n_arcs': c['n_arcs'],
                'construction': c.get('construction'),
                'observed_global_efficiency': e,
                'density': rho,
                'E_minus_density': e - rho,
            })

    result = {
        'source_file': SRC,
        'family_B_action_id': FAMILY_B_ID,
        'n_cases_familyB': n_cases,
        'familyB_E_eq_density': {
            'tolerance': TOL,
            'e_eq_density_all': abs(b_max_diff) <= TOL and not b_violations,
            'max_abs_diff': b_max_diff,
            'violations': b_violations,
        },
        'universal_bound_E_ge_density_minus_1e12': {
            'total_cases_checked': len(all_cases),
            'holds': not lb_violations,
            'violations': lb_violations,
        },
        'familyA_case_count': len(all_cases) - n_cases,
    }

    with open(OUT, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
