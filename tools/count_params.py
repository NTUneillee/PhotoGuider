#!/usr/bin/env python3
"""Count learned scalars in the shipped decision layer (trusted local pickles only)."""
import argparse
from pathlib import Path
import joblib


def count(models):
    totals = [0, 0, 0]
    print(f"{'model':24} {'trees':>8} {'leaf values':>12} {'thresholds':>12}")
    paths = sorted(models.glob('gate_*.pkl')) + sorted(models.glob('ranker_*.pkl'))
    knots = 0
    tree_models = 0
    for path in paths:
        model = joblib.load(path)
        if path.stem == 'ranker_iso':
            knots = len(model.X_thresholds_)
            continue
        booster = model.booster_ if hasattr(model, 'booster_') else model
        trees = booster.dump_model()['tree_info']
        def nodes(node):
            if 'leaf_value' in node:
                return (1, 0)
            left, right = nodes(node['left_child']), nodes(node['right_child'])
            return (left[0] + right[0], left[1] + right[1] + 1)
        counts = [nodes(tree['tree_structure']) for tree in trees]
        row = [len(trees), sum(n[0] for n in counts), sum(n[1] for n in counts)]
        totals = [a+b for a,b in zip(totals, row)]
        tree_models += 1
        print(f'{path.stem:24} {row[0]:8,d} {row[1]:12,d} {row[2]:12,d}')
    print(f"{'TOTAL':24} {totals[0]:8,d} {totals[1]:12,d} {totals[2]:12,d}")
    print(f'{tree_models} LightGBM models + 1 isotonic map = {tree_models+1} learned model artifacts')
    print(f'Isotonic map: {knots} knots; tree learned scalars: {sum(totals[1:]):,}')
    print(f'Leaf values + isotonic output values: {totals[1] + knots:,}')
    print('Excluded: cache-resident DPP quality model; static dpp_meta and aes_ref_stats.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', type=Path, default=Path(__file__).resolve().parents[1]/'inference/models')
    count(ap.parse_args().models)
