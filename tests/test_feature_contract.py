#!/usr/bin/env python3
"""CPU regressions for decoded-view geometry, feature coverage and scratch isolation."""
import ast
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'inference'))
from _lib.feature_contract import validate_boxes, validate_rows


def rejects(fn, error=ValueError):
    try:
        fn()
    except error:
        return
    raise AssertionError('invalid input was accepted')


def main():
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        view = tmp / 'view.png'
        Image.new('RGB', (1024, 1024), 'white').save(view)
        # Execute the actual panorama phase with CPU stubs for the heavy feature stages.
        tree = ast.parse((ROOT / 'PanoCropping/scripts/score_views.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'phase_deep')
        def extract(frame, *args, **kwargs):
            validate_boxes(frame)
            return frame
        ns = dict(pd=pd, map_triple=lambda m: (None, None, None),
                  M02=SimpleNamespace(stage_c_features=lambda f, maps: pd.DataFrame(index=f.index)),
                  INFER=SimpleNamespace(Extractors=lambda tmp, runner: SimpleNamespace(deep=extract),
                                        with_pi=lambda f: f))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<phase_deep>', 'exec'), ns)
        args = ([dict(vid=0, path=str(view))], {0: {'mask': True}})
        frame = ns['phase_deep'](*args, tmp)
        assert frame[['x2', 'y2', 'img_w', 'img_h']].iloc[0].tolist() == [1024.] * 4
        bad = frame.assign(x2=8192., y2=4096., img_w=8192., img_h=4096.)
        rejects(lambda: validate_boxes(bad), AssertionError)
        rejects(lambda: validate_boxes(frame.assign(x2=1025.)), AssertionError)
        print('unequal panorama/view dimensions and bounds: PASS')

        expected = pd.DataFrame({'photo_id': [1, 1], 'dpp_rank': [0, 1]})
        good = expected.assign(feature=[1., 2.])
        validate_rows(expected, good.iloc[::-1], 'test')
        for bad in (good.iloc[:1], pd.concat([good, good.iloc[:1]]),
                    good.assign(feature=[np.nan, 2.]), good.assign(feature=[1., np.inf]),
                    good.assign(dpp_rank=[0, 2])):
            rejects(lambda: validate_rows(expected, bad, 'test'))
        print('missing/duplicate/unexpected/non-finite feature rows: PASS')

        # Exercise the actual parent merge (feature_extractors.Extractors.deep) through its
        # interface, with a fake runner that writes malicious/partial worker output.
        sys.path.insert(0, str(ROOT / 'inference/_lib'))
        from feature_extractors import Extractors, DEEP_OUTPUTS
        outputs = [f for fs in DEEP_OUTPUTS.values() for f in fs]
        fault = [None]
        def runner(op, args):
            for name in outputs:
                col = name.replace('feat_', '')
                result = frame[['photo_id', 'dpp_rank']].assign(**{col: 1.})
                if name == 'feat_clip' and fault[0] == 'missing':
                    result = result.iloc[:0]
                if name == 'feat_clip' and fault[0] == 'duplicate':
                    result = pd.concat([result, result])
                if name == 'feat_clip' and fault[0] == 'nonfinite':
                    result[col] = np.inf
                result.to_parquet(Path(args['scratch']) / f'{name}.parquet')
            return True
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            ex = Extractors(tmp, runner=runner)
            assert len(ex.deep(frame)) == 1
            for fault[0] in ('missing', 'duplicate', 'nonfinite'):
                rejects(lambda: ex.deep(frame))
            # a runner that reports failure must name the extractor, not surface a KeyError later
            rejects(lambda: Extractors(tmp, runner=lambda op, args: False).deep(frame), RuntimeError)
        print('parent rejects incomplete/duplicate/non-finite worker output: PASS')

        sys.path.insert(0, str(ROOT / 'inference/_lib/extractors'))
        from job_contract import extraction_job
        @extraction_job({'CANDS': 'cands.parquet', 'OUT': 'feat_test.parquet'})
        def fake(args=None):
            f = pd.read_parquet(CANDS)
            f[['photo_id', 'dpp_rank']].assign(value=7.).to_parquet(OUT)
        for name in ('first', 'second'):
            scratch = tmp / name
            scratch.mkdir()
            frame.to_parquet(scratch / 'cands.parquet')
            fake({'scratch': str(scratch)})
            assert (scratch / 'feat_test.parquet').is_file()
        assert OUT.parent == tmp / 'second'
        rejects(lambda: fake(), RuntimeError)
        print('persistent extractor scratch rebinding and required path: PASS')
    _test_metadata_columns_allowed()


def _test_metadata_columns_allowed():
    """Extractors echo metadata columns (image_path, src) next to their features; only numeric
    feature columns are checked, key-only frames are pure coverage checks."""
    exp = pd.DataFrame(dict(photo_id=[1, 1], dpp_rank=[0, 1]))
    validate_rows(exp, exp.copy(), "candidate input")
    ok = pd.DataFrame(dict(photo_id=[1, 1], dpp_rank=[0, 1], image_path=["a", "a"], src=["C", "C"], aes=[0.1, 0.2]))
    validate_rows(exp, ok, "laion")
    for bad in (ok.assign(aes=[np.nan, 0.2]), ok.drop(columns=["aes"])):
        try:
            validate_rows(exp, bad, "laion"); raise AssertionError("should have been rejected")
        except ValueError:
            pass
    print("metadata columns allowed, non-finite/no-numeric rejected: PASS")



if __name__ == '__main__':
    main()
