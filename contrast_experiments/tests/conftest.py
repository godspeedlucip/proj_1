import pytest
from stent_repro.configuration import resolve_config


@pytest.fixture
def workflow_config():
    def make(method='A'):
        return resolve_config({'method':method,'profile':f'{method}-adapted-3view','seed':17,
            'geometry':{'sdd_mm':200,'sid_mm':100,'pixel_size_mm':2,'detector_shape':[16,16],
                        'angles_deg':[0,30,60],'grid':{'shape':[16]*3,'spacing_mm':[1]*3,'origin_mm':[-7.5]*3}},
            'network':{'architecture':'compact-g0'},
            'training':{s:{'epochs':1,'batch_size':2,'train_pairs':4,'validation_pairs':2,
                          'patch_shape':[16]* (3 if method=='B' and s=='dtr' else 2)} for s in ('dte','dtr')},
            'provenance':{'purpose':'g0','dataset_version':'unit-test'}})
    return make
