'''
Meta stats for the datasets
'''

import pandas as pd

Organ2FileID = {
    'Colon': {
        'train': [
            'TCGA-AY-A8YK-01A-01-TS1',
            'TCGA-NH-A8F7-01A-01-TS1'
            ],
        'test': [
            'TCGA-A6-6782-01A-01-BS1'
            ]
    },
    'Breast': {
        'train': [
            'TCGA-A7-A13E-01Z-00-DX1',
            'TCGA-A7-A13F-01Z-00-DX1',
            'TCGA-AR-A1AK-01Z-00-DX1',
            'TCGA-AR-A1AS-01Z-00-DX1',
            'TCGA-E2-A1B5-01Z-00-DX1',
            'TCGA-E2-A14V-01Z-00-DX1'
        ],
        'test': [
            'TCGA-AC-A2FO-01A-01-TS1',
            'TCGA-AO-A0J2-01A-01-BSA'
        ]
    },
    'Kidney': {
        'train': [
            'TCGA-B0-5711-01Z-00-DX1',
            'TCGA-HE-7128-01Z-00-DX1',
            'TCGA-HE-7129-01Z-00-DX1',
            'TCGA-HE-7130-01Z-00-DX1',
            'TCGA-B0-5710-01Z-00-DX1',
            'TCGA-B0-5698-01Z-00-DX1'
        ],
        'test': [
            'TCGA-2Z-A9J9-01A-01-TS1',
            'TCGA-GL-6846-01A-01-BS1',
            'TCGA-IZ-8196-01A-01-BS1'
        ]
    },
}