'''
Meta stats for the datasets
'''

import pandas as pd

GLySAC_Organ2FileID = {
    'Normal': {
        'train': [
            'DB-0001_normal_2',
            'DB-0003_normal_1',
            'DB-0466_normal_1',
            'DB-0466_normal_2',
            'DB-0466_normal_3',
            'EGC1_new_normal_1',
            'EGC1_new_normal_3',
        ],
        'test': [
            'DB-0001_normal_1',
            'DB-0037_normal_1',
            'EGC1_new_normal_2',
            'EGC1_new_normal_5'
        ],
    },
    'Tumor': {
        'train': [
            'AGC1_tumor_1',
            'AGC1_tumor_3',
            'AGC1_tumor_5',
            'AGC1_tumor_7',
            'AGC1_tumor_8',
            'AGC1_tumor_9',
            'AGC1_tumor_10',
            'DB-0001_tumor_1',
            'DB-0001_tumor_3',
            'DB-0003_tumor_1',
            'DB-0003_tumor_2',
            'DB-0446_tumor_2',
            'DB-0446_tumor_3'
        ],
        'test': [
            'AGC1_tumor_2',
            'AGC1_tumor_4',
            'AGC1_tumor_11',
            'DB-0001_tumor_2',
            'DB-0466_tumor_1',
            'EGC1_new_tumor_1',
            'EGC1_new_tumor_2',
            'EGC1_new_tumor_3',
            'EGC1_new_tumor_4',
            'EGC1_new_tumor_5',
            'EGC1_new_tumor_6',
            'EGC1_new_tumor_7',
            'EGC1_new_tumor_10',
            'EGC1_new_tumor_11',
        ]
    }
}

MoNuSeg_Organ2FileID = {
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
    'Prostate': {
        'train': [
            'TCGA-G9-6336-01Z-00-DX1',
            'TCGA-G9-6348-01Z-00-DX1',
            'TCGA-G9-6356-01Z-00-DX1',
            'TCGA-G9-6358-01Z-00-DX1',
            'TCGA-G9-6363-01Z-00-DX1',
            'TCGA-CH-5767-01Z-00-DX1',
            'TCGA-G9-6362-01Z-00-DX1',
        ],
        'test': [
            'TCGA-EJ-A46H-01A-03-TSC',
            'TCGA-HC-7209-01A-01-TS1',
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