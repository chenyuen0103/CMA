model_attributes = {
    'bert': {
        'feature_type': 'text'
    },
    'inception_v3': {
        'feature_type': 'image',
        'target_resolution': (299, 299),
        'flatten': False
    },
    'wideresnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'resnet34': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': False
    },
    'raw_logistic_regression': {
        'feature_type': 'image',
        'target_resolution': None,
        'flatten': True,
    },
    'clip': {
            'feature_type': 'image',
            'target_resolution': (224, 224),
            'flatten': True,
    },
    'clip512': {
        'feature_type': 'image',
        'target_resolution': (512, 512),
        'flatten': True,
    },
    'vits': {
        'feature_type': 'image',
        'target_resolution': (384, 384),
        'flatten': False
    }
}
