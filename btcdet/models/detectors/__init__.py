from .detector3d_template import Detector3DTemplate
from .btcnet import BtcNet

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'BtcNet': BtcNet
}


def build_detector(model_cfg, num_class, dataset, full_config=None):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset, full_config=full_config
    )

    return model
