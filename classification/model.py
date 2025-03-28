import torch
import numpy as np
from protogcn.models import build_model
from ntu_classes import classes


class BaseModel:
    def __init__(
        self, checkpoint="checkpoints/ntu120_xsub/k_1/best_top1_acc_epoch_150.pth", num_classes=120, device="cpu"
    ):
        import os
        print(os.getcwd())
        graph = "nturgb+d"
        model = dict(
            type="RecognizerGCN",
            backbone=dict(
                type="ProtoGCN",
                num_prototype=100,
                tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"],
                graph_cfg=dict(layout=graph, mode="random", num_filter=8, init_off=0.04, init_std=0.02),
            ),
            cls_head=dict(
                type="SimpleHead", joint_cfg="nturgb+d", num_classes=num_classes, in_channels=384, weight=0.1
            ),
        )

        self.model = build_model(model)
        loaded = torch.load(checkpoint, weights_only=False, map_location=device)
        self.model.load_state_dict(loaded["state_dict"])
        self.model.to(device)
        self.model.eval()

    def run(self, data):
        results = self.model(return_loss=False, **data)
        ind = np.argmax(results[0])
        result = classes[ind]
        return ind, result
