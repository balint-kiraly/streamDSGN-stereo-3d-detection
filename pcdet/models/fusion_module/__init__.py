from .indentity_fusion import IndentityFusion
from .dual_flow_fusion import DualFlowFusion
from .dual_flow_fusion_v2 import DualFlowFusionV2
from .timewise_fusion import TimeWiseFusion
from .feature_alignment import FeatureAlignment
from .feature_flow import FeatureFlow
from .feature_flow_cascade import FeatureFlowCascade
from .base_fusion import BaseFusion

__all__ = {
    'IndentityFusion': IndentityFusion,
    'DualFlowFusion': DualFlowFusion,
    'DualFlowFusionV2': DualFlowFusionV2,
    'TimeWiseFusion': TimeWiseFusion,
    'FeatureAlignment': FeatureAlignment,
    'FeatureFlow': FeatureFlow,
    'FeatureFlowCascade': FeatureFlowCascade,
    'BaseFusion': BaseFusion
}