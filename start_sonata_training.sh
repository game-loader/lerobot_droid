#!/usr/bin/env bash
printf '%s\n' 'Sonata training is disabled: the inherited encoder is an MLP placeholder. See docs/EXPERIMENTAL_POINT_ENCODERS.md.' >&2
exit 1
set -Eeuo pipefail

echo "=========================================="
echo "  开始 Sonata 训练 (50k steps)"
echo "=========================================="
echo ""

SONATA_WEIGHTS="/home/droid/project/lerobot_droid/pretrained_weights/sonata.pth"

# 验证权重文件
if [ ! -f "$SONATA_WEIGHTS" ]; then
    echo "❌ 错误: Sonata 权重文件不存在"
    echo "   期望位置: $SONATA_WEIGHTS"
    exit 1
fi

SIZE=$(ls -lh "$SONATA_WEIGHTS" | awk '{print $5}')
echo "✅ Sonata 权重文件已就绪"
echo "   位置: $SONATA_WEIGHTS"
echo "   大小: $SIZE"
echo ""

echo "🚀 启动训练..."
echo "   编码器: Sonata (108.5M params, CVPR 2025 Highlight)"
echo "   训练步数: 50,000"
echo "   批次大小: 8"
echo ""

# 使用训练脚本
python scripts/train_dp3_compare.py \
    --encoder sonata \
    --sonata-weights "$SONATA_WEIGHTS" \
    --steps 50000 \
    --batch-size 8
