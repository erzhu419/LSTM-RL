#!/bin/bash
# 快速测试内存修复是否有效

echo "=================================="
echo "测试内存泄漏修复"
echo "=================================="
echo ""

LOG_FILE="test_memory_fix.log"

echo "运行10个episodes测试..."
echo "监控内存使用情况..."
echo ""

python sac_ensemble_original.py \
  --weight_reg=0.03 \
  --maximum_alpha=0.3 \
  --critic_actor_ratio=2 \
  --replay_buffer_size=100000 \
  --max_episodes=10 \
  --train \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "进程PID: $PID"
echo "日志文件: $LOG_FILE"
echo ""

# 监控内存使用
for i in {1..30}; do
    if ps -p $PID > /dev/null; then
        MEM=$(ps -o rss= -p $PID | awk '{print $1/1024}')
        echo "[$(date '+%H:%M:%S')] Episode进度: 检查中... | 内存: ${MEM}MB"

        # 从日志中提取最新的episode信息
        if [ -f "$LOG_FILE" ]; then
            LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null | grep "Episode:")
            if [ ! -z "$LAST_LINE" ]; then
                echo "  最新: $LAST_LINE"
            fi
        fi
    else
        echo ""
        echo "进程已结束"
        break
    fi
    sleep 10
done

echo ""
echo "=================================="
echo "测试完成！查看日志："
echo "  tail -50 $LOG_FILE"
echo "=================================="
