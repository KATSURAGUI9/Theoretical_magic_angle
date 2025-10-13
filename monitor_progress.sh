#!/bin/bash

echo "=============================="
echo "监控代码运行进度"
echo "=============================="
echo ""

# 检查进程
echo "检查运行中的 Python 进程："
ps aux | grep -E "Theoretical_magic_angle.*\.py" | grep -v grep

echo ""
echo "=============================="
echo "检查输出文件："
echo "=============================="

# 检查生成的文件
for file in alpha_sample_map.png alpha_sample_map_improved.png \
            alpha_sample_fine_scan.png alpha_sample_fine_scan_improved.png; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        time=$(ls -l "$file" | awk '{print $6, $7, $8}')
        echo "✓ $file ($size, $time)"
    else
        echo "✗ $file (未生成)"
    fi
done

echo ""
echo "=============================="
