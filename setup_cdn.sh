#!/bin/bash
# ==============================================================
#  比赛离线准备：下载前端 CDN 依赖到本地 vendor 目录
#  运行一次后，Web 前端可完全离线工作
# ==============================================================
set -e
cd "$(dirname "$0")"

VENDOR_DIR="web/static/vendor"
mkdir -p "$VENDOR_DIR"

echo "下载前端依赖到 $VENDOR_DIR/ ..."

# Vue 3
curl -L -o "$VENDOR_DIR/vue.global.prod.js" \
  "https://unpkg.com/vue@3/dist/vue.global.prod.js"
echo "  [OK] Vue 3"

# ECharts 5
curl -L -o "$VENDOR_DIR/echarts.min.js" \
  "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"
echo "  [OK] ECharts 5"

# jsPDF
curl -L -o "$VENDOR_DIR/jspdf.umd.min.js" \
  "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"
echo "  [OK] jsPDF"

# html2canvas
curl -L -o "$VENDOR_DIR/html2canvas.min.js" \
  "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"
echo "  [OK] html2canvas"

# Three.js
curl -L -o "$VENDOR_DIR/three.min.js" \
  "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"
echo "  [OK] Three.js"

echo ""
echo "替换 index.html 中的 CDN 链接为本地路径..."

HTML="web/templates/index.html"
cp "$HTML" "$HTML.bak"

sed -i 's|https://unpkg.com/vue@3/dist/vue.global.prod.js|/static/vendor/vue.global.prod.js|' "$HTML"
sed -i 's|https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js|/static/vendor/echarts.min.js|' "$HTML"
sed -i 's|https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js|/static/vendor/jspdf.umd.min.js|' "$HTML"
sed -i 's|https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js|/static/vendor/html2canvas.min.js|' "$HTML"
sed -i 's|https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js|/static/vendor/three.min.js|' "$HTML"

echo ""
echo "全部完成。Web 前端现在可以完全离线运行。"
echo "原始 HTML 备份在 $HTML.bak"
