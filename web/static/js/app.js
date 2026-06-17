/* ═══════════════════════════════════════════════
   FatigueAI Vue 3 App — 全功能版
   实时监测 + 多页分析 + 模型仪表盘 + 3D模型
   ═══════════════════════════════════════════════ */

const { createApp, ref, reactive, computed, onMounted, onUnmounted, watch, nextTick } = Vue;

const app = createApp({
  setup() {
    // ══════════════════════════════════════
    // 状态
    // ══════════════════════════════════════
    const participants = ref([]);
    const selectedPid = ref('01');
    const selectedSid = ref('01');
    const streamActive = ref(false);
    const sessionProgress = ref(0);
    const dataPointCount = ref(0);
    const activeTab = ref('hr');
    const currentSessionId = ref('--');
    const currentActivityLabel = ref('--');
    const cycleCount = ref(0);
    const elapsedTime = ref('00:00:00');
    const sessionDataLength = ref(1500);
    let streamStartTime = null;
    let elapsedTimer = null;

    // 页面导航
    const currentPage = ref('monitor');

    // 主题
    const isDark = ref(true);

    // 声音
    const soundEnabled = ref(false);

    // 告警
    const showAlert = ref(false);
    const alertDismissed = ref(false);
    let lastFatigueLevel = null;

    // SHAP
    const bottomLeftTab = ref('trend');

    // 雷达图对比
    const comparePid = ref('');

    // 数据分析页
    const compareSessionA = ref('01');
    const compareSessionB = ref('02');

    // 过劳预测
    const overtrainPrediction = ref(null);
    let predictionHistory = [];

    // 当前数据
    const currentData = reactive({
      hr: null, hrv: null, eda: null, temp: null,
      br: null,
      fatigue_score: null, fatigue_level: '--', confidence: 0,
    });

    // 聊天
    const chatMessages = ref([
      { role: 'ai', text: '你好！我是 FatigueAI 健康顾问。启动监测后，你可以问我关于疲劳和损伤预防的问题。\n\n例如：\n• 为什么我会疲劳？\n• 怎么恢复？\n• 有什么损伤风险？' }
    ]);
    const chatInput = ref('');
    const chatArea = ref(null);

    // 图表实例
    let realtimeChart = null;
    let fatigueChart = null;
    let radarChart = null;
    let shapChart = null;
    let ecgCanvas = null;
    let ecgCtx = null;
    let ecgAnimId = null;
    let body3dScene = null;
    let compareChartA = null;
    let compareChartB = null;
    let heatmapChart = null;
    let modelCompareChart = null;
    let ablationChart = null;
    let confusionChart = null;
    let trainingCurveChart = null;
    let ws = null;
    let chatWs = null;

    // 数据缓冲区
    const MAX_POINTS = 300;
    const timeData = [];
    const hrData = [];
    const hrvData = [];
    const edaData = [];
    const tempData = [];
    // EEG removed in v4
    const brData = [];
    const fatigueData = [];

    const chartTabs = [
      { key: 'hr', label: '心率' },
      { key: 'hrv', label: 'HRV' },
      { key: 'eda', label: '皮电' },
      { key: 'multi', label: '综合' },
    ];

    // 指标定义
    const metricDefs = [
      {
        key: 'hr', icon: '❤️', label: '心率 (bpm)',
        alert: d => d.hr > 120,
        trendClass: d => d.hr > 120 ? 'up' : 'normal',
        trendText: d => d.hr > 120 ? '↑ 偏高' : '正常',
      },
      {
        key: 'hrv', icon: '💓', label: 'HRV-RMSSD (ms)',
        alert: d => d.hrv < 30,
        trendClass: d => d.hrv < 30 ? 'down' : 'normal',
        trendText: d => d.hrv < 30 ? '↓ 偏低' : '正常',
      },
      {
        key: 'eda', icon: '⚡', label: '皮电活动 (μS)',
        alert: d => d.eda > 0.5,
        trendClass: d => d.eda > 0.5 ? 'up' : 'normal',
        trendText: d => d.eda > 0.5 ? '↑ 偏高' : '正常',
      },
      {
        key: 'temp', icon: '🌡️', label: '皮肤温度 (°C)',
        alert: () => false,
        trendClass: () => 'normal',
        trendText: () => '正常',
      },

      {
        key: 'br', icon: '🫁', label: '呼吸频率 (次/分)',
        alert: d => d.br > 20,
        trendClass: d => d.br > 20 ? 'up' : 'normal',
        trendText: d => d.br > 20 ? '↑ 偏高' : '正常',
      },
    ];

    // 训练历史时间线标记
    const timelineMarks = ref([]);

    // ══════════════════════════════════════
    // 计算属性
    // ══════════════════════════════════════
    const fatigueLevelClass = computed(() => {
      const level = currentData.fatigue_level;
      if (level === '低') return 'low';
      if (level === '中') return 'medium';
      return 'high';
    });
    const fatigueColor = computed(() => {
      const cls = fatigueLevelClass.value;
      if (cls === 'low') return '#10b981';
      if (cls === 'medium') return '#f59e0b';
      return '#ef4444';
    });
    const fatigueDasharray = computed(() => {
      const score = currentData.fatigue_score || 0;
      const circ = 2 * Math.PI * 52;
      return `${(score / 100) * circ} ${circ}`;
    });

    // ══════════════════════════════════════
    // 页面导航
    // ══════════════════════════════════════
    function switchPage(page) {
      currentPage.value = page;
      nextTick(() => {
        if (page === 'monitor') {
          realtimeChart?.resize();
          fatigueChart?.resize();
          radarChart?.resize();
          shapChart?.resize();
        } else if (page === 'analysis') {
          loadComparison();
          loadHeatmap();
        } else if (page === 'models') {
          loadModelData();
        }
      });
    }

    // ══════════════════════════════════════
    // 主题切换
    // ══════════════════════════════════════
    function toggleTheme() {
      isDark.value = !isDark.value;
      document.documentElement.setAttribute('data-theme', isDark.value ? 'dark' : 'light');
      // 重新初始化图表（主题变化需要重建）
      nextTick(() => {
        disposeCharts();
        initCharts();
      });
    }

    function disposeCharts() {
      realtimeChart?.dispose(); realtimeChart = null;
      fatigueChart?.dispose(); fatigueChart = null;
      radarChart?.dispose(); radarChart = null;
      shapChart?.dispose(); shapChart = null;
      modelCompareChart?.dispose(); modelCompareChart = null;
      ablationChart?.dispose(); ablationChart = null;
      confusionChart?.dispose(); confusionChart = null;
      trainingCurveChart?.dispose(); trainingCurveChart = null;
      compareChartA?.dispose(); compareChartA = null;
      compareChartB?.dispose(); compareChartB = null;
      heatmapChart?.dispose(); heatmapChart = null;
    }

    // ══════════════════════════════════════
    // 声音告警
    // ══════════════════════════════════════
    function playAlertSound() {
      if (!soundEnabled.value) return;
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = 880;
        osc.type = 'sine';
        gain.gain.setValueAtTime(0.3, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.5);
      } catch (e) {}
    }

    // ══════════════════════════════════════
    // 告警逻辑
    // ══════════════════════════════════════
    function dismissAlert() {
      alertDismissed.value = true;
      showAlert.value = false;
    }
    function checkFatigueAlert(level) {
      if (level === '高' && lastFatigueLevel !== '高' && !alertDismissed.value) {
        showAlert.value = true;
        playAlertSound();
        const card = document.querySelector('.fatigue-card');
        if (card) { card.classList.add('shake'); setTimeout(() => card.classList.remove('shake'), 600); }
      }
      if (level !== '高' && lastFatigueLevel === '高') {
        showAlert.value = false;
        alertDismissed.value = false;
      }
      lastFatigueLevel = level;
    }

    // ══════════════════════════════════════
    // 过劳预测
    // ══════════════════════════════════════
    function updateOvertrainPrediction(score) {
      if (!score) return;
      predictionHistory.push(score);
      if (predictionHistory.length > 60) predictionHistory.shift();
      if (predictionHistory.length < 20 || score < 40) { overtrainPrediction.value = null; return; }
      // 线性回归预测
      const n = predictionHistory.length;
      let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
      for (let i = 0; i < n; i++) {
        sumX += i; sumY += predictionHistory[i];
        sumXY += i * predictionHistory[i]; sumX2 += i * i;
      }
      const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
      if (slope <= 0) { overtrainPrediction.value = null; return; }
      const stepsToHigh = (66 - score) / slope;
      const minutesLeft = Math.max(1, Math.round(stepsToHigh * 0.15 / 60));
      overtrainPrediction.value = minutesLeft > 120 ? null : minutesLeft;
    }

    // ══════════════════════════════════════
    // 计时器
    // ══════════════════════════════════════
    function startElapsedTimer() {
      streamStartTime = Date.now();
      if (elapsedTimer) clearInterval(elapsedTimer);
      elapsedTimer = setInterval(() => {
        const sec = Math.floor((Date.now() - streamStartTime) / 1000);
        const h = String(Math.floor(sec / 3600)).padStart(2, '0');
        const m = String(Math.floor((sec % 3600) / 60)).padStart(2, '0');
        const s = String(sec % 60).padStart(2, '0');
        elapsedTime.value = `${h}:${m}:${s}`;
      }, 1000);
    }

    // ══════════════════════════════════════
    // 初始化图表
    // ══════════════════════════════════════
    function initCharts() {
      const theme = isDark.value ? 'dark' : undefined;
      realtimeChart = echarts.init(document.getElementById('realtimeChart'), theme);
      updateRealtimeChart();
      radarChart = echarts.init(document.getElementById('radarChart'), theme);
      updateRadarChart();
      shapChart = echarts.init(document.getElementById('shapChart'), theme);
      updateShapChart([]);
      fatigueChart = echarts.init(document.getElementById('fatigueChart'), theme);
      fatigueChart.setOption({
        backgroundColor: 'transparent',
        grid: { top: 20, right: 20, bottom: 30, left: 50 },
        xAxis: { type: 'category', data: [], axisLine: { lineStyle: { color: '#2a3548' } }, axisLabel: { color: '#64748b', fontSize: 10 } },
        yAxis: { type: 'value', min: 0, max: 100, splitLine: { lineStyle: { color: '#1f2b3d' } }, axisLabel: { color: '#64748b', fontSize: 10 } },
        series: [{
          type: 'line', data: [], smooth: true, symbol: 'none',
          lineStyle: { width: 2, color: '#3b82f6' },
          areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'rgba(59,130,246,0.3)'},{offset:1,color:'rgba(59,130,246,0)'}]) },
          markLine: {
            silent: true,
            data: [
              { yAxis: 33, lineStyle: { color: '#10b981', type: 'dashed' }, label: { formatter: '低/中', color: '#10b981', fontSize: 10 } },
              { yAxis: 66, lineStyle: { color: '#ef4444', type: 'dashed' }, label: { formatter: '中/高', color: '#ef4444', fontSize: 10 } },
            ]
          }
        }],
        tooltip: { trigger: 'axis', backgroundColor: '#1a2332', borderColor: '#2a3548' },
      });
      window.addEventListener('resize', () => {
        realtimeChart?.resize(); fatigueChart?.resize(); radarChart?.resize(); shapChart?.resize();
      });
    }

    // ══════════════════════════════════════
    // 实时图表
    // ══════════════════════════════════════
    function updateRealtimeChart() {
      if (!realtimeChart) return;
      const configs = {
        hr: [{ name: '心率 (bpm)', data: hrData, type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 2, color: '#ef4444' }, areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'rgba(239,68,68,0.2)'},{offset:1,color:'rgba(239,68,68,0)'}]) } }],
        hrv: [{ name: 'HRV-RMSSD (ms)', data: hrvData, type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 2, color: '#8b5cf6' }, areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'rgba(139,92,246,0.2)'},{offset:1,color:'rgba(139,92,246,0)'}]) } }],
        eda: [{ name: '皮电活动 (μS)', data: edaData, type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 2, color: '#f59e0b' }, areaStyle: { color: new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'rgba(245,158,11,0.2)'},{offset:1,color:'rgba(245,158,11,0)'}]) } }],
        multi: [
          { name: '心率', data: hrData, type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 1.5, color: '#ef4444' } },
          { name: 'HRV', data: hrvData, type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 1.5, color: '#8b5cf6' } },
          { name: '皮电×100', data: edaData.map(v => v * 100), type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 1.5, color: '#f59e0b' } },
        ],
      };
      realtimeChart.setOption({
        backgroundColor: 'transparent',
        grid: { top: 30, right: 20, bottom: 30, left: 60 },
        legend: { show: activeTab.value === 'multi', top: 0, textStyle: { color: '#94a3b8', fontSize: 11 } },
        xAxis: { type: 'category', data: [...timeData], axisLine: { lineStyle: { color: '#2a3548' } }, axisLabel: { color: '#64748b', fontSize: 10, formatter: v => (v / 60).toFixed(1) + 'm' } },
        yAxis: { type: 'value', splitLine: { lineStyle: { color: '#1f2b3d' } }, axisLabel: { color: '#64748b', fontSize: 10 } },
        series: configs[activeTab.value] || configs.hr,
        tooltip: { trigger: 'axis', backgroundColor: '#1a2332', borderColor: '#2a3548', textStyle: { color: '#e2e8f0' } },
        animation: false,
      }, true);
    }

    function updateFatigueChart() {
      if (!fatigueChart) return;
      fatigueChart.setOption({ xAxis: { data: [...timeData] }, series: [{ data: [...fatigueData] }] });
    }

    // ══════════════════════════════════════
    // 雷达图（支持双人对比）
    // ══════════════════════════════════════
    function updateRadarChart() {
      if (!radarChart) return;
      const hrN = Math.min(100, Math.max(0, ((currentData.hr || 70) - 60) / 100 * 100));
      const hrvN = Math.min(100, Math.max(0, (1 - (currentData.hrv || 50) / 60) * 100));
      const edaN = Math.min(100, Math.max(0, ((currentData.eda || 0.1) - 0.05) / 1.0 * 100));
      const brN = Math.min(100, Math.max(0, ((currentData.br || 12) - 8) / 25 * 100));
      const tempN = Math.min(100, Math.max(0, ((currentData.temp || 36.0) - 35.0) / 3.0 * 100));
      const cls = fatigueLevelClass.value;
      const areaColor = cls === 'low' ? 'rgba(16,185,129,0.25)' : cls === 'medium' ? 'rgba(245,158,11,0.25)' : 'rgba(239,68,68,0.25)';
      const lineColor = cls === 'low' ? '#10b981' : cls === 'medium' ? '#f59e0b' : '#ef4444';

      const seriesData = [{
        value: [hrN.toFixed(0), hrvN.toFixed(0), edaN.toFixed(0), brN.toFixed(0), tempN.toFixed(0)],
        name: '当前状态', symbol: 'circle', symbolSize: 5,
        lineStyle: { width: 2, color: lineColor }, areaStyle: { color: areaColor }, itemStyle: { color: lineColor },
      }];

      // 如果选择了对比参与者，添加第二条线
      if (comparePid.value) {
        seriesData.push({
          value: [
            (Math.random() * 60 + 20).toFixed(0),
            (Math.random() * 60 + 20).toFixed(0),
            (Math.random() * 60 + 20).toFixed(0),
            (Math.random() * 60 + 20).toFixed(0),
            (Math.random() * 60 + 20).toFixed(0),
            (Math.random() * 60 + 20).toFixed(0),
          ],
          name: 'P' + comparePid.value, symbol: 'diamond', symbolSize: 5,
          lineStyle: { width: 2, color: '#3b82f6', type: 'dashed' },
          areaStyle: { color: 'rgba(59,130,246,0.1)' }, itemStyle: { color: '#3b82f6' },
        });
      }

      radarChart.setOption({
        backgroundColor: 'transparent',
        legend: { show: comparePid.value ? true : false, bottom: 0, textStyle: { color: '#94a3b8', fontSize: 10 } },
        radar: {
          indicator: [
            { name: '心率', max: 100 }, { name: 'HRV↓', max: 100 }, { name: '皮电', max: 100 },
            { name: '呼吸', max: 100 }, { name: '体温', max: 100 },
          ],
          shape: 'polygon', radius: '60%', center: ['50%', '50%'],
          axisName: { color: '#94a3b8', fontSize: 10 },
          splitArea: { areaStyle: { color: ['rgba(255,255,255,0.02)', 'rgba(255,255,255,0.04)'] } },
          splitLine: { lineStyle: { color: '#2a3548' } }, axisLine: { lineStyle: { color: '#2a3548' } },
        },
        series: [{ type: 'radar', data: seriesData }],
        tooltip: { trigger: 'item', backgroundColor: '#1a2332', borderColor: '#2a3548', textStyle: { color: '#e2e8f0', fontSize: 11 } },
        animation: true, animationDuration: 300,
      });
    }

    // ══════════════════════════════════════
    // SHAP 特征重要性
    // ══════════════════════════════════════
    function switchBottomLeft(tab) {
      bottomLeftTab.value = tab;
      if (tab === 'shap') { fetchFeatureImportance(); nextTick(() => shapChart?.resize()); }
      else { nextTick(() => fatigueChart?.resize()); }
    }
    async function fetchFeatureImportance() {
      try {
        const resp = await fetch(`/api/feature-importance/${selectedPid.value}/${selectedSid.value}`);
        const data = await resp.json();
        updateShapChart(data.features || []);
      } catch (e) { console.error('获取特征重要性失败:', e); }
    }
    function updateShapChart(features) {
      if (!shapChart) return;
      const names = features.map(f => f.name).reverse();
      const values = features.map(f => f.importance).reverse();
      const colors = features.map(f => f.direction === 'up' ? '#ef4444' : f.direction === 'down' ? '#f59e0b' : '#10b981').reverse();
      shapChart.setOption({
        backgroundColor: 'transparent',
        grid: { top: 10, right: 50, bottom: 20, left: 120 },
        xAxis: { type: 'value', splitLine: { lineStyle: { color: '#1f2b3d' } }, axisLabel: { color: '#64748b', fontSize: 10 } },
        yAxis: { type: 'category', data: names, axisLabel: { color: '#94a3b8', fontSize: 11 }, axisLine: { lineStyle: { color: '#2a3548' } } },
        series: [{
          type: 'bar', barWidth: 14,
          data: values.map((v, i) => ({ value: v, itemStyle: { color: new echarts.graphic.LinearGradient(0,0,1,0,[{offset:0,color:colors[i]+'33'},{offset:1,color:colors[i]}]), borderRadius: [0,4,4,0] } })),
          label: { show: true, position: 'right', color: '#94a3b8', fontSize: 10, formatter: p => p.value.toFixed(3) },
        }],
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, backgroundColor: '#1a2332', borderColor: '#2a3548', textStyle: { color: '#e2e8f0', fontSize: 11 } },
        animation: true, animationDuration: 600,
      }, true);
    }

    // ══════════════════════════════════════
    // ECG 波形动画
    // ══════════════════════════════════════
    function initECG() {
      ecgCanvas = document.getElementById('ecgCanvas');
      if (!ecgCanvas) return;
      ecgCtx = ecgCanvas.getContext('2d');
      resizeECG();
      window.addEventListener('resize', resizeECG);
      animateECG();
    }
    function resizeECG() {
      if (!ecgCanvas) return;
      const rect = ecgCanvas.parentElement.getBoundingClientRect();
      ecgCanvas.width = rect.width - 24;
      ecgCanvas.height = rect.height - 12;
    }
    let ecgX = 0;
    function animateECG() {
      if (!ecgCtx || !ecgCanvas) return;
      const w = ecgCanvas.width, h = ecgCanvas.height;
      ecgCtx.fillStyle = isDark.value ? '#0a0e17' : '#f1f5f9';
      ecgCtx.fillRect(0, 0, w, h);
      // 网格线
      ecgCtx.strokeStyle = isDark.value ? '#1f2b3d' : '#e2e8f0';
      ecgCtx.lineWidth = 0.5;
      for (let y = 0; y < h; y += 15) { ecgCtx.beginPath(); ecgCtx.moveTo(0, y); ecgCtx.lineTo(w, y); ecgCtx.stroke(); }
      for (let x = 0; x < w; x += 15) { ecgCtx.beginPath(); ecgCtx.moveTo(x, 0); ecgCtx.lineTo(x, h); ecgCtx.stroke(); }
      // ECG 波形
      ecgCtx.beginPath();
      ecgCtx.strokeStyle = '#10b981';
      ecgCtx.lineWidth = 2;
      ecgCtx.shadowColor = '#10b981';
      ecgCtx.shadowBlur = 4;
      const mid = h / 2;
      const hr = currentData.hr || 72;
      const period = w / (hr / 60 * 2);
      for (let x = 0; x < w; x++) {
        const phase = ((x + ecgX) % period) / period;
        let y;
        if (phase < 0.1) y = mid;
        else if (phase < 0.15) y = mid - h * 0.08;
        else if (phase < 0.2) y = mid + h * 0.4;
        else if (phase < 0.25) y = mid - h * 0.15;
        else if (phase < 0.35) y = mid;
        else if (phase < 0.4) y = mid - h * 0.1;
        else if (phase < 0.45) y = mid;
        else y = mid;
        y += Math.sin(x * 0.02 + ecgX * 0.01) * 1.5;
        if (x === 0) ecgCtx.moveTo(x, y); else ecgCtx.lineTo(x, y);
      }
      ecgCtx.stroke();
      ecgCtx.shadowBlur = 0;
      ecgX += 2;
      ecgAnimId = requestAnimationFrame(animateECG);
    }

    // ══════════════════════════════════════
    // 3D 人体模型
    // ══════════════════════════════════════
    function initBody3D() {
      const container = document.getElementById('body3dCanvas');
      if (!container || typeof THREE === 'undefined') return;
      const rect = container.getBoundingClientRect();
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(45, rect.width / rect.height, 0.1, 100);
      camera.position.set(0, 0, 4);
      const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
      renderer.setSize(rect.width, rect.height);
      renderer.setPixelRatio(window.devicePixelRatio);
      container.appendChild(renderer.domElement);

      // 简单人形：头部 + 身体 + 四肢
      const mat = new THREE.MeshPhongMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.8 });
      const matHighlight = new THREE.MeshPhongMaterial({ color: 0xef4444, emissive: 0xef4444, emissiveIntensity: 0.3 });
      const group = new THREE.Group();
      // 头
      const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 16, 16), mat);
      head.position.y = 1.1; group.add(head);
      // 身体
      const body = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.2, 0.7, 8), mat);
      body.position.y = 0.55; group.add(body);
      // 骨盆
      const pelvis = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.25, 0.2, 8), mat);
      pelvis.position.y = 0.1; group.add(pelvis);
      // 手臂
      [-1, 1].forEach(side => {
        const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.05, 0.55, 6), mat);
        arm.position.set(side * 0.35, 0.55, 0); arm.rotation.z = side * 0.2; group.add(arm);
        const forearm = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.04, 0.5, 6), mat);
        forearm.position.set(side * 0.45, 0.1, 0); forearm.rotation.z = side * 0.1; group.add(forearm);
      });
      // 腿
      [-1, 1].forEach(side => {
        const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.06, 0.6, 6), mat);
        leg.position.set(side * 0.12, -0.35, 0); group.add(leg);
        const shin = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.05, 0.55, 6), mat);
        shin.position.set(side * 0.12, -0.85, 0); group.add(shin);
      });
      scene.add(group);
      // 灯光
      scene.add(new THREE.AmbientLight(0xffffff, 0.6));
      const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
      dirLight.position.set(2, 3, 4); scene.add(dirLight);

      let isDragging = false, prevX = 0;
      container.addEventListener('mousedown', e => { isDragging = true; prevX = e.clientX; });
      container.addEventListener('mousemove', e => { if (isDragging) { group.rotation.y += (e.clientX - prevX) * 0.01; prevX = e.clientX; } });
      container.addEventListener('mouseup', () => isDragging = false);
      container.addEventListener('mouseleave', () => isDragging = false);

      body3dScene = { scene, camera, renderer, group, mat, matHighlight };
      function animate() {
        requestAnimationFrame(animate);
        if (!isDragging) group.rotation.y += 0.005;
        renderer.render(scene, camera);
      }
      animate();
    }

    // 更新3D模型颜色（基于疲劳等级）
    function updateBody3D() {
      if (!body3dScene) return;
      const cls = fatigueLevelClass.value;
      const color = cls === 'low' ? 0x10b981 : cls === 'medium' ? 0xf59e0b : 0xef4444;
      body3dScene.mat.color.setHex(color);
    }

    // ══════════════════════════════════════
    // 多会话对比
    // ══════════════════════════════════════
    function getSessionLabel(sid) {
      const match = participants.value.find(x => x.participant_id === selectedPid.value && x.session_id === sid);
      if (match) {
        const map = { low: '低强度', medium: '中强度', high: '高强度' };
        return map[match.activity_level] || match.activity_level;
      }
      return '--';
    }

    async function loadComparison() {
      const theme = isDark.value ? 'dark' : undefined;
      await nextTick();
      const elA = document.getElementById('compareChartA');
      const elB = document.getElementById('compareChartB');
      if (!elA || !elB) return;
      compareChartA?.dispose(); compareChartA = echarts.init(elA, theme);
      compareChartB?.dispose(); compareChartB = echarts.init(elB, theme);

      // 模拟两个会话的疲劳曲线数据
      const len = 100;
      const labels = Array.from({ length: len }, (_, i) => i);
      function genCurve(base, slope, noise) {
        return labels.map(i => Math.min(100, Math.max(0, base + slope * (i / len) + (Math.random() - 0.5) * noise)));
      }
      const commonOpt = {
        backgroundColor: 'transparent',
        grid: { top: 20, right: 20, bottom: 30, left: 50 },
        xAxis: { type: 'category', data: labels, axisLabel: { color: '#64748b', fontSize: 10 }, axisLine: { lineStyle: { color: '#2a3548' } } },
        yAxis: { type: 'value', min: 0, max: 100, splitLine: { lineStyle: { color: '#1f2b3d' } }, axisLabel: { color: '#64748b', fontSize: 10 } },
        tooltip: { trigger: 'axis', backgroundColor: '#1a2332', borderColor: '#2a3548', textStyle: { color: '#e2e8f0' } },
        series: [{
          type: 'line', smooth: true, symbol: 'none',
          lineStyle: { width: 2 }, areaStyle: { opacity: 0.2 },
          markLine: { silent: true, data: [{ yAxis: 66, lineStyle: { color: '#ef4444', type: 'dashed' }, label: { formatter: '高疲劳', color: '#ef4444', fontSize: 10 } }] },
        }],
      };
      const sida = compareSessionA.value, sidb = compareSessionB.value;
      const levels = { '01': [20, 0.4, 8], '02': [30, 0.5, 10], '03': [40, 0.6, 12] };
      const la = levels[sida] || levels['01'], lb = levels[sidb] || levels['02'];
      compareChartA.setOption({ ...commonOpt, series: [{ ...commonOpt.series[0], data: genCurve(...la), lineStyle: { ...commonOpt.series[0].lineStyle, color: '#10b981' }, areaStyle: { color: 'rgba(16,185,129,0.2)' } }] });
      compareChartB.setOption({ ...commonOpt, series: [{ ...commonOpt.series[0], data: genCurve(...lb), lineStyle: { ...commonOpt.series[0].lineStyle, color: '#ef4444' }, areaStyle: { color: 'rgba(239,68,68,0.2)' } }] });
    }

    // ══════════════════════════════════════
    // 疲劳热力图
    // ══════════════════════════════════════
    function loadHeatmap() {
      const theme = isDark.value ? 'dark' : undefined;
      const el = document.getElementById('heatmapChart');
      if (!el) return;
      heatmapChart?.dispose(); heatmapChart = echarts.init(el, theme);
      const metrics = ['心率', 'HRV', '皮电', '呼吸', '疲劳评分'];
      const times = Array.from({ length: 30 }, (_, i) => `${i}m`);
      const data = [];
      metrics.forEach((_, mi) => { times.forEach((_, ti) => { data.push([ti, mi, Math.round(Math.random() * 100)]); }); });
      heatmapChart.setOption({
        backgroundColor: 'transparent',
        grid: { top: 10, right: 20, bottom: 40, left: 80 },
        xAxis: { type: 'category', data: times, axisLabel: { color: '#64748b', fontSize: 10 }, splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.02)', 'rgba(255,255,255,0.04)'] } } },
        yAxis: { type: 'category', data: metrics, axisLabel: { color: '#94a3b8', fontSize: 11 } },
        visualMap: { min: 0, max: 100, calculable: true, orient: 'horizontal', left: 'center', bottom: 0, inRange: { color: ['#10b981', '#f59e0b', '#ef4444'] }, textStyle: { color: '#64748b' } },
        series: [{ type: 'heatmap', data: data, emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } } }],
        tooltip: { backgroundColor: '#1a2332', borderColor: '#2a3548', textStyle: { color: '#e2e8f0' }, formatter: p => `${metrics[p.value[1]]} @ ${times[p.value[0]]}<br/>值: ${p.value[2]}` },
      });
    }

    // ══════════════════════════════════════
    // 模型性能仪表盘
    // ══════════════════════════════════════
    async function loadModelData() {
      try {
        const resp = await fetch('/api/model-compare');
        const data = await resp.json();
        await nextTick();
        renderModelCompare(data);
        renderAblation(data);
        renderConfusion(data);
        renderTrainingCurve(data);
      } catch (e) { console.error('加载模型数据失败:', e); }
    }

    function renderModelCompare(data) {
      const el = document.getElementById('modelCompareChart');
      if (!el) return;
      const theme = isDark.value ? 'dark' : undefined;
      modelCompareChart?.dispose(); modelCompareChart = echarts.init(el, theme);
      const models = data.models;
      modelCompareChart.setOption({
        backgroundColor: 'transparent',
        grid: { top: 30, right: 20, bottom: 30, left: 50 },
        legend: { data: ['Accuracy', 'F1', 'Precision', 'Recall'], top: 0, textStyle: { color: '#94a3b8', fontSize: 10 } },
        xAxis: { type: 'category', data: models.map(m => m.name), axisLabel: { color: '#94a3b8', fontSize: 11 } },
        yAxis: { type: 'value', min: 0.7, max: 1, splitLine: { lineStyle: { color: '#1f2b3d' } }, axisLabel: { color: '#64748b', fontSize: 10 } },
        series: [
          { name: 'Accuracy', type: 'bar', data: models.map(m => m.accuracy), itemStyle: { color: '#3b82f6', borderRadius: [4,4,0,0] } },
          { name: 'F1', type: 'bar', data: models.map(m => m.f1), itemStyle: { color: '#10b981', borderRadius: [4,4,0,0] } },
          { name: 'Precision', type: 'bar', data: models.map(m => m.precision), itemStyle: { color: '#f59e0b', borderRadius: [4,4,0,0] } },
          { name: 'Recall', type: 'bar', data: models.map(m => m.recall), itemStyle: { color: '#8b5cf6', borderRadius: [4,4,0,0] } },
        ],
        tooltip: { trigger: 'axis', backgroundColor: '#1a2332', borderColor: '#2a3548', textStyle: { color: '#e2e8f0' } },
      });
    }

    function renderAblation(data) {
      const el = document.getElementById('ablationChart');
      if (!el) return;
      const theme = isDark.value ? 'dark' : undefined;
      ablationChart?.dispose(); ablationChart = echarts.init(el, theme);
      const abl = data.ablation;
      ablationChart.setOption({
        backgroundColor: 'transparent',
        grid: { top: 20, right: 40, bottom: 20, left: 120 },
        xAxis: { type: 'value', min: 0.6, max: 1, splitLine: { lineStyle: { color: '#1f2b3d' } }, axisLabel: { color: '#64748b', fontSize: 10 } },
        yAxis: { type: 'category', data: abl.map(a => a.name).reverse(), axisLabel: { color: '#94a3b8', fontSize: 11 } },
        series: [{
          type: 'bar', barWidth: 18,
          data: abl.map(a => a.accuracy).reverse().map((v, i) => ({
            value: v,
            itemStyle: { color: new echarts.graphic.LinearGradient(0,0,1,0,[{offset:0,color:'rgba(6,182,212,0.3)'},{offset:1,color:'#06b6d4'}]), borderRadius: [0,4,4,0] },
          })),
          label: { show: true, position: 'right', color: '#94a3b8', fontSize: 10, formatter: p => (p.value * 100).toFixed(1) + '%' },
        }],
        tooltip: { trigger: 'axis', backgroundColor: '#1a2332', borderColor: '#2a3548', textStyle: { color: '#e2e8f0' } },
      });
    }

    function renderConfusion(data) {
      const el = document.getElementById('confusionChart');
      if (!el) return;
      const theme = isDark.value ? 'dark' : undefined;
      confusionChart?.dispose(); confusionChart = echarts.init(el, theme);
      const cm = data.confusion_matrix;
      const labels = data.confusion_labels || ['低', '中', '高'];
      const cmData = [];
      cm.forEach((row, i) => { row.forEach((val, j) => { cmData.push([j, i, val]); }); });
      confusionChart.setOption({
        backgroundColor: 'transparent',
        grid: { top: 10, right: 80, bottom: 40, left: 60 },
        xAxis: { type: 'category', data: labels, name: '预测', nameLocation: 'center', nameGap: 25, axisLabel: { color: '#94a3b8', fontSize: 12 } },
        yAxis: { type: 'category', data: labels, name: '实际', nameLocation: 'center', nameGap: 40, axisLabel: { color: '#94a3b8', fontSize: 12 } },
        visualMap: { min: 0, max: 40, calculable: false, orient: 'vertical', right: 10, top: 'center', inRange: { color: ['#1a2332', '#3b82f6', '#ef4444'] }, textStyle: { color: '#64748b' } },
        series: [{
          type: 'heatmap', data: cmData, label: { show: true, color: '#e2e8f0', fontSize: 14, fontWeight: 700 },
          emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.5)' } },
        }],
        tooltip: { backgroundColor: '#1a2332', borderColor: '#2a3548', textStyle: { color: '#e2e8f0' }, formatter: p => `实际: ${labels[p.value[1]]}<br/>预测: ${labels[p.value[0]]}<br/>数量: ${p.value[2]}` },
      });
    }

    function renderTrainingCurve(data) {
      const el = document.getElementById('trainingCurveChart');
      if (!el) return;
      const theme = isDark.value ? 'dark' : undefined;
      trainingCurveChart?.dispose(); trainingCurveChart = echarts.init(el, theme);
      const th = data.training_history;
      trainingCurveChart.setOption({
        backgroundColor: 'transparent',
        grid: { top: 30, right: 60, bottom: 30, left: 50 },
        legend: { data: ['训练损失', '验证损失', '训练准确率', '验证准确率'], top: 0, textStyle: { color: '#94a3b8', fontSize: 10 } },
        xAxis: { type: 'category', data: th.epochs.map(e => 'Epoch ' + e), axisLabel: { color: '#64748b', fontSize: 10 } },
        yAxis: [
          { type: 'value', name: 'Loss', min: 0, splitLine: { lineStyle: { color: '#1f2b3d' } }, axisLabel: { color: '#64748b', fontSize: 10 }, nameTextStyle: { color: '#64748b' } },
          { type: 'value', name: 'Accuracy', min: 0, max: 1, splitLine: { show: false }, axisLabel: { color: '#64748b', fontSize: 10 }, nameTextStyle: { color: '#64748b' } },
        ],
        series: [
          { name: '训练损失', type: 'line', data: th.train_loss, smooth: true, symbol: 'circle', symbolSize: 4, lineStyle: { color: '#ef4444', width: 2 }, itemStyle: { color: '#ef4444' } },
          { name: '验证损失', type: 'line', data: th.val_loss, smooth: true, symbol: 'circle', symbolSize: 4, lineStyle: { color: '#f59e0b', width: 2, type: 'dashed' }, itemStyle: { color: '#f59e0b' } },
          { name: '训练准确率', type: 'line', yAxisIndex: 1, data: th.train_acc, smooth: true, symbol: 'circle', symbolSize: 4, lineStyle: { color: '#10b981', width: 2 }, itemStyle: { color: '#10b981' } },
          { name: '验证准确率', type: 'line', yAxisIndex: 1, data: th.val_acc, smooth: true, symbol: 'circle', symbolSize: 4, lineStyle: { color: '#3b82f6', width: 2, type: 'dashed' }, itemStyle: { color: '#3b82f6' } },
        ],
        tooltip: { trigger: 'axis', backgroundColor: '#1a2332', borderColor: '#2a3548', textStyle: { color: '#e2e8f0' } },
      });
    }

    // ══════════════════════════════════════
    // 数据导出
    // ══════════════════════════════════════
    function exportCSV() {
      if (timeData.length === 0) { alert('暂无数据可导出，请先启动监测。'); return; }
      var csv = 'Time,HR,HRV,EDA,Temp,BR,Fatigue_Score\n';
      for (var i = 0; i < timeData.length; i++) {
        var time = timeData[i] || '';
        var hr = hrData[i] != null ? hrData[i] : '';
        var hrv = hrvData[i] != null ? hrvData[i] : '';
        var eda = edaData[i] != null ? edaData[i] : '';
        var temp = tempData[i] != null ? tempData[i] : '';
        var br = brData[i] != null ? brData[i] : '';
        var score = fatigueData[i] != null ? fatigueData[i] : '';
        csv += time + ',' + hr + ',' + hrv + ',' + eda + ',' + temp + ',' + br + ',' + score + '\n';
      }
      const bom = new Uint8Array([0xEF, 0xBB, 0xBF]);
      const blob = new Blob([bom, csv], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `fatigue_data_P${selectedPid.value}_S${selectedSid.value}_${new Date().toISOString().slice(0,10)}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    }

function exportPDF() {
      const { jsPDF } = window.jspdf;
      const pdf = new jsPDF('p', 'mm', 'a4');
      pdf.setFontSize(18);
      pdf.text('FatigueAI Analysis Report', 20, 20);
      pdf.setFontSize(11);
      pdf.text(`Participant: ${selectedPid.value}  |  Session: ${selectedSid.value}  |  Date: ${new Date().toLocaleString()}`, 20, 30);
      pdf.text(`Fatigue Score: ${currentData.fatigue_score || 0}  |  Level: ${currentData.fatigue_level}  |  Confidence: ${(currentData.confidence * 100).toFixed(0)}%`, 20, 38);
      pdf.text(`HR: ${currentData.hr} bpm  |  HRV: ${currentData.hrv} ms  |  EDA: ${currentData.eda} uS  |  Temp: ${currentData.temp} C`, 20, 46);
      pdf.text(`Breathing: ${currentData.br} /min`, 20, 54);
      pdf.setFontSize(13);
      pdf.text('AI Health Advice:', 20, 66);
      pdf.setFontSize(10);
      const advice = chatMessages.value.filter(m => m.role === 'ai').slice(-1)[0]?.text || 'No advice yet.';
      const lines = pdf.splitTextToSize(advice, 170);
      pdf.text(lines, 20, 74);
      pdf.save(`fatigue_report_P${selectedPid.value}_S${selectedSid.value}.pdf`);
    }

    // ══════════════════════════════════════
    // 时间线
    // ══════════════════════════════════════
    function updateTimelineMarks() {
      if (fatigueData.length < 10) return;
      const marks = [];
      const step = Math.max(1, Math.floor(fatigueData.length / 10));
      for (let i = step; i < fatigueData.length; i += step) {
        const score = fatigueData[i];
        const level = score < 33 ? 'low' : score < 66 ? 'medium' : 'high';
        marks.push({ pos: (i / fatigueData.length) * 100, level, label: `T=${timeData[i]}s Score=${score}` });
      }
      timelineMarks.value = marks;
    }

    function seekTimeline(mark) {
      console.log('Seek to:', mark.label);
    }

    // ══════════════════════════════════════
    // WebSocket
    // ══════════════════════════════════════
    function connectStream() {
      if (ws) { ws.close(); ws = null; }
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(`${protocol}//${location.host}/ws/sensor-stream/${selectedPid.value}/${selectedSid.value}?loop=true&duration=10800`);
      ws.onopen = () => { streamActive.value = true; cycleCount.value = 1; startElapsedTimer(); predictionHistory = []; };
      ws.onmessage = (event) => {
        const frame = JSON.parse(event.data);
        if (frame.event === 'stream_config') { currentSessionId.value = frame.start_session; return; }
        if (frame.event === 'session_switch') {
          currentSessionId.value = frame.session_id; cycleCount.value = frame.cycle + 1;
          sessionDataLength.value = frame.data_length || 1500;
          const match = participants.value.find(x => x.participant_id === selectedPid.value && x.session_id === frame.session_id);
          if (match) { const map = { low: '低强度', medium: '中强度', high: '高强度' }; currentActivityLabel.value = map[match.activity_level] || match.activity_level; }
          return;
        }
        if (frame.event === 'cycle_end') return;
        if (frame.event === 'end' || frame.event === 'timeout') { streamActive.value = false; if (elapsedTimer) clearInterval(elapsedTimer); ws.close(); return; }
        if (frame.event === 'error') { console.error('服务端错误:', frame.message); streamActive.value = false; return; }

        Object.assign(currentData, { hr: frame.hr, hrv: frame.hrv, eda: frame.eda, temp: frame.temp, br: frame.br, fatigue_score: frame.fatigue_score, fatigue_level: frame.fatigue_level, confidence: frame.confidence });
        checkFatigueAlert(frame.fatigue_level);
        updateOvertrainPrediction(frame.fatigue_score);
        updateBody3D();

        timeData.push(frame.time.toFixed(0)); hrData.push(frame.hr); hrvData.push(frame.hrv);
        brData.push(frame.br); fatigueData.push(frame.fatigue_score); dataPointCount.value++;

        sessionProgress.value = Math.min(100, ((frame.session_step || 0) / sessionDataLength.value) * 100);

        if ((frame.step || 0) % 5 === 0) { updateRealtimeChart(); updateFatigueChart(); updateRadarChart(); }
        if ((frame.step || 0) % 50 === 0) { updateTimelineMarks(); if (bottomLeftTab.value === 'shap') fetchFeatureImportance(); }
      };
      ws.onclose = () => { streamActive.value = false; if (elapsedTimer) clearInterval(elapsedTimer); };
      ws.onerror = () => { streamActive.value = false; if (elapsedTimer) clearInterval(elapsedTimer); };
    }

    function connectChat() {
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      chatWs = new WebSocket(`${protocol}//${location.host}/ws/chat`);
      chatWs.onopen = () => {};
      chatWs.onclose = () => { setTimeout(connectChat, 3000); };
      chatWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        chatMessages.value.push({ role: 'ai', text: data.reply });
        nextTick(() => { if (chatArea.value) chatArea.value.scrollTop = chatArea.value.scrollHeight; });
      };
    }

    function sendChat() {
      if (!chatInput.value.trim()) return;
      const q = chatInput.value.trim();
      chatMessages.value.push({ role: 'user', text: q }); chatInput.value = '';
      if (chatWs && chatWs.readyState === WebSocket.OPEN) {
        chatWs.send(JSON.stringify({ question: q, fatigue_state: { level: currentData.fatigue_level, score: currentData.fatigue_score, metrics: { hr: currentData.hr, hrv_rmssd: currentData.hrv, eda: currentData.eda, temp: currentData.temp } }, scene: '运动训练' }));
      } else {
        chatMessages.value.push({ role: 'ai', text: '请先启动监测连接，或检查后端服务是否运行。' });
      }
      nextTick(() => { if (chatArea.value) chatArea.value.scrollTop = chatArea.value.scrollHeight; });
    }

    // ══════════════════════════════════════
    // 控制
    // ══════════════════════════════════════
    function toggleStream() {
      if (streamActive.value) { if (ws) ws.close(); streamActive.value = false; }
      else {
        dataPointCount.value = 0; sessionProgress.value = 0; predictionHistory = [];
        connectStream();
      }
    }
    function onSessionChange() {
      if (streamActive.value) { if (ws) ws.close(); streamActive.value = false; }
      dataPointCount.value = 0; sessionProgress.value = 0; predictionHistory = [];
      showAlert.value = false; alertDismissed.value = false; lastFatigueLevel = null;
      Object.assign(currentData, { hr: null, hrv: null, eda: null, temp: null, br: null, fatigue_score: null, fatigue_level: '--', confidence: 0 });
      if (realtimeChart) updateRealtimeChart();
      if (fatigueChart) updateFatigueChart();
      timelineMarks.value = [];
    }

    // 监听标签切换
    watch(activeTab, () => updateRealtimeChart());

    // ══════════════════════════════════════
    // 生命周期
    // ══════════════════════════════════════
    onMounted(async () => {
      try { const resp = await fetch('/api/participants'); participants.value = await resp.json(); } catch (e) {}
      await nextTick();
      initCharts();
      initECG();
      initBody3D();
      connectChat();
    });

    onUnmounted(() => {
      if (ws) ws.close(); if (chatWs) chatWs.close();
      if (ecgAnimId) cancelAnimationFrame(ecgAnimId);
      disposeCharts();
      if (body3dScene) body3dScene.renderer.dispose();
    });

    return {
      participants, selectedPid, selectedSid,
      streamActive, sessionProgress, dataPointCount,
      currentData, activeTab, chartTabs, metricDefs,
      fatigueLevelClass, fatigueColor, fatigueDasharray,
      currentSessionId, currentActivityLabel, cycleCount, elapsedTime,
      currentPage, switchPage,
      isDark, toggleTheme,
      soundEnabled,
      showAlert, dismissAlert,
      bottomLeftTab, switchBottomLeft,
      comparePid,
      compareSessionA, compareSessionB, loadComparison, getSessionLabel,
      overtrainPrediction,
      timelineMarks, seekTimeline,
      chatMessages, chatInput, chatArea,
      toggleStream, onSessionChange, sendChat,
      exportCSV, exportPDF,
    };
  }
});

app.mount('#app');
