/* E²-JEPA 研究阶段报告 — 图表逻辑（数据来自 external/*.json，2026-10-21） */
(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  var baseAxis = {
    axisLine: { lineStyle: { color: rule } },
    axisLabel: { color: muted, fontSize: 12 },
    splitLine: { lineStyle: { color: rule, type: 'dashed' } }
  };

  /* ---------- 图 1 · C1 四源对比 ---------- */
  var c1 = echarts.init(document.getElementById('chart-c1'), null, { renderer: 'svg' });
  c1.setOption({
    animation: false,
    grid: { left: 56, right: 20, top: 56, bottom: 42 },
    legend: { top: 10, textStyle: { color: muted, fontSize: 12 } },
    tooltip: { trigger: 'axis', appendToBody: true, valueFormatter: function (v) { return v == null ? '—' : Number(v).toFixed(3); } },
    xAxis: Object.assign({ type: 'category', data: ['PushT', 'AntMaze · LeWM', 'AntMaze · H-JEPA'] }, baseAxis, { splitLine: { show: false } }),
    yAxis: Object.assign({ type: 'value', max: 0.55, name: 'dispR2', nameTextStyle: { color: muted, fontSize: 12 } }, baseAxis),
    series: [
      {
        name: '底座 raw z', type: 'bar', data: [0.100, 0.358, 0.483],
        itemStyle: { color: bg2, borderColor: muted, borderWidth: 1 },
        markLine: {
          symbol: 'none', silent: true, label: { position: 'insideEndTop', fontSize: 11, color: accent2 },
          lineStyle: { type: 'dashed' },
          data: [
            { yAxis: 0.37, lineStyle: { color: accent }, label: { formatter: 'C1 阈值 0.37', color: accent } },
            { yAxis: 0.137, lineStyle: { color: accent2 }, label: { formatter: 'PushT 地板 0.137', color: accent2 } }
          ]
        }
      },
      { name: 'v1.2 · 纯SFA脚手架+训练', type: 'bar', data: [0.193, 0.121, 0.251], itemStyle: { color: muted } },
      { name: 'v1.3 · 切空间增广基+训练', type: 'bar', data: [0.106, 0.314, 0.354], itemStyle: { color: accent } }
    ]
  });

  /* ---------- 图 2 · H-JEPA 带中位 τ vs 参考阶梯 ---------- */
  var ladder = echarts.init(document.getElementById('chart-ladder'), null, { renderer: 'svg' });
  ladder.setOption({
    animation: false,
    grid: { left: 60, right: 24, top: 56, bottom: 42 },
    legend: { top: 10, textStyle: { color: muted, fontSize: 12 } },
    tooltip: { trigger: 'axis', appendToBody: true, valueFormatter: function (v) { return Number(v).toFixed(2) + ' 步'; } },
    xAxis: Object.assign({ type: 'category', name: '带（慢→快）', nameTextStyle: { color: muted, fontSize: 12 }, data: ['1', '2', '3', '4', '5', '6', '7', '8'] }, baseAxis, { splitLine: { show: false } }),
    yAxis: Object.assign({ type: 'log', logBase: 10, name: 'τ (步, log)', nameTextStyle: { color: muted, fontSize: 12 } }, baseAxis),
    series: [
      {
        name: '参考阶梯（SFA/DMD 测量）', type: 'line', data: [2.80, 3.41, 4.51, 6.21, 8.55, 13.06, 27.68, 102.10],
        symbol: 'circle', symbolSize: 8, lineStyle: { color: muted, type: 'dashed', width: 1.5 }, itemStyle: { color: muted }
      },
      {
        name: 'v1.3 实测带中位', type: 'line', data: [1.97, 3.53, 4.58, 6.77, 8.75, 14.88, 28.40, 122.53],
        symbol: 'circle', symbolSize: 9, lineStyle: { color: accent, width: 2.5 }, itemStyle: { color: accent },
        areaStyle: { color: accent, opacity: 0.06 }
      }
    ]
  });

  /* ---------- 图 3 · 内容按时间尺度落位（探针 R² by 带） ---------- */
  var place = echarts.init(document.getElementById('chart-place'), null, { renderer: 'svg' });
  place.setOption({
    animation: false,
    grid: { left: 56, right: 20, top: 56, bottom: 42 },
    legend: { top: 10, textStyle: { color: muted, fontSize: 12 } },
    tooltip: { trigger: 'axis', appendToBody: true, valueFormatter: function (v) { return Number(v).toFixed(3); } },
    xAxis: Object.assign({ type: 'category', name: '带（慢→快）', nameTextStyle: { color: muted, fontSize: 12 }, data: ['1', '2', '3', '4', '5', '6', '7', '8'] }, baseAxis, { splitLine: { show: false } }),
    yAxis: Object.assign({ type: 'value', max: 1.0, name: '探针 R²', nameTextStyle: { color: muted, fontSize: 12 } }, baseAxis),
    series: [
      { name: 'block 位姿（慢内容）', type: 'bar', data: [0.857, -0.048, 0.064, -0.058, 0.209, 0.206, 0.194, -0.126], itemStyle: { color: accent } },
      { name: 'agent 位姿（快内容）', type: 'bar', data: [0.002, 0.001, 0.001, 0.010, 0.001, 0.002, 0.043, 0.277], itemStyle: { color: accent2 } },
      { name: 'agent 速度（快内容）', type: 'bar', data: [0.000, 0.000, 0.000, 0.001, 0.002, 0.000, 0.004, 0.312], itemStyle: { color: bg2, borderColor: muted, borderWidth: 1 } }
    ]
  });

  window.addEventListener('resize', function () {
    c1.resize(); ladder.resize(); place.resize();
  });
})();
