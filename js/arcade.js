// arcade.js — 咖啡街机厅内核（daily 页彩蛋）
// 触发：像素办公室里的咖啡杯 .pixel-mug 2 秒内三连击
// 结构：内核负责选单 / 输入分发 / 暂停 / 结算 / 高分存档，各游戏文件向 ARCADE.register 注册
(function () {
  var games = [];
  var overlay = null, canvas = null, ctx = null, W = 480, H = 360;
  var marqueeCn, marqueeEn, panelEl;
  var mode = 'closed'; // closed | menu | game | over
  var sel = 0;
  var active = null, activeDef = null;
  var overInfo = null;   // {score, best, isNew, name}
  var rafId = null, acc = 0, lastT = 0;
  var paused = false;
  var blink = 0;

  var COL = {
    bg: '#0d1420', card: '#141d2e', cardSel: '#1c2a40',
    line: '#2c3a54', gold: '#ffd166', hot: '#ff8a5c',
    text: '#d3c9d4', dim: '#8899bb', green: '#3fae5a', bug: '#e06c75'
  };

  // ---------- 高分存档 ----------
  function bestKey(id) { return 'arcade.best.' + id; }
  function getBest(id) {
    try { return +localStorage.getItem(bestKey(id)) || 0; } catch (e) { return 0; }
  }
  function commitBest(id, score) {
    try {
      if (score > getBest(id)) { localStorage.setItem(bestKey(id), score); return true; }
    } catch (e) {}
    return false;
  }

  function register(def) { games.push(def); }

  // ---------- DOM 面板 ----------
  function setMarquee(cn, en) {
    if (marqueeCn) marqueeCn.textContent = cn;
    if (marqueeEn) marqueeEn.textContent = en;
  }
  function setPanel(hints, tip) {
    if (!panelEl) return;
    var html = '';
    for (var i = 0; i < hints.length; i++) {
      html += '<span class="game-key">' + hints[i][0] + '</span><span class="game-label">' + hints[i][1] + '</span>';
    }
    if (tip) html += '<span class="game-tip">' + tip + '</span>';
    panelEl.innerHTML = html;
  }
  function menuPanel() {
    setMarquee('咖啡街机厅', 'SKYFURY ARCADE');
    setPanel([['↑↓←→', '选择'], ['ENTER', '开始'], ['ESC', '离开']],
      '机台秘技：每台机器都单独记录你的最高分');
  }

  // ---------- 游戏生命周期 ----------
  function makeApi(def) {
    return {
      canvas: canvas, ctx: ctx, W: W, H: H,
      best: getBest(def.id),
      panel: function (hints, tip) { setPanel(hints, tip); },
      gameOver: function (score, opts) {
        if (activeDef !== def || mode !== 'game') return; // 已离开本局，过期调用作废
        opts = opts || {};
        var isNew = commitBest(def.id, score);
        overInfo = { score: score, best: getBest(def.id), isNew: isNew, name: def.cn, big: opts.big || '游 戏 结 束', def: def };
        if (active && active.destroy) { try { active.destroy(); } catch (e) {} }
        active = null; activeDef = null;
        mode = 'over';
        setMarquee('游戏结束', 'GAME OVER');
        setPanel([['ENTER', '再来一局'], ['ESC', '返回片库']], null);
      }
    };
  }

  function startGame(def) {
    activeDef = def;
    overInfo = null;
    active = def.create(makeApi(def));
    mode = 'game';
    paused = false;
    setMarquee(def.cn, def.en);
  }

  function restartGame() {
    var def = activeDef || (overInfo && overInfo.def);
    if (def) startGame(def);
  }

  function toMenu() {
    if (active && active.destroy) { try { active.destroy(); } catch (e) {} }
    active = null; activeDef = null;
    mode = 'menu';
    paused = false;
    menuPanel();
  }

  // ---------- 选单绘制 ----------
  function drawIcon(id, x, y, s) { // s = 图标区域大小
    ctx.save();
    ctx.translate(x, y);
    if (id === 'tank') {
      ctx.fillStyle = COL.green;
      ctx.fillRect(s * 0.15, s * 0.35, s * 0.7, s * 0.4);
      ctx.fillRect(s * 0.42, s * 0.12, s * 0.16, s * 0.3);
      ctx.fillStyle = '#2a7a43';
      ctx.fillRect(s * 0.25, s * 0.45, s * 0.5, s * 0.18);
      ctx.fillStyle = COL.gold;
      ctx.fillRect(s * 0.3, s * 0.82, s * 0.4, s * 0.12);
    } else if (id === 'breakout') {
      var cols = ['#e06c75', '#e5c07b', '#3fae5a'];
      for (var r = 0; r < 3; r++)
        for (var c = 0; c < 4; c++) {
          ctx.fillStyle = cols[r];
          ctx.fillRect(s * 0.1 + c * s * 0.21, s * 0.12 + r * s * 0.16, s * 0.17, s * 0.11);
        }
      ctx.fillStyle = '#f5f7fa';
      ctx.beginPath(); ctx.arc(s * 0.55, s * 0.75, s * 0.07, 0, 7); ctx.fill();
      ctx.fillRect(s * 0.3, s * 0.92, s * 0.4, s * 0.06);
    } else if (id === 'flappy') {
      ctx.fillStyle = COL.hot;
      ctx.fillRect(s * 0.22, s * 0.28, s * 0.5, s * 0.42);
      ctx.fillRect(s * 0.72, s * 0.38, s * 0.1, s * 0.2);
      ctx.fillStyle = '#f5e9d6';
      ctx.fillRect(s * 0.32, s * 0.4, s * 0.1, s * 0.08);
      ctx.fillStyle = '#fff';
      ctx.fillRect(s * 0.3, s * 0.16, s * 0.4, s * 0.07);
      ctx.fillStyle = COL.dim;
      ctx.fillRect(s * 0.06, s * 0.34, s * 0.16, s * 0.24);
    } else if (id === 'drop100') {
      ctx.fillStyle = COL.dim;
      for (var f = 0; f < 4; f++) {
        if (f % 2) ctx.fillRect(s * 0.08, s * 0.1 + f * s * 0.24, s * 0.36, s * 0.12);
        else ctx.fillRect(s * 0.56, s * 0.1 + f * s * 0.24, s * 0.36, s * 0.12);
      }
      ctx.fillStyle = COL.gold;
      ctx.fillRect(s * 0.46, s * 0.44, s * 0.1, s * 0.2);
      ctx.fillRect(s * 0.42, s * 0.36, s * 0.18, s * 0.1);
    } else if (id === 'pinball') {
      ctx.fillStyle = COL.bug;
      ctx.beginPath(); ctx.arc(s * 0.5, s * 0.28, s * 0.11, 0, 7); ctx.fill();
      ctx.fillStyle = '#e5c07b';
      ctx.beginPath(); ctx.arc(s * 0.26, s * 0.2, s * 0.08, 0, 7); ctx.fill();
      ctx.beginPath(); ctx.arc(s * 0.74, s * 0.2, s * 0.08, 0, 7); ctx.fill();
      ctx.strokeStyle = '#9fb3d9'; ctx.lineWidth = s * 0.06; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(s * 0.16, s * 0.85); ctx.lineTo(s * 0.42, s * 0.62); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(s * 0.84, s * 0.85); ctx.lineTo(s * 0.58, s * 0.62); ctx.stroke();
      ctx.fillStyle = '#f5f7fa';
      ctx.beginPath(); ctx.arc(s * 0.5, s * 0.62, s * 0.05, 0, 7); ctx.fill();
    } else {
      ctx.fillStyle = COL.gold;
      ctx.font = 'bold ' + (s * 0.7) + 'px monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('?', s / 2, s / 2);
    }
    ctx.restore();
  }

  function menuLayout() {
    // 5 卡片：上排 3 张，下排 2 张（居中）
    var cw = 138, ch = 116, gx = 14, gy = 14;
    var topY = 74, botY = topY + ch + gy;
    var pos = [];
    var row1 = (W - (cw * 3 + gx * 2)) / 2;
    var row2 = (W - (cw * 2 + gx)) / 2;
    for (var i = 0; i < games.length; i++) {
      if (i < 3) pos.push({ x: row1 + i * (cw + gx), y: topY });
      else pos.push({ x: row2 + (i - 3) * (cw + gx), y: botY });
    }
    return { pos: pos, cw: cw, ch: ch };
  }

  function drawMenu() {
    ctx.fillStyle = COL.bg;
    ctx.fillRect(0, 0, W, H);
    // 背景星点
    for (var s = 0; s < 30; s++) {
      ctx.fillStyle = '#1a2538';
      ctx.fillRect((s * 197) % W, (s * 89) % H, 1, 1);
    }
    // 标题
    ctx.textAlign = 'center';
    ctx.font = 'bold 24px monospace';
    ctx.fillStyle = COL.gold;
    ctx.shadowColor = 'rgba(255,209,102,.55)';
    ctx.shadowBlur = 10;
    ctx.fillText('咖 啡 街 机 厅', W / 2, 36);
    ctx.shadowBlur = 0;
    ctx.font = '10px monospace';
    ctx.fillStyle = COL.dim;
    ctx.fillText('SKYFURY ARCADE · FIVE MACHINES · INSERT COFFEE ☕', W / 2, 54);

    var L = menuLayout();
    blink++;
    for (var i = 0; i < games.length; i++) {
      var g = games[i];
      var p = L.pos[i];
      var on = i === sel;
      ctx.fillStyle = on ? COL.cardSel : COL.card;
      ctx.fillRect(p.x, p.y, L.cw, L.ch);
      ctx.strokeStyle = on ? ((blink >> 4) % 2 ? COL.gold : COL.hot) : COL.line;
      ctx.lineWidth = on ? 2 : 1;
      ctx.strokeRect(p.x + .5, p.y + .5, L.cw - 1, L.ch - 1);

      drawIcon(g.id, p.x + L.cw / 2 - 22, p.y + 12, 44);

      ctx.textAlign = 'center';
      ctx.font = 'bold 14px monospace';
      ctx.fillStyle = on ? '#f5f0e6' : COL.text;
      ctx.fillText(g.cn, p.x + L.cw / 2, p.y + 74);
      ctx.font = '8px monospace';
      ctx.fillStyle = COL.dim;
      ctx.fillText(g.en, p.x + L.cw / 2, p.y + 87);
      var b = getBest(g.id);
      ctx.fillStyle = b > 0 ? COL.gold : '#4a5878';
      ctx.fillText(b > 0 ? 'HI ' + b : '— 无纪录 —', p.x + L.cw / 2, p.y + 102);
    }

    ctx.font = '11px monospace';
    ctx.fillStyle = ((blink >> 4) % 2) ? COL.green : COL.dim;
    ctx.fillText('↑↓←→ 选择 · ENTER 投币开始', W / 2, H - 14);
  }

  function drawOver() {
    ctx.fillStyle = 'rgba(6,10,16,.82)';
    ctx.fillRect(0, 0, W, H);
    ctx.textAlign = 'center';
    ctx.font = 'bold 22px monospace';
    ctx.fillStyle = COL.bug;
    ctx.fillText(overInfo.big, W / 2, H / 2 - 62);
    ctx.font = '13px monospace';
    ctx.fillStyle = COL.text;
    ctx.fillText('「' + overInfo.name + '」', W / 2, H / 2 - 38);
    ctx.font = 'bold 20px monospace';
    ctx.fillStyle = '#f5f7fa';
    ctx.fillText('得分  ' + overInfo.score, W / 2, H / 2 - 8);
    if (overInfo.isNew) {
      ctx.fillStyle = COL.gold;
      ctx.shadowColor = 'rgba(255,209,102,.7)'; ctx.shadowBlur = 8;
      ctx.font = 'bold 13px monospace';
      ctx.fillText('★ 新纪录 ★', W / 2, H / 2 + 16);
      ctx.shadowBlur = 0;
    } else {
      ctx.font = '11px monospace';
      ctx.fillStyle = COL.dim;
      ctx.fillText('最高纪录 ' + overInfo.best, W / 2, H / 2 + 16);
    }
    ctx.font = '12px monospace';
    ctx.fillStyle = ((blink >> 4) % 2) ? COL.green : COL.dim;
    ctx.fillText('ENTER 再来一局 · ESC 返回片库', W / 2, H / 2 + 56);
  }

  // ---------- 主循环（固定 60Hz 步进） ----------
  function step() {
    if (mode === 'game' && active && !paused && active.update) active.update();
  }
  function frame(t) {
    if (!rafId) return;
    var dt = Math.min(100, t - lastT || 16.7);
    lastT = t;
    acc += dt;
    var n = 0;
    while (acc >= 16.667 && n < 4) { step(); acc -= 16.667; n++; }
    if (acc > 200) acc = 0;

    if (mode === 'menu') drawMenu();
    else if (mode === 'over') { blink++; drawOver(); }
    else if (mode === 'game' && active && active.draw) {
      active.draw();
      if (paused) {
        ctx.fillStyle = 'rgba(6,10,16,.66)';
        ctx.fillRect(0, 0, W, H);
        ctx.textAlign = 'center';
        ctx.font = 'bold 20px monospace';
        ctx.fillStyle = COL.gold;
        ctx.fillText('暂 停', W / 2, H / 2 - 6);
        ctx.font = '11px monospace';
        ctx.fillStyle = COL.dim;
        ctx.fillText('按 P 继续 · ESC 返回片库', W / 2, H / 2 + 20);
      }
    }
    rafId = requestAnimationFrame(frame);
  }

  // ---------- 输入 ----------
  var GAME_KEYS = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', ' ', 'a', 'A', 'd', 'D', 'w', 'W', 's', 'S', 'z', 'Z', 'm', 'M', 'p', 'P', 'r', 'R', 'Enter'];
  document.addEventListener('keydown', function (e) {
    if (mode === 'closed') return;
    if (GAME_KEYS.indexOf(e.key) >= 0) e.preventDefault();
    if (e.key === 'Escape') {
      if (mode === 'menu') closeArcade();
      else toMenu();
      return;
    }
    if (mode === 'menu') {
      var last = games.length - 1;
      if (e.key === 'ArrowLeft' && sel > 0 && sel !== 3) sel--;
      else if (e.key === 'ArrowRight' && sel < last && sel !== 2) sel++;
      else if (e.key === 'ArrowUp' && sel >= 3) sel -= 3;
      else if (e.key === 'ArrowDown' && sel <= 2) sel = Math.min(last, sel + 3);
      else if (e.key === 'Enter' || e.key === ' ') { if (games[sel]) startGame(games[sel]); }
      return;
    }
    if (mode === 'over') {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'r' || e.key === 'R') restartGame();
      return;
    }
    if (mode === 'game') {
      if (e.key === 'p' || e.key === 'P') { paused = !paused; return; }
      if ((e.key === 'r' || e.key === 'R') && !paused) { restartGame(); return; }
      if (!paused && active && active.onKey) active.onKey(e.key, true);
    }
  });
  document.addEventListener('keyup', function (e) {
    // 暂停期间也要放行 keyup：否则按住方向键时按 P，松键会被吞掉，
    // 恢复后 keys 状态卡在 true，坦克/小人不受控
    if (mode === 'game' && active && active.onKey) active.onKey(e.key, false);
  });

  // 触屏 / 鼠标：坐标换算后交给当前状态处理
  function canvasXY(ev) {
    var r = canvas.getBoundingClientRect();
    var pt = ev.touches ? ev.touches[0] : ev;
    return {
      x: (pt.clientX - r.left) * (W / r.width),
      y: (pt.clientY - r.top) * (H / r.height)
    };
  }
  function pointerTap(x, y) {
    if (mode === 'menu') {
      var L = menuLayout();
      for (var i = 0; i < games.length; i++) {
        var p = L.pos[i];
        if (x >= p.x && x <= p.x + L.cw && y >= p.y && y <= p.y + L.ch) {
          if (i === sel) startGame(games[i]); else sel = i;
          return;
        }
      }
    } else if (mode === 'over') {
      restartGame();
    } else if (mode === 'game' && active && active.onPointer) {
      active.onPointer('down', x, y);
    }
  }
  function bindPointer() {
    canvas.addEventListener('mousedown', function (ev) { var p = canvasXY(ev); pointerTap(p.x, p.y); });
    canvas.addEventListener('mousemove', function (ev) {
      if (mode === 'game' && active && active.onPointer) { var p = canvasXY(ev); active.onPointer('move', p.x, p.y); }
    });
    canvas.addEventListener('touchstart', function (ev) {
      ev.preventDefault();
      var p = canvasXY(ev);
      pointerTap(p.x, p.y);
    }, { passive: false });
    canvas.addEventListener('touchmove', function (ev) {
      ev.preventDefault();
      if (mode === 'game' && active && active.onPointer) { var p = canvasXY(ev); active.onPointer('move', p.x, p.y); }
    }, { passive: false });
  }

  // ---------- 开关 ----------
  function openArcade() {
    mode = 'menu';
    sel = 0;
    overlay.style.display = 'block';
    document.body.style.overflow = 'hidden';
    toMenu();
    if (!rafId) { lastT = 0; acc = 0; rafId = requestAnimationFrame(frame); }
  }
  function closeArcade() {
    if (active && active.destroy) { try { active.destroy(); } catch (e) {} }
    active = null; activeDef = null;
    mode = 'closed';
    overlay.style.display = 'none';
    document.body.style.overflow = '';
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  }

  // ---------- 启动 ----------
  function boot() {
    overlay = document.getElementById('game-overlay');
    canvas = document.getElementById('game-canvas');
    if (!overlay || !canvas) return;
    ctx = canvas.getContext('2d');
    W = canvas.width; H = canvas.height;
    marqueeCn = overlay.querySelector('.game-marquee-cn');
    marqueeEn = overlay.querySelector('.game-marquee-en');
    panelEl = overlay.querySelector('.game-panel');

    var closeBtn = document.getElementById('game-close');
    if (closeBtn) closeBtn.addEventListener('click', closeArcade);
    bindPointer();

    var mug = document.querySelector('.pixel-mug');
    if (mug) {
      var times = [];
      mug.addEventListener('click', function (e) {
        e.stopPropagation();
        var now = Date.now();
        times.push(now);
        times = times.filter(function (t) { return now - t < 2000; });
        if (times.length >= 3) { times = []; openArcade(); }
      });
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();

  window.ARCADE = { register: register, open: openArcade, close: closeArcade };
})();
