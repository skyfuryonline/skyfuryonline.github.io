// games/barista.js — 浓缩时序（BARISTA · 1982）
// 一键时机游戏：扫针停进判定区，按订单做咖啡。3 张糊单结束。
ARCADE.register({
  id: 'barista',
  cn: '浓缩时序',
  en: 'BARISTA · 1982',
  create: function (api) {
    var ctx = api.ctx, W = api.W, H = api.H;
    var C = {
      bg: '#0d1420', panel: '#151d2c', panelEdge: '#232f45',
      bar: '#1c2638', barEdge: '#39506e',
      zone: 'rgba(63,174,90,.4)', zoneHi: '#3fae5a',
      core: 'rgba(255,209,102,.55)', needle: '#f5f7fa',
      good: '#3fae5a', perfect: '#ffd166', miss: '#e06c75',
      hud: '#8899bb', text: '#f5f7fa', gold: '#ffd166',
      cup: '#f5e9d6', cupDark: '#0a0f18', steam: '#b9c7d9'
    };
    var BARX = 60, BARW = 360, BARY = 148, BARH = 18;
    var CORE = 8;   // 完美核心半宽
    var CUPX = 240, CUPTOP = 236, CUPBOT = 328, CUPHALF_T = 42, CUPHALF_B = 30;

    // 阶段机制：linear 单程针（到头即糊）/ pingpong 往返针 / drift 判定区漂移
    var KINDS = {
      extract: { label: '萃取', hint: '单程针 · 停进绿区', style: 'linear',   spd: 1.0,  zw: 1.0,  color: '#6b4226' },
      water:   { label: '加水', hint: '单程针 · 飞快!',   style: 'linear',   spd: 1.35, zw: 0.9,  color: '#a9744a' },
      steam:   { label: '打奶', hint: '往返针 · 稳住',    style: 'pingpong', spd: 0.9,  zw: 1.0,  color: '#f5e9d6' },
      sauce:   { label: '加酱', hint: '单程针 · 区很小!', style: 'linear',   spd: 1.0,  zw: 0.6,  color: '#3a2418' },
      art:     { label: '拉花', hint: '区在漂移 · 追它!', style: 'drift',    spd: 1.1,  zw: 0.85, color: '#fffdf5' }
    };
    var DRINKS = [
      { name: '浓缩', en: 'ESPRESSO',  phases: ['extract'],                           w: function (s) { return s < 2 ? 99 : 2; } },
      { name: '美式', en: 'AMERICANO', phases: ['extract', 'water'],                  w: function (s) { return s >= 1 ? 3 : 0; } },
      { name: '拿铁', en: 'LATTE',     phases: ['extract', 'steam', 'art'],           w: function (s) { return s >= 4 ? 3 : 0; } },
      { name: '摩卡', en: 'MOCHA',     phases: ['extract', 'steam', 'sauce', 'art'],  w: function (s) { return s >= 8 ? 2 : 0; } }
    ];

    var state, stateT, order, phaseIdx, needle, needleDir, zoneC, zoneBase, zoneH, driftSeed;
    var score, combo, perfectStreak, served, ruined, feverUntil;
    var patience, patienceMax, bannerT, frame, over, rating;
    var layers, parts, pops, shakeT, steamT;

    api.panel([['SPACE/点击', '停针'], ['P', '暂停'], ['ESC', '片库']],
      '机台秘技：连击 5 次 PERFECT 触发「手冲大师」——指针减速、得分翻倍');

    function onKey(k, down) {
      if (down && (k === ' ' || k === 'ArrowUp' || k === 'w' || k === 'W')) press();
    }
    function onPointer(type) {
      if (type === 'down') press();
    }

    function reset() {
      score = 0; combo = 0; perfectStreak = 0; served = 0; ruined = 0;
      feverUntil = 0; frame = 0; over = false;
      layers = []; parts = []; pops = []; shakeT = 0; steamT = 0;
      newOrder();
    }

    function pickDrink() {
      var total = 0, i;
      for (i = 0; i < DRINKS.length; i++) total += DRINKS[i].w(served);
      var r = Math.random() * total;
      for (i = 0; i < DRINKS.length; i++) {
        var w = DRINKS[i].w(served);
        if (r < w) return DRINKS[i];
        r -= w;
      }
      return DRINKS[0];
    }

    function newOrder() {
      order = pickDrink();
      phaseIdx = 0;
      layers = [];
      patienceMax = Math.max(720, 1320 - served * 40);
      patience = patienceMax;
      bannerT = 55;
      startPhase();
    }

    function startPhase() {
      var k = KINDS[order.phases[phaseIdx]];
      zoneH = Math.max(12, Math.round(Math.max(20, 42 - served * 1.2) * k.zw));
      zoneBase = BARX + zoneH + 8 + Math.random() * (BARW - (zoneH + 8) * 2);
      zoneC = zoneBase;
      driftSeed = Math.random() * 6.28;
      needle = BARX; needleDir = 1;
      state = 'sweep'; stateT = 0;
    }

    function needleSpeed() {
      var v = Math.min(13, 6 + served * 0.3) * KINDS[order.phases[phaseIdx]].spd;
      if (frame < feverUntil) v *= 0.5;
      return v;
    }

    function boom(x, y, n, color, up) {
      for (var i = 0; i < n; i++) parts.push({
        x: x, y: y,
        vx: (Math.random() - 0.5) * 2.6,
        vy: up ? -Math.random() * 1.8 : (Math.random() - 0.5) * 2,
        life: 20 + Math.random() * 12, color: color
      });
    }
    function pop(x, y, txt, color, big) {
      pops.push({ x: Math.max(70, Math.min(W - 70, x)), y: y, txt: txt, color: color, big: big, life: 46 });
    }

    function press() {
      if (over || state !== 'sweep') return;
      var d = Math.abs(needle - zoneC);
      if (d <= CORE) hit('perfect');
      else if (d <= zoneH) hit('good');
      else fail('糊 了 !');
    }

    function hit(kind) {
      var fever = frame < feverUntil;
      rating = kind;
      var pts = (kind === 'perfect' ? 100 : 50) * (fever ? 2 : 1) + combo * 10;
      score += pts;
      combo++;
      layers.push(KINDS[order.phases[phaseIdx]].color);
      pop(needle, BARY - 30, kind === 'perfect' ? 'PERFECT!' : '不错', kind === 'perfect' ? C.perfect : C.good, kind === 'perfect');
      pop(needle, BARY - 12, '+' + pts, C.text, false);
      boom(needle, BARY + BARH, kind === 'perfect' ? 8 : 4, kind === 'perfect' ? C.perfect : C.good, true);
      if (kind === 'perfect') {
        perfectStreak++;
        if (perfectStreak >= 5) {
          perfectStreak = 0;
          feverUntil = frame + 180;
          pop(W / 2, 100, '☕ 手冲大师! ×2', C.gold, true);
        }
      } else perfectStreak = 0;
      state = 'resolve'; stateT = 30;
    }

    function fail(txt) {
      combo = 0; perfectStreak = 0;
      rating = 'miss';
      pop(BARX + BARW / 2, BARY - 30, txt, C.miss, true);
      shakeT = 12;
      boom(CUPX, CUPBOT - 20, 16, C.cup, true);
      state = 'ruin'; stateT = 62;
    }

    function update() {
      frame++;
      if (shakeT > 0) shakeT--;
      if (bannerT > 0) bannerT--;
      for (var q = parts.length - 1; q >= 0; q--) {
        var pt = parts[q];
        pt.x += pt.vx; pt.y += pt.vy; pt.vy += 0.03; pt.life--;
        if (pt.life <= 0) parts.splice(q, 1);
      }
      for (var p = pops.length - 1; p >= 0; p--) {
        pops[p].y -= 0.6; pops[p].life--;
        if (pops[p].life <= 0) pops.splice(p, 1);
      }
      if (over) return;

      if (state === 'sweep') {
        patience--;
        if (patience <= 0) { patience = 0; fail('超时了!'); return; }
        var k = KINDS[order.phases[phaseIdx]];
        if (k.style === 'pingpong') {
          needle += needleSpeed() * needleDir;
          if (needle <= BARX) { needle = BARX; needleDir = 1; }
          if (needle >= BARX + BARW) { needle = BARX + BARW; needleDir = -1; }
        } else {
          needle += needleSpeed();
          if (needle >= BARX + BARW) { fail('错过了!'); return; }
        }
        if (k.style === 'drift') {
          zoneC = zoneBase + Math.sin(frame * 0.045 + driftSeed) * 70;
          zoneC = Math.max(BARX + zoneH, Math.min(BARX + BARW - zoneH, zoneC));
        }
        if (layers.length > 0 && --steamT <= 0) {
          steamT = 9;
          parts.push({ x: CUPX - 10 + Math.random() * 20, y: CUPTOP - 4, vx: (Math.random() - 0.5) * 0.4, vy: -0.5 - Math.random() * 0.3, life: 34, color: C.steam });
        }
      } else if (state === 'resolve') {
        stateT--;
        if (stateT <= 0) {
          phaseIdx++;
          if (phaseIdx >= order.phases.length) {
            var bonus = Math.floor(patience / 60) * 5;
            score += bonus;
            served++;
            if (bonus > 0) pop(CUPX, 210, '+' + bonus + ' 时间分', C.hud, false);
            state = 'serve'; stateT = 46;
          } else startPhase();
        }
      } else if (state === 'serve') {
        stateT--;
        if (stateT <= 0) newOrder();
      } else if (state === 'ruin') {
        stateT--;
        if (stateT <= 0) {
          ruined++;
          if (ruined >= 3) { over = true; api.gameOver(score); return; }
          newOrder();
        }
      }
    }

    function rect(x, y, w, h, c) { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), w, h); }

    function drawCup() {
      var t = CUPHALF_T, b = CUPHALF_B, top = CUPTOP, bot = CUPBOT;
      // 杯内层（裁剪到杯形梯台，逐层自底向上）
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(CUPX - t + 4, top + 4); ctx.lineTo(CUPX + t - 4, top + 4);
      ctx.lineTo(CUPX + b - 3, bot); ctx.lineTo(CUPX - b + 3, bot);
      ctx.closePath(); ctx.clip();
      rect(CUPX - t, top, t * 2, bot - top, C.cupDark);
      var n = order.phases.length, ih = (bot - top - 6) / n;
      for (var j = 0; j < layers.length; j++) {
        rect(CUPX - t, bot - (j + 1) * ih - 2, t * 2, ih, layers[j]);
      }
      ctx.restore();
      // 杯体描边 + 把手 + 杯托
      ctx.strokeStyle = C.cup; ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(CUPX - t, top); ctx.lineTo(CUPX - b, bot); ctx.lineTo(CUPX + b, bot); ctx.lineTo(CUPX + t, top);
      ctx.stroke();
      rect(CUPX - t - 3, top - 5, t * 2 + 6, 5, C.cup);
      ctx.beginPath(); ctx.arc(CUPX + t + 9, top + 30, 14, -1.4, 1.4); ctx.stroke();
      rect(CUPX - b - 12, bot, b * 2 + 24, 4, C.cupDark);
      if (state === 'serve') {
        // 出杯蒸汽礼花
        if (stateT % 6 === 0) boom(CUPX - 8 + Math.random() * 16, top - 6, 2, C.gold, true);
      }
    }

    function drawTicket() {
      rect(70, 12, W - 140, 52, C.panel);
      ctx.strokeStyle = C.panelEdge; ctx.lineWidth = 1;
      ctx.strokeRect(70.5, 12.5, W - 141, 51);
      ctx.textAlign = 'left';
      ctx.font = 'bold 16px monospace';
      ctx.fillStyle = C.text;
      ctx.fillText(order.name, 84, 34);
      ctx.font = '9px monospace';
      ctx.fillStyle = C.hud;
      ctx.fillText(order.en, 84, 50);
      // 阶段圆点
      for (var i = 0; i < order.phases.length; i++) {
        var px = 170 + i * 22;
        ctx.beginPath();
        ctx.arc(px, 30, i === phaseIdx ? 6 : 4, 0, 7);
        if (i < phaseIdx) { ctx.fillStyle = C.good; ctx.fill(); }
        else if (i === phaseIdx) {
          ctx.strokeStyle = (frame >> 3) % 2 ? C.gold : C.text; ctx.lineWidth = 2; ctx.stroke();
        } else { ctx.strokeStyle = C.hud; ctx.lineWidth = 1; ctx.stroke(); }
      }
      ctx.font = '9px monospace';
      ctx.fillStyle = C.hud;
      ctx.textAlign = 'right';
      ctx.fillText('第 ' + Math.min(phaseIdx + 1, order.phases.length) + '/' + order.phases.length + ' 步', W - 84, 34);
      // 耐心条
      var ratio = patience / patienceMax;
      var bw = W - 200;
      rect(100, 44, bw, 7, '#0a0f18');
      rect(100, 44, bw * ratio, 7, ratio > 0.5 ? C.good : ratio > 0.25 ? C.gold : C.miss);
      ctx.textAlign = 'right';
      ctx.fillStyle = ratio < 0.25 && (frame >> 3) % 2 ? C.miss : C.hud;
      ctx.fillText('耐心', 92, 51);
    }

    function drawBar() {
      // serve/ruin 态 phaseIdx 可能越界（出杯时已指向 phases.length），钳到最后一步
      var idx = Math.min(phaseIdx, order.phases.length - 1);
      var k = KINDS[order.phases[idx]];
      // 阶段说明
      ctx.textAlign = 'center';
      ctx.font = 'bold 13px monospace';
      ctx.fillStyle = C.text;
      ctx.fillText(state === 'serve' ? '出 杯 !' : k.label + ' — ' + k.hint, W / 2, BARY - 22);
      // 轨道
      rect(BARX, BARY, BARW, BARH, C.bar);
      ctx.strokeStyle = C.barEdge; ctx.lineWidth = 1;
      ctx.strokeRect(BARX + 0.5, BARY + 0.5, BARW - 1, BARH - 1);
      if (state === 'sweep' || state === 'resolve') {
        // 判定区
        rect(zoneC - zoneH, BARY - 4, zoneH * 2, BARH + 8, C.zone);
        rect(zoneC - zoneH, BARY - 4, 2, BARH + 8, C.zoneHi);
        rect(zoneC + zoneH - 2, BARY - 4, 2, BARH + 8, C.zoneHi);
        // 完美核心
        rect(zoneC - CORE, BARY - 4, CORE * 2, BARH + 8, C.core);
        // 针
        var nc = state === 'resolve' ? (rating === 'miss' ? C.miss : rating === 'perfect' ? C.perfect : C.good) : C.needle;
        rect(needle - 1, BARY - 14, 2, BARH + 28, nc);
        ctx.fillStyle = nc;
        ctx.beginPath();
        ctx.moveTo(needle - 5, BARY - 14); ctx.lineTo(needle + 5, BARY - 14); ctx.lineTo(needle, BARY - 7);
        ctx.closePath(); ctx.fill();
      }
      // 超时进度提示（pingpong 无到头失败，靠耐心压）
      if (state === 'sweep' && patienceMax - patience > 300 && KINDS[order.phases[phaseIdx]].style === 'pingpong') {
        ctx.font = '9px monospace';
        ctx.fillStyle = C.miss;
        ctx.fillText('别磨蹭!', W / 2, BARY + BARH + 22);
      }
    }

    function draw() {
      ctx.save();
      if (shakeT > 0) ctx.translate((Math.random() - 0.5) * 5, (Math.random() - 0.5) * 5);
      rect(0, 0, W, H, C.bg);
      // 背景星点
      for (var s = 0; s < 20; s++) {
        ctx.fillStyle = '#141d2e';
        ctx.fillRect((s * 211 + 40) % W, (s * 97 + 70) % H, 1, 1);
      }

      drawTicket();
      drawBar();
      drawCup();

      // 粒子 / 浮字
      for (var q = 0; q < parts.length; q++) {
        ctx.globalAlpha = Math.max(0, parts[q].life / 28);
        rect(parts[q].x, parts[q].y, 2, 2, parts[q].color);
      }
      ctx.globalAlpha = 1;
      for (var p = 0; p < pops.length; p++) {
        var pp = pops[p];
        ctx.globalAlpha = Math.min(1, pp.life / 16);
        ctx.textAlign = 'center';
        ctx.font = (pp.big ? 'bold 20px' : 'bold 12px') + ' monospace';
        ctx.fillStyle = pp.color;
        ctx.fillText(pp.txt, pp.x, pp.y);
      }
      ctx.globalAlpha = 1;

      // Fever 滤镜
      if (frame < feverUntil) {
        ctx.strokeStyle = 'rgba(255,209,102,' + (0.4 + 0.3 * Math.sin(frame * 0.2)) + ')';
        ctx.lineWidth = 6;
        ctx.strokeRect(3, 3, W - 6, H - 6);
        ctx.textAlign = 'center';
        ctx.font = 'bold 11px monospace';
        ctx.fillStyle = C.gold;
        ctx.fillText('手冲大师 ×2 · ' + Math.ceil((feverUntil - frame) / 60) + 's', W / 2, 80);
      }

      // 新订单横幅
      if (bannerT > 0) {
        ctx.globalAlpha = Math.min(1, bannerT / 12);
        ctx.fillStyle = 'rgba(6,10,16,.7)';
        ctx.fillRect(0, H / 2 - 40, W, 54);
        ctx.textAlign = 'center';
        ctx.font = 'bold 18px monospace';
        ctx.fillStyle = C.gold;
        ctx.fillText('新订单: ' + order.name + ' ☕', W / 2, H / 2 - 12);
        ctx.font = '10px monospace';
        ctx.fillStyle = C.hud;
        ctx.fillText(order.phases.length + ' 道工序 · 别让客人等急了', W / 2, H / 2 + 6);
        ctx.globalAlpha = 1;
      }

      // HUD
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      ctx.fillStyle = C.text;
      ctx.fillText('SCORE ' + score, 10, 20);
      if (combo > 1) {
        ctx.fillStyle = C.gold;
        ctx.fillText('连击 x' + combo, 10, 34);
      }
      // 右上：卖出数 + 剩余机会（3 个杯子，糊一个碎一个）
      ctx.textAlign = 'right';
      ctx.fillStyle = C.hud;
      ctx.fillText('卖出 ' + served, W - 10, 20);
      for (var h = 0; h < 3; h++) {
        var hx = W - 16 - h * 16, alive = h >= ruined;
        ctx.fillStyle = alive ? C.cup : '#333c46';
        ctx.fillRect(hx - 6, 26 + h * 0, 10, 8);
        ctx.fillRect(hx + 4, 28, 3, 4);
      }
      ctx.restore();
    }

    reset();
    return { update: update, draw: draw, onKey: onKey, onPointer: onPointer };
  }
});
