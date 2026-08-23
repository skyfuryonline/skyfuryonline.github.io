// ground-defense.js — 《地面保卫军》像素守卫小游戏（daily 页彩蛋）
// 触发方式：像素办公室里的咖啡杯 .pixel-mug 2 秒内三连击（与 jzxm 植物彩蛋同款机制）
(function () {
  var overlay = document.getElementById('game-overlay');
  var canvas = document.getElementById('game-canvas');
  if (!overlay || !canvas) return;
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;
  var GROUND_Y = H - 26;
  var C = {
    bg: '#0d1420', ground: '#1f3d2b', groundTop: '#3fae5a',
    player: '#66fcf1', bullet: '#f5f7fa', hud: '#8899bb',
    bug: '#e06c75', dart: '#e5c07b', heavy: '#c678dd',
    coffee: '#ff9f64', boom: '#f5f7fa'
  };

  var state = 'title';
  var score, wave, groundHP, kills, spawnTimer, waveTimer, powerUntil, shake;
  var player, bullets, enemies, drops, particles;
  var keys = {};
  var rafId = null, lastT = 0;

  function reset() {
    score = 0; wave = 1; groundHP = 5; kills = 0;
    spawnTimer = 0; waveTimer = 0; powerUntil = 0; shake = 0;
    player = { x: W / 2, w: 26, h: 14, cooldown: 0 };
    bullets = []; enemies = []; drops = []; particles = [];
  }

  function spawnEnemy() {
    var r = Math.random();
    var type = r < 0.55 ? 'bug' : (r < 0.85 ? 'dart' : 'heavy');
    var e = {
      type: type,
      x: 16 + Math.random() * (W - 48),
      y: -14,
      seed: Math.random() * 100,
      hp: type === 'heavy' ? 3 : 1
    };
    if (type === 'bug')      { e.w = 10; e.h = 10; e.vy = 0.55 + wave * 0.05; }
    if (type === 'dart')     { e.w = 6;  e.h = 12; e.vy = 1.05 + wave * 0.08; }
    if (type === 'heavy')    { e.w = 14; e.h = 14; e.vy = 0.38 + wave * 0.03; }
    e.vy *= (0.9 + Math.random() * 0.25);
    enemies.push(e);
  }

  function fire() {
    var now = performance.now();
    var active = now < powerUntil;
    if (player.cooldown > 0 || bullets.length >= (active ? 9 : 4)) return;
    player.cooldown = active ? 10 : 15; // 帧数冷却（约 160/240ms）
    var cx = player.x;
    if (active) {
      bullets.push({ x: cx, y: GROUND_Y - 18, vy: -5 });
      bullets.push({ x: cx - 7, y: GROUND_Y - 14, vy: -5 });
      bullets.push({ x: cx + 7, y: GROUND_Y - 14, vy: -5 });
    } else {
      bullets.push({ x: cx, y: GROUND_Y - 18, vy: -5 });
    }
  }

  function boom(x, y, color, n) {
    for (var i = 0; i < n; i++) {
      particles.push({
        x: x, y: y,
        vx: (Math.random() - 0.5) * 3.4,
        vy: (Math.random() - 0.8) * 3,
        life: 22 + Math.random() * 14,
        color: color
      });
    }
  }

  function hit(a, b) {
    return Math.abs(a.x - b.x) < (a.w + b.w) / 2 && (a.y - b.y) < (a.h + b.h) && (a.y - b.y) > -(a.h + b.h);
  }

  function update() {
    // 玩家移动
    var spd = 3.1;
    if (keys.left)  player.x -= spd;
    if (keys.right) player.x += spd;
    player.x = Math.max(18, Math.min(W - 18, player.x));
    if (player.cooldown > 0) player.cooldown--;
    if (keys.fire) fire();

    // 波次
    waveTimer++;
    if (waveTimer > 60 * 18) { wave++; waveTimer = 0; }

    // 生成敌人
    var interval = Math.max(28, 66 - wave * 4);
    spawnTimer++;
    if (spawnTimer >= interval) { spawnTimer = 0; spawnEnemy(); }

    // 咖啡道具：每击落 16 个掉一个
    if (kills > 0 && kills % 16 === 0 && !drops.length && Math.random() < 0.03) {
      drops.push({ x: 20 + Math.random() * (W - 40), y: -10, vy: 0.7, w: 10, h: 10 });
    }

    // 子弹
    for (var i = bullets.length - 1; i >= 0; i--) {
      var b = bullets[i];
      b.y += b.vy;
      if (b.y < -6) { bullets.splice(i, 1); continue; }
      for (var j = enemies.length - 1; j >= 0; j--) {
        var e = enemies[j];
        if (hit(b, e)) {
          bullets.splice(i, 1);
          e.hp--;
          if (e.hp <= 0) {
            enemies.splice(j, 1);
            kills++;
            score += e.type === 'bug' ? 10 : (e.type === 'dart' ? 20 : 30);
            boom(e.x, e.y, e.type === 'bug' ? C.bug : (e.type === 'dart' ? C.dart : C.heavy), 10);
          } else {
            boom(b.x, b.y, C.boom, 3);
          }
          break;
        }
      }
    }

    // 敌人
    for (var k = enemies.length - 1; k >= 0; k--) {
      var en = enemies[k];
      en.y += en.vy;
      en.x += Math.sin((en.y + en.seed) / 16) * 0.55;
      en.x = Math.max(8, Math.min(W - 8, en.x));
      if (en.y + en.h >= GROUND_Y) {
        enemies.splice(k, 1);
        groundHP--;
        shake = 10;
        boom(en.x, GROUND_Y, C.groundTop, 14);
        if (groundHP <= 0) { state = 'over'; }
      }
    }

    // 道具
    for (var d = drops.length - 1; d >= 0; d--) {
      var co = drops[d];
      co.y += co.vy;
      if (co.y + co.h >= GROUND_Y - 6 && Math.abs(co.x - player.x) < 20) {
        drops.splice(d, 1);
        powerUntil = performance.now() + 5000;
        score += 5;
        boom(co.x, co.y, C.coffee, 8);
      } else if (co.y > H) {
        drops.splice(d, 1);
      }
    }

    // 粒子
    for (var p = particles.length - 1; p >= 0; p--) {
      var pt = particles[p];
      pt.x += pt.vx; pt.y += pt.vy; pt.vy += 0.06; pt.life--;
      if (pt.life <= 0) particles.splice(p, 1);
    }
    if (shake > 0) shake--;
  }

  function rect(x, y, w, h, color) {
    ctx.fillStyle = color;
    ctx.fillRect(Math.round(x), Math.round(y), w, h);
  }

  function drawEnemy(e) {
    var col = e.type === 'bug' ? C.bug : (e.type === 'dart' ? C.dart : C.heavy);
    rect(e.x - e.w / 2, e.y, e.w, e.h, col);
    if (e.type === 'bug') { rect(e.x - e.w / 2 - 2, e.y + 2, 2, 3, col); rect(e.x + e.w / 2, e.y + 2, 2, 3, col); }
    if (e.type === 'heavy') { rect(e.x - 2, e.y - 3, 4, 3, col); }
  }

  function draw() {
    ctx.save();
    if (shake > 0) ctx.translate((Math.random() - 0.5) * 4, (Math.random() - 0.5) * 4);

    rect(0, 0, W, H, C.bg);
    // 星空
    for (var s = 0; s < 26; s++) {
      var sx = (s * 197) % W, sy = (s * 89) % (GROUND_Y - 60);
      rect(sx, sy, 1, 1, '#26324a');
    }
    // 地面
    rect(0, GROUND_Y, W, H - GROUND_Y, C.ground);
    rect(0, GROUND_Y, W, 3, C.groundTop);

    // 道具（咖啡）
    for (var d = 0; d < drops.length; d++) {
      var co = drops[d];
      rect(co.x - 5, co.y, 10, 8, C.coffee);
      rect(co.x + 5, co.y + 2, 3, 4, C.coffee);
      rect(co.x - 1, co.y - 4, 2, 3, '#f5f7fa');
    }

    // 敌人 / 子弹 / 玩家
    for (var i = 0; i < enemies.length; i++) drawEnemy(enemies[i]);
    for (var b = 0; b < bullets.length; b++) rect(bullets[b].x - 1, bullets[b].y, 2, 6, C.bullet);
    if (state !== 'over') {
      rect(player.x - 13, GROUND_Y - 12, 26, 8, C.player);       // 车体
      rect(player.x - 2, GROUND_Y - 18, 4, 6, C.player);         // 炮管
      rect(player.x - 9, GROUND_Y - 16, 18, 3, '#3ba7a0');       // 舱盖
    }

    // 粒子
    for (var p = 0; p < particles.length; p++) {
      var pt = particles[p];
      ctx.globalAlpha = Math.max(0, pt.life / 30);
      rect(pt.x, pt.y, 2, 2, pt.color);
    }
    ctx.globalAlpha = 1;

    // HUD
    ctx.font = '10px monospace';
    ctx.fillStyle = C.hud;
    ctx.textAlign = 'left';
    ctx.fillText('SCORE ' + score, 8, 16);
    ctx.fillText('WAVE ' + wave, 8, 30);
    ctx.textAlign = 'right';
    var hpText = '';
    for (var g = 0; g < groundHP; g++) hpText += '▮';
    ctx.fillStyle = groundHP <= 2 ? C.bug : C.groundTop;
    ctx.fillText('地面 ' + hpText, W - 8, 16);
    if (performance.now() < powerUntil) {
      ctx.fillStyle = C.coffee;
      ctx.fillText('☕ 三连射', W - 8, 30);
    }

    // 标题 / 结束画面
    ctx.textAlign = 'center';
    if (state === 'title') {
      ctx.fillStyle = 'rgba(6,10,16,.72)';
      ctx.fillRect(0, 0, W, H);
      ctx.font = 'bold 26px monospace';
      ctx.fillStyle = C.player;
      ctx.fillText('地 面 保 卫 军', W / 2, H / 2 - 52);
      ctx.font = '11px monospace';
      ctx.fillStyle = C.hud;
      ctx.fillText('GROUND DEFENSE FORCE', W / 2, H / 2 - 32);
      ctx.fillText('← → / A D 移动 · 空格 / 点击 射击', W / 2, H / 2 + 2);
      ctx.fillText('别让敌人踏上地面 · 接住 ☔ 咖啡获得三连射', W / 2, H / 2 + 20);
      ctx.fillStyle = C.groundTop;
      ctx.fillText('— 点击或按空格开始 —', W / 2, H / 2 + 52);
    }
    if (state === 'over') {
      ctx.fillStyle = 'rgba(6,10,16,.78)';
      ctx.fillRect(0, 0, W, H);
      ctx.font = 'bold 22px monospace';
      ctx.fillStyle = C.bug;
      ctx.fillText('防 线 失 守', W / 2, H / 2 - 36);
      ctx.font = '12px monospace';
      ctx.fillStyle = C.hud;
      ctx.fillText('得分 ' + score + ' · 波次 ' + wave + ' · 击落 ' + kills, W / 2, H / 2 - 8);
      ctx.fillStyle = C.groundTop;
      ctx.fillText('按 R 或点击重新开始', W / 2, H / 2 + 24);
    }
    ctx.restore();
  }

  function loop(t) {
    if (!rafId) return;
    var dt = Math.min(50, t - lastT || 16);
    lastT = t;
    if (state === 'playing') {
      // 以 16ms 为一步做固定步进，简单起见按帧累计
      update();
    }
    draw();
    rafId = requestAnimationFrame(loop);
  }

  function openGame() {
    reset();
    state = 'title';
    overlay.style.display = 'block';
    document.body.style.overflow = 'hidden';
    if (!rafId) { lastT = 0; rafId = requestAnimationFrame(loop); }
  }

  function closeGame() {
    overlay.style.display = 'none';
    document.body.style.overflow = '';
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    state = 'title';
  }

  function startOrRestart() {
    if (state === 'title' || state === 'over') {
      reset();
      state = 'playing';
    }
  }

  // ---- 输入 ----
  document.addEventListener('keydown', function (e) {
    if (overlay.style.display !== 'block') return;
    if (['ArrowLeft', 'ArrowRight', ' ', 'ArrowUp'].indexOf(e.key) >= 0) e.preventDefault();
    if (e.key === 'Escape') { closeGame(); return; }
    if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') keys.left = true;
    if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') keys.right = true;
    if (e.key === ' ' || e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') {
      if (state !== 'playing') startOrRestart(); else keys.fire = true;
    }
    if ((e.key === 'r' || e.key === 'R') && state === 'over') startOrRestart();
  });
  document.addEventListener('keyup', function (e) {
    if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') keys.left = false;
    if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') keys.right = false;
    if (e.key === ' ' || e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') keys.fire = false;
  });

  function canvasX(ev) {
    var r = canvas.getBoundingClientRect();
    var cx = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
    return cx * (W / r.width);
  }
  canvas.addEventListener('mousemove', function (ev) {
    if (state === 'playing') player.x = Math.max(18, Math.min(W - 18, canvasX(ev)));
  });
  canvas.addEventListener('mousedown', function () {
    if (state !== 'playing') startOrRestart(); else fire();
  });
  canvas.addEventListener('touchmove', function (ev) {
    ev.preventDefault();
    if (state === 'playing') player.x = Math.max(18, Math.min(W - 18, canvasX(ev)));
  }, { passive: false });
  canvas.addEventListener('touchstart', function (ev) {
    ev.preventDefault();
    if (state !== 'playing') startOrRestart(); else { player.x = canvasX(ev); fire(); }
  }, { passive: false });

  var closeBtn = document.getElementById('game-close');
  if (closeBtn) closeBtn.addEventListener('click', closeGame);

  // ---- 触发：咖啡杯 2 秒内三连击 ----
  var mug = document.querySelector('.pixel-mug');
  if (mug) {
    var times = [];
    mug.addEventListener('click', function (e) {
      e.stopPropagation();
      var now = Date.now();
      times.push(now);
      times = times.filter(function (t) { return now - t < 2000; });
      if (times.length >= 3) {
        times = [];
        openGame();
      }
    });
  }
})();
