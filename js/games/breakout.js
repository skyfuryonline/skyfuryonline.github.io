// games/breakout.js — 打砖块 · 咖啡特调
ARCADE.register({
  id: 'breakout',
  cn: '打砖块',
  en: 'COFFEE BREAKOUT · 1976',
  create: function (api) {
    var ctx = api.ctx, W = api.W, H = api.H;
    var C = {
      bg: '#0d1420', hud: '#8899bb', paddle: '#e8c15a', ball: '#f5f7fa',
      rows: ['#e06c75', '#e5c07b', '#3fae5a', '#5aa7c0', '#a06cc0', '#ff9f64'],
      power: '#ff9f64', gold: '#ffd166'
    };
    var BW = 40, BH = 14, COLS = 12, TOP = 52;

    var paddle, balls, bricks, drops, parts;
    var score, lives, level, speed, launched, combo, flash;

    var LEVELS = [
      // 1: 满阵
      function (b) { for (var r = 0; r < 5; r++) for (var c = 0; c < COLS; c++) b[r][c] = 1; },
      // 2: 棋盘
      function (b) { for (var r = 0; r < 6; r++) for (var c = 0; c < COLS; c++) b[r][c] = (r + c) % 2 ? 1 : 0; },
      // 3: 金字塔
      function (b) { for (var r = 0; r < 6; r++) for (var c = 0; c < COLS; c++) b[r][c] = (c >= r && c < COLS - r) ? 1 : 0; },
      // 4: 双塔
      function (b) { for (var r = 0; r < 6; r++) for (var c = 0; c < COLS; c++) b[r][c] = (c < 3 || c > COLS - 4 || (r > 2 && c > 4 && c < 7)) ? 1 : 0; }
    ];

    function reset(full) {
      if (full) { score = 0; lives = 3; level = 0; }
      speed = 2.6 + level * 0.35;
      paddle = { x: W / 2, w: 64, until: 0 };
      balls = [];
      bricks = [];
      for (var r = 0; r < 7; r++) { bricks.push(new Array(COLS).fill(0)); }
      LEVELS[level % LEVELS.length](bricks);
      drops = []; parts = []; combo = 0; flash = 0;
      newBall();
    }
    function newBall() {
      balls.push({ x: paddle.x, y: H - 40, vx: 0, vy: 0, stuck: true });
      launched = false;
    }
    function launch() {
      for (var i = 0; i < balls.length; i++)
        if (balls[i].stuck) {
          balls[i].stuck = false;
          var a = -Math.PI / 2 + (Math.random() - 0.5) * 0.7;
          balls[i].vx = Math.cos(a) * speed; balls[i].vy = Math.sin(a) * speed;
        }
      launched = true;
    }
    function boom(x, y, color, n) {
      for (var i = 0; i < n; i++) parts.push({
        x: x, y: y, vx: (Math.random() - 0.5) * 2.6, vy: -Math.random() * 2.4,
        life: 20 + Math.random() * 10, color: color
      });
    }
    function brickLeft() {
      var n = 0;
      for (var r = 0; r < bricks.length; r++) for (var c = 0; c < COLS; c++) if (bricks[r][c]) n++;
      return n;
    }
    function dropPower(c, r) {
      if (Math.random() > 0.22) return;
      var types = ['wide', 'multi', 'slow', 'life'];
      drops.push({ x: c * BW + BW / 2, y: r * BH + TOP, t: types[Math.floor(Math.random() * types.length)] });
    }

    api.panel([['←→/鼠标', '接球'], ['SPACE', '发球'], ['P', '暂停'], ['ESC', '片库']],
      '机台秘技：接住掉落的咖啡豆 — 加宽、三球、减速、加命');
    var keys = {};
    function onKey(k, down) {
      if (k === 'ArrowLeft' || k === 'a' || k === 'A') keys.left = down;
      if (k === 'ArrowRight' || k === 'd' || k === 'D') keys.right = down;
      if (k === ' ' && down) { if (!launched) launch(); }
    }
    function onPointer(type, x) {
      if (type === 'move') paddle.x = Math.max(paddle.w / 2, Math.min(W - paddle.w / 2, x));
      else if (type === 'down' && !launched) launch();
    }

    function update() {
      // 挡板
      var pw = performance.now() < paddle.until ? 96 : 64;
      if (pw !== paddle.w) { paddle.w = pw; paddle.x = Math.max(pw / 2, Math.min(W - pw / 2, paddle.x)); }
      var spd = 5.2;
      if (keys.left) paddle.x -= spd;
      if (keys.right) paddle.x += spd;
      paddle.x = Math.max(paddle.w / 2, Math.min(W - paddle.w / 2, paddle.x));

      // 球
      for (var i = balls.length - 1; i >= 0; i--) {
        var b = balls[i];
        if (b.stuck) { b.x = paddle.x; b.y = H - 40; continue; }
        b.x += b.vx; b.y += b.vy;
        if (b.x < 5) { b.x = 5; b.vx = Math.abs(b.vx); }
        if (b.x > W - 5) { b.x = W - 5; b.vx = -Math.abs(b.vx); }
        if (b.y < 5) { b.y = 5; b.vy = Math.abs(b.vy); }
        // 挡板
        var py = H - 28;
        if (b.vy > 0 && b.y > py - 5 && b.y < py + 8 && Math.abs(b.x - paddle.x) < paddle.w / 2 + 4) {
          var rel = (b.x - paddle.x) / (paddle.w / 2);
          var ang = -Math.PI / 2 + rel * 1.05;
          var sp = Math.min(6.5, Math.hypot(b.vx, b.vy) * 1.012);
          b.vx = Math.cos(ang) * sp; b.vy = Math.sin(ang) * sp;
          b.y = py - 5;
          combo = 0;
        }
        // 掉落
        if (b.y > H + 8) {
          balls.splice(i, 1);
          if (!balls.length) {
            lives--;
            if (lives <= 0) { api.gameOver(score); return; }
            newBall();
          }
          continue;
        }
        // 砖块
        var r = Math.floor((b.y - TOP) / BH), c = Math.floor(b.x / BW);
        if (r >= 0 && r < bricks.length && c >= 0 && c < COLS && bricks[r][c]) {
          bricks[r][c] = 0;
          combo++;
          score += 50 + (combo - 1) * 10;
          boom(c * BW + BW / 2, r * BH + TOP + BH / 2, C.rows[r % C.rows.length], 6);
          dropPower(c, r);
          // 反弹方向：比较进入深度
          var cx = Math.abs(b.x - (c * BW + BW / 2)) / BW, cy = Math.abs(b.y - (r * BH + TOP + BH / 2)) / BH;
          if (cx > cy) b.vx = -b.vx; else b.vy = -b.vy;
          if (brickLeft() === 0) {
            level++;
            score += 500;
            reset(false);
            flash = 60;
            return;
          }
        }
      }

      // 道具
      for (var p = drops.length - 1; p >= 0; p--) {
        var dr = drops[p];
        dr.y += 1.1;
        if (dr.y > H) { drops.splice(p, 1); continue; }
        if (dr.y > H - 34 && Math.abs(dr.x - paddle.x) < paddle.w / 2 + 8) {
          drops.splice(p, 1);
          if (dr.t === 'wide') paddle.until = performance.now() + 12000;
          if (dr.t === 'life') lives = Math.min(6, lives + 1);
          if (dr.t === 'slow') for (var s = 0; s < balls.length; s++) { balls[s].vx *= 0.8; balls[s].vy *= 0.8; }
          if (dr.t === 'multi' && balls.length) {
            var b0 = balls[0];
            for (var m = 0; m < 2; m++) {
              var a2 = Math.atan2(b0.vy, b0.vx) + (m ? 0.5 : -0.5);
              balls.push({ x: b0.x, y: b0.y, vx: Math.cos(a2) * speed, vy: Math.sin(a2) * speed, stuck: false });
            }
          }
          boom(dr.x, dr.y, C.power, 6);
        }
      }

      // 粒子
      for (var q = parts.length - 1; q >= 0; q--) {
        var pt = parts[q];
        pt.x += pt.vx; pt.y += pt.vy; pt.vy += 0.08; pt.life--;
        if (pt.life <= 0) parts.splice(q, 1);
      }
      if (flash > 0) flash--;
    }

    function rect(x, y, w, h, c) { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), w, h); }
    function draw() {
      rect(0, 0, W, H, C.bg);
      // 砖
      for (var r = 0; r < bricks.length; r++)
        for (var c = 0; c < COLS; c++)
          if (bricks[r][c]) {
            rect(c * BW + 1, TOP + r * BH + 1, BW - 2, BH - 3, C.rows[r % C.rows.length]);
            rect(c * BW + 1, TOP + r * BH + 1, BW - 2, 3, 'rgba(255,255,255,.25)');
          }
      // 道具（咖啡豆）
      for (var p = 0; p < drops.length; p++) {
        var dr = drops[p];
        rect(dr.x - 4, dr.y - 5, 8, 10, C.power);
        rect(dr.x - 1, dr.y - 6, 2, 12, '#7a4020');
      }
      // 球
      for (var i = 0; i < balls.length; i++) {
        var b = balls[i];
        ctx.fillStyle = C.ball;
        ctx.beginPath(); ctx.arc(b.x, b.y, 4.5, 0, 7); ctx.fill();
      }
      // 挡板（拉花样式）
      rect(paddle.x - paddle.w / 2, H - 28, paddle.w, 9, C.paddle);
      rect(paddle.x - paddle.w / 2, H - 28, paddle.w, 3, '#f7e2ae');
      // 粒子
      for (var q = 0; q < parts.length; q++) {
        ctx.globalAlpha = Math.max(0, parts[q].life / 24);
        rect(parts[q].x, parts[q].y, 2, 2, parts[q].color);
      }
      ctx.globalAlpha = 1;
      // HUD
      ctx.font = '10px monospace'; ctx.fillStyle = C.hud; ctx.textAlign = 'left';
      ctx.fillText('SCORE ' + score, 8, 16);
      ctx.fillText('LEVEL ' + (level + 1), 8, 30);
      ctx.textAlign = 'right';
      var life = ''; for (var l = 0; l < lives; l++) life += '▮';
      ctx.fillStyle = C.paddle;
      ctx.fillText('命 ' + (life || '—'), W - 8, 16);
      if (combo > 1) { ctx.fillStyle = C.gold; ctx.fillText('COMBO ×' + combo, W - 8, 30); }
      if (!launched) {
        ctx.textAlign = 'center';
        ctx.fillStyle = ((Date.now() >> 9) % 2) ? '#3fae5a' : C.hud;
        ctx.fillText('← → 移动 · SPACE 发球', W / 2, H - 52);
      }
      if (flash > 0) {
        ctx.fillStyle = 'rgba(255,209,102,' + (flash / 120) + ')';
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = C.gold; ctx.textAlign = 'center'; ctx.font = 'bold 18px monospace';
        ctx.fillText('LEVEL ' + (level + 1), W / 2, H / 2);
        ctx.font = '10px monospace';
      }
    }

    reset(true);
    return { update: update, draw: draw, onKey: onKey, onPointer: onPointer };
  }
});
