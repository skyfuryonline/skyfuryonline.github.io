// games/flappy.js — Flappy 咖啡杯
ARCADE.register({
  id: 'flappy',
  cn: 'Flappy 咖啡杯',
  en: 'FLAPPY MUG · 2013',
  create: function (api) {
    var ctx = api.ctx, W = api.W, H = api.H;
    var C = {
      skyTop: '#1a2536', skyBot: '#2c3e58',
      pipe: '#3d6b4f', pipeHi: '#5a9e73', pipeDark: '#2a4c39',
      cup: '#f5e9d6', coffee: '#6b4226', steam: '#b9c7d9',
      wing: '#ff9f64', hud: '#8899bb', gold: '#ffd166', bug: '#e06c75'
    };
    var GROUND = H - 36;
    var GAP0 = 130, GAPMIN = 98, SPACING = 175;

    var bird, pipes, parts, groundX, bgBeans;
    var score, best, dead, started, frame, flapAnim;

    function reset() {
      bird = { x: 110, y: H / 2, vy: 0, rot: 0 };
      pipes = []; parts = [];
      groundX = 0; frame = 0; flapAnim = 0;
      score = 0; dead = false; started = false;
      bgBeans = [];
      for (var i = 0; i < 8; i++)
        bgBeans.push({ x: Math.random() * W, y: 40 + Math.random() * (GROUND - 120), r: 2 + Math.random() * 3, s: 0.18 + Math.random() * 0.2 });
    }
    function flap() {
      if (dead) return;
      started = true;
      bird.vy = -5.4;
      flapAnim = 12;
      for (var i = 0; i < 3; i++)
        parts.push({ x: bird.x - 8, y: bird.y + 6, vx: -1 - Math.random(), vy: Math.random() * 1.2 - 0.4, life: 18, color: C.steam });
    }
    function boom(x, y, n) {
      for (var i = 0; i < n; i++)
        parts.push({ x: x, y: y, vx: (Math.random() - 0.5) * 3.4, vy: (Math.random() - 0.7) * 3, life: 24 + Math.random() * 12, color: Math.random() < 0.5 ? C.cup : C.coffee });
    }

    api.panel([['SPACE/点击', '扇翅膀'], ['P', '暂停'], ['ESC', '片库']],
      '机台秘技：越飞越快越窄，12 分后管道开始漂移——注意 HUD 上的速度倍率');
    function onKey(k, down) {
      if (down && (k === ' ' || k === 'ArrowUp' || k === 'w' || k === 'W')) {
        if (!dead) flap();
      }
    }
    function onPointer(type, x, y) {
      if (type === 'down' && !dead) flap();
    }

    function gapSize() { return Math.max(GAPMIN, GAP0 - score * 1.2); }
    function speed() { return 2.1 + Math.min(1.7, score * 0.045); }

    function update() {
      frame++;
      groundX = (groundX - (started ? speed() : 1.2)) % 24;
      for (var i = 0; i < bgBeans.length; i++) {
        bgBeans[i].x -= bgBeans[i].s * (started ? 1.6 : 0.6);
        if (bgBeans[i].x < -6) { bgBeans[i].x = W + 6; bgBeans[i].y = 40 + Math.random() * (GROUND - 120); }
      }
      if (flapAnim > 0) flapAnim--;

      if (!started) { bird.y = H / 2 + Math.sin(frame / 22) * 8; }
      if (!dead && started) {
        bird.vy += 0.3;
        bird.vy = Math.min(8, bird.vy);
        bird.y += bird.vy;
        bird.rot = Math.max(-0.5, Math.min(1.2, bird.vy / 9));

        // 管道
        if (frame % Math.round(SPACING / speed() * 2.1) === 0 || (pipes.length === 0 && frame > 30)) {
          var g = gapSize();
          var top = 44 + Math.random() * (GROUND - g - 108);
          pipes.push({
            x: W + 30, top: top, baseTop: top, g: g, passed: false,
            amp: score >= 12 ? Math.min(30, (score - 12) * 1.2) : 0, // 12 分后管道漂移
            phase: Math.random() * 6.28
          });
        }
        for (var p = pipes.length - 1; p >= 0; p--) {
          var pi = pipes[p];
          pi.x -= speed();
          if (pi.amp > 0) {
            pi.top = pi.baseTop + Math.sin(frame * 0.03 + pi.phase) * pi.amp;
            pi.top = Math.max(24, Math.min(GROUND - pi.g - 24, pi.top));
          }
          if (pi.x < -64) { pipes.splice(p, 1); continue; }
          if (!pi.passed && pi.x + 28 < bird.x) {
            pi.passed = true; score++;
          }
          // 碰撞：杯体 22×18，管道 28 宽
          var inX = bird.x + 10 > pi.x && bird.x - 10 < pi.x + 28;
          if (inX && (bird.y - 8 < pi.top || bird.y + 9 > pi.top + pi.g)) {
            die();
          }
        }
        if (bird.y + 9 >= GROUND || bird.y - 8 < 0) die();
      }
      // 粒子
      for (var q = parts.length - 1; q >= 0; q--) {
        var pt = parts[q];
        pt.x += pt.vx; pt.y += pt.vy; pt.vy += 0.04; pt.life--;
        if (pt.life <= 0) parts.splice(q, 1);
      }
    }
    function die() {
      if (dead) return;
      dead = true;
      boom(bird.x, bird.y, 12);
      setTimeout(function () { if (dead) api.gameOver(score); }, 900);
    }

    function rect(x, y, w, h, c) { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), w, h); }
    function drawCup() {
      ctx.save();
      ctx.translate(bird.x, bird.y);
      ctx.rotate(bird.rot);
      // 翅膀
      var wingY = flapAnim > 0 ? -4 : 3;
      rect(-16, wingY, 10, 6, C.wing);
      rect(-19, wingY + 1, 5, 4, C.wing);
      // 杯身
      rect(-10, -9, 18, 17, C.cup);
      rect(8, -5, 5, 8, C.cup);
      rect(-8, -7, 14, 5, C.coffee);
      // 高光 + 眼睛
      rect(-6, 5, 3, 5, '#fff');
      rect(2, -4, 3, 3, '#222');
      // 蒸汽
      if ((frame >> 3) % 3 !== 0) {
        rect(-3, -14, 2, 4, C.steam);
        rect(2, -13, 2, 3, C.steam);
      }
      ctx.restore();
    }
    function drawPipe(p) {
      var botY = p.top + p.g;
      // 上管
      rect(p.x, 0, 28, p.top - 14, C.pipe);
      rect(p.x, p.top - 14, 28, 14, C.pipeDark);
      rect(p.x - 3, p.top - 14, 34, 14, C.pipe);
      rect(p.x - 3, p.top - 14, 34, 3, C.pipeHi);
      // 蒸汽从管口冒出
      if ((frame >> 4) % 2 === 0) rect(p.x + 8, p.top - 20, 4, 5, C.steam);
      // 下管
      rect(p.x, botY + 14, 28, GROUND - botY - 14, C.pipe);
      rect(p.x, botY, 28, 14, C.pipeDark);
      rect(p.x - 3, botY, 34, 14, C.pipe);
      rect(p.x - 3, botY + 11, 34, 3, C.pipeHi);
      if ((frame >> 4) % 2 === 1) rect(p.x + 16, botY + 15, 4, 5, C.steam);
    }
    function draw() {
      // 天空
      var g = ctx.createLinearGradient(0, 0, 0, H);
      g.addColorStop(0, C.skyTop); g.addColorStop(1, C.skyBot);
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
      // 星
      for (var s = 0; s < 24; s++) {
        ctx.fillStyle = 'rgba(255,255,255,.35)';
        var sx = (s * 173 + 40) % W, sy = (s * 79) % (GROUND - 60);
        if ((frame + s * 13) % 97 > 10) ctx.fillRect(sx, sy, 1, 1);
      }
      // 背景咖啡豆
      for (var b = 0; b < bgBeans.length; b++) {
        var be = bgBeans[b];
        ctx.fillStyle = 'rgba(120,80,50,.35)';
        ctx.beginPath(); ctx.ellipse(be.x, be.y, be.r + 1, be.r, 0.6, 0, 7); ctx.fill();
      }
      for (var p = 0; p < pipes.length; p++) drawPipe(pipes[p]);
      // 地面
      rect(0, GROUND, W, H - GROUND, '#20293a');
      rect(0, GROUND, W, 4, '#39506e');
      for (var tx = groundX; tx < W; tx += 24) rect(tx, GROUND + 10, 12, 3, '#2c3a54');
      if (!dead) drawCup();
      // 粒子
      for (var q = 0; q < parts.length; q++) {
        ctx.globalAlpha = Math.max(0, parts[q].life / 28);
        rect(parts[q].x, parts[q].y, 2, 2, parts[q].color);
      }
      ctx.globalAlpha = 1;
      // HUD
      ctx.textAlign = 'center';
      ctx.font = 'bold 26px monospace';
      ctx.fillStyle = '#f5f7fa';
      ctx.strokeStyle = 'rgba(0,0,0,.5)'; ctx.lineWidth = 4;
      ctx.strokeText('' + score, W / 2, 44);
      ctx.fillText('' + score, W / 2, 44);
      ctx.font = '10px monospace';
      ctx.fillStyle = C.hud;
      ctx.textAlign = 'left';
      ctx.fillText('SPD ×' + (speed() / 2.1).toFixed(1), 10, 20);
      if (!started) {
        ctx.font = '11px monospace'; ctx.fillStyle = C.hud;
        ctx.fillText('SPACE / 点击 扇动翅膀', W / 2, H / 2 - 60);
        ctx.fillStyle = ((Date.now() >> 9) % 2) ? '#3fae5a' : C.hud;
        ctx.fillText('— 准备好了就起飞 —', W / 2, H / 2 - 42);
      }
    }

    reset();
    return { update: update, draw: draw, onKey: onKey, onPointer: onPointer };
  }
});
