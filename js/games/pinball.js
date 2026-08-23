// games/pinball.js — 3D 弹珠台 · 太空军校生（2D 物理模拟 + 伪立体渲染）
ARCADE.register({
  id: 'pinball',
  cn: '3D 弹珠台',
  en: 'SPACE CADET PINBALL',
  create: function (api) {
    var ctx = api.ctx, W = api.W, H = api.H;
    var C = {
      bg: '#0a0e1a', table: '#141b30', rail: '#6a7ba0', railHi: '#9fb3d9',
      ball: '#e8ecf4', ballHi: '#ffffff',
      bumper: '#e06c75', bumperHi: '#ff9aa2', bumperGlow: 'rgba(224,108,117,.35)',
      sling: '#e5c07b', target: '#3fae5a', targetOn: '#7be39a',
      flipper: '#c47bd6', flipperHi: '#e3aef0',
      text: '#8899bb', gold: '#ffd166', lane: '#1d2742'
    };
    var GRAV = 0.045, REST = 0.62, BALL_R = 6;

    var ball, balls, score, lives, ballsLeft;
    var flippers, bumpers, slings, targets, plunger;
    var parts, pops, litLane, frame, launched, charge, dead;
    var keys = {};

    // ---------- 台面几何 ----------
    var WALL_L = 14, WALL_R = W - 14, WALL_T = 14;
    var BASE_Y = H - 78; // 弹板转轴高度
    function buildTable() {
      flippers = {
        left: { px: W / 2 - 74, py: BASE_Y, angle: 0.42, len: 62, side: -1, pressed: false, va: 0 },
        right: { px: W / 2 + 74, py: BASE_Y, angle: Math.PI - 0.42, len: 62, side: 1, pressed: false, va: 0 }
      };
      bumpers = [
        { x: W / 2 - 62, y: 92, r: 17, glow: 0, pts: 150 },
        { x: W / 2, y: 64, r: 17, glow: 0, pts: 150 },
        { x: W / 2 + 62, y: 92, r: 17, glow: 0, pts: 150 }
      ];
      slings = [
        { a: { x: W / 2 - 96, y: 190 }, b: { x: W / 2 - 44, y: 258 }, glow: 0 },
        { a: { x: W / 2 + 96, y: 190 }, b: { x: W / 2 + 44, y: 258 }, glow: 0 }
      ];
      targets = [];
      var letters = 'COFFEE';
      for (var i = 0; i < letters.length; i++)
        targets.push({ x: WALL_L + 26 + (i % 3) * 26, y: 150 + Math.floor(i / 3) * 26, w: 18, h: 12, ch: letters[i], on: false });
      plunger = { x: 0, y: 0 };
    }

    function resetBall() {
      balls = [{ x: W - 34, y: H - 150, vx: 0, vy: 0, inLane: true }];
      launched = false; charge = 0;
    }
    function fullReset() {
      score = 0; ballsLeft = 3; frame = 0; dead = false;
      parts = []; pops = []; litLane = -1;
      buildTable();
      resetBall();
    }

    api.panel([['← → / Z M', '弹板'], ['SPACE', '按住蓄力发射'], ['P', '暂停'], ['ESC', '片库']],
      '机台秘技：点亮 COFFEE 六个字母，奖励 5000 分');
    function onKey(k, down) {
      if (k === 'ArrowLeft' || k === 'z' || k === 'Z' || k === 'a' || k === 'A') flippers.left.pressed = down;
      if (k === 'ArrowRight' || k === 'm' || k === 'M' || k === 'd' || k === 'D') flippers.right.pressed = down;
      if (k === ' ' && down && !launched) charge = Math.min(60, charge + 1);
      if (k === ' ' && !down && !launched) launch();
    }
    function onPointer(type, x, y) {
      if (type === 'down') {
        if (!launched) { charge = Math.min(60, charge + 1); launch(); return; }
        if (x < W / 2) flippers.left.pressed = true;
        else flippers.right.pressed = true;
      }
    }

    function launch() {
      if (launched || !balls.length) return;
      var b = balls[0];
      if (b.inLane) {
        b.vy = -(6.2 + charge * 0.11);
        b.vx = 0;
        b.inLane = false;
        launched = true;
      }
      charge = 0;
    }
    function pop(x, y, text, color) { pops.push({ x: x, y: y, text: text, color: color || C.gold, life: 46 }); }
    function boom(x, y, n, color) {
      for (var i = 0; i < n; i++) parts.push({
        x: x, y: y, vx: (Math.random() - 0.5) * 3, vy: (Math.random() - 0.6) * 3,
        life: 16 + Math.random() * 12, color: color
      });
    }
    function allTargetsLit() {
      for (var i = 0; i < targets.length; i++) if (!targets[i].on) return false;
      return true;
    }

    // ---------- 碰撞辅助 ----------
    function circleSeg(b, ax, ay, bx, by, rad) {
      // 返回 {push, nx, ny, dist} 或 null
      var dx = bx - ax, dy = by - ay;
      var L2 = dx * dx + dy * dy;
      var t = ((b.x - ax) * dx + (b.y - ay) * dy) / L2;
      t = Math.max(0, Math.min(1, t));
      var cx = ax + t * dx, cy = ay + t * dy;
      var ox = b.x - cx, oy = b.y - cy;
      var d2 = ox * ox + oy * oy;
      if (d2 >= rad * rad || d2 === 0) return null;
      var d = Math.sqrt(d2);
      return { push: rad - d, nx: ox / d, ny: oy / d, cx: cx, cy: cy };
    }
    function bounce(b, hit, rest, extra) {
      b.x += hit.nx * hit.push; b.y += hit.ny * hit.push;
      var vn = b.vx * hit.nx + b.vy * hit.ny;
      if (vn < 0) {
        b.vx -= (1 + rest) * vn * hit.nx;
        b.vy -= (1 + rest) * vn * hit.ny;
      }
      if (extra) { b.vx += hit.nx * extra; b.vy += hit.ny * extra; }
    }
    function flipperTip(f) {
      var a = f.angle;
      return { x: f.px + Math.cos(a) * f.len, y: f.py + Math.sin(a) * f.len };
    }

    function update() {
      frame++;
      if (dead) return;

      // 弹板动画
      ['left', 'right'].forEach(function (s) {
        var f = flippers[s];
        var targetA = f.side === -1
          ? (f.pressed ? -0.62 : 0.42)
          : (f.pressed ? Math.PI + 0.62 : Math.PI - 0.42);
        var old = f.angle;
        f.angle += (targetA - f.angle) * 0.4;
        f.va = (f.angle - old);
      });

      // 发射槽蓄力显示
      if (!launched && balls.length && balls[0].inLane) {
        // 球随蓄力下压
        balls[0].y = H - 150 + Math.min(26, charge * 0.45);
      }

      for (var bi = balls.length - 1; bi >= 0; bi--) {
        var b = balls[bi];
        if (b.inLane) continue;

        b.vy += GRAV;
        b.vx *= 0.9985;
        b.x += b.vx; b.y += b.vy;

        // 外墙
        if (b.x - BALL_R < WALL_L) { b.x = WALL_L + BALL_R; b.vx = Math.abs(b.vx) * REST; }
        if (b.x + BALL_R > WALL_R) { b.x = WALL_R - BALL_R; b.vx = -Math.abs(b.vx) * REST; }
        if (b.y - BALL_R < WALL_T) { b.y = WALL_T + BALL_R; b.vy = Math.abs(b.vy) * REST; }

        // 顶部弧形导轨（把球导回台面）
        var arcCx = W / 2, arcCy = WALL_T + 240, arcR = 206;
        var dox = b.x - arcCx, doy = b.y - arcCy;
        var dd = Math.sqrt(dox * dox + doy * doy);
        if (b.y < 250 && dd > arcR - BALL_R) {
          var nx = -dox / dd, ny = -doy / dd;
          b.x = arcCx - nx * (arcR - BALL_R); b.y = arcCy - ny * (arcR - BALL_R);
          var vn = b.vx * nx + b.vy * ny;
          if (vn < 0) { b.vx -= (1 + REST) * vn * nx; b.vy -= (1 + REST) * vn * ny; }
        }

        // 缓冲器
        for (var u = 0; u < bumpers.length; u++) {
          var bp = bumpers[u];
          var ddx = b.x - bp.x, ddy = b.y - bp.y;
          var dist = Math.sqrt(ddx * ddx + ddy * ddy);
          if (dist < bp.r + BALL_R) {
            var n2 = dist || 1;
            var nx2 = ddx / n2, ny2 = ddy / n2;
            b.x = bp.x + nx2 * (bp.r + BALL_R);
            b.y = bp.y + ny2 * (bp.r + BALL_R);
            b.vx = nx2 * 3.4; b.vy = ny2 * 3.4 - 0.8;
            bp.glow = 14;
            score += bp.pts;
            pop(bp.x, bp.y - 24, '+' + bp.pts);
            boom(b.x, b.y, 6, C.bumperHi);
          }
        }
        // 三连击奖励
        if (frame % 30 === 0 && bumpers[0].glow + bumpers[1].glow + bumpers[2].glow > 30) {
          score += 300; pop(W / 2, 130, 'COMBO +300');
        }

        // 打弹器（三角弹射）
        for (var sg = 0; sg < slings.length; sg++) {
          var s = slings[sg];
          var hit = circleSeg(b, s.a.x, s.a.y, s.b.x, s.b.y, BALL_R + 2);
          if (hit) {
            bounce(b, hit, REST, 3.4);
            s.glow = 12;
            score += 25;
          }
        }

        // 目标块（COFFEE）
        for (var t = 0; t < targets.length; t++) {
          var tg = targets[t];
          if (b.x > tg.x - BALL_R && b.x < tg.x + tg.w + BALL_R &&
              b.y > tg.y - BALL_R && b.y < tg.y + tg.h + BALL_R) {
            // 从哪个面弹开
            var ox = b.x - (tg.x + tg.w / 2), oy = b.y - (tg.y + tg.h / 2);
            if (Math.abs(ox) / (tg.w / 2 + BALL_R) > Math.abs(oy) / (tg.h / 2 + BALL_R)) {
              b.vx = Math.abs(b.vx) * (ox > 0 ? 1 : -1) + (ox > 0 ? 1.2 : -1.2);
              b.x = tg.x + (ox > 0 ? tg.w + BALL_R : -BALL_R);
            } else {
              b.vy = Math.abs(b.vy) * (oy > 0 ? 1 : -1) + (oy > 0 ? 1.2 : -1.2);
              b.y = tg.y + (oy > 0 ? tg.h + BALL_R : -BALL_R);
            }
            if (!tg.on) {
              tg.on = true;
              score += 200;
              pop(tg.x + tg.w / 2, tg.y - 10, tg.ch + ' +200', C.targetOn);
              boom(b.x, b.y, 5, C.targetOn);
              if (allTargetsLit()) {
                score += 5000;
                pop(W / 2, 190, 'COFFEE! +5000');
                for (var tt = 0; tt < targets.length; tt++) targets[tt].on = false;
              }
            }
          }
        }

        // 弹板（旋转胶囊）
        ['left', 'right'].forEach(function (s2) {
          var f = flippers[s2];
          var tip = flipperTip(f);
          var hit2 = circleSeg(b, f.px, f.py, tip.x, tip.y, BALL_R + 5);
          if (hit2) {
            bounce(b, hit2, 0.3, 0);
            // 弹板角速度甩球
            var kick = Math.abs(f.va) * 90;
            b.vx += hit2.nx * kick;
            b.vy += hit2.ny * kick - Math.abs(f.va) * 30;
            // 限速
            var sp = Math.hypot(b.vx, b.vy);
            if (sp > 9.5) { b.vx *= 9.5 / sp; b.vy *= 9.5 / sp; }
          }
        });

        // 底部内导轨（防止从侧面漏球）
        var guideL = circleSeg(b, WALL_L, BASE_Y + 26, W / 2 - 96, BASE_Y + 4, BALL_R);
        if (guideL) bounce(b, guideL, 0.5, 0);
        var guideR = circleSeg(b, WALL_R, BASE_Y + 26, W / 2 + 96, BASE_Y + 4, BALL_R);
        if (guideR) bounce(b, guideR, 0.5, 0);

        // 发射道隔板物理（台内侧的球不能穿回发射道）
        if (b.x < W - 46) {
          var lane = circleSeg(b, W - 52, 260, W - 52, H - 44, BALL_R);
          if (lane) bounce(b, lane, 0.5, 0);
        }
        // 发射道底部托底：弱发射回落后允许再次蓄力
        if (b.x > W - 52 && b.y > H - 158 && b.vy > -0.5) {
          b.y = H - 158;
          b.vy = -Math.abs(b.vy) * REST;
          b.vx = 0;
          if (Math.abs(b.vy) < 0.6) { launched = false; b.vy = 0; }
        }

        // 漏球
        if (b.y > H + 16) {
          balls.splice(bi, 1);
          if (!balls.length) {
            ballsLeft--;
            if (ballsLeft <= 0) { dead = true; setTimeout(function () { api.gameOver(score); }, 700); return; }
            pop(W / 2, H / 2, 'BALL LOST', '#e06c75');
            resetBall();
            return;
          }
        }
      }

      // 特效衰减
      for (var p = parts.length - 1; p >= 0; p--) {
        var pt = parts[p];
        pt.x += pt.vx; pt.y += pt.vy; pt.vy += 0.06; pt.life--;
        if (pt.life <= 0) parts.splice(p, 1);
      }
      for (var q = pops.length - 1; q >= 0; q--) {
        pops[q].y -= 0.5; pops[q].life--;
        if (pops[q].life <= 0) pops.splice(q, 1);
      }
      for (var u2 = 0; u2 < bumpers.length; u2++) if (bumpers[u2].glow > 0) bumpers[u2].glow--;
      for (var s3 = 0; s3 < slings.length; s3++) if (slings[s3].glow > 0) slings[s3].glow--;
    }

    // ---------- 绘制 ----------
    function circle(x, y, r, c) {
      ctx.fillStyle = c;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
    }
    function draw() {
      ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);

      // 台面
      ctx.fillStyle = C.table;
      ctx.beginPath();
      ctx.moveTo(WALL_L, H);
      ctx.lineTo(WALL_L, 250);
      ctx.arc(W / 2, WALL_T + 240, 206, Math.PI, 0);
      ctx.lineTo(WALL_R, H);
      ctx.closePath();
      ctx.fill();
      // 台面网格光
      ctx.strokeStyle = 'rgba(90,120,200,.08)'; ctx.lineWidth = 1;
      for (var gy = 40; gy < H; gy += 32) { ctx.beginPath(); ctx.moveTo(WALL_L, gy); ctx.lineTo(WALL_R, gy); ctx.stroke(); }

      // 外墙
      ctx.strokeStyle = C.rail; ctx.lineWidth = 6;
      ctx.beginPath();
      ctx.moveTo(WALL_L + 3, H);
      ctx.lineTo(WALL_L + 3, 250);
      ctx.arc(W / 2, WALL_T + 240, 203, Math.PI, 0);
      ctx.lineTo(WALL_R - 3, H);
      ctx.stroke();

      // 发射通道
      ctx.strokeStyle = C.rail; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(W - 52, 258); ctx.lineTo(W - 52, H); ctx.stroke();
      if (charge > 0 && !launched) {
        ctx.fillStyle = 'rgba(255,209,102,' + (0.25 + charge / 90) + ')';
        ctx.fillRect(W - 49, H - 150 + Math.min(26, charge * 0.45), 32, 150);
        ctx.fillStyle = C.gold; ctx.font = '10px monospace'; ctx.textAlign = 'center';
        ctx.fillText('↑' + Math.round(charge / 60 * 100) + '%', W - 33, H - 168);
      }

      // 缓冲器
      for (var u = 0; u < bumpers.length; u++) {
        var bp = bumpers[u];
        if (bp.glow > 0) {
          ctx.fillStyle = C.bumperGlow;
          ctx.beginPath(); ctx.arc(bp.x, bp.y, bp.r + 10, 0, 7); ctx.fill();
        }
        circle(bp.x, bp.y, bp.r, '#5c2a35');
        circle(bp.x, bp.y, bp.r - 3, bp.glow > 0 ? C.bumperHi : C.bumper);
        circle(bp.x - 4, bp.y - 5, 3.5, 'rgba(255,255,255,.5)');
        ctx.fillStyle = '#fff'; ctx.font = 'bold 10px monospace'; ctx.textAlign = 'center';
        ctx.fillText('150', bp.x, bp.y + 4);
      }

      // 打弹器
      for (var s = 0; s < slings.length; s++) {
        var sl = slings[s];
        ctx.strokeStyle = sl.glow > 0 ? '#fff' : C.sling;
        ctx.lineWidth = 7; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(sl.a.x, sl.a.y); ctx.lineTo(sl.b.x, sl.b.y); ctx.stroke();
      }
      ctx.lineCap = 'butt';

      // COFFEE 目标块
      for (var t = 0; t < targets.length; t++) {
        var tg = targets[t];
        ctx.fillStyle = tg.on ? C.targetOn : C.target;
        ctx.fillRect(tg.x, tg.y, tg.w, tg.h);
        ctx.fillStyle = tg.on ? '#0a3a1a' : '#123f24';
        ctx.font = 'bold 9px monospace'; ctx.textAlign = 'center';
        ctx.fillText(tg.ch, tg.x + tg.w / 2, tg.y + tg.h - 3);
      }

      // 弹板
      ['left', 'right'].forEach(function (s2) {
        var f = flippers[s2];
        var tip = flipperTip(f);
        ctx.strokeStyle = C.flipper; ctx.lineWidth = 12; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(f.px, f.py); ctx.lineTo(tip.x, tip.y); ctx.stroke();
        ctx.strokeStyle = C.flipperHi; ctx.lineWidth = 4;
        ctx.beginPath(); ctx.moveTo(f.px, f.py - 2); ctx.lineTo(tip.x, tip.y - 2); ctx.stroke();
        circle(f.px, f.py, 5, C.railHi);
      });
      ctx.lineCap = 'butt';

      // 底部漏斗警示
      ctx.fillStyle = '#1c2338';
      ctx.beginPath();
      ctx.moveTo(W / 2 - 30, H); ctx.lineTo(W / 2 + 30, H);
      ctx.lineTo(W / 2 + 12, H - 14); ctx.lineTo(W / 2 - 12, H - 14);
      ctx.closePath(); ctx.fill();

      // 球（带高光与拖影）
      for (var bi = 0; bi < balls.length; bi++) {
        var b = balls[bi];
        if (b.vx || b.vy) circle(b.x - b.vx * 1.5, b.y - b.vy * 1.5, BALL_R - 2, 'rgba(232,236,244,.25)');
        circle(b.x, b.y, BALL_R, C.ball);
        circle(b.x - 2, b.y - 2, 2.2, C.ballHi);
      }

      // 粒子 / 分数飘字
      for (var p = 0; p < parts.length; p++) {
        ctx.globalAlpha = Math.max(0, parts[p].life / 24);
        ctx.fillStyle = parts[p].color;
        ctx.fillRect(parts[p].x, parts[p].y, 2, 2);
      }
      ctx.globalAlpha = 1;
      for (var q = 0; q < pops.length; q++) {
        var pp = pops[q];
        ctx.globalAlpha = Math.min(1, pp.life / 20);
        ctx.fillStyle = pp.color; ctx.font = 'bold 11px monospace'; ctx.textAlign = 'center';
        ctx.fillText(pp.text, pp.x, pp.y);
      }
      ctx.globalAlpha = 1;

      // HUD
      ctx.font = '10px monospace'; ctx.textAlign = 'left'; ctx.fillStyle = C.text;
      ctx.fillText('SCORE ' + score, 22, 26);
      var bl = ''; for (var l = 0; l < ballsLeft; l++) bl += '●';
      ctx.fillStyle = C.railHi;
      ctx.fillText('BALL ' + (bl || '—'), 22, 40);
      if (!launched && !dead) {
        ctx.textAlign = 'center';
        ctx.fillStyle = ((Date.now() >> 9) % 2) ? '#3fae5a' : C.text;
        ctx.fillText('按住 SPACE 蓄力，松手发射', W / 2 - 30, H - 120);
      }
    }

    fullReset();
    return { update: update, draw: draw, onKey: onKey, onPointer: onPointer };
  }
});
