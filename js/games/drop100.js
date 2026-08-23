// games/drop100.js — 是男人就下一百层
ARCADE.register({
  id: 'drop100',
  cn: '下一百层',
  en: 'DROP 100 FLOORS',
  create: function (api) {
    var ctx = api.ctx, W = api.W, H = api.H;
    var C = {
      bg: '#0d1420', floor: '#3a4a66', floorHi: '#5a719a', floorDead: '#8d99a6',
      spike: '#e06c75', player: '#ffd166', hud: '#8899bb',
      green: '#3fae5a', gold: '#ffd166'
    };
    var FH = 14;            // 楼层厚度
    var MAXF = 100;

    var floors, man, score, dead, won, speed, spawnY, frame, deadTimer;

    function newFloor(y) {
      // 楼层 = [起点x, 宽度] 的实心段 + 尖刺段
      var gap = 70 + Math.random() * 60; // 缺口宽
      var gx = 50 + Math.random() * (W - 100 - gap);
      var f = {
        y: y, segs: [], spike: null,
        scroll: true
      };
      // 左段与右段
      f.segs.push([0, gx]);
      f.segs.push([gx + gap, W - gx - gap]);
      // 部分楼层带尖刺（不在缺口边缘）
      if (Math.random() < 0.28) {
        var sw = 34;
        var side = f.segs[0][1] > sw + 30 ? 0 : (f.segs[1][1] > sw + 30 ? 1 : -1);
        if (side >= 0) {
          var sx = side === 0 ? Math.max(4, gx - sw - 4 - Math.random() * 40)
                              : Math.min(W - 4, gx + gap + 4 + Math.random() * 40);
          if (side === 0) f.spike = [sx, sw]; else f.spike = [sx, sw];
        }
      }
      return f;
    }

    function reset() {
      floors = []; score = 0; dead = false; won = false;
      speed = 0.9; frame = 0; deadTimer = 0;
      man = { x: W / 2, y: H - 60, vy: 0, onGround: false };
      // 初始垫脚层：带缺口（否则第一层就无路可下，被天花板压死），且不放尖刺
      var start = newFloor(H - 40);
      start.spike = null;
      floors.push(start);
      // 预铺两层在屏幕下方，穿落首层后立刻有落脚点
      floors.push(newFloor(H + 16));
      floors.push(newFloor(H + 72));
      spawnY = H + 40;
    }

    api.panel([['←→', '移动'], ['P', '暂停'], ['ESC', '片库']],
      '机台秘技：红色尖刺碰不得，落差太大也摔不死你——只有天花板会');
    var keys = {};
    function onKey(k, down) {
      if (k === 'ArrowLeft' || k === 'a' || k === 'A') keys.left = down;
      if (k === 'ArrowRight' || k === 'd' || k === 'D') keys.right = down;
    }
    function onPointer(type, x) {
      if (type === 'move') man.x = Math.max(6, Math.min(W - 6, x));
    }

    function segAt(y, x) { // 该高度是否存在实心段
      for (var i = 0; i < floors.length; i++) {
        var f = floors[i];
        if (y >= f.y && y < f.y + FH) {
          for (var s = 0; s < f.segs.length; s++)
            if (x >= f.segs[s][0] && x <= f.segs[s][0] + f.segs[s][1]) return f;
        }
      }
      return null;
    }

    function update() {
      frame++;
      if (dead || won) {
        deadTimer++;
        if (dead && deadTimer === 50) api.gameOver(score);
        if (won && deadTimer === 100) api.gameOver(score, { big: '是 男 人 ！' });
        return;
      }

      // 加速与生成
      speed = 0.9 + Math.min(3.4, score * 0.032);
      spawnY -= speed;
      while (spawnY <= H - FH) {
        floors.push(newFloor(H));
        spawnY += 46 + Math.random() * 26;
      }
      // 楼层滚动
      for (var i = floors.length - 1; i >= 0; i--) {
        floors[i].y -= speed;
        if (floors[i].y < -FH) {
          // 离开屏幕底 = 下一层
          if (!floors[i].counted) { floors[i].counted = true; score++; }
          if (score >= MAXF) { won = true; deadTimer = 0; return; }
          floors.splice(i, 1);
        }
      }

      // 小人
      var spd = 3.2;
      if (keys.left) man.x -= spd;
      if (keys.right) man.x += spd;
      man.x = Math.max(6, Math.min(W - 6, man.x));

      var under = segAt(man.y + 8 + 1, man.x); // 脚下一格
      if (under && man.vy >= 0 && man.y + 8 <= under.y + 6) {
        man.y = under.y - 8;
        man.vy = 0;
        // 尖刺判定
        if (under.spike && man.x > under.spike[0] - 4 && man.x < under.spike[0] + under.spike[1] + 4) {
          dead = true; deadTimer = 0; return;
        }
      } else {
        man.vy = Math.min(4.2, man.vy + 0.35);
        man.y += man.vy;
      }

      // 被顶出屏幕顶端
      if (man.y < 6) { dead = true; deadTimer = 0; return; }
      // 掉出底部（几乎不可能，容错）
      if (man.y > H + 20) { dead = true; deadTimer = 0; }
    }

    function rect(x, y, w, h, c) { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), w, h); }
    function draw() {
      rect(0, 0, W, H, C.bg);
      // 深度计（右侧标尺）
      for (var d = 10; d < H; d += 46) {
        rect(W - 12, d, 8, 1, '#1c2739');
      }

      // 楼层
      for (var i = 0; i < floors.length; i++) {
        var f = floors[i];
        for (var s = 0; s < f.segs.length; s++) {
          var x = f.segs[s][0], w = f.segs[s][1];
          if (w <= 0) continue;
          rect(x, f.y, w, FH, C.floor);
          rect(x, f.y, w, 3, C.floorHi);
          rect(x, f.y + FH - 2, w, 2, '#242f45');
        }
        if (f.spike) {
          var sx = f.spike[0], sw = f.spike[1];
          ctx.fillStyle = C.spike;
          for (var t = 0; t < sw; t += 7) {
            ctx.beginPath();
            ctx.moveTo(sx + t, f.y);
            ctx.lineTo(sx + t + 3.5, f.y - 7);
            ctx.lineTo(sx + t + 7, f.y);
            ctx.closePath(); ctx.fill();
          }
        }
      }

      // 小人（像素小人 + 安全帽）
      var px = man.x, py = man.y;
      rect(px - 4, py - 8, 8, 8, '#4a5a7a');          // 身体
      rect(px - 3, py - 13, 6, 5, '#e8c1a0');         // 头
      rect(px - 4, py - 15, 8, 3, C.gold);            // 安全帽
      if (keys.left) rect(px - 7, py - 6, 3, 5, '#e8c1a0');
      else if (keys.right) rect(px + 4, py - 6, 3, 5, '#e8c1a0');

      // 天花板危险区
      var danger = ctx.createLinearGradient(0, 0, 0, 26);
      danger.addColorStop(0, 'rgba(224,108,117,.5)');
      danger.addColorStop(1, 'rgba(224,108,117,0)');
      ctx.fillStyle = danger; ctx.fillRect(0, 0, W, 26);

      // HUD
      ctx.font = 'bold 20px monospace';
      ctx.fillStyle = score > 80 ? C.gold : '#f5f7fa';
      ctx.textAlign = 'left';
      ctx.strokeStyle = 'rgba(0,0,0,.6)'; ctx.lineWidth = 3;
      ctx.strokeText('第 ' + score + ' 层', 10, 30);
      ctx.fillText('第 ' + score + ' 层', 10, 30);
      ctx.font = '10px monospace';
      ctx.fillStyle = C.hud;
      ctx.textAlign = 'right';
      ctx.fillText('速度 ' + speed.toFixed(1) + 'x · 目标 100', W - 16, 18);

      if (won) {
        ctx.fillStyle = 'rgba(6,10,16,.72)'; ctx.fillRect(0, 0, W, H);
        ctx.textAlign = 'center';
        ctx.font = 'bold 24px monospace'; ctx.fillStyle = C.gold;
        ctx.fillText('是 男 人 ！', W / 2, H / 2 - 30);
        ctx.font = '12px monospace'; ctx.fillStyle = '#f5f7fa';
        ctx.fillText('100 层达成 · 你证明了你自己', W / 2, H / 2);
        ctx.fillStyle = ((Date.now() >> 9) % 2) ? C.green : C.hud;
        ctx.fillText('ENTER / R 再来一次', W / 2, H / 2 + 34);
      } else if (dead && deadTimer > 12) {
        ctx.fillStyle = 'rgba(6,10,16,.6)'; ctx.fillRect(0, 0, W, H);
        ctx.textAlign = 'center';
        ctx.font = 'bold 18px monospace'; ctx.fillStyle = C.spike;
        ctx.fillText(man.y < 6 ? '被 天 花 板 顶 碎 了' : '踩 到 尖 刺 了', W / 2, H / 2 - 8);
      }
    }

    reset();
    return { update: update, draw: draw, onKey: onKey, onPointer: onPointer };
  }
});
