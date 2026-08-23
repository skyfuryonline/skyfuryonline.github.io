// games/drop100.js — 是男人就下一百层（带跳跃 / 跑步机 / 脆板）
ARCADE.register({
  id: 'drop100',
  cn: '下一百层',
  en: 'DROP 100 FLOORS',
  create: function (api) {
    var ctx = api.ctx, W = api.W, H = api.H;
    var C = {
      bg: '#0d1420', floor: '#3a4a66', floorHi: '#5a719a',
      spike: '#e06c75', player: '#ffd166', hud: '#8899bb',
      green: '#3fae5a', gold: '#ffd166',
      conv: '#2d7a8c', convHi: '#4db8cc',        // 跑步机
      crumb: '#a8862f', crumbHi: '#d4b05e'       // 脆板
    };
    var FH = 14;            // 楼层厚度
    var MAXF = 100;

    var floors, man, score, dead, won, speed, spawnY, frame, deadTimer;
    var keys = {};
    var parts = [];

    function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

    function newFloor(y) {
      var gap = 68 + Math.random() * 56;
      // 缺口位置偏向玩家可达范围：以玩家横坐标为中心 ±160px 抖动，杜绝"实际无路"
      var gx;
      if (man) gx = clamp(man.x - gap / 2 + (Math.random() - 0.5) * 240, 22, W - 22 - gap);
      else gx = 50 + Math.random() * (W - 100 - gap);
      var f = {
        y: y, segs: [[0, gx], [gx + gap, W - gx - gap]],
        spike: null, type: 'normal', dir: 0, timer: -1
      };
      // 楼层类型：跑步机 22% / 脆板 12%
      var r = Math.random();
      if (r < 0.22) { f.type = 'conveyor'; f.dir = Math.random() < 0.5 ? -1 : 1; }
      else if (r < 0.34) f.type = 'crumble';
      // 尖刺只出现在普通层，且给两侧留出落脚空间
      if (f.type === 'normal' && Math.random() < 0.3) {
        var sw = 32;
        var side = f.segs[0][1] >= 84 ? 0 : (f.segs[1][1] >= 84 ? 1 : -1);
        if (side >= 0) {
          var segX = f.segs[side][0], segW = f.segs[side][1];
          f.spike = [segX + 22 + Math.random() * Math.max(0, segW - sw - 44), sw];
        }
      }
      return f;
    }

    function reset() {
      floors = []; score = 0; dead = false; won = false;
      speed = 0.9; frame = 0; deadTimer = 0; parts = [];
      man = { x: W / 2, y: H - 60, vy: 0, onGround: false, stand: null };
      var start = newFloor(H - 40);
      start.spike = null; start.type = 'normal';
      floors.push(start);
      floors.push(newFloor(H + 16));
      floors.push(newFloor(H + 72));
      spawnY = H + 40;
    }

    api.panel([['←→', '移动'], ['SPACE/↑', '跳跃'], ['P', '暂停'], ['ESC', '片库']],
      '机台秘技：跑步机推人、脆板踩了就碎，红刺碰不得——该跳就跳');
    function onKey(k, down) {
      if (k === 'ArrowLeft' || k === 'a' || k === 'A') keys.left = down;
      if (k === 'ArrowRight' || k === 'd' || k === 'D') keys.right = down;
      if (k === ' ' || k === 'ArrowUp' || k === 'w' || k === 'W') keys.jump = down;
    }
    function onPointer(type, x) {
      if (type === 'move') man.x = clamp(x, 6, W - 6);
    }

    function segAt(y, x) {
      for (var i = 0; i < floors.length; i++) {
        var f = floors[i];
        if (y >= f.y && y < f.y + FH) {
          for (var s = 0; s < f.segs.length; s++)
            if (x >= f.segs[s][0] && x <= f.segs[s][0] + f.segs[s][1]) return f;
        }
      }
      return null;
    }
    function boom(x, y, n, color) {
      for (var i = 0; i < n; i++) parts.push({
        x: x, y: y, vx: (Math.random() - 0.5) * 2.6, vy: -Math.random() * 2,
        life: 18 + Math.random() * 10, color: color
      });
    }

    function update() {
      frame++;
      for (var q = parts.length - 1; q >= 0; q--) {
        var pt = parts[q];
        pt.x += pt.vx; pt.y += pt.vy; pt.vy += 0.07; pt.life--;
        if (pt.life <= 0) parts.splice(q, 1);
      }
      if (dead || won) {
        deadTimer++;
        if (dead && deadTimer === 50) api.gameOver(score);
        if (won && deadTimer === 100) api.gameOver(score, { big: '是 男 人 ！' });
        return;
      }

      // 加速与生成（曲线调陡）
      speed = 0.9 + Math.min(4.2, score * 0.04);
      spawnY -= speed;
      while (spawnY <= H - FH) {
        floors.push(newFloor(H));
        spawnY += 44 + Math.random() * 26;
      }
      // 楼层滚动 + 脆板计时
      for (var i = floors.length - 1; i >= 0; i--) {
        var fl = floors[i];
        fl.y -= speed;
        if (fl.timer > 0) {
          fl.timer--;
          if (fl.timer === 0) {
            boom(fl.segs[0][0] + fl.segs[0][1] / 2, fl.y + FH / 2, 8, C.crumbHi);
            if (fl.segs[1][1] > 0) boom(fl.segs[1][0] + fl.segs[1][1] / 2, fl.y + FH / 2, 8, C.crumbHi);
            if (man.stand === fl) { man.stand = null; man.onGround = false; }
            floors.splice(i, 1);
            continue;
          }
        }
        if (fl.y < -FH) {
          if (!fl.counted) { fl.counted = true; score++; }
          if (score >= MAXF) { won = true; deadTimer = 0; return; }
          floors.splice(i, 1);
        }
      }

      // 小人：横移
      var spd = 3.2;
      if (keys.left) man.x -= spd;
      if (keys.right) man.x += spd;
      man.x = clamp(man.x, 6, W - 6);

      // 跳跃（仅站地面时）
      if (keys.jump && man.onGround) {
        man.vy = -5.6;
        man.onGround = false;
        man.stand = null;
      }

      // 落地 / 下落
      var under = segAt(man.y + 9, man.x);
      if (under && man.vy >= 0 && man.y + 8 <= under.y + 6) {
        man.y = under.y - 8;
        man.vy = 0;
        if (under.spike && man.x > under.spike[0] - 4 && man.x < under.spike[0] + under.spike[1] + 4) {
          dead = true; deadTimer = 0; return;
        }
        man.onGround = true;
        man.stand = under;
        if (under.type === 'crumble' && under.timer < 0) under.timer = 42;
        if (under.type === 'conveyor') man.x = clamp(man.x + under.dir * 1.1, 6, W - 6);
      } else {
        man.onGround = false;
        man.stand = null;
        man.vy = Math.min(4.4, man.vy + 0.34);
        man.y += man.vy;
      }

      // 被顶出屏幕顶端 / 掉出底部
      if (man.y < 6) { dead = true; deadTimer = 0; return; }
      if (man.y > H + 20) { dead = true; deadTimer = 0; }
    }

    function rect(x, y, w, h, c) { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), w, h); }
    function drawFloorSeg(f, x, w) {
      if (w <= 0) return;
      var shake = (f.timer > 0) ? (Math.random() - 0.5) * 3 : 0;
      x += shake;
      var base = C.floor, hi = C.floorHi;
      if (f.type === 'conveyor') { base = C.conv; hi = C.convHi; }
      if (f.type === 'crumble') { base = C.crumb; hi = C.crumbHi; }
      rect(x, f.y, w, FH, base);
      rect(x, f.y, w, 3, hi);
      rect(x, f.y + FH - 2, w, 2, '#242f45');
      if (f.type === 'conveyor') {
        // 移动箭头纹理
        var off = (frame * 1.6 * f.dir) % 16;
        ctx.fillStyle = 'rgba(255,255,255,.35)';
        for (var c = -16 + off; c < w; c += 16) {
          var ax = x + c;
          ctx.beginPath();
          if (f.dir > 0) {
            ctx.moveTo(ax, f.y + 5); ctx.lineTo(ax + 8, f.y + 9); ctx.lineTo(ax, f.y + 13);
          } else {
            ctx.moveTo(ax + 8, f.y + 5); ctx.lineTo(ax, f.y + 9); ctx.lineTo(ax + 8, f.y + 13);
          }
          ctx.closePath(); ctx.fill();
        }
      }
      if (f.type === 'crumble') {
        // 裂纹
        ctx.strokeStyle = 'rgba(0,0,0,.4)'; ctx.lineWidth = 1;
        ctx.beginPath();
        for (var k = 6; k < w; k += 14) {
          ctx.moveTo(x + k, f.y + 2);
          ctx.lineTo(x + k + 5, f.y + FH - 2);
        }
        ctx.stroke();
      }
    }
    function draw() {
      rect(0, 0, W, H, C.bg);
      for (var d = 10; d < H; d += 46) rect(W - 12, d, 8, 1, '#1c2739');

      for (var i = 0; i < floors.length; i++) {
        var f = floors[i];
        for (var s = 0; s < f.segs.length; s++)
          drawFloorSeg(f, f.segs[s][0], f.segs[s][1]);
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

      // 小人（起跳/下落时腿收起）
      var px = man.x, py = man.y;
      rect(px - 4, py - 8, 8, 8, '#4a5a7a');
      rect(px - 3, py - 13, 6, 5, '#e8c1a0');
      rect(px - 4, py - 15, 8, 3, C.gold);
      if (!man.onGround) {
        rect(px - 6, py - 6, 3, 4, '#e8c1a0');
        rect(px + 3, py - 6, 3, 4, '#e8c1a0');
      } else if (keys.left) rect(px - 7, py - 6, 3, 5, '#e8c1a0');
      else if (keys.right) rect(px + 4, py - 6, 3, 5, '#e8c1a0');

      // 粒子
      for (var q = 0; q < parts.length; q++) {
        ctx.globalAlpha = Math.max(0, parts[q].life / 24);
        rect(parts[q].x, parts[q].y, 2, 2, parts[q].color);
      }
      ctx.globalAlpha = 1;

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
        ctx.fillText(man.y < 6 ? '被 天 花 板 顶 碎 了' : (man.y > H ? '摔 出 了 世 界' : '踩 到 尖 刺 了'), W / 2, H / 2 - 8);
      }
    }

    reset();
    return { update: update, draw: draw, onKey: onKey, onPointer: onPointer };
  }
});
