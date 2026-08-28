// games/drop100.js — 是男人就下一百层（蓄力跳 / 跑步机 / 脆板 / 可读性打磨）
ARCADE.register({
  id: 'drop100',
  cn: '下一百层',
  en: 'DROP 100 FLOORS',
  create: function (api) {
    var ctx = api.ctx, W = api.W, H = api.H;
    var C = {
      bg: '#0d1420', bgDeep: '#0a0f1a',
      floor: '#3a4a66', floorHi: '#5a719a', floorLo: '#242f45',
      edge: 'rgba(255, 209, 102, .75)',          // 缺口边缘高亮：一眼找到出路
      spike: '#e06c75', spikeLo: '#8c3138',
      player: '#ffd166', hud: '#8899bb',
      green: '#3fae5a', gold: '#ffd166',
      conv: '#2d7a8c', convHi: '#4db8cc',        // 跑步机
      crumb: '#a8862f', crumbHi: '#d4b05e',      // 脆板
      spring: '#3fae5a', springHi: '#8fe0a2'     // 弹簧板
    };
    var FH = 14;            // 楼层厚度
    var MAXF = 100;

    var floors, man, score, dead, won, speed, spawnY, frame, deadTimer;
    var keys = {};
    var parts = [];
    var landT = 0, shakeT = 0, mileT = 0, mileN = 0;

    function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

    function newFloor(y) {
      var gap = 68 + Math.random() * 56;
      // 缺口位置偏向玩家可达范围：以玩家横坐标为中心 ±120px 抖动，杜绝"实际无路"
      var gx;
      if (man) gx = clamp(man.x - gap / 2 + (Math.random() - 0.5) * 240, 22, W - 22 - gap);
      else gx = 50 + Math.random() * (W - 100 - gap);
      var f = {
        y: y, segs: [[0, gx], [gx + gap, W - gx - gap]],
        spike: null, spring: null, type: 'normal', dir: 0, timer: -1
      };
      // 楼层类型：跑步机 20% / 脆板 12% / 弹簧板 12%
      var r = Math.random();
      if (r < 0.20) { f.type = 'conveyor'; f.dir = Math.random() < 0.5 ? -1 : 1; }
      else if (r < 0.32) f.type = 'crumble';
      else if (r < 0.44) f.type = 'spring';
      // 弹簧板：放在某一段的中部，两侧留落脚空间
      if (f.type === 'spring') {
        var side = f.segs[0][1] >= 60 ? 0 : (f.segs[1][1] >= 60 ? 1 : -1);
        if (side >= 0) {
          var segX = f.segs[side][0], segW = f.segs[side][1];
          f.spring = [segX + 16 + Math.random() * Math.max(0, segW - 60), 28];
        } else f.type = 'normal';
      }
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
      landT = 0; shakeT = 0; mileT = 0; mileN = 0;
      man = { x: W / 2, y: H - 60, vy: 0, onGround: false, stand: null };
      var start = newFloor(H - 40);
      start.spike = null; start.type = 'normal';
      floors.push(start);
      floors.push(newFloor(H + 16));
      floors.push(newFloor(H + 72));
      spawnY = H + 40;
    }

    api.panel([['←→', '移动'], ['SPACE/↑', '跳跃(长按更高)'], ['P', '暂停'], ['ESC', '片库']],
      '机台秘技：金色亮边是出路；尖刺能跳过去；绿色弹簧板会把你弹上天花板——躲着走');
    function onKey(k, down) {
      if (k === 'ArrowLeft' || k === 'a' || k === 'A') keys.left = down;
      if (k === 'ArrowRight' || k === 'd' || k === 'D') keys.right = down;
      if (k === ' ' || k === 'ArrowUp' || k === 'w' || k === 'W') {
        // 松手截断上升：轻点小跳，长按高跳
        if (!down && man && man.vy < -1.2) man.vy *= 0.5;
        keys.jump = down;
      }
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
    function boom(x, y, n, color, up) {
      for (var i = 0; i < n; i++) parts.push({
        x: x, y: y, vx: (Math.random() - 0.5) * 2.6, vy: up ? -Math.random() * 1.6 : -Math.random() * 2,
        life: 18 + Math.random() * 10, color: color
      });
    }

    function update() {
      frame++;
      if (landT > 0) landT--;
      if (shakeT > 0) shakeT--;
      if (mileT > 0) mileT--;
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

      // 加速与生成。速度上限 4.3 必须低于下落上限 4.4：
      // 否则高分段地板上升比极限下落还快，站着只会被一路顶上天花板，100 层物理上不可达
      speed = 0.9 + Math.min(3.4, score * 0.04);
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
          if (!fl.counted) {
            fl.counted = true; score++;
            if (score % 10 === 0 && score < MAXF) { mileT = 70; mileN = score; }
          }
          if (score >= MAXF) { won = true; deadTimer = 0; return; }
          floors.splice(i, 1);
        }
      }

      // 小人：横移
      var spd = 3.2;
      if (keys.left) man.x -= spd;
      if (keys.right) man.x += spd;
      man.x = clamp(man.x, 6, W - 6);

      // 蓄力跳：上升期间按住跳跃键则重力减半 → 长按跳得高，轻点小跳
      if (keys.jump && man.onGround) {
        man.vy = -5.2;
        man.onGround = false;
        man.stand = null;
        boom(man.x, man.y + 8, 3, '#5a719a', true);
      }

      var under = segAt(man.y + 9, man.x);
      if (under && man.vy >= 0 && man.y + 8 <= under.y + 6) {
        if (!man.onGround) {
          landT = 6;
          boom(man.x, under.y - 1, 4, '#5a719a', true);
        }
        man.y = under.y - 8;
        man.vy = 0;
        if (under.spike && man.x > under.spike[0] - 4 && man.x < under.spike[0] + under.spike[1] + 4) {
          dead = true; deadTimer = 0; shakeT = 10; return;
        }
        if (under.type === 'spring' && under.spring &&
            man.x > under.spring[0] - 2 && man.x < under.spring[0] + under.spring[1] + 2) {
          // 弹簧板：立刻弹起（长按跳跃键会弹得更高，小心天花板）
          man.vy = -7.6;
          man.onGround = false;
          man.stand = null;
          shakeT = 4;
          boom(man.x, under.y - 3, 6, C.springHi, true);
        } else {
          man.onGround = true;
          man.stand = under;
          if (under.type === 'crumble' && under.timer < 0) under.timer = 42;
          if (under.type === 'conveyor') man.x = clamp(man.x + under.dir * 1.1, 6, W - 6);
        }
      } else {
        man.onGround = false;
        man.stand = null;
        var g = (man.vy < 0 && keys.jump) ? 0.17 : 0.36;
        man.vy = Math.min(4.4, man.vy + g);
        man.y += man.vy;
      }

      // 被顶出屏幕顶端 / 掉出底部
      if (man.y < 6) { dead = true; deadTimer = 0; shakeT = 10; return; }
      if (man.y > H + 20) { dead = true; deadTimer = 0; }
    }

    function rect(x, y, w, h, c) { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), w, h); }

    function drawFloorSeg(f, x, w, segIdx) {
      if (w <= 0) return;
      var shake = (f.timer > 0) ? (Math.random() - 0.5) * 3 : 0;
      x += shake;
      var base = C.floor, hi = C.floorHi;
      if (f.type === 'conveyor') { base = C.conv; hi = C.convHi; }
      if (f.type === 'crumble') { base = C.crumb; hi = C.crumbHi; }
      rect(x, f.y, w, FH, base);
      rect(x, f.y, w, 3, hi);
      rect(x, f.y + FH - 2, w, 2, C.floorLo);
      // 缺口边缘高亮：段端面画金色竖条，出路一眼可见
      ctx.fillStyle = C.edge;
      if (segIdx === 0) ctx.fillRect(Math.round(x + w - 3), f.y - 2, 3, FH + 2);
      else ctx.fillRect(Math.round(x), f.y - 2, 3, FH + 2);
      if (f.type === 'conveyor') {
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
        ctx.strokeStyle = 'rgba(0,0,0,.4)'; ctx.lineWidth = 1;
        ctx.beginPath();
        for (var k = 6; k < w; k += 14) {
          ctx.moveTo(x + k, f.y + 2);
          ctx.lineTo(x + k + 5, f.y + FH - 2);
        }
        ctx.stroke();
      }
    }

    function drawCeiling() {
      // 移动斜纹危险带
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, W, 16); ctx.clip();
      ctx.fillStyle = 'rgba(224,108,117,.22)';
      ctx.fillRect(0, 0, W, 16);
      ctx.fillStyle = 'rgba(224,108,117,.5)';
      var off = (frame * 0.8) % 28;
      for (var x = -28 + off; x < W + 28; x += 28) {
        ctx.beginPath();
        ctx.moveTo(x, 16); ctx.lineTo(x + 14, 0); ctx.lineTo(x + 24, 0); ctx.lineTo(x + 10, 16);
        ctx.closePath(); ctx.fill();
      }
      ctx.restore();
    }

    function draw() {
      ctx.save();
      if (shakeT > 0) ctx.translate((Math.random() - 0.5) * 5, (Math.random() - 0.5) * 5);

      // 背景纵向渐深 + 侧壁刻度
      var bgG = ctx.createLinearGradient(0, 0, 0, H);
      bgG.addColorStop(0, C.bg); bgG.addColorStop(1, C.bgDeep);
      ctx.fillStyle = bgG; ctx.fillRect(0, 0, W, H);
      for (var d = 16; d < H; d += 46) {
        rect(2, d, 6, 1, '#1c2739');
        rect(W - 8, d, 6, 1, '#1c2739');
      }

      for (var i = 0; i < floors.length; i++) {
        var f = floors[i];
        for (var s = 0; s < f.segs.length; s++)
          drawFloorSeg(f, f.segs[s][0], f.segs[s][1], s);
        if (f.spike) {
          var sx = f.spike[0], sw = f.spike[1];
          // 矮刺贴地 + 底座条：明确"这是能跳过去的障碍"，不是墙
          rect(sx, f.y - 2, sw, 2, C.spikeLo);
          ctx.fillStyle = C.spike;
          for (var t = 0; t < sw; t += 8) {
            ctx.beginPath();
            ctx.moveTo(sx + t, f.y - 2);
            ctx.lineTo(sx + t + 4, f.y - 7);
            ctx.lineTo(sx + t + 8, f.y - 2);
            ctx.closePath(); ctx.fill();
          }
        }
        if (f.spring) {
          // 弹簧板：绿色垫子 + 向上箭头
          var px0 = f.spring[0], pw = f.spring[1];
          rect(px0, f.y - 3, pw, 3, C.spring);
          ctx.fillStyle = C.springHi;
          for (var t2 = 0; t2 < pw; t2 += 14) {
            ctx.beginPath();
            ctx.moveTo(px0 + t2 + 3, f.y - 4);
            ctx.lineTo(px0 + t2 + 7, f.y - 10);
            ctx.lineTo(px0 + t2 + 11, f.y - 4);
            ctx.closePath(); ctx.fill();
          }
        }
      }

      // 小人：落地压扁、腾空收腿、跑动摆腿
      var px = man.x, py = man.y;
      var ph = landT > 0 ? 5 : 8;                       // squash
      rect(px - 4, py - ph, 8, ph, '#4a5a7a');
      rect(px - 3, py - ph - 5, 6, 5, '#e8c1a0');
      rect(px - 4, py - ph - 7, 8, 3, C.gold);
      if (!man.onGround) {
        rect(px - 6, py - ph + 1, 3, 4, '#e8c1a0');
        rect(px + 3, py - ph + 1, 3, 4, '#e8c1a0');
      } else if (keys.left || keys.right) {
        var legPhase = ((frame >> 3) % 2) ? 1 : 0;
        var dir = keys.right ? 1 : -1;
        rect(px - 3 + dir * legPhase, py, 3, 3, '#e8c1a0');
        rect(px + 0 - dir * legPhase, py, 3, 3, '#e8c1a0');
        if ((frame >> 3) % 2 === 0) rect(px + dir * 4, py - 4, 2, 2, '#5a719a');
      } else {
        rect(px - 4, py, 3, 3, '#e8c1a0');
        rect(px + 1, py, 3, 3, '#e8c1a0');
      }

      // 粒子
      for (var q = 0; q < parts.length; q++) {
        ctx.globalAlpha = Math.max(0, parts[q].life / 24);
        rect(parts[q].x, parts[q].y, 2, 2, parts[q].color);
      }
      ctx.globalAlpha = 1;

      drawCeiling();

      // HUD
      ctx.font = 'bold 20px monospace';
      ctx.fillStyle = score > 80 ? C.gold : '#f5f7fa';
      ctx.textAlign = 'left';
      ctx.strokeStyle = 'rgba(0,0,0,.6)'; ctx.lineWidth = 3;
      ctx.strokeText('第 ' + score + ' 层', 10, 34);
      ctx.fillText('第 ' + score + ' 层', 10, 34);
      ctx.font = '10px monospace';
      ctx.fillStyle = C.hud;
      ctx.textAlign = 'right';
      ctx.fillText('速度 ' + speed.toFixed(1) + 'x · 目标 100', W - 16, 30);

      // 里程碑闪现
      if (mileT > 0) {
        ctx.globalAlpha = Math.min(1, mileT / 25);
        ctx.textAlign = 'center';
        ctx.font = 'bold 16px monospace';
        ctx.fillStyle = C.gold;
        ctx.fillText('▼ ' + mileN + ' 层', W / 2, 58);
        ctx.globalAlpha = 1;
      }

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
      ctx.restore();
    }

    reset();
    return { update: update, draw: draw, onKey: onKey, onPointer: onPointer };
  }
});
