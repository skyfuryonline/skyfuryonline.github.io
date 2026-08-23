// games/tank.js — 坦克大战 · 保卫最后一杯咖啡
ARCADE.register({
  id: 'tank',
  cn: '坦克大战',
  en: 'BATTLE COFFEE · 1985',
  create: function (api) {
    var ctx = api.ctx, W = api.W, H = api.H;
    var T = 20, COLS = W / T, ROWS = H / T;
    var C = {
      bg: '#101418', grid: '#161c22',
      brick: '#b4622d', brickDark: '#8a4a20', steel: '#8d99a6', steelHi: '#c4cdd6',
      player: '#e8c15a', playerDark: '#a8862f',
      e1: '#c05a5a', e2: '#5aa7c0', e3: '#a06cc0', e4: '#5ac07a',
      cup: '#f5e9d6', coffee: '#6b4226', hud: '#8899bb', gold: '#ffd166'
    };

    var EMPTY = 0, BRICK = 1, STEEL = 2, BASE = 9;
    var map = [];
    var player, enemies, bullets, powerups, booms;
    var score, lives, wave, left, spawnTimer, splash, banner, invuln, powerUntil;
    var keys = {};

    // ---------- 地图 ----------
    function buildMap() {
      map = [];
      for (var y = 0; y < ROWS; y++) {
        var row = [];
        for (var x = 0; x < COLS; x++) row.push(EMPTY);
        map.push(row);
      }
      // 砖墙群（经典对称布局）
      var clusters = [
        [3, 1], [6, 1], [9, 1], [12, 1], [15, 1], [18, 1],
        [3, 4], [6, 4], [9, 4], [12, 4], [15, 4], [18, 4],
        [3, 7], [6, 7], [9, 7], [12, 7], [15, 7], [18, 7],
        [3, 10], [6, 10], [15, 10], [18, 10]
      ];
      for (var i = 0; i < clusters.length; i++) {
        var cx = clusters[i][0], cy = clusters[i][1];
        for (var dy = 0; dy < 2; dy++)
          for (var dx = 0; dx < 2; dx++)
            if (cy + dy < ROWS - 4 && cx + dx < COLS) map[cy + dy][cx + dx] = BRICK;
      }
      // 中央钢块
      map[8][11] = map[8][12] = STEEL;
      map[9][11] = map[9][12] = STEEL;
      // 出生点强制清空（敌方上排三点 + 玩家点），避免坦克生成在砖里卡死
      [1, 12, 22].forEach(function (c) { map[1][c] = EMPTY; map[2][c] = EMPTY; });
      map[ROWS - 2][8] = EMPTY;

      // 基地（咖啡杯）+ 护砖
      map[ROWS - 3][11] = map[ROWS - 3][12] = BASE;
      map[ROWS - 4][10] = map[ROWS - 4][11] = map[ROWS - 4][12] = map[ROWS - 4][13] = BRICK;
      map[ROWS - 3][10] = map[ROWS - 3][13] = BRICK;
    }

    function solidAt(px, py) {
      var tx = Math.floor(px / T), ty = Math.floor(py / T);
      if (tx < 0 || ty < 0 || tx >= COLS || ty >= ROWS) return true;
      var v = map[ty][tx];
      return v === BRICK || v === STEEL || v === BASE;
    }
    function hitsWall(x, y, w, h) {
      var l = x - w / 2, t = y - h / 2;
      return solidAt(l + 1, t + 1) || solidAt(l + w - 1, t + 1) ||
             solidAt(l + 1, t + h - 1) || solidAt(l + w - 1, t + h - 1);
    }

    // ---------- 实体 ----------
    function spawnPlayer() {
      // 第 8 列格子中心（基地左侧隔两格），不再压到基地护砖
      player = { x: 8 * T + 10, y: (ROWS - 2) * T + 10, dir: 0, cooldown: 0, alive: true };
      invuln = 100;
    }
    function spawnEnemy() {
      if (left <= 0) return;
      var spots = [1, COLS / 2, COLS - 2];
      var sx = spots[(enemies.length + wave) % 3] * T + 10;
      for (var i = 0; i < enemies.length; i++)
        if (Math.abs(enemies[i].x - sx) < T * 2 && enemies[i].y < T * 2) return;
      var r = Math.random(), type = r < 0.45 ? 0 : r < 0.75 ? 1 : r < 0.92 ? 2 : 3;
      enemies.push({
        x: sx, y: T + 10, dir: 2, type: type, hp: type === 2 ? 3 : 1,
        speed: [0.55, 1.1, 0.5, 0.7][type] + wave * 0.03,
        cooldown: 60 + Math.random() * 120, think: 0, spawnAnim: 40
      });
      left--;
    }
    function boom(x, y, n, color) {
      for (var i = 0; i < n; i++) booms.push({
        x: x, y: y,
        vx: (Math.random() - 0.5) * 3, vy: (Math.random() - 0.8) * 3,
        life: 20 + Math.random() * 12, color: color
      });
    }

    // ---------- 输入 ----------
    api.panel([['←→↑↓', '驾驶'], ['SPACE', '开炮'], ['P', '暂停'], ['ESC', '片库']],
      '机台秘技：击毁紫色重坦会掉 ☕ 咖啡之力，双管齐发 10 秒');
    function onKey(k, down) {
      if (k === 'ArrowLeft' || k === 'a' || k === 'A') keys.left = down;
      if (k === 'ArrowRight' || k === 'd' || k === 'D') keys.right = down;
      if (k === 'ArrowUp' || k === 'w' || k === 'W') keys.up = down;
      if (k === 'ArrowDown' || k === 's' || k === 'S') keys.down = down;
      if (k === ' ') keys.fire = down;
    }

    // ---------- 逻辑 ----------
    var DIRV = [[0, -1], [1, 0], [0, 1], [-1, 0]];
    function moveTank(t, spd) {
      var v = DIRV[t.dir];
      var nx = t.x + v[0] * spd, ny = t.y + v[1] * spd;
      // 拐弯时贴齐半格车道，方便钻缝
      if (v[0] !== 0) ny = Math.round((t.y - 10) / 10) * 10 + 10;
      else nx = Math.round((t.x - 10) / 10) * 10 + 10;
      if (nx < 10 || nx > W - 10 || ny < 10 || ny > H - 10) return false;
      if (hitsWall(nx, ny, 16, 16)) return false;
      t.x = nx; t.y = ny;
      return true;
    }
    function fire(t, isPlayer) {
      var powered = isPlayer && performance.now() < powerUntil;
      if (t.cooldown > 0) return;
      t.cooldown = isPlayer ? 22 : 70 + Math.random() * 60;
      var v = DIRV[t.dir];
      var spd = isPlayer ? 4 : (t.type === 3 ? 4 : 2.6);
      function one(ox, oy) {
        bullets.push({
          x: t.x + v[0] * 10 + ox, y: t.y + v[1] * 10 + oy,
          vx: v[0] * spd, vy: v[1] * spd, player: isPlayer, power: powered
        });
      }
      if (powered) {
        if (v[0] !== 0) { one(0, -5); one(0, 5); } else { one(-5, 0); one(5, 0); }
      } else one(0, 0);
    }
    function bulletHitTile(b) {
      var tx = Math.floor(b.x / T), ty = Math.floor(b.y / T);
      if (tx < 0 || ty < 0 || tx >= COLS || ty >= ROWS) return true;
      var v = map[ty][tx];
      if (v === BASE && b.player !== undefined) { destroyBase(); return true; }
      if (v === BRICK) {
        map[ty][tx] = EMPTY;
        boom(b.x, b.y, 4, C.brickDark);
        return true;
      }
      if (v === STEEL) { boom(b.x, b.y, 2, C.steelHi); return true; }
      return false;
    }
    function destroyBase() {
      if (splash) return;
      score += 0;
      boom(11.5 * T, (ROWS - 2.5) * T, 30, C.gold);
      endGame();
    }
    function endGame() {
      splash = 90; // 播放爆炸后再结算
    }
    function rectsHit(a, b, r) {
      var dx = a.x - b.x, dy = a.y - b.y;
      return dx * dx + dy * dy < r * r;
    }

    function update() {
      if (splash !== 0) {
        splash--;
        if (splash === 0) api.gameOver(score);
        updateBooms();
        return;
      }
      if (respawnT > 0) {
        respawnT--;
        if (respawnT === 0) spawnPlayer();
      }
      // 玩家
      if (player.alive) {
        var want = -1;
        if (keys.left) want = 3; else if (keys.right) want = 1;
        else if (keys.up) want = 0; else if (keys.down) want = 2;
        if (want >= 0 && want !== player.dir) {
          player.dir = want;
          moveTank(player, 1.3); // 先对齐车道
        } else if (want >= 0) {
          if (!moveTank(player, 1.3)) moveTank(player, 0);
        }
        if (player.cooldown > 0) player.cooldown--;
        if (keys.fire) fire(player, true);
      }
      if (invuln > 0) invuln--;

      // 敌人
      if (spawnTimer > 0) spawnTimer--;
      if (spawnTimer === 0 && enemies.length < 4 && left > 0) { spawnEnemy(); spawnTimer = 110 - Math.min(60, wave * 6); }
      for (var i = enemies.length - 1; i >= 0; i--) {
        var e = enemies[i];
        if (e.spawnAnim > 0) { e.spawnAnim--; continue; }
        e.think--;
        if (e.think <= 0 || !moveTank(e, e.speed)) {
          // 换向：偏向基地/玩家
          e.think = 40 + Math.random() * 80;
          var r = Math.random();
          if (r < 0.45) {
            var ddx = 11.5 * T - e.x, ddy = (ROWS - 3) * T - e.y;
            e.dir = Math.abs(ddx) > Math.abs(ddy) ? (ddx > 0 ? 1 : 3) : (ddy > 0 ? 2 : 0);
          } else e.dir = Math.floor(Math.random() * 4);
        }
        if (e.cooldown > 0) e.cooldown--;
        else if (Math.random() < 0.03) fire(e, false);
        // 压到玩家
        if (player.alive && invuln <= 0 && rectsHit(e, player, 14)) killPlayer();
      }

      // 子弹
      for (var b = bullets.length - 1; b >= 0; b--) {
        var bl = bullets[b];
        bl.x += bl.vx; bl.y += bl.vy;
        if (bl.x < 0 || bl.y < 0 || bl.x > W || bl.y > H) { bullets.splice(b, 1); continue; }
        if (bulletHitTile(bl)) { bullets.splice(b, 1); continue; }
        var gone = false;
        // 子弹互消
        for (var o = bullets.length - 1; o >= 0; o--) {
          if (o !== b && bullets[o].player !== bl.player && rectsHit(bl, bullets[o], 4)) {
            bullets.splice(Math.max(o, b)); bullets.splice(Math.min(o, b));
            gone = true; break;
          }
        }
        if (gone) continue;
        if (bl.player) {
          for (var j = enemies.length - 1; j >= 0; j--) {
            var en = enemies[j];
            if (en.spawnAnim > 0) continue;
            if (rectsHit(bl, en, 10)) {
              bullets.splice(b, 1);
              en.hp--;
              if (en.hp <= 0) {
                score += [100, 200, 400, 300][en.type];
                boom(en.x, en.y, 10, [C.e1, C.e2, C.e3, C.e4][en.type]);
                if (en.type === 2) powerups.push({ x: en.x, y: en.y, t: 400 });
                enemies.splice(j, 1);
              } else boom(bl.x, bl.y, 2, '#fff');
              gone = true; break;
            }
          }
        } else if (player.alive && invuln <= 0 && rectsHit(bl, player, 10)) {
          bullets.splice(b, 1);
          killPlayer();
        }
        if (gone) continue;
      }

      // 道具
      for (var p = powerups.length - 1; p >= 0; p--) {
        var pu = powerups[p];
        pu.t--;
        if (pu.t <= 0) { powerups.splice(p, 1); continue; }
        if (player.alive && rectsHit(pu, player, 14)) {
          powerups.splice(p, 1);
          powerUntil = performance.now() + 10000;
          boom(player.x, player.y, 8, C.gold);
        }
      }

      updateBooms();

      // 波次清空
      if (left === 0 && enemies.length === 0 && respawnT === 0 && player.alive) {
        wave++;
        left = 5 + wave * 2;
        spawnTimer = 30;
        buildMap();
        powerups = [];
        banner = 90;
      }
      if (banner > 0) banner--;
    }
    function updateBooms() {
      for (var i = booms.length - 1; i >= 0; i--) {
        var pt = booms[i];
        pt.x += pt.vx; pt.y += pt.vy; pt.vy += 0.05; pt.life--;
        if (pt.life <= 0) booms.splice(i, 1);
      }
    }
    function killPlayer() {
      boom(player.x, player.y, 14, C.player);
      lives--;
      if (lives <= 0) endGame();
      else { player.alive = false; respawnT = 60; }
    }
    var respawnT = 0;

    // ---------- 绘制 ----------
    function rect(x, y, w, h, c) { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), w, h); }
    function drawTank(t, body, dark) {
      var x = Math.round(t.x) - 8, y = Math.round(t.y) - 8;
      var d = t.dir;
      ctx.save();
      ctx.translate(x + 8, y + 8);
      ctx.rotate(d === 0 ? 0 : d === 1 ? Math.PI / 2 : d === 2 ? Math.PI : -Math.PI / 2);
      rect(-8, -6, 16, 12, body);       // 履带
      rect(-6, -8, 12, 16, dark);       // 车体
      rect(-3, -3, 6, 6, body);         // 舱盖
      rect(-1.5, -12, 3, 6, body);      // 炮管
      ctx.restore();
    }
    function drawBase() {
      var x = 11 * T, y = (ROWS - 3) * T;
      rect(x, y, 2 * T, 2 * T, C.bg);
      // 咖啡杯
      rect(x + 10, y + 6, 20, 22, C.cup);
      rect(x + 30, y + 12, 5, 8, C.cup);
      rect(x + 13, y + 8, 14, 6, C.coffee);
      rect(x + 15, y + 2, 3, 5, '#b9c7d9');
      rect(x + 21, y + 1, 3, 5, '#b9c7d9');
    }
    function draw() {
      rect(0, 0, W, H, C.bg);
      // 网格底纹
      ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
      ctx.beginPath();
      for (var gx = 0; gx <= COLS; gx++) { ctx.moveTo(gx * T + .5, 0); ctx.lineTo(gx * T + .5, H); }
      for (var gy = 0; gy <= ROWS; gy++) { ctx.moveTo(0, gy * T + .5); ctx.lineTo(W, gy * T + .5); }
      ctx.stroke();

      // 砖 / 钢
      for (var ty = 0; ty < ROWS; ty++)
        for (var tx = 0; tx < COLS; tx++) {
          var v = map[ty][tx];
          if (v === BRICK) {
            rect(tx * T, ty * T, T, T, C.brickDark);
            for (var b = 0; b < 2; b++) {
              rect(tx * T + 1, ty * T + 1 + b * 10, 18, 8, C.brick);
              rect(tx * T + 1 + (b % 2 ? 9 : 0), ty * T + 1 + b * 10, 9, 8, C.brickDark);
            }
          } else if (v === STEEL) {
            rect(tx * T + 2, ty * T + 2, 16, 16, C.steel);
            rect(tx * T + 4, ty * T + 4, 6, 6, C.steelHi);
          }
        }
      drawBase();

      // 玩家
      if (player.alive && (invuln <= 0 || (invuln >> 2) % 2 === 0))
        drawTank(player, C.player, C.playerDark);

      // 敌人
      for (var i = 0; i < enemies.length; i++) {
        var e = enemies[i];
        if (e.spawnAnim > 0) {
          var r = 12 - e.spawnAnim / 8;
          ctx.strokeStyle = C.gold; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(e.x, e.y, Math.max(2, r), 0, 7); ctx.stroke();
          continue;
        }
        drawTank(e, [C.e1, C.e2, C.e3, C.e4][e.type], '#333c46');
      }

      // 子弹
      ctx.fillStyle = '#f5f7fa';
      for (var bl = 0; bl < bullets.length; bl++)
        ctx.fillRect(Math.round(bullets[bl].x) - 1.5, Math.round(bullets[bl].y) - 1.5, 3, 3);

      // 道具
      for (var p = 0; p < powerups.length; p++) {
        var pu = powerups[p];
        if ((pu.t >> 3) % 2 === 0) {
          rect(pu.x - 6, pu.y - 5, 12, 10, '#ff9f64');
          rect(pu.x + 6, pu.y - 3, 3, 6, '#ff9f64');
          rect(pu.x - 3, pu.y - 8, 2, 3, '#f5f7fa');
          rect(pu.x + 1, pu.y - 9, 2, 3, '#f5f7fa');
        }
      }

      // 爆炸粒子
      for (var bo = 0; bo < booms.length; bo++) {
        var pt = booms[bo];
        ctx.globalAlpha = Math.max(0, pt.life / 26);
        rect(pt.x, pt.y, 2, 2, pt.color);
      }
      ctx.globalAlpha = 1;

      // HUD
      ctx.font = '10px monospace'; ctx.fillStyle = C.hud; ctx.textAlign = 'left';
      ctx.fillText('SCORE ' + score, 8, 14);
      ctx.fillText('WAVE ' + wave, 8, 26);
      var life = ''; for (var l = 0; l < lives; l++) life += '▮';
      ctx.fillStyle = C.player;
      ctx.fillText('命 ' + (life || '—'), W - 70, 14);
      var foes = ''; for (var f = 0; f < Math.min(left, 12); f++) foes += '▪';
      ctx.fillStyle = C.e1; ctx.textAlign = 'right';
      ctx.fillText(foes, W - 8, 26);
      if (performance.now() < powerUntil) {
        ctx.fillStyle = '#ff9f64'; ctx.textAlign = 'center';
        ctx.fillText('☕ 双管炮', W / 2, 14);
      }

      // 过关提示
      if (banner > 0) {
        ctx.fillStyle = 'rgba(6,10,16,.6)'; ctx.fillRect(0, H / 2 - 26, W, 52);
        ctx.textAlign = 'center';
        ctx.font = 'bold 18px monospace'; ctx.fillStyle = C.gold;
        ctx.fillText('第 ' + wave + ' 波来袭', W / 2, H / 2 + 6);
      }
    }

    // ---------- 初始化 ----------
    function reset() {
      score = 0; lives = 3; wave = 1;
      left = 5 + wave * 2; spawnTimer = 30;
      powerUntil = 0; splash = 0; banner = 0; respawnT = 0;
      enemies = []; bullets = []; powerups = []; booms = [];
      buildMap();
      spawnPlayer();
    }
    reset();

    return { update: update, draw: draw, onKey: onKey };
  }
});
