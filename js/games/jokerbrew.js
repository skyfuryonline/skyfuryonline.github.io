// games/jokerbrew.js — 小丑杯（JOKER BREW · 2024）
// Balatro 式扑克肉鸽迷你版：牌型 → 筹码×倍率 → 商店买小丑滚雪球 → ante 递增。
// 3 条命，打不过盲注扣命重打；高分为本局累计总分。

// ================= 顶层纯函数（module 可单测） =================
var JOKERBREW_SUITS = ['♠', '♥', '♦', '♣'];

function jokerbrewRankStr(r) {
  return r === 14 ? 'A' : r === 13 ? 'K' : r === 12 ? 'Q' : r === 11 ? 'J' : '' + r;
}
function jokerbrewCardChips(r) {
  return r === 14 ? 11 : r > 10 ? 10 : r;
}

// cards: [{r:2..14, s:0..3}]，1~5 张，返回最优牌型
function jokerbrewDetect(cards) {
  var n = cards.length, i;
  var counts = {};
  for (i = 0; i < n; i++) counts[cards[i].r] = (counts[cards[i].r] || 0) + 1;
  var countsArr = [];
  for (var k in counts) countsArr.push(counts[k]);
  countsArr.sort(function (a, b) { return b - a; });
  var flush = n === 5;
  for (i = 1; i < n; i++) if (cards[i].s !== cards[0].s) flush = false;
  var ranks = cards.map(function (c) { return c.r; }).sort(function (a, b) { return a - b; });
  var straight = false;
  if (n === 5) {
    straight = true;
    for (i = 0; i < 4; i++) if (ranks[i + 1] !== ranks[i] + 1) straight = false;
    if (!straight && ranks[0] === 2 && ranks[1] === 3 && ranks[2] === 4 && ranks[3] === 5 && ranks[4] === 14) straight = true; // A2345
  }
  if (straight && flush) return { key: 'sf', label: '同花顺', chips: 100, mult: 8 };
  if (countsArr[0] === 4) return { key: 'four', label: '四条', chips: 60, mult: 3 };
  if (n === 5 && countsArr[0] === 3 && countsArr[1] === 2) return { key: 'full', label: '葫芦', chips: 40, mult: 3 };
  if (flush) return { key: 'flush', label: '同花', chips: 35, mult: 4 };
  if (straight) return { key: 'straight', label: '顺子', chips: 30, mult: 4 };
  if (countsArr[0] === 3) return { key: 'three', label: '三条', chips: 30, mult: 3 };
  if (countsArr[0] === 2 && countsArr[1] === 2) return { key: 'two', label: '两对', chips: 20, mult: 2 };
  if (countsArr[0] === 2) return { key: 'pair', label: '对子', chips: 10, mult: 2 };
  return { key: 'high', label: '高牌', chips: 5, mult: 1 };
}

// 16 个小丑（数据驱动；ch 像素图标字符）
var JOKERBREW_JOKERS = [
  { id: 'espresso', name: '双份浓缩', ch: '浓', color: '#c05a5a', price: 4, mult: 4, desc: '+4 倍率' },
  { id: 'oat', name: '燕麦奶', ch: '奶', color: '#e8d8b0', price: 4, chips: 80, desc: '+80 筹码' },
  { id: 'arabica', name: '精品豆', ch: '豆', color: '#3fae5a', price: 4, perCardChips: { suit: 1, v: 15 }, desc: '每张♥ +15 筹码' },
  { id: 'darkroast', name: '深烘豆', ch: '烘', color: '#6b4226', price: 5, perCardMult: { suit: 0, v: 2 }, desc: '每张♠ +2 倍率' },
  { id: 'latteart', name: '拉花艺术', ch: '花', color: '#f5e9d6', price: 5, xIf: { keys: ['flush'], v: 3 }, desc: '出同花 ×3 倍率' },
  { id: 'moka', name: '摩卡壶', ch: '壶', color: '#a0522d', price: 6, xIf: { keys: ['full'], v: 4 }, desc: '出葫芦 ×4 倍率' },
  { id: 'cinnamon', name: '肉桂粉', ch: '桂', color: '#d4a05e', price: 4, multIf: { keys: ['pair', 'two'], v: 8 }, desc: '出对子/两对 +8 倍率' },
  { id: 'caffeine', name: '咖啡因上头', ch: '因', color: '#ffd166', price: 6, growOn: { keys: ['three', 'full', 'four', 'sf'], v: 1 }, desc: '出三条以上，永久 +1 倍率' },
  { id: 'iced', name: '冰美式', ch: '冰', color: '#5aa7c0', price: 5, xIf: { keys: ['straight'], v: 3 }, desc: '出顺子 ×3 倍率' },
  { id: 'coaster', name: '幸运杯垫', ch: '运', color: '#8899bb', price: 4, xChance: { p: 0.25, v: 2 }, desc: '出牌后 25% 概率 ×2' },
  { id: 'refill', name: '续杯', ch: '续', color: '#3fae5a', price: 3, income: { onWin: 3 }, desc: '过关时额外 +$3' },
  { id: 'tips', name: '小费罐', ch: '费', color: '#ffd166', price: 3, income: { onDiscard: 1 }, desc: '每次弃牌 +$1' },
  { id: 'roaster', name: '烘焙师', ch: '焙', color: '#b4622d', price: 6, xIf: { keys: ['four'], v: 6 }, desc: '出四条 ×6 倍率' },
  { id: 'kettle', name: '手冲壶', ch: '冲', color: '#5a9e73', price: 4, perCardChips: { suit: 3, v: 15 }, desc: '每张♣ +15 筹码' },
  { id: 'caramel', name: '焦糖玛奇朵', ch: '糖', color: '#e5a04c', price: 5, chips: 40, mult: 2, desc: '+40 筹码 +2 倍率' },
  { id: 'decaf', name: '低因咖啡', ch: '低', color: '#8899bb', price: 4, xIf: { keys: ['high'], v: 4 }, desc: '出高牌 ×4 倍率' }
];

ARCADE.register({
  id: 'jokerbrew',
  cn: '小丑杯',
  en: 'JOKER BREW · 2024',
  create: function (api) {
    var ctx = api.ctx, W = api.W, H = api.H;
    var C = {
      bg: '#171009', felt: '#1d150b', feltDot: '#241a0e',
      panel: '#241a10', panelEdge: '#3d2e1a',
      text: '#f5f0e6', dim: '#8899bb', gold: '#ffd166',
      chip: '#2d7a8c', chipHi: '#4db8cc', multC: '#c05a5a', multHi: '#e06c75',
      good: '#3fae5a', miss: '#e06c75',
      cardFace: '#f5f0e6', cardEdge: '#b8ab90', cardSel: '#ffd166',
      red: '#c0392b', black: '#1a2530'
    };
    // ---- 布局 ----
    var CARD_W = 44, CARD_H = 62, HAND_N = 8, GAP = 5;
    var HAND_X = Math.round((W - (CARD_W * HAND_N + GAP * (HAND_N - 1))) / 2);
    var HAND_Y = 268, HAND_Y_SEL = 254;
    var BTN_PLAY = { x: 46, y: 336, w: 110, h: 18 };
    var BTN_DISC = { x: 166, y: 336, w: 110, h: 18 };
    var BTN_TABLE = { x: 10, y: 58, w: 76, h: 20 };
    var JOKER_N = 5, JK_W = 44, JK_H = 40, JK_GAP = 6;
    var JK_X0 = W - 10 - JOKER_N * JK_W - (JOKER_N - 1) * JK_GAP; // 226
    var JK_Y = 52;
    var STAGE_Y = 104, BIG_W = 56, BIG_H = 78, BIG_GAP = 8;

    // ---- 状态 ----
    var deck, hand, sel, cursor;
    var phase, phaseT, anim;
    var ante, target, roundScore, hands, discards, lives, score, money;
    var jokers, shopOffers, shopReroll, tableOpen, tipIdx, tipT;
    var popups, parts, shakeT, bannerT, frame, over;

    api.panel([['←→+ENTER', '选牌'], ['SPACE', '出牌'], ['D', '弃牌'], ['P', '暂停']],
      '机台秘技：对子是前期主力；「咖啡因上头」滚到后期是核弹——小丑要买得早');

    function onKey(k, down) {
      if (!down || over) return;
      if (tableOpen) { tableOpen = false; return; }
      if (phase === 'anim') { if (anim) anim.t = 999; return; }
      if (phase === 'won' || phase === 'lost') { phaseT = 1; return; }
      if (phase === 'shop') {
        if (k === 'Enter') leaveShop();
        if (k === 'r' || k === 'R') doReroll();
        return;
      }
      if (phase !== 'play') return;
      if (k === 'ArrowLeft') { cursor = (cursor + HAND_N - 1) % HAND_N; }
      else if (k === 'ArrowRight') { cursor = (cursor + 1) % HAND_N; }
      else if (k === 'Enter') { toggleSel(cursor); }
      else if (k === ' ') { play(); }
      else if (k === 'd' || k === 'D') { doDiscard(); }
      else if (k === '?') { tableOpen = true; }
    }

    function onPointer(type, x, y) {
      if (type !== 'down' || over) return;
      if (tableOpen) { tableOpen = false; return; }
      if (phase === 'anim') { if (anim) anim.t = 999; return; }
      if (phase === 'won' || phase === 'lost') { phaseT = 1; return; }
      if (phase === 'shop') {
        // 小丑栏 tooltip
        for (var i = 0; i < jokers.length; i++) {
          var jx = JK_X0 + i * (JK_W + JK_GAP);
          if (x >= jx && x <= jx + JK_W && y >= JK_Y && y <= JK_Y + JK_H) { tipIdx = i; tipT = 130; return; }
        }
        // 商品
        for (var o = 0; o < 2; o++) {
          if (shopOffers[o] && inRect(x, y, SHOP_OFFER_X[o], 92, 130, 130)) { buy(o); return; }
        }
        if (inRect(x, y, 70, 240, 120, 24)) { doReroll(); return; }
        if (inRect(x, y, 240, 240, 120, 24)) { leaveShop(); return; }
        return;
      }
      if (phase !== 'play') return;
      if (inRect(x, y, BTN_TABLE.x, BTN_TABLE.y, BTN_TABLE.w, BTN_TABLE.h)) { tableOpen = true; return; }
      for (var c = 0; c < hand.length; c++) {
        var cy = sel[c] ? HAND_Y_SEL : HAND_Y;
        if (x >= HAND_X + c * (CARD_W + GAP) && x <= HAND_X + c * (CARD_W + GAP) + CARD_W &&
            y >= cy && y <= cy + CARD_H) { toggleSel(c); return; }
      }
      if (inRect(x, y, BTN_PLAY.x, BTN_PLAY.y, BTN_PLAY.w, BTN_PLAY.h)) { play(); return; }
      if (inRect(x, y, BTN_DISC.x, BTN_DISC.y, BTN_DISC.w, BTN_DISC.h)) { doDiscard(); return; }
    }
    function inRect(x, y, rx, ry, rw, rh) { return x >= rx && x <= rx + rw && y >= ry && y <= ry + rh; }

    // ---------- 牌堆与回合 ----------
    function newDeck() {
      var d = [];
      for (var s = 0; s < 4; s++) for (var r = 2; r <= 14; r++) d.push({ r: r, s: s });
      for (var i = d.length - 1; i > 0; i--) {
        var j = (Math.random() * (i + 1)) | 0, t = d[i]; d[i] = d[j]; d[j] = t;
      }
      return d;
    }
    function anteTarget(n) { return Math.round(250 * Math.pow(2.6, n - 1) / 10) * 10; }

    function newRound() {
      deck = newDeck();
      hand = [];
      for (var i = 0; i < HAND_N; i++) hand.push(deck.pop());
      sel = []; for (i = 0; i < HAND_N; i++) sel.push(false);
      cursor = 0;
      hands = 5; discards = 3;
      roundScore = 0;
      target = anteTarget(ante);
      bannerT = 50;
      phase = 'play'; phaseT = 0; anim = null;
      tipIdx = -1; tipT = 0;
    }

    function reset() {
      ante = 1; lives = 3; score = 0; money = 3;
      jokers = []; shopOffers = [null, null]; shopReroll = 2;
      tableOpen = false; tipIdx = -1; tipT = 0;
      popups = []; parts = []; shakeT = 0; bannerT = 0;
      frame = 0; over = false;
      newRound();
    }

    // ---------- 选择 / 出牌 / 弃牌 ----------
    function selectedIdx() {
      var r = [];
      for (var i = 0; i < hand.length; i++) if (sel[i]) r.push(i);
      return r;
    }
    function toggleSel(i) {
      if (phase !== 'play') return;
      if (!sel[i] && selectedIdx().length >= 5) { pop(240, 240, '最多选 5 张', C.miss, false); return; }
      sel[i] = !sel[i];
    }

    function resolveScore(cards, forPreview) {
      var d = jokerbrewDetect(cards);
      var chips = d.chips, mult = d.mult, i, j;
      for (i = 0; i < cards.length; i++) chips += jokerbrewCardChips(cards[i].r);
      // 加算
      for (i = 0; i < jokers.length; i++) {
        var def = jokers[i].def, v = jokers[i].v || 0;
        if (def.chips) chips += def.chips;
        if (def.mult) mult += def.mult + v;
        if (def.perCardChips) for (j = 0; j < cards.length; j++) if (cards[j].s === def.perCardChips.suit) chips += def.perCardChips.v;
        if (def.perCardMult) for (j = 0; j < cards.length; j++) if (cards[j].s === def.perCardMult.suit) mult += def.perCardMult.v;
        if (def.multIf && def.multIf.keys.indexOf(d.key) >= 0) mult += def.multIf.v;
      }
      // 乘算（预览时跳过概率型）
      for (i = 0; i < jokers.length; i++) {
        var def2 = jokers[i].def;
        if (def2.xIf && def2.xIf.keys.indexOf(d.key) >= 0) mult *= def2.xIf.v;
        if (def2.xChance && !forPreview && Math.random() < def2.xChance.p) mult *= def2.xChance.v;
      }
      // 成长（预览不成长）
      if (!forPreview) {
        for (i = 0; i < jokers.length; i++) {
          var def3 = jokers[i].def;
          if (def3.growOn && def3.growOn.keys.indexOf(d.key) >= 0) jokers[i].v += def3.growOn.v;
        }
      }
      return { d: d, chips: chips, mult: mult, total: chips * mult };
    }

    function play() {
      if (phase !== 'play' || hands <= 0) return;
      var idx = selectedIdx();
      if (!idx.length) return;
      var cards = [];
      for (var i = 0; i < idx.length; i++) cards.push(hand[idx[i]]);
      var r = resolveScore(cards, false);
      score += r.total;
      roundScore += r.total;
      hands--;
      anim = { idx: idx, cards: cards, d: r.d, chips: r.chips, mult: r.mult, total: r.total, t: 0, burst: false };
      phase = 'anim'; phaseT = 0;
    }

    function doDiscard() {
      if (phase !== 'play' || discards <= 0) return;
      var idx = selectedIdx();
      if (!idx.length) return;
      var flag = [];
      for (var i = 0; i < hand.length; i++) flag[i] = false;
      for (i = 0; i < idx.length; i++) flag[idx[i]] = true;
      var kept = [];
      for (i = 0; i < hand.length; i++) if (!flag[i]) kept.push(hand[i]);
      for (i = 0; i < idx.length; i++) kept.push(deck.pop());
      hand = kept;
      sel = []; for (i = 0; i < HAND_N; i++) sel.push(false);
      discards--;
      for (i = 0; i < jokers.length; i++) {
        var inc = jokers[i].def.income;
        if (inc && inc.onDiscard) { money += inc.onDiscard; pop(300, 344, '+$' + inc.onDiscard, C.gold, false); }
      }
      pop(240, 250, '弃了 ' + idx.length + ' 张', C.dim, false);
    }

    function finishPlay() {
      var flag = [];
      for (var i = 0; i < hand.length; i++) flag[i] = false;
      for (i = 0; i < anim.idx.length; i++) flag[anim.idx[i]] = true;
      var kept = [];
      for (i = 0; i < hand.length; i++) if (!flag[i]) kept.push(hand[i]);
      while (kept.length < HAND_N && deck.length) kept.push(deck.pop());
      hand = kept;
      sel = []; for (i = 0; i < HAND_N; i++) sel.push(false);
      anim = null;
      if (roundScore >= target) {
        var income = 3 + hands;
        for (i = 0; i < jokers.length; i++) {
          var inc = jokers[i].def.income;
          if (inc && inc.onWin) income += inc.onWin;
        }
        money += income;
        pop(240, 200, '过关! +$' + income, C.gold, true);
        confetti(240, 160, 26);
        phase = 'won'; phaseT = 80;
      } else if (hands > 0) {
        phase = 'play';
      } else {
        pop(240, 200, '差 ' + (target - roundScore) + ' 分…', C.miss, true);
        shakeT = 14;
        phase = 'lost'; phaseT = 90;
      }
    }

    // ---------- 商店 ----------
    var SHOP_OFFER_X = [70, 240];
    function randomDef(exclude) {
      for (var tries = 0; tries < 30; tries++) {
        var d = JOKERBREW_JOKERS[(Math.random() * JOKERBREW_JOKERS.length) | 0];
        if (!exclude || exclude.indexOf(d) < 0) return d;
      }
      return JOKERBREW_JOKERS[0];
    }
    function startShop() {
      shopOffers = [randomDef(null), randomDef(null)];
      if (shopOffers[1] === shopOffers[0]) shopOffers[1] = randomDef([shopOffers[0]]);
      shopReroll = 2;
      phase = 'shop';
    }
    function buy(i) {
      var def = shopOffers[i];
      if (!def) return;
      if (jokers.length >= JOKER_N) { pop(240, 250, '小丑栏已满', C.miss, false); return; }
      if (money < def.price) { pop(240, 250, '钱不够', C.miss, false); return; }
      money -= def.price;
      jokers.push({ def: def, v: 0 });
      shopOffers[i] = null;
      pop(240, 200, def.name + ' 入队!', C.gold, false);
    }
    function doReroll() {
      if (money < shopReroll) { pop(240, 250, '钱不够', C.miss, false); return; }
      money -= shopReroll;
      shopReroll++;
      shopOffers = [randomDef(null), randomDef(null)];
      if (shopOffers[1] === shopOffers[0]) shopOffers[1] = randomDef([shopOffers[0]]);
    }
    function leaveShop() {
      ante++;
      newRound();
    }

    // ---------- 粒子 / 浮字 ----------
    function pop(x, y, txt, color, big) {
      if (popups.length > 12) popups.shift();
      popups.push({ x: x, y: y, txt: txt, color: color, big: big, life: 50 });
    }
    function burst(x, y, n, color) {
      for (var i = 0; i < n; i++) parts.push({
        x: x, y: y, vx: (Math.random() - 0.5) * 4, vy: -Math.random() * 3 - 0.5,
        life: 24 + Math.random() * 16, color: color
      });
      if (parts.length > 120) parts.splice(0, parts.length - 120);
    }
    function confetti(x, y, n) {
      var cols = [C.gold, C.good, C.multHi, C.chipHi, '#f5e9d6'];
      for (var i = 0; i < n; i++) parts.push({
        x: x + (Math.random() - 0.5) * 160, y: y, vx: (Math.random() - 0.5) * 2, vy: -Math.random() * 2.4 - 0.6,
        life: 30 + Math.random() * 24, color: cols[(Math.random() * cols.length) | 0]
      });
      if (parts.length > 120) parts.splice(0, parts.length - 120);
    }

    // ---------- 主循环 ----------
    function update() {
      frame++;
      if (shakeT > 0) shakeT--;
      if (bannerT > 0) bannerT--;
      if (tipT > 0) { tipT--; if (tipT === 0) tipIdx = -1; }
      for (var q = parts.length - 1; q >= 0; q--) {
        var pt = parts[q];
        pt.x += pt.vx; pt.y += pt.vy; pt.vy += 0.06; pt.life--;
        if (pt.life <= 0) parts.splice(q, 1);
      }
      for (var p = popups.length - 1; p >= 0; p--) {
        popups[p].y -= 0.55; popups[p].life--;
        if (popups[p].life <= 0) popups.splice(p, 1);
      }
      if (over) return;

      if (phase === 'anim') {
        anim.t++;
        // 大分震动 + 爆点
        if (!anim.burst && anim.t >= 46) {
          anim.burst = true;
          if (anim.total >= 250) {
            shakeT = anim.total >= 800 ? 16 : 9;
            burst(240, 210, anim.total >= 800 ? 26 : 14, C.gold);
          }
        }
        if (anim.t >= 84) finishPlay();
      } else if (phase === 'won') {
        phaseT--;
        if (phaseT <= 0) startShop();
      } else if (phase === 'lost') {
        phaseT--;
        if (phaseT <= 0) {
          lives--;
          if (lives <= 0) { over = true; api.gameOver(score); return; }
          newRound();
        }
      }
    }

    // ---------- 绘制 ----------
    function rect(x, y, w, h, c) { ctx.fillStyle = c; ctx.fillRect(Math.round(x), Math.round(y), w, h); }
    function txt(x, y, str, size, color, align, bold) {
      ctx.font = (bold ? 'bold ' : '') + size + 'px monospace';
      ctx.textAlign = align || 'left';
      ctx.fillStyle = color;
      ctx.fillText(str, x, y);
    }

    function drawCard(x, y, card, w, h, selected, ghost) {
      var red = card.s === 1 || card.s === 2;
      ctx.globalAlpha = ghost ? 0.45 : 1;
      rect(x, y, w, h, selected ? '#fff6d8' : C.cardFace);
      ctx.strokeStyle = selected ? C.cardSel : C.cardEdge;
      ctx.lineWidth = selected ? 2.5 : 1;
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
      ctx.fillStyle = red ? C.red : C.black;
      ctx.font = 'bold ' + Math.round(h * 0.21) + 'px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(jokerbrewRankStr(card.r), x + 5, y + h * 0.25);
      ctx.font = Math.round(h * 0.17) + 'px monospace';
      ctx.fillText(JOKERBREW_SUITS[card.s], x + 5, y + h * 0.45);
      ctx.font = Math.round(h * 0.34) + 'px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(JOKERBREW_SUITS[card.s], x + w / 2, y + h * 0.82);
      ctx.globalAlpha = 1;
    }

    function drawJokerBadge(x, y, w, h, def, dimmed) {
      rect(x, y, w, h, def.color);
      rect(x, y, w, 3, 'rgba(255,255,255,.25)');
      rect(x, y + h - 3, w, 3, 'rgba(0,0,0,.3)');
      ctx.font = 'bold ' + Math.round(h * 0.42) + 'px monospace';
      ctx.textAlign = 'center';
      ctx.fillStyle = '#1a1208';
      ctx.fillText(def.ch, x + w / 2, y + h * 0.66);
      if (dimmed) { rect(x, y, w, h, 'rgba(23,16,9,.55)'); }
    }

    function drawTop() {
      txt(10, 22, 'ANTE ' + ante, 15, C.gold, 'left', true);
      txt(10, 40, '目标 ' + target + ' · 当前 ' + roundScore, 10, roundScore >= target ? C.good : C.text, 'left');
      txt(W - 10, 22, 'SCORE ' + score, 11, C.text, 'right', true);
      // 命：3 只小杯
      for (var i = 0; i < 3; i++) {
        var hx = W - 18 - i * 16;
        rect(hx, 30, 11, 9, i < lives ? C.cardFace : '#3a2f22');
        rect(hx + 11, 32, 3, 4, i < lives ? C.cardFace : '#3a2f22');
      }
    }

    function drawJokerTip() {
      if (tipIdx < 0 || tipT <= 0 || !jokers[tipIdx]) return;
      var def = jokers[tipIdx].def;
      var tx = Math.min(JK_X0, W - 170);
      rect(tx, JK_Y + JK_H + 4, 164, 34, 'rgba(6,4,2,.92)');
      ctx.strokeStyle = C.gold; ctx.lineWidth = 1;
      ctx.strokeRect(tx + 0.5, JK_Y + JK_H + 4.5, 163, 33);
      txt(tx + 6, JK_Y + JK_H + 18, def.name, 10, C.gold, 'left', true);
      var d = def.desc;
      if (def.growOn) d += '（当前 +' + (jokers[tipIdx].v || 0) + '）';
      txt(tx + 6, JK_Y + JK_H + 31, d, 9, C.text, 'left');
    }

    function drawJokerRow() {
      // 牌型表按钮
      rect(BTN_TABLE.x, BTN_TABLE.y, BTN_TABLE.w, BTN_TABLE.h, C.panel);
      ctx.strokeStyle = C.panelEdge; ctx.lineWidth = 1;
      ctx.strokeRect(BTN_TABLE.x + 0.5, BTN_TABLE.y + 0.5, BTN_TABLE.w - 1, BTN_TABLE.h - 1);
      txt(BTN_TABLE.x + BTN_TABLE.w / 2, BTN_TABLE.y + 14, '牌 型 表', 10, C.dim, 'center', true);
      // 小丑栏
      txt(JK_X0 - 6, JK_Y + 26, '小丑', 10, C.dim, 'right', true);
      for (var i = 0; i < JOKER_N; i++) {
        var jx = JK_X0 + i * (JK_W + JK_GAP);
        rect(jx, JK_Y, JK_W, JK_H, '#100b06');
        ctx.strokeStyle = C.panelEdge; ctx.strokeRect(jx + 0.5, JK_Y + 0.5, JK_W - 1, JK_H - 1);
        if (i < jokers.length) {
          drawJokerBadge(jx, JK_Y, JK_W, JK_H, jokers[i].def, tipIdx === i && tipT > 0);
        }
      }
      drawJokerTip();
    }

    // 牌型表覆盖层
    var PAYTABLE = [
      ['同花顺', 100, 8], ['四条', 60, 3], ['葫芦', 40, 3], ['同花', 35, 4],
      ['顺子', 30, 4], ['三条', 30, 3], ['两对', 20, 2], ['对子', 10, 2], ['高牌', 5, 1]
    ];
    function drawPaytable() {
      ctx.fillStyle = 'rgba(6,4,2,.86)';
      ctx.fillRect(0, 0, W, H);
      rect(90, 40, 300, 280, C.panel);
      ctx.strokeStyle = C.gold; ctx.lineWidth = 2;
      ctx.strokeRect(91, 41, 298, 278);
      txt(240, 66, '牌 型 一 览', 16, C.gold, 'center', true);
      txt(240, 84, '每张所选牌再加面值筹码', 9, C.dim, 'center');
      for (var i = 0; i < PAYTABLE.length; i++) {
        var y = 106 + i * 22;
        txt(130, y, PAYTABLE[i][0], 12, C.text, 'left', true);
        rect(250, y - 12, 56, 16, C.chip);
        txt(278, y, PAYTABLE[i][1], 10, '#fff', 'center', true);
        txt(314, y, '×', 11, C.dim, 'center', true);
        rect(326, y - 12, 44, 16, C.multC);
        txt(348, y, PAYTABLE[i][2], 10, '#fff', 'center', true);
      }
      txt(240, 306, '按任意键 / 点击返回', 9, C.dim, 'center');
    }

    function drawStage() {
      if (phase === 'anim' && anim) {
        // 打出的牌上浮到舞台
        var n = anim.cards.length;
        var x0 = 240 - (n * BIG_W + (n - 1) * BIG_GAP) / 2;
        var rise = Math.min(1, anim.t / 12);
        for (var i = 0; i < n; i++) {
          var fx = HAND_X + anim.idx[i] * (CARD_W + GAP);
          var fy = sel[anim.idx[i]] ? HAND_Y_SEL : HAND_Y;
          var gx = x0 + i * (BIG_W + BIG_GAP);
          var x = fx + (gx - fx) * rise, y = fy + ((STAGE_Y + 8) - fy) * rise;
          drawCard(x, y, anim.cards[i], BIG_W, BIG_H, false, false);
        }
        var label = anim.d.label;
        txt(240, 200, label, 15, C.text, 'center', true);
        // 数字条：筹码 × 倍率 = 总分
        var cShow = anim.t < 24 ? Math.round(anim.chips * Math.min(1, anim.t / 24)) : anim.chips;
        var mShow = anim.t < 40 ? Math.round(anim.mult * Math.max(0, Math.min(1, (anim.t - 24) / 16))) : anim.mult;
        var tShow = anim.t < 56 ? Math.round(anim.total * Math.max(0, Math.min(1, (anim.t - 40) / 16))) : anim.total;
        var scale = anim.t >= 56 ? 1 + Math.max(0, 1 - (anim.t - 56) / 10) * 0.5 : 1;
        rect(122, 210, 78, 24, C.chip);
        txt(161, 227, '' + cShow, 14, '#fff', 'center', true);
        txt(210, 227, '×', 14, C.dim, 'center', true);
        rect(222, 210, 62, 24, C.multC);
        txt(253, 227, '' + mShow, 14, '#fff', 'center', true);
        txt(296, 227, '=', 14, C.dim, 'center', true);
        ctx.save();
        ctx.translate(240 + 128, 222);
        ctx.scale(scale, scale);
        txt(0, 6, '' + tShow, 19, C.gold, 'center', true);
        ctx.restore();
      } else if (phase === 'play') {
        // 静置提示 + 上一手余韵
        txt(240, 160, '点选手牌 · 组成牌型', 11, C.dim, 'center');
        txt(240, 178, '打到目标分就过关', 9, '#5a4a30', 'center');
      }
    }

    function drawPreview() {
      if (phase !== 'play') return;
      var idx = selectedIdx();
      if (!idx.length) { txt(240, 246, '', 10, C.dim, 'center'); return; }
      var cards = [];
      for (var i = 0; i < idx.length; i++) cards.push(hand[idx[i]]);
      var r = resolveScore(cards, true);
      var col = r.total + roundScore >= target ? C.good : C.text;
      txt(240, 244, '→ ' + r.d.label + '  ' + r.chips + ' × ' + r.mult + '  ≈ ' + r.total, 12, col, 'center', true);
    }

    function drawHand() {
      if (phase === 'shop') return;
      for (var i = 0; i < hand.length; i++) {
        var y = sel[i] ? HAND_Y_SEL : HAND_Y;
        drawCard(HAND_X + i * (CARD_W + GAP), y, hand[i], CARD_W, CARD_H, sel[i], false);
        if (phase === 'play' && cursor === i) {
          ctx.strokeStyle = 'rgba(245,240,230,.5)';
          ctx.lineWidth = 1;
          ctx.strokeRect(HAND_X + i * (CARD_W + GAP) - 3.5, y - 3.5, CARD_W + 7, CARD_H + 7);
        }
      }
    }

    function drawButtons() {
      if (phase === 'shop') return;
      // 出牌
      var canPlay = phase === 'play' && hands > 0 && selectedIdx().length > 0;
      rect(BTN_PLAY.x, BTN_PLAY.y, BTN_PLAY.w, BTN_PLAY.h, canPlay ? C.good : '#2a3a2c');
      txt(BTN_PLAY.x + BTN_PLAY.w / 2, BTN_PLAY.y + 13, '出 牌 ×' + hands, 11, canPlay ? '#08130a' : '#5a6a5c', 'center', true);
      // 弃牌
      var canDisc = phase === 'play' && discards > 0 && selectedIdx().length > 0;
      rect(BTN_DISC.x, BTN_DISC.y, BTN_DISC.w, BTN_DISC.h, canDisc ? C.chip : '#22303a');
      txt(BTN_DISC.x + BTN_DISC.w / 2, BTN_DISC.y + 13, '弃 牌 ×' + discards, 11, canDisc ? '#06131a' : '#4a5a64', 'center', true);
      // 钱 + 牌堆
      txt(292, 350, '$ ' + money, 14, C.gold, 'left', true);
      txt(W - 12, 350, '牌堆 ' + deck.length, 10, C.dim, 'right');
    }

    function drawShop() {
      ctx.fillStyle = 'rgba(6,4,2,.82)';
      ctx.fillRect(0, 0, W, H);
      rect(40, 30, 400, 300, C.panel);
      ctx.strokeStyle = C.gold; ctx.lineWidth = 2;
      ctx.strokeRect(41, 31, 398, 298);
      txt(240, 58, '咖 啡 商 店', 17, C.gold, 'center', true);
      txt(240, 76, '下一关 ANTE ' + (ante + 1) + ' · 目标 ' + anteTarget(ante + 1), 10, C.dim, 'center');
      txt(428, 58, '栏位 ' + jokers.length + '/' + JOKER_N, 10, C.dim, 'right', true);
      for (var o = 0; o < 2; o++) {
        var x = SHOP_OFFER_X[o], def = shopOffers[o];
        rect(x, 92, 130, 130, '#1a1208');
        ctx.strokeStyle = def ? C.panelEdge : '#2a2013';
        ctx.strokeRect(x + 0.5, 92.5, 129, 129);
        if (def) {
          drawJokerBadge(x + 41, 102, 48, 30, def, false);
          txt(x + 65, 152, def.name, 11, C.text, 'center', true);
          // 描述按 12 字折行
          var words = def.desc, line = '', ly = 168, lines = [];
          for (var wi = 0; wi < words.length; wi++) {
            line += words[wi];
            if (line.length >= 10) { lines.push(line); line = ''; }
          }
          if (line) lines.push(line);
          for (var li = 0; li < lines.length && li < 3; li++) txt(x + 65, ly + li * 12, lines[li], 9, C.dim, 'center');
          txt(x + 65, 212, '$ ' + def.price, 12, money >= def.price ? C.gold : C.miss, 'center', true);
        } else {
          txt(x + 65, 160, '已 购', 13, '#5a4a30', 'center', true);
        }
      }
      // 按钮
      rect(70, 240, 120, 24, money >= shopReroll ? C.chip : '#22303a');
      txt(130, 256, '重掷 $' + shopReroll, 11, money >= shopReroll ? '#06131a' : '#4a5a64', 'center', true);
      rect(240, 240, 120, 24, C.good);
      txt(300, 256, '继续 →', 12, '#08130a', 'center', true);
      txt(240, 288, '点击小丑卡购买 · 点击上方小丑栏查看说明', 9, C.dim, 'center');
      txt(240, 304, 'R 重掷 / ENTER 出发', 9, C.dim, 'center');
      drawJokerTip(); // 商店覆盖层之上再画一次，否则被压暗
    }

    function draw() {
      ctx.save();
      if (shakeT > 0) ctx.translate((Math.random() - 0.5) * 6, (Math.random() - 0.5) * 6);
      // 咖啡馆毛毡桌面
      rect(0, 0, W, H, C.bg);
      rect(0, 0, W, 46, C.felt);
      rect(0, 46, W, 2, C.panelEdge);
      for (var d = 0; d < 26; d++) {
        rect((d * 173 + 21) % W, 52 + (d * 89) % 290, 2, 2, C.feltDot);
      }
      drawTop();
      drawJokerRow();
      drawStage();
      drawPreview();
      drawHand();
      drawButtons();

      // 粒子 / 浮字
      for (var q = 0; q < parts.length; q++) {
        ctx.globalAlpha = Math.max(0, parts[q].life / 30);
        rect(parts[q].x, parts[q].y, 2, 2, parts[q].color);
      }
      ctx.globalAlpha = 1;
      for (var p = 0; p < popups.length; p++) {
        var pp = popups[p];
        ctx.globalAlpha = Math.min(1, pp.life / 18);
        txt(pp.x, pp.y, pp.txt, pp.big ? 17 : 11, pp.color, 'center', true);
      }
      ctx.globalAlpha = 1;

      // 新回合横幅
      if (bannerT > 0 && phase === 'play') {
        ctx.globalAlpha = Math.min(1, bannerT / 12);
        ctx.fillStyle = 'rgba(6,4,2,.72)';
        ctx.fillRect(0, 148, W, 52);
        txt(240, 172, 'ANTE ' + ante, 19, C.gold, 'center', true);
        txt(240, 190, '目标 ' + target + ' · 打不过扣一条命', 10, C.dim, 'center');
        ctx.globalAlpha = 1;
      }
      // 过关 / 失败横幅
      if (phase === 'won') {
        ctx.fillStyle = 'rgba(6,4,2,.6)';
        ctx.fillRect(0, 140, W, 68);
        txt(240, 172, '过 关 !', 22, C.gold, 'center', true);
        txt(240, 194, '去商店给自己添个帮手', 10, C.text, 'center');
      } else if (phase === 'lost') {
        ctx.fillStyle = 'rgba(60,10,10,.5)';
        ctx.fillRect(0, 140, W, 68);
        txt(240, 172, '没 打 够…', 22, C.miss, 'center', true);
        txt(240, 194, lives > 1 ? '还剩 ' + (lives - 1) + ' 条命，重打 ANTE ' + ante : '最后一条命也没了…', 10, C.text, 'center');
      }

      if (phase === 'shop') drawShop();
      if (tableOpen) drawPaytable();
      ctx.restore();
    }

    reset();
    return { update: update, draw: draw, onKey: onKey, onPointer: onPointer };
  }
});

// 供 Node 单元测试（浏览器环境无 module，跳过）
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    jokerbrewDetect: jokerbrewDetect,
    jokerbrewCardChips: jokerbrewCardChips,
    jokerbrewRankStr: jokerbrewRankStr,
    JOKERBREW_JOKERS: JOKERBREW_JOKERS
  };
}
