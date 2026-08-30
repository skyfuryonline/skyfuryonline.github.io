    // -------- GitHub Intelligence Modal Logic --------
    document.addEventListener('DOMContentLoaded', function () {
      var cards = document.querySelectorAll('.daily-article-card');
      var modalJq = $('#summaryModal');
      var modalContent = document.getElementById('summaryModal').querySelector('.modal-content');
      var modalTitle = document.getElementById('summaryModalLabel');
      var modalSummary = document.getElementById('modalSummaryContent');
      var modalGallery = document.getElementById('modalImageGallery');
      var modalReadMoreBtn = document.getElementById('modalReadMoreBtn');

      modalReadMoreBtn.style.display = 'none';

      cards.forEach(function (card) {
        card.addEventListener('click', function () {
          var title = this.getAttribute('data-title');
          var summary = this.getAttribute('data-summary');
          var cachePath = this.getAttribute('data-cache-path');
          var images = this.getAttribute('data-images').split(',').filter(Boolean);

          modalTitle.textContent = title;

          if (typeof marked !== 'undefined') {
            modalSummary.innerHTML = marked.parse(summary);
            modalSummary.className = 'markdown-body';
          } else {
            modalSummary.textContent = summary;
            modalSummary.className = '';
          }

          modalContent.classList.add('trend-radar-theme');

          modalGallery.innerHTML = '';
          if (images.length > 0) {
            images.forEach(function(imageFile) {
              var galleryItem = document.createElement('div');
              galleryItem.className = 'gallery-item';
              var img = document.createElement('img');
              img.src = SITE_BASEURL + '/' + cachePath + '/' + imageFile;
              galleryItem.appendChild(img);
              modalGallery.appendChild(galleryItem);
            });
            modalGallery.style.display = 'flex';
          } else {
            modalGallery.style.display = 'none';
          }

          modalJq.modal('show');
        });
      });
    });
    // -------- 时钟与倒计时 --------
function updateClock() {
  const clock = document.getElementById("clock");
  if (!clock) return;
  const now = new Date();
  // 设置时区
  const options = { timeZone: 'Asia/Shanghai', hour: '2-digit', minute: '2-digit', second: '2-digit' };
  clock.textContent = now.toLocaleTimeString('zh-CN', options);
}
function updateCountdown() {
  const countdown = document.getElementById("countdown");
  if (!countdown) return;
  // 倒计时的目标时间
  const targetDate = new Date("2026-08-28T00:00:00");
  const now = new Date();
  const diff = targetDate - now;
  const days = Math.max(0, Math.floor(diff / (1000 * 60 * 60 * 24)));
  // 倒计时文案
  countdown.textContent = `距离暑假结束还有 ${days} 天`;
}
setInterval(updateClock, 1000);
setInterval(updateCountdown, 3600000);
updateClock();
updateCountdown();




function renderGwyInfoBar() {
  var el = document.getElementById('gwyInfoBar');
  if (!el) return;
  var now = new Date();
  var weekdays = ['日','一','二','三','四','五','六'];
  var y = now.getFullYear();
  var m = now.getMonth() + 1;
  var d = now.getDate();
  var wd = weekdays[now.getDay()];
  el.innerHTML =
    '<span class="bar-label">📋 公考日报</span>' +
    '<span class="bar-date">' + y + '年' + m + '月' + d + '日 周' + wd + '</span>';
}

function renderGwyDetail(report) {
  var el = document.getElementById('gwyDetail');
  if (!el || !report) return;
  var baseUrl = SITE_BASEURL;
  var dp = report.date.split('-');
  var dateStr = dp[0] + '-' + dp[1] + '-' + dp[2];
  var reportUrl = baseUrl + '/' + report.path;
  el.style.cursor = 'pointer';
  el.onclick = function() { window.open(reportUrl, '_blank', 'noopener'); };
  el.innerHTML =
    '<span class="detail-tag" style="background:#f59e0b;">📋 GWY</span>' +
    '<div class="detail-title">' + report.title + '</div>' +
    (report.summary ? '<div class="detail-summary">' + report.summary + '</div>' : '') +
    '<div class="detail-meta">' + dateStr + (report.source ? ' · ' + report.source : '') + '</div>' +
    '<div class="detail-hint">点击打开报告 →</div>';
}

function renderGwyDateStrip(activeDate) {
  var el = document.getElementById('gwyDateStrip');
  if (!el || GWY_REPORTS.length === 0) return;
  var weekdays = ['日','一','二','三','四','五','六'];
  var html = '<span class="date-label">近期</span>';
  var count = Math.min(GWY_REPORTS.length, 7);
  for (var i = 0; i < count; i++) {
    var r = GWY_REPORTS[i];
    var dp = r.date.split('-');
    var num = parseInt(dp[2]);
    var wd = weekdays[new Date(r.date + 'T00:00:00').getDay()];
    html += '<div class="date-cell' + (r.date === activeDate ? ' active' : '') + '" data-date="' + r.date + '">' +
      '<span class="d-num">' + num + '</span><span class="d-weekday">' + wd + '</span></div>';
  }
  el.innerHTML = html;

  el.querySelectorAll('.date-cell').forEach(function(cell) {
    cell.addEventListener('click', function() {
      var date = this.dataset.date;
      renderGwyDetail(gwyMap[date]);
      el.querySelectorAll('.date-cell').forEach(function(c) { c.classList.remove('active'); });
      this.classList.add('active');
    });
  });
}

// -------- Tuitui rendering --------
function renderTuituiDetail(entry) {
  var el = document.getElementById('tuituiDetail');
  if (!el || !entry) return;
  var qCount = entry.questions ? entry.questions.length : 0;
  var tuCount = entry.questions ? entry.questions.filter(function(q){return q.section==='图推'}).length : 0;
  var lbCount = entry.questions ? entry.questions.filter(function(q){return q.section==='类比'}).length : 0;
  el.style.cursor = 'pointer';
  el.onclick = function() { openTuituiModal(entry.date); };
  el.innerHTML =
    '<span class="detail-tag" style="background:#7b5ea7;">🧩 图推</span>' +
    '<div class="detail-title">每日图推类比 · ' + entry.date + '</div>' +
    '<div class="detail-summary">今日共 ' + qCount + ' 题 — 图形推理 ' + tuCount + ' 题，类比推理 ' + lbCount + ' 题。点击查看答案与解析。</div>' +
    '<div class="detail-hint">点击打开做题 →</div>';
}

function openTuituiModal(date) {
  var entry = tuituiMap[date];
  if (!entry) return;
  var body = document.getElementById('tuituiModalBody');
  var dateEl = document.getElementById('tuituiModalDate');
  dateEl.textContent = entry.date + (entry.pub_time ? ' · ' + entry.pub_time.split(' ')[1] : '');

  var html = '';
  var sections = [
    { key: '图推', label: '图形推理', badgeClass: '' },
    { key: '类比', label: '类比推理', badgeClass: ' leibi' }
  ];

  sections.forEach(function(sec) {
    var qs = entry.questions.filter(function(q){ return q.section === sec.key; });
    if (qs.length === 0) return;
    html += '<div class="tuitui-section-group">';
    html += '<div class="tuitui-section-heading">';
    html += '<span class="badge' + sec.badgeClass + '">' + sec.key + '</span>';
    html += '<h4>' + sec.label + ' · ' + qs.length + ' 题</h4>';
    html += '</div>';

    qs.forEach(function(q, idx) {
      html += '<div class="tuitui-q-block">';
      html += '<div class="tuitui-q-header">第 ' + q.num + ' 题</div>';
      if (q.q_text && q.q_text.trim() !== '') {
        html += '<div class="tuitui-q-text">' + q.q_text.replace(/\n/g, '<br>') + '</div>';
      }
      if (q.img) {
        html += '<div class="tuitui-q-img"><img src="/cache/tuitui/' + entry.date + '/' + q.img + '" alt="' + sec.key + '第' + q.num + '题" loading="lazy"></div>';
      }
      html += '<div class="tuitui-q-toggle" onclick="toggleTuituiAnswer(this)">';
      html += '<span class="arrow">&#9654;</span> 点击查看答案';
      html += '</div>';
      html += '<div class="tuitui-q-answer">';
      if (q.ref_answer) {
        html += '<div class="ref">参考答案：' + q.ref_answer + '</div>';
      }
      if (q.explanation) {
        html += '<div class="exp"><strong>解析：</strong>' + q.explanation + '</div>';
      }
      html += '</div>';
      html += '</div>';
    });

    html += '</div>';
  });

  if (!html) {
    html = '<div class="tuitui-empty">当日暂无图推类比数据。</div>';
  }

  body.innerHTML = html;
  $('#tuituiModal').modal('show');
}

function toggleTuituiAnswer(el) {
  var answer = el.nextElementSibling;
  var arrow = el.querySelector('.arrow');
  if (answer.classList.contains('open')) {
    answer.classList.remove('open');
    arrow.classList.remove('open');
  } else {
    answer.classList.add('open');
    arrow.classList.add('open');
  }
}

// -------- Chengyu rendering --------
function renderChengyuDetail(entry) {
  var el = document.getElementById('chengyuDetail');
  if (!el || !entry) return;
  var idiomCount = entry.idioms ? entry.idioms.length : 0;
  el.style.cursor = 'pointer';
  el.onclick = function() { openChengyuModal(entry.date); };
  el.innerHTML =
    '<span class="detail-tag" style="background:#8b5e3c;">📖 成语</span>' +
    '<div class="detail-title">' + entry.title + '</div>' +
    '<div class="detail-summary">第 ' + entry.period + ' 期 · 共 ' + idiomCount + ' 个成语。</div>' +
    '<div class="detail-hint">点击查看详情 →</div>';
}

function openChengyuModal(date) {
  var entry = chengyuMap[date];
  if (!entry) return;
  var body = document.getElementById('chengyuModalBody');
  var meta = document.getElementById('chengyuModalMeta');
  meta.textContent = entry.date + (entry.period ? ' · 第 ' + entry.period + ' 期' : '');
  var html = '';
  if (entry.idioms && entry.idioms.length > 0) {
    entry.idioms.forEach(function(item) {
      html += '<div class="chengyu-card"><div style="display:flex;align-items:center;">';
      html += '<span class="cy-num">' + item.num + '</span>';
      html += '<span class="cy-idiom">' + item.idiom + '</span></div>';
      html += '<div class="cy-def">' + item.definition + '</div></div>';
    });
  }
  if (!html) html = '<div class="chengyu-empty">当日暂无成语积累数据。</div>';
  body.innerHTML = html;
  $('#chengyuModal').modal('show');
}

function renderInfoBar() {
  var el = document.getElementById('newsInfoBar');
  if (!el) return;
  var now = new Date();
  var weekdays = ['日','一','二','三','四','五','六'];
  var y = now.getFullYear();
  var m = now.getMonth() + 1;
  var d = now.getDate();
  var wd = weekdays[now.getDay()];
  el.innerHTML =
    '<span class="bar-label">Daily Briefing</span>' +
    '<span class="bar-date">' + y + '年' + m + '月' + d + '日 周' + wd + '</span>';
}

function renderDetail(report) {
  var el = document.getElementById('newsDetail');
  if (!el || !report) return;
  var baseUrl = SITE_BASEURL;
  var isFirst = DAILY_REPORTS.length > 0 && DAILY_REPORTS[0].date === report.date;
  var dp = report.date.split('-');
  var dateStr = dp[0] + '-' + dp[1] + '-' + dp[2];
  var reportUrl = baseUrl + '/' + report.path;
  el.style.cursor = 'pointer';
  el.onclick = function() { window.open(reportUrl, '_blank', 'noopener'); };
  el.innerHTML =
    '<span class="detail-tag">' + (isFirst ? '🔥 NEWS' : (report.type || 'NEWS')) + '</span>' +
    '<div class="detail-title">' + report.title + '</div>' +
    (report.summary ? '<div class="detail-summary">' + report.summary + '</div>' : '') +
    '<div class="detail-meta">' + dateStr + (report.source ? ' · ' + report.source : '') + '</div>' +
    '<div class="detail-hint">点击打开报告 →</div>';
}

function renderDateStrip(activeDate) {
  var el = document.getElementById('newsDateStrip');
  if (!el) return;

  // Build unified date list from ALL sources
  var dateSet = {};
  DAILY_REPORTS.forEach(function(r) { dateSet[r.date] = true; });
  GWY_REPORTS.forEach(function(r) { dateSet[r.date] = true; });
  TUITUI_DATA.forEach(function(d) { dateSet[d.date] = true; });
  CHENGYU_DATA.forEach(function(d) { dateSet[d.date] = true; });

  var allDates = Object.keys(dateSet).sort().reverse();
  if (allDates.length === 0) return;

  var weekdays = ['日','一','二','三','四','五','六'];
  var html = '<span class="date-label">近期</span>';
  var count = Math.min(allDates.length, 10);
  for (var i = 0; i < count; i++) {
    var date = allDates[i];
    var dp = date.split('-');
    var num = parseInt(dp[2]);
    var month = parseInt(dp[1]);
    var wd = weekdays[new Date(date + 'T00:00:00').getDay()];
    // Show month prefix when month changes or for first item
    var showMonth = (i === 0) || (allDates[i-1] && allDates[i-1].split('-')[1] !== dp[1]);
    var label = showMonth ? month + '/' + num : num;
    html += '<div class="date-cell' + (date === activeDate ? ' active' : '') + '" data-date="' + date + '">' +
      '<span class="d-num">' + label + '</span><span class="d-weekday">' + wd + '</span></div>';
  }
  html += '<span class="cal-toggle" id="calToggle">更多 ▸</span>';
  el.innerHTML = html;

  el.querySelectorAll('.date-cell').forEach(function(cell) {
    cell.addEventListener('click', function() {
      var date = this.dataset.date;
      selectDate(date);
      el.querySelectorAll('.date-cell').forEach(function(c) { c.classList.remove('active'); });
      this.classList.add('active');
      renderCalendar(new Date(date + 'T00:00:00'), date);
    });
  });

  document.getElementById('calToggle').addEventListener('click', function() {
    var panel = document.getElementById('newsCalendarPanel');
    var isOpen = panel.classList.contains('open');
    if (isOpen) {
      panel.classList.remove('open');
      this.textContent = '更多 ▸';
    } else {
      panel.classList.add('open');
      this.textContent = '收起 ▾';
      if (!panel.innerHTML.trim()) {
        renderCalendar(new Date(allDates[0] + 'T00:00:00'), activeDate);
      }
    }
  });
}

// Select a date: update ALL sections based on available data
function selectDate(date) {
  // News
  var newsEl = document.getElementById('newsSection');
  if (reportMap[date]) {
    renderDetail(reportMap[date]);
    newsEl.style.display = '';
  } else {
    document.getElementById('newsDetail').innerHTML =
      '<div class="news-empty-state">当日无新闻日报。</div>';
    newsEl.style.display = '';
  }

  // GWY
  var gwyEl = document.getElementById('gwySection');
  if (gwyMap[date]) {
    renderGwyDetail(gwyMap[date]);
    gwyEl.style.display = '';
  } else {
    document.getElementById('gwyDetail').innerHTML =
      '<div class="news-empty-state">当日无公考日报。</div>';
    gwyEl.style.display = '';
  }

  // Tuitui
  var tuituiEl = document.getElementById('tuituiSection');
  if (tuituiMap[date]) {
    renderTuituiDetail(tuituiMap[date]);
    tuituiEl.style.display = '';
  } else {
    document.getElementById('tuituiDetail').innerHTML =
      '<div class="news-empty-state">当日无图推类比。</div>';
    tuituiEl.style.display = '';
  }

  // Chengyu
  var chengyuEl = document.getElementById('chengyuSection');
  if (chengyuMap[date]) {
    renderChengyuDetail(chengyuMap[date]);
    chengyuEl.style.display = '';
  } else {
    document.getElementById('chengyuDetail').innerHTML =
      '<div class="news-empty-state">当日无成语积累。</div>';
    chengyuEl.style.display = '';
  }
}

function renderCalendar(viewDate, activeDate) {
  var panel = document.getElementById('newsCalendarPanel');
  if (!panel) return;
  var year = viewDate.getFullYear();
  var month = viewDate.getMonth();
  var weekdays = ['一','二','三','四','五','六','日'];
  var firstDay = new Date(year, month, 1);
  var startWeekday = (firstDay.getDay() + 6) % 7;
  var daysInMonth = new Date(year, month + 1, 0).getDate();
  var today = new Date().toISOString().slice(0, 10);

  // Unified date set for calendar dots
  var anyDateSet = {};
  DAILY_REPORTS.forEach(function(r) { anyDateSet[r.date] = true; });
  GWY_REPORTS.forEach(function(r) { anyDateSet[r.date] = true; });
  TUITUI_DATA.forEach(function(d) { anyDateSet[d.date] = true; });
  CHENGYU_DATA.forEach(function(d) { anyDateSet[d.date] = true; });

  var html = '<div class="cal-card">' +
    '<div class="cal-nav">' +
      '<button id="calPrev">◀</button>' +
      '<span class="cal-month-label">' + year + '年' + (month + 1) + '月</span>' +
      '<button id="calNext">▶</button>' +
    '</div>' +
    '<div class="cal-grid">';

  weekdays.forEach(function(w) { html += '<div class="cal-hd">' + w + '</div>'; });

  for (var i = 0; i < startWeekday; i++) html += '<div class="cal-day empty"></div>';

  for (var d = 1; d <= daysInMonth; d++) {
    var dateStr = year + '-' + String(month + 1).padStart(2, '0') + '-' + String(d).padStart(2, '0');
    var hasAny = !!anyDateSet[dateStr];
    var isActive = dateStr === activeDate;
    var isToday = dateStr === today;
    var cls = 'cal-day';
    if (hasAny) cls += ' has-report';
    if (isActive) cls += ' active';
    if (isToday) cls += ' today';
    html += '<div class="' + cls + '"' + (hasAny ? ' data-date="' + dateStr + '"' : '') + '>' + d + '</div>';
  }

  html += '</div></div>';
  panel.innerHTML = html;

  var calViewDate = new Date(year, month, 1);
  document.getElementById('calPrev').addEventListener('click', function() {
    calViewDate.setMonth(calViewDate.getMonth() - 1);
    renderCalendar(calViewDate, activeDate);
  });
  document.getElementById('calNext').addEventListener('click', function() {
    calViewDate.setMonth(calViewDate.getMonth() + 1);
    renderCalendar(calViewDate, activeDate);
  });

  panel.querySelectorAll('.cal-day.has-report').forEach(function(day) {
    day.addEventListener('click', function() {
      var date = this.dataset.date;
      selectDate(date);
      var strip = document.getElementById('newsDateStrip');
      strip.querySelectorAll('.date-cell').forEach(function(c) {
        c.classList.toggle('active', c.dataset.date === date);
      });
      panel.querySelectorAll('.cal-day').forEach(function(c) { c.classList.remove('active'); });
      this.classList.add('active');
    });
  });
}

// -------- 像素办公室 / 日报内容 点击切换 --------
(function() {
  var area = document.getElementById('pixelToggleArea');
  if (!area) return;
  var pixelCard = area.querySelector('.pixel-office-card');
  var newsCard = document.getElementById('pixelNewsCard');
  var closeBtn = area.querySelector('.pixel-news-close');

  if (!pixelCard || !newsCard) return;

  var initialized = false;

  function showNews(e) {
    if (e) e.stopPropagation();
    if (newsCard.style.display !== 'none') return;
    pixelCard.style.display = 'none';
    newsCard.style.display = '';
    area.classList.add('news-open');
    newsCard.setAttribute('aria-expanded', 'true');

    if (!initialized) {
      initialized = true;
      renderInfoBar();

      // Find the latest date across ALL sources
      var allDates = [];
      DAILY_REPORTS.forEach(function(r) { allDates.push(r.date); });
      GWY_REPORTS.forEach(function(r) { allDates.push(r.date); });
      TUITUI_DATA.forEach(function(d) { allDates.push(d.date); });
      CHENGYU_DATA.forEach(function(d) { allDates.push(d.date); });
      allDates.sort().reverse();

      if (allDates.length > 0) {
        selectDate(allDates[0]);
        renderDateStrip(allDates[0]);
      } else {
        document.getElementById('newsDetail').innerHTML =
          '<div class="news-empty-state">等待数据接入。</div>';
      }
    }
  }

  function showPixel(e) {
    if (e) e.stopPropagation();
    if (pixelCard.style.display !== 'none') return;
    newsCard.style.display = 'none';
    pixelCard.style.display = '';
    area.classList.remove('news-open');
    newsCard.setAttribute('aria-expanded', 'false');
  }

  area.addEventListener('click', showNews);
  if (closeBtn) closeBtn.addEventListener('click', showPixel);
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') showPixel(e);
  });
})();

// -------- Easter Egg: Pixel Clock → Backroom --------
(function() {
  var clock = document.querySelector('.pixel-clock');
  if (!clock) return;

  var clickTimes = [];
  clock.addEventListener('click', function(e) {
    e.stopPropagation();
    var now = Date.now();
    clickTimes.push(now);
    clickTimes = clickTimes.filter(function(t) { return now - t < 2000; });
    if (clickTimes.length >= 3) {
      clickTimes = [];
      var base = (typeof SITE_BASEURL !== 'undefined') ? SITE_BASEURL : '';
      window.location.href = base + '/backroom/';
    }
  });
})();

// -------- Easter Egg: Pixel Cat → 午後のコンビニ ヨルマート --------
(function() {
  var cat = document.querySelector('.pixel-cat');
  if (!cat) return;

  var clickTimes = [];
  cat.addEventListener('click', function(e) {
    e.stopPropagation();
    var now = Date.now();
    clickTimes.push(now);
    clickTimes = clickTimes.filter(function(t) { return now - t < 2000; });
    if (clickTimes.length >= 3) {
      clickTimes = [];
      var base = (typeof SITE_BASEURL !== 'undefined') ? SITE_BASEURL : '';
      window.location.href = base + '/scenes/?v=' + Date.now();   // 时间戳防缓存：进入微缩景观选集
    }
  });
})();

// -------- Easter Egg: Pixel Plant → JZXM Terminal --------
(function() {
  var plant = document.querySelector('.pixel-plant');
  var overlay = document.getElementById('jzxm-overlay');
  var closeBtn = document.getElementById('jzxm-close');
  if (!plant || !overlay) return;

  var clickTimes = [];
  plant.addEventListener('click', function(e) {
    e.stopPropagation();
    var now = Date.now();
    clickTimes.push(now);
    clickTimes = clickTimes.filter(function(t) { return now - t < 2000; });
    if (clickTimes.length >= 3) {
      clickTimes = [];
      overlay.style.display = 'block';
      document.body.style.overflow = 'hidden';
    }
  });

  function closeOverlay() {
    overlay.style.display = 'none';
    document.body.style.overflow = '';
  }
  closeBtn.addEventListener('click', closeOverlay);
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && overlay.style.display === 'block') closeOverlay();
  });

  // Inject jzxm post data (only rendered for the hidden group)
  // Inject jzxm post data (built inline in the page for the hidden group)
  var jzxmData = (typeof JZXM_DATA !== 'undefined' && JZXM_DATA) ? JZXM_DATA : [];

var content = document.getElementById('jzxm-content');
  var globalIdx = 1;
  jzxmData.forEach(function(cat) {
    if (cat.posts.length === 0) return;
    var catEl = document.createElement('div');
    catEl.className = 'jzxm-cat';
    var hdr = document.createElement('div');
    hdr.className = 'jzxm-cat-header';
    hdr.innerHTML = '<span class="jzxm-cat-badge">' + cat.name + '</span><span class="jzxm-cat-count">' + cat.posts.length + ' posts</span>';
    catEl.appendChild(hdr);
    var items = document.createElement('div');
    items.className = 'jzxm-items';
    cat.posts.forEach(function(p) {
      var a = document.createElement('a');
      a.className = 'jzxm-item';
      a.href = p.url;
      a.innerHTML =
        '<span class="jzxm-idx">[' + String(globalIdx).padStart(2, '0') + ']</span>' +
        '<span class="jzxm-name">' + p.title + '</span>' +
        '<span class="jzxm-date">' + p.date + '</span>';
      items.appendChild(a);
      globalIdx++;
    });
    catEl.appendChild(items);
    content.appendChild(catEl);
  });

})();
