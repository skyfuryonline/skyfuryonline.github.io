function updateClock() {
  const clock = document.getElementById("clock");
  if (!clock) return;
  const now = new Date();
  const options = { timeZone: 'Asia/Shanghai', hour: '2-digit', minute: '2-digit', second: '2-digit' };
  clock.textContent = now.toLocaleTimeString('zh-CN', options);  // 'zh-CN' 用于中文格式，你可以调整为'en-US'等
}

function updateCountdown() {
  const countdown = document.getElementById("countdown");
  if (!countdown) return;
  const targetDate = new Date("2026-06-30T00:00:00"); // 毕业日自行修改
  const now = new Date();
  const diff = targetDate - now;
  const days = Math.max(0, Math.floor(diff / (1000 * 60 * 60 * 24)));
  countdown.textContent = `距离毕业还有 ${days} 天`;
}

setInterval(updateClock, 1000);
setInterval(updateCountdown, 3600000);
updateClock();
updateCountdown();
