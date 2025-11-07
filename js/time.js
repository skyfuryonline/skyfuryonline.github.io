function updateClock() {
  const clock = document.getElementById("clock");
  if (!clock) return;
  const now = new Date();
  clock.textContent = now.toLocaleTimeString();
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
