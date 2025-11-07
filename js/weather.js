async function loadWeather() {
  const weatherEl = document.getElementById("weather");
  if (!weatherEl) return;

  try {
    const res = await fetch("https://wttr.in/?format=j1");
    const data = await res.json();
    const area = data.nearest_area?.[0]?.areaName?.[0]?.value || "你的城市";
    const condition = data.current_condition?.[0]?.weatherDesc?.[0]?.value || "";
    const temp = data.current_condition?.[0]?.temp_C || "--";
    weatherEl.textContent = `${area}: ${condition} ${temp}°C`;
  } catch (e) {
    weatherEl.textContent = "天气加载失败";
  }
}

loadWeather();
