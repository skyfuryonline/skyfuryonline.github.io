document.addEventListener("mousemove", function (e) {
  const cat = document.getElementById("cat");
  if (!cat) return;
  cat.style.transform = `translate(${e.clientX - 25}px, ${e.clientY - 25}px)`;
});
