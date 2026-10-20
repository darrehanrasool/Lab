// Project 2: Titanic Survival Prediction Client Script
document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form");
  if (form) {
    form.addEventListener("submit", (e) => {
      const submitBtn = form.querySelector("button[type='submit']");
      if (submitBtn) {
        submitBtn.textContent = "Calculating Survival Odds...";
        submitBtn.style.opacity = "0.7";
      }
    });
  }
});
