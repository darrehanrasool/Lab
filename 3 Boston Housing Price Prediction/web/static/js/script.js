// Project 3: Boston Housing Price Prediction Client Script
document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form");
  if (form) {
    form.addEventListener("submit", () => {
      const btn = form.querySelector("button[type='submit']");
      if (btn) {
        btn.textContent = "Estimating Valuation...";
        btn.style.opacity = "0.7";
      }
    });
  }
});
