// script.js – Handles form submission and result display
document
  .getElementById("prediction-form")
  .addEventListener("submit", async function (e) {
    e.preventDefault();

    // Collect input values
    const data = {
      sepal_length: document.getElementById("sepal_length").value,
      sepal_width: document.getElementById("sepal_width").value,
      petal_length: document.getElementById("petal_length").value,
      petal_width: document.getElementById("petal_width").value,
    };

    try {
      // Send data to Flask backend
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (result.error) {
        alert("Error: " + result.error);
        return;
      }

      // Show prediction
      document.getElementById("predicted-species").textContent =
        result.prediction;

      // Update probability bars
      const probs = result.probabilities;
      updateBar("prob-setosa", "prob-setosa-text", probs["Setosa"]);
      updateBar("prob-versicolor", "prob-versicolor-text", probs["Versicolor"]);
      updateBar("prob-virginica", "prob-virginica-text", probs["Virginica"]);

      // Reveal result section
      document.getElementById("result").classList.remove("hidden");
    } catch (err) {
      alert("Something went wrong. Please try again.");
      console.error(err);
    }
  });

function updateBar(barId, textId, value) {
  const bar = document.getElementById(barId);
  const text = document.getElementById(textId);
  bar.style.width = value + "%";
  text.textContent = value + "%";
}
