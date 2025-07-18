document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const output = document.getElementById("output");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById("csv-file");
    const model = document.getElementById("model-select").value;

    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
      output.innerHTML = `<p style="color:red;">❌ Please upload a CSV file.</p>`;
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("model_type", model);

    output.innerHTML = "⏳ Uploading file and predicting...";

    try {
      const response = await fetch("https://churn-prediction-udq1.onrender.com/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const result = await response.json();

      if (result.error) {
        output.innerHTML = `<p style="color: red;">❌ ${result.error}</p>`;
        return;
      }

      // Handle list of predictions
      if (Array.isArray(result.predictions) && result.predictions.length > 0) {
        let table = "<table><thead><tr><th>Prediction</th></tr></thead><tbody>";
        result.predictions.forEach(p => {
          table += `<tr><td>${p}</td></tr>`;
        });
        table += "</tbody></table>";
        output.innerHTML = table;
      } else {
        output.innerHTML = "<p>No data returned.</p>";
      }

    } catch (err) {
      output.innerHTML = `<p style="color:red;">❌ Error: ${err.message}</p>`;
      console.error("Error during fetch:", err);
    }
  });
});
