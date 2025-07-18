document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById("file");
  const modelInput = document.getElementById("model");
  const outputDiv = document.getElementById("output");

  outputDiv.innerHTML = "⏳ Uploading file and predicting...";

  if (!fileInput.files.length) {
    outputDiv.innerHTML = `<p style="color:red">Please upload a CSV file.</p>`;
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("model_type", modelInput.value);

  try {
    const response = await fetch("https://churn-prediction-udq1.onrender.com/predict", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}`);
    }

    const result = await response.json();
    outputDiv.innerHTML = "";  // Clear loading text

    if (result.error) {
      outputDiv.innerHTML = `<p style="color:red">⚠️ ${result.error}</p>`;
      return;
    }

    if (result.predictions && result.predictions.length > 0) {
      let table = "<table border='1'><thead><tr><th>Prediction</th></tr></thead><tbody>";
      result.predictions.forEach(pred => {
        table += `<tr><td>${pred}</td></tr>`;
      });
      table += "</tbody></table>";
      outputDiv.innerHTML = table;
    } else {
      outputDiv.innerHTML = "<p>No predictions returned.</p>";
    }

  } catch (error) {
    console.error("Fetch failed:", error);
    outputDiv.innerHTML = `<p style="color:red">❌ Fetch failed: ${error.message}</p>`;
  }
});
