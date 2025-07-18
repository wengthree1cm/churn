document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const fileInput = document.getElementById("file");
  const modelInput = document.getElementById("model");
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("model_type", modelInput.value);

  const response = await fetch("https://churn-prediction-udq1.onrender.com/predict", {
    method: "POST",
    body: formData
  });

  const result = await response.json();

  const outputDiv = document.getElementById("output");
  outputDiv.innerHTML = "";

  if (result.error) {
    outputDiv.innerHTML = `<p style="color:red">${result.error}</p>`;
    return;
  }

  // Render result table
  if (result.length > 0) {
    const keys = Object.keys(result[0]);
    let table = "<table><thead><tr>" + keys.map(k => `<th>${k}</th>`).join("") + "</tr></thead><tbody>";
    result.forEach(row => {
      table += "<tr>" + keys.map(k => `<td>${row[k]}</td>`).join("") + "</tr>";
    });
    table += "</tbody></table>";
    outputDiv.innerHTML = table;
  } else {
    outputDiv.innerHTML = "<p>No data returned.</p>";
  }
});
