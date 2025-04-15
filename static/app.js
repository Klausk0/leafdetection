// Set up webcam stream
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Start video feed from webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
  })
  .catch((err) => {
    console.log("Error: ", err);
  });

// Capture image from webcam
function captureImage() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg');

  const formData = new FormData();
  formData.append('image', dataUrl);

  fetch('/predict', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    displayResult(data);
  })
  .catch(error => console.error('Error:', error));
}

// Handle file upload prediction
document.getElementById('file-input').addEventListener('change', predictFromFile);

// Handle file input prediction
function predictFromFile() {
  const fileInput = document.getElementById('file-input');
  const file = fileInput.files[0];
  if (file) {
    const formData = new FormData();
    formData.append('image', file);

    fetch('/predict', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      displayResult(data);
    })
    .catch(error => console.error('Error:', error));
  }
}

// Display prediction result
function displayResult(data) {
  const resultDiv = document.getElementById('result');
  if (data.error) {
    resultDiv.innerHTML = `Error: ${data.error}`;
  } else {
    resultDiv.innerHTML = `
      <strong>Predicted Disease:</strong> ${data.predicted_class}<br>
      <strong>Confidence:</strong> ${data.confidence}%<br>
      <strong>Disease Percentage:</strong> ${data.disease_percentage}%<br>
      <strong>Days Left to Full Infection:</strong> ${data.days_left} days
    `;
  }
}

// Trigger file input from a button
function triggerFileInput() {
  document.getElementById('file-input').click();
}

// Handling the image upload and prediction process
document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    // Get the file from the input
    const fileInput = document.getElementById("fileInput");
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // Show loading spinner or message
    document.getElementById("result").style.display = "none";

    try {
        // Send the image to the Flask server for prediction
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        if (result.error) {
            alert(result.error);
            return;
        }

        // Display the result
        document.getElementById("diseaseName").textContent = `Disease: ${result.prediction.plant} - ${result.prediction.disease}`;
        document.getElementById("confidence").textContent = `Confidence: ${result.prediction.label}`;
        document.getElementById("diseasePercentage").textContent = `Disease Percentage: ${result.prediction.percentage_disease}`;
        document.getElementById("daysLeft").textContent = `Days Left to Full Infection: ${result.prediction.days_left}`;

        document.getElementById("result").style.display = "block";
    } catch (error) {
        console.error("Error:", error);
        alert("There was an error during the prediction.");
    }
});
