const form = document.getElementById('analysis-form');
const fileInput = document.getElementById('dataset-input');
const fileLabel = document.getElementById('file-label');
const resultsBody = document.getElementById('results-body');
const statusBox = document.getElementById('status-box');
const tableState = document.querySelector('.table-state');

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) {
    fileLabel.textContent = file.name;
    statusBox.textContent = 'Dataset selected. Review the settings and run the analysis.';
  }
});

function updateSummary(data) {
  const summary = data.summary;
  document.getElementById('summary-observations').textContent = summary.observations.toLocaleString();
  document.getElementById('summary-windows').textContent = summary.windows.toLocaleString();
  document.getElementById('summary-anomalies').textContent = summary.anomalous_windows.toLocaleString();
  document.getElementById('summary-threshold').textContent = Number(summary.threshold).toFixed(3);
}

function renderRows(rows) {
  if (!rows || rows.length === 0) {
    resultsBody.innerHTML = '<tr><td colspan="5" class="empty-state"><span>✓</span><strong>No anomaly windows detected</strong><small>The configured run did not exceed its training threshold.</small></td></tr>';
    return;
  }

  resultsBody.innerHTML = rows.map((row) => `
    <tr>
      <td>${new Date(row.start).toLocaleString()}</td>
      <td>${new Date(row.end).toLocaleString()}</td>
      <td><strong>${Number(row.UAS).toFixed(4)}</strong></td>
      <td>${row.event_type}</td>
      <td>${row.decision}</td>
    </tr>
  `).join('');
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) {
    statusBox.textContent = 'Select a real CSV dataset before running detection.';
    return;
  }

  const submitButton = form.querySelector('button[type="submit"]');
  submitButton.disabled = true;
  submitButton.innerHTML = 'Analyzing dataset <span>...</span>';
  statusBox.textContent = 'Running the detection pipeline. Larger datasets may take a moment.';
  statusBox.style.background = '#f5f8fd';

  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: new FormData(form) });
    const data = await response.json();
    if (!response.ok || data.status !== 'success') throw new Error(data.message || 'Analysis failed');

    updateSummary(data);
    renderRows(data.top_anomalies);
    tableState.textContent = 'Latest run completed';
    statusBox.textContent = `Analysis complete. ${data.anomaly_count.toLocaleString()} anomaly windows require review.`;
    statusBox.style.background = '#eefaf6';
    statusBox.style.borderColor = '#bdebd9';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    statusBox.textContent = error.message;
    statusBox.style.background = '#fff3f3';
    statusBox.style.borderColor = '#f0caca';
    tableState.textContent = 'Run failed';
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = 'Run detection <span>→</span>';
  }
});
