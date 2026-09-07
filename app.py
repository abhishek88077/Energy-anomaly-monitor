from __future__ import annotations

import tempfile
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.pipeline import Config, run_pipeline

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'project': 'Outlier Detection Framework',
        'dataset': 'Energy consumption anomaly analytics',
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    form = request.form
    uploaded = request.files.get('dataset')

    try:
        if not uploaded or not uploaded.filename:
            return jsonify({
                'status': 'error',
                'message': 'Upload a real energy CSV before running an analysis.',
            }), 400

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            uploaded.save(tmp.name)
            dataset_path = tmp.name

        cfg = Config(
            window=int(form.get('window', 24) or 24),
            stride=int(form.get('stride', 6) or 6),
            epochs=int(form.get('epochs', 30) or 30),
            threshold_quantile=float(form.get('threshold_quantile', 0.97) or 0.97),
            seed=int(form.get('seed', 42) or 42),
        )
        output_dir = Path(tempfile.mkdtemp(prefix='outlier_results_'))
        result, summary = run_pipeline(dataset_path, form.get('value_col') or None, cfg, output_dir)

        top_anomalies = result.sort_values('UAS', ascending=False).head(10)
        payload = {
            'status': 'success',
            'summary': summary,
            'records_total': int(len(result)),
            'anomaly_count': int(result['anomaly'].sum()),
            'top_anomalies': top_anomalies[['start', 'end', 'UAS', 'event_type', 'decision']].to_dict('records'),
            'max_uas': float(result['UAS'].max()) if not result.empty else 0.0,
        }
        return jsonify(payload)
    except Exception as exc:  # pragma: no cover - defensive API handling
        return jsonify({'status': 'error', 'message': str(exc)}), 400


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', '5000')),
        debug=False,
    )
