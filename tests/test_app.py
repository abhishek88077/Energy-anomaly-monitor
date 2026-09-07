from app import app


def test_index_and_health_routes():
    client = app.test_client()
    resp = client.get('/')
    assert resp.status_code == 200
    assert b'Energy anomaly monitor' in resp.data
    assert b'No analysis results yet' in resp.data

    health = client.get('/api/health')
    assert health.status_code == 200
    assert health.json['status'] == 'ok'


def test_analysis_requires_real_upload():
    response = app.test_client().post('/api/analyze', data={})
    assert response.status_code == 400
    assert response.json['message'] == 'Upload a real energy CSV before running an analysis.'
