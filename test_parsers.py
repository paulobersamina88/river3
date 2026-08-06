import pandas as pd

from providers.bulacan_pdrrmo import parse_html as parse_bulacan
from providers.official_reports_csv import parse as parse_official
from providers.pagasa_pmt import parse_html as parse_pmt


def test_pmt_parser():
    html = """
    <html><body><h4>Time : 2026-08-06 10:30</h4>
    <table>
      <thead>
       <tr><th rowspan='2'>Station</th><th colspan='4'>Observed WL [EL.m]</th><th colspan='3'>Warning WL [EL.m]</th></tr>
       <tr><th>Current</th><th>-30 min</th><th>-1 hr</th><th>-2hr</th><th>Alert</th><th>Alarm</th><th>Critical</th></tr>
      </thead>
      <tbody>
       <tr><td>Sto. Nino</td><td>14.80</td><td>14.70</td><td>14.50</td><td>14.20</td><td>15.0</td><td>16.0</td><td>18.0</td></tr>
       <tr><td>Quirino Tullahan</td><td>39.40</td><td>39.30</td><td>39.10</td><td>38.90</td><td>39.0</td><td>40.0</td><td>41.0</td></tr>
      </tbody>
    </table></body></html>
    """
    frame = parse_pmt(html)
    assert len(frame) == 2
    assert set(frame["river_system"]) == {"Marikina River", "Tullahan River"}
    assert round(frame.iloc[0]["rise_rate_m_hr"], 2) == 0.30


def test_bulacan_parser():
    html = """
    <html><body>August 6, 2026 11:08 am
    <table>
      <thead><tr><th>Station</th><th>Actual Level</th><th>Alert</th><th>Alarm</th><th>Critical</th><th>Date</th></tr></thead>
      <tbody>
       <tr><td>Sulipan Bridge, Apalit, Pampanga</td><td>0.10 meter</td><td>1.0 meter</td><td>2.0 meters</td><td>3.0 meters</td><td>08/06/2026</td></tr>
       <tr><td>Northville Bridge River, Marilao, Bulacan</td><td>0.35 meter</td><td>1.0 meter</td><td>1.5 meters</td><td>2.0 meters</td><td>08/06/2026</td></tr>
      </tbody>
    </table></body></html>
    """
    frame = parse_bulacan(html)
    assert len(frame) == 2
    assert frame.iloc[0]["river_system"] == "Pampanga River"
    assert "Meycauayan-Marilao-Obando" in frame.iloc[1]["river_system"]


def test_official_csv_units():
    raw = pd.DataFrame([
        {
            "source_page": "MDRRMO Calasiao",
            "post_url": "https://example.com/post",
            "river_name": "Sinucalan River",
            "monitoring_point": "Marusay Bridge",
            "observed_at": "2026-08-06 09:00",
            "water_level": 6.3,
            "unit": "ft",
            "status": "Alarm",
        }
    ])
    frame = parse_official(raw)
    assert len(frame) == 1
    assert abs(frame.iloc[0]["level_m"] - 1.92024) < 1e-6
    assert frame.iloc[0]["threshold_status"] == "Alarm"
