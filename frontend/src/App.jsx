import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  Boxes,
  CalendarDays,
  ChevronDown,
  CircleHelp,
  Clock3,
  LayoutDashboard,
  Globe2,
  MapPinned,
  Package,
  Route,
  Settings2,
  Sparkles,
  Truck,
  TrendingUp
} from "lucide-react";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import {
  cityMetrics,
  hourlyDemo,
  projectSummary
} from "./data";

import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
  useMap
} from "react-leaflet";

const modelComparison = [
  { model: "Naive", mae: 9.341, rmse: 26.379, wape: 133.19 },
  { model: "Seasonal Naive", mae: 4.723, rmse: 14.162, wape: 67.34 },
  { model: "XGBoost", mae: 4.001, rmse: 10.865, wape: 57.05 }
];

const navigation = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "forecast", label: "Demand Forecast", icon: TrendingUp },
  { id: "performance", label: "Model Performance", icon: BarChart3 },
  { id: "hotspots", label: "Hotspot Intelligence", icon: MapPinned },
  { id: "fleet", label: "Fleet & Routes", icon: Truck }
];

function MetricCard({ label, value, detail, icon: Icon, accent = "blue" }) {
  return (
    <div className="metric-card">
      <div className={`metric-icon ${accent}`}>
        <Icon size={19} />
      </div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div>
    </div>
  );
}

function SectionTitle({ eyebrow, title, description, right }) {
  return (
    <div className="section-title">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
      {right && <div className="section-actions">{right}</div>}
    </div>
  );
}


function DemandGlobe() {
  return (
    <div className="command-layout">
      <div className="command-panel command-left">
        <div className="command-panel-heading">
          <span>REGIONAL SNAPSHOT</span>
          <span className="live-indicator"><i /> HISTORICAL DATA</span>
        </div>
        <div className="command-stat">
          <span>Dataset observations</span>
          <strong>{projectSummary.observations.toLocaleString("en-US")}</strong>
          <small>Processed hourly records</small>
        </div>
        <div className="command-divider" />
        <div className="command-stat-row">
          <div><span>Cities</span><strong>{projectSummary.cities}</strong></div>
          <div><span>Regions</span><strong>{projectSummary.regions}</strong></div>
        </div>
        <div className="command-divider" />
        <div className="command-mini-title">MODEL VALIDATION MAE</div>
        <div className="command-stat">
          <strong className="gold-number">{projectSummary.validationMAE.toFixed(3)}</strong>
          <small>XGBoost · September 2022</small>
        </div>
        <div className="command-meter"><span style={{ width: "72%" }} /></div>
        <small className="command-muted">Model evaluation summary, not live telemetry</small>
      </div>

      <div className="globe-stage">
        <div className="globe-label"><span>DEMAND INTELLIGENCE</span><b>REGIONAL OVERVIEW</b></div>
        <div className="globe-orbit orbit-one" />
        <div className="globe-orbit orbit-two" />
        <div className="globe">
          <svg viewBox="0 0 420 420" role="img" aria-label="Abstract wireframe globe representing regional delivery demand">
            <defs>
              <radialGradient id="globeShade" cx="38%" cy="32%" r="70%">
                <stop offset="0%" stopColor="#496b55" stopOpacity=".55" />
                <stop offset="72%" stopColor="#203b2d" stopOpacity=".35" />
                <stop offset="100%" stopColor="#101d17" stopOpacity=".95" />
              </radialGradient>
              <clipPath id="globeClip"><circle cx="210" cy="210" r="174" /></clipPath>
              <filter id="goldGlow"><feGaussianBlur stdDeviation="5" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
            </defs>
            <circle cx="210" cy="210" r="174" fill="url(#globeShade)" stroke="#78927b" strokeOpacity=".7" />
            <g clipPath="url(#globeClip)" fill="none" stroke="#9ab09b" strokeOpacity=".35">
              <ellipse cx="210" cy="210" rx="174" ry="54"/>
              <ellipse cx="210" cy="210" rx="174" ry="105"/>
              <ellipse cx="210" cy="210" rx="174" ry="145"/>
              <ellipse cx="210" cy="210" rx="62" ry="174"/>
              <ellipse cx="210" cy="210" rx="118" ry="174"/>
              <path d="M36 210H384M48 145H372M48 275H372M86 82L334 338M334 82L86 338"/>
              <path d="M210 36V384"/>
            </g>
            <g clipPath="url(#globeClip)" fill="#77977e" fillOpacity=".22" stroke="#a5bba5" strokeOpacity=".48" strokeWidth="1.2">
              <path d="M95 116l27-21 29 8 17 24-8 18-23 7-9 24-23-5-15-24z"/>
              <path d="M158 185l30-12 24 17 10 33-18 28-7 46-22 27-17-33 5-37-19-26z"/>
              <path d="M230 94l28-18 39 11 25 26-13 24-29 4-13 24-27-8-14-27z"/>
              <path d="M280 204l31-12 30 19 11 35-21 27-35-5-17-26z"/>
            </g>
            <g filter="url(#goldGlow)">
              <circle cx="280" cy="139" r="5" fill="#d2b879"/>
              <circle cx="302" cy="157" r="4" fill="#d2b879"/>
              <circle cx="250" cy="177" r="3.5" fill="#d2b879"/>
              <circle cx="174" cy="252" r="3.5" fill="#9bc2a2"/>
              <circle cx="120" cy="184" r="3" fill="#9bc2a2"/>
            </g>
            <g stroke="#c7ad70" strokeOpacity=".75" fill="none">
              <path d="M280 139Q330 90 365 72"/>
              <path d="M302 157Q344 142 375 151"/>
            </g>
          </svg>
          <span className="globe-pulse pulse-a" />
          <span className="globe-pulse pulse-b" />
        </div>
        <div className="globe-caption"><span className="caption-dot" /> Conceptual regional visualization <span>·</span> Map data integration pending</div>
      </div>

      <div className="command-panel command-right">
        <div className="command-panel-heading"><span>MODEL PERFORMANCE</span><span className="panel-index">01 / 03</span></div>
        <div className="side-metric"><span>MAE</span><strong>{projectSummary.validationMAE.toFixed(2)}</strong><small>XGBoost validation</small></div>
        <div className="side-metric"><span>RMSE</span><strong>10.86</strong><small>September 2022</small></div>
        <div className="side-metric"><span>WAPE</span><strong>57.05%</strong><small>Weighted error</small></div>
        <div className="command-divider" />
        <div className="command-panel-heading"><span>PROJECT MODULES</span></div>
        <div className="module-status"><i className="status-ready" /><span>Demand forecasting</span><b>Ready</b></div>
        <div className="module-status"><i className="status-pending" /><span>Hotspot detection</span><b>Pending</b></div>
        <div className="module-status"><i className="status-pending" /><span>Fleet optimization</span><b>Pending</b></div>
      </div>
    </div>
  );
}


function DemandMap({ selectedCity }) {
  const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadMapData() {
      try {
        setLoading(true);
        setError("");

        const response = await fetch(`${API}/api/hotspots`);
        if (!response.ok) {
          throw new Error(`Hotspot API returned ${response.status}`);
        }

        const result = await response.json();
        const rows = Array.isArray(result)
          ? result
          : result.data || result.results || result.hotspots || [];

        if (!Array.isArray(rows)) {
          throw new Error("Unexpected hotspot response format.");
        }

        if (!cancelled) setRecords(rows);
      } catch (err) {
        if (!cancelled) setError(err.message || "Could not load map data.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadMapData();
    return () => { cancelled = true; };
  }, [API]);

  const filtered = records.filter((item) => {
    const cityMatches =
      selectedCity === "All cities" || item.city === selectedCity;

    const lat = Number(item.latitude);
    const lng = Number(item.longitude);

    return cityMatches &&
      Number.isFinite(lat) &&
      Number.isFinite(lng) &&
      lat >= -90 && lat <= 90 &&
      lng >= -180 && lng <= 180;
  });

  const center = selectedCity === "Chongqing"
    ? [29.56, 106.55]
    : selectedCity === "Hangzhou"
    ? [30.27, 120.15]
    : selectedCity === "Shanghai"
    ? [31.23, 121.47]
    : selectedCity === "Yantai"
    ? [37.46, 121.45]
    : [33.5, 115.5];

  const maxDemand = Math.max(
    1,
    ...filtered.map((item) => Number(item.predicted_demand) || 0)
  );

  return (
    <div className="panel demand-map-panel">
      <div className="panel-heading">
        <div>
          <h3>Regional demand map</h3>
          <p>
            {filtered.length} mapped regions · {selectedCity}
          </p>
        </div>
        <span className="tag">Backend coordinates</span>
      </div>

      {loading && <div className="notice">Loading regional map...</div>}

      {error && (
        <div className="notice warning">
          Could not load map data: {error}
        </div>
      )}

      {!loading && !error && filtered.length === 0 && (
        <div className="notice warning">
          No records with valid latitude and longitude were returned.
        </div>
      )}

      {!loading && !error && filtered.length > 0 && (
        <MapContainer
          key={selectedCity}
          center={center}
          zoom={selectedCity === "All cities" ? 4 : 9}
          scrollWheelZoom={true}
          className="demand-leaflet-map"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {filtered.map((item, index) => {
            const demand = Number(item.predicted_demand) || 0;
            const isHotspot = item.is_hotspot === true;

            return (
              <CircleMarker
                key={`${item.city}-${item.region_id}-${index}`}
                center={[
                  Number(item.latitude),
                  Number(item.longitude)
                ]}
                radius={5 + (demand / maxDemand) * 9}
                pathOptions={{
                  color: isHotspot ? "#b94b3c" : "#397b58",
                  fillColor: isHotspot ? "#df725e" : "#82b994",
                  fillOpacity: 0.75,
                  weight: 1
                }}
              >
                <Popup>
                  <strong>{item.city}</strong>
                  <br />
                  Region: {item.region_id}
                  <br />
                  Predicted demand: {demand.toFixed(3)}
                  <br />
                  Hotspot score: {Number(item.hotspot_score || 0).toFixed(4)}
                  <br />
                  Classification: {item.hotspot_classification || "—"}
                  <br />
                  Hotspot: {isHotspot ? "Detected" : "No"}
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
      )}

      <div className="map-legend">
        <span><i className="map-dot normal" /> Regional demand</span>
        <span><i className="map-dot hotspot" /> Detected hotspot</span>
        <span>Circle size represents relative predicted demand</span>
      </div>
    </div>
  );
}


function Overview({ selectedCity, setSelectedCity }) {
  const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadOverview() {
      try {
        setLoading(true);
        setError("");

        const params = new URLSearchParams();
        if (selectedCity !== "All cities") {
          params.set("city", selectedCity);
        }

        // The overview endpoint currently returns aggregate data.
        const response = await fetch(`${API}/api/overview`);

        if (!response.ok) {
          throw new Error(`Overview API returned ${response.status}`);
        }

        const result = await response.json();

        if (!cancelled) setSummary(result);
      } catch (err) {
        if (!cancelled) setError(err.message || "Could not load overview.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadOverview();
    return () => { cancelled = true; };
  }, [API, selectedCity]);

  const format = (value, digits = 2) =>
    Number(value ?? 0).toLocaleString("en-US", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    });

  return (
    <>
      <SectionTitle
        eyebrow="PROJECT OVERVIEW"
        title="Delivery Demand Intelligence"
        description="Demand forecasting, regional hotspot analysis and fleet optimization."
        right={
          <select
            className="select-control"
            value={selectedCity}
            onChange={(event) => setSelectedCity(event.target.value)}
          >
            <option>All cities</option>
            <option>Chongqing</option>
            <option>Hangzhou</option>
            <option>Shanghai</option>
            <option>Yantai</option>
          </select>
        }
      />

      {loading && (
        <div className="notice">
          <Clock3 size={16} /> Loading project metrics...
        </div>
      )}

      {error && (
        <div className="notice warning">
          <CircleHelp size={16} /> {error}
        </div>
      )}

      {!loading && summary && (
        <>
          <div className="notice">
            <DemandMap selectedCity={selectedCity} />
            <Activity size={16} />
            Forecast target: {summary.forecast_target || "Not available"}
          </div>

          <div className="metrics-grid">
            <MetricCard
              label="Cities covered"
              value={summary.cities ?? "—"}
              detail="Cities in the planning snapshot"
              icon={MapPinned}
              accent="green"
            />
            <MetricCard
              label="Regions analysed"
              value={summary.regions ?? "—"}
              detail="Regional forecast records"
              icon={Package}
            />
            <MetricCard
              label="Predicted demand"
              value={format(summary.predicted_demand, 3)}
              detail="Selected planning-hour demand"
              icon={TrendingUp}
              accent="orange"
            />
            <MetricCard
              label="Detected hotspots"
              value={summary.hotspots ?? "—"}
              detail="Regions flagged by backend"
              icon={Activity}
              accent="purple"
            />
          </div>

          <div className="two-col">
            <div className="panel">
              <h3>Forecast model validation</h3>
              <p className="chart-footnote">
                XGBoost · September 2022 validation
              </p>
              <div className="explain-list">
                <div><b>MAE</b><p>4.0009</p></div>
                <div><b>RMSE</b><p>10.8646</p></div>
                <div><b>WAPE</b><p>57.05%</p></div>
                <div><b>MAE improvement over Seasonal Naive</b><p>15.3%</p></div>
              </div>
            </div>

            <div className="panel">
              <h3>Fleet planning</h3>
              <div className="explain-list">
                <div>
                  <b>Fleet size</b>
                  <p>{summary.fleet_size ?? "—"} vehicles</p>
                </div>
                <div>
                  <b>Spatial network</b>
                  <p>
                    {summary.graph
  ? `${summary.graph.node_count} nodes · ${summary.graph.edge_count} edges`
  : "Not available"}
                  </p>
                </div>
                <div>
  <b>Isolated regions</b>
  <p>{summary.graph?.isolated_node_count ?? "—"}</p>
</div>
<div>
  <b>Connected components</b>
  <p>{summary.graph?.component_count ?? "—"}</p>
</div>
                <div>
                  <b>Forecast target</b>
                  <p>{summary.forecast_target || "—"}</p>
                </div>
                <div>
                  <b>Dataset</b>
                  <p>LaDe-D · Four Chinese cities</p>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
}

  

function Forecast() {
  const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

  const [city, setCity] = useState("All cities");
  const [horizon, setHorizon] = useState("6");
  const [forecastData, setForecastData] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [target, setTarget] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadForecast() {
      try {
        setLoading(true);
        setError("");

        const params = new URLSearchParams({
          horizon,
          planning_hour: "1"
        });

        if (city !== "All cities") {
          params.set("city", city);
        }

        const response = await fetch(`${API}/api/forecast?${params}`);

        if (!response.ok) {
          throw new Error(`Forecast API returned ${response.status}`);
        }

        const result = await response.json();

        if (!Array.isArray(result.data)) {
          throw new Error("Unexpected forecast response format.");
        }

        if (!cancelled) {
          setForecastData(result.data);
          setMetrics(result.metrics || null);
          setTarget(result.target || "");
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Could not load forecast results.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadForecast();

    return () => {
      cancelled = true;
    };
  }, [API, city, horizon]);

  const formatNumber = (value, digits = 3) =>
    Number(value || 0).toLocaleString("en-US", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    });

  // Aggregate regional predictions into a city-level or all-city
  // forecast for each forecast step.
  const chartData = useMemo(() => {
    const grouped = new Map();

    forecastData.forEach((row) => {
      const step = Number(row.forecast_hour);
      const predicted = Number(row.predicted_demand);

      if (!Number.isFinite(step) || !Number.isFinite(predicted)) return;

      if (!grouped.has(step)) {
        grouped.set(step, {
          hour: `H+${step}`,
          forecastHour: step,
          predicted: 0,
          actual: 0,
          actualCount: 0,
          regions: 0
        });
      }

      const item = grouped.get(step);
      item.predicted += predicted;
      item.regions += 1;

      const actual = Number(row.actual_demand);
      if (row.actual_demand !== null && Number.isFinite(actual)) {
        item.actual += actual;
        item.actualCount += 1;
      }
    });

    return Array.from(grouped.values())
      .sort((a, b) => a.forecastHour - b.forecastHour)
      .map((item) => ({
        ...item,
        actual: item.actualCount ? item.actual : null
      }));
  }, [forecastData]);

  const totalPredicted = chartData.reduce(
    (sum, row) => sum + row.predicted,
    0
  );

  const firstStep = chartData[0];
  const peakStep = chartData.reduce(
    (peak, row) => !peak || row.predicted > peak.predicted ? row : peak,
    null
  );

  const actualAvailable = chartData.some((row) => row.actual !== null);

  return (
    <>
      <SectionTitle
        eyebrow="PREDICTIVE ANALYTICS"
        title="Demand Forecast"
        description="Regional parcel demand predictions generated by the XGBoost forecasting pipeline."
        right={
          <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
            <select
              className="select-control"
              value={city}
              onChange={(event) => setCity(event.target.value)}
            >
              <option value="All cities">All cities</option>
              {["Chongqing", "Hangzhou", "Shanghai", "Yantai"].map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>

            <select
              className="select-control"
              value={horizon}
              onChange={(event) => setHorizon(event.target.value)}
            >
              <option value="1">1-hour horizon</option>
              <option value="3">3-hour horizon</option>
              <option value="6">6-hour horizon</option>
            </select>
          </div>
        }
      />

      {loading && (
        <div className="notice">
          <Clock3 size={16} />
          Loading forecast results from the backend...
        </div>
      )}

      {error && (
        <div className="notice warning">
          <CircleHelp size={16} />
          Forecast API error: {error}
        </div>
      )}

      {!loading && !error && (
        <>
          <div className="notice">
            <Activity size={16} />
            <span>
              Backend forecast · {city} · {horizon}-hour horizon
              {target ? ` · First target: ${target}` : ""}
            </span>
          </div>

          <div className="metrics-grid">
            <MetricCard
              label="Regional records"
              value={forecastData.length}
              detail="Records returned by forecast API"
              icon={MapPinned}
              accent="green"
            />

            <MetricCard
              label="Forecast steps"
              value={chartData.length}
              detail="Available prediction horizons"
              icon={Clock3}
            />

            <MetricCard
              label="Total predicted demand"
              value={formatNumber(totalPredicted)}
              detail="Sum across displayed forecast steps"
              icon={Package}
              accent="orange"
            />

            <MetricCard
              label="Highest forecast step"
              value={peakStep ? peakStep.hour : "—"}
              detail={peakStep
                ? `${formatNumber(peakStep.predicted)} predicted demand`
                : "No forecast records"}
              icon={TrendingUp}
              accent="purple"
            />
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <h3>Predicted demand by forecast step</h3>
                <p>
                  Sum of regional XGBoost predictions for each forecast hour
                </p>
              </div>
            </div>

            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={330}>
                <LineChart
                  data={chartData}
                  margin={{ top: 10, right: 20, left: 5, bottom: 5 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    vertical={false}
                    stroke="#355141"
                  />
                  <XAxis dataKey="hour" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{
                      background: "#1b3025",
                      border: "1px solid #49634f",
                      borderRadius: "8px",
                      color: "#f1eee3"
                    }}
                    formatter={(value, name) => [
                      formatNumber(value),
                      name
                    ]}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    name="Predicted demand"
                    stroke="#c3a76d"
                    strokeWidth={3}
                    dot={{ r: 5 }}
                    connectNulls={false}
                  />
                  {actualAvailable && (
                    <Line
                      type="monotone"
                      dataKey="actual"
                      name="Actual demand"
                      stroke="#77a88b"
                      strokeWidth={2}
                      dot={{ r: 4 }}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ padding: "32px", textAlign: "center" }}>
                No forecast records were returned for this selection.
              </div>
            )}
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <h3>Regional forecast records</h3>
                <p>Predictions returned directly by the backend</p>
              </div>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>City</th>
                    <th>Region</th>
                    <th>Forecast step</th>
                    <th>Predicted demand</th>
                    <th>High-demand probability</th>
                    <th>High-demand prediction</th>
                  </tr>
                </thead>
                <tbody>
                  {forecastData.slice(0, 100).map((row, index) => (
                    <tr key={`${row.city}-${row.region_id}-${row.forecast_hour}-${index}`}>
                      <td>{row.city || "—"}</td>
                      <td>{row.region_id || "—"}</td>
                      <td>H+{row.forecast_hour ?? "—"}</td>
                      <td>{formatNumber(row.predicted_demand)}</td>
                      <td>
                        {row.high_demand_probability == null
                          ? "—"
                          : `${formatNumber(Number(row.high_demand_probability) * 100, 2)}%`}
                      </td>
                      <td>{row.predicted_high_demand ? "Yes" : "No"}</td>
                    </tr>
                  ))}
                  {!forecastData.length && (
                    <tr>
                      <td colSpan="6" style={{ textAlign: "center", padding: "24px" }}>
                        No forecast records available.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            {forecastData.length > 100 && (
              <p className="chart-footnote">
                Showing 100 of {forecastData.length} regional records.
              </p>
            )}
          </div>

          <div className="two-col">
            <div className="panel">
              <h3>Forecast configuration</h3>
              <div className="explain-list">
                <div>
                  <b>Forecast model</b>
                  <p>XGBoost forecasting pipeline</p>
                </div>
                <div>
                  <b>Forecast target</b>
                  <p>{target || "Returned by the backend"}</p>
                </div>
                <div>
                  <b>Forecast horizon</b>
                  <p>{horizon} hour(s)</p>
                </div>
                <div>
                  <b>Selected city</b>
                  <p>{city}</p>
                </div>
              </div>
            </div>

            <div className="panel">
              <h3>Validation performance</h3>
              <div className="explain-list">
                <div>
                  <b>MAE</b>
                  <p>{metrics?.mae == null ? "Not returned" : formatNumber(metrics.mae)}</p>
                </div>
                <div>
                  <b>RMSE</b>
                  <p>{metrics?.rmse == null ? "Not returned" : formatNumber(metrics.rmse)}</p>
                </div>
                <div>
                  <b>WAPE</b>
                  <p>{metrics?.wape == null ? "Not returned" : `${formatNumber(metrics.wape, 2)}%`}</p>
                </div>
              </div>
              <p className="chart-footnote">
                Validation metrics are separate from future forecast predictions.
              </p>
            </div>
          </div>
        </>
      )}
    </>
  );
}

  
function Performance() {
  const featureImportance = [
    { feature: "lag_24", importance: 0.173707 },
    { feature: "lag_48", importance: 0.069126 },
    { feature: "lag_72", importance: 0.023663 },
    { feature: "rolling_mean_3h", importance: 0.023113 },
    { feature: "region_id_100", importance: 0.020571 },
    { feature: "region_id_14", importance: 0.015921 },
    { feature: "lag_168", importance: 0.014207 },
    { feature: "city_Hangzhou", importance: 0.011350 },
    { feature: "region_id_155", importance: 0.011103 },
    { feature: "region_id_111", importance: 0.010059 }
  ];

  const maeImprovement = (
    ((4.722561 - 4.000929) / 4.722561) * 100
  ).toFixed(1);

  const rmseImprovement = (
    ((14.162156 - 10.864607) / 14.162156) * 100
  ).toFixed(1);

  return (
    <>
      <SectionTitle
        eyebrow="MODEL EVALUATION"
        title="Model Performance"
        description="Evaluate forecasting accuracy and understand the contribution of model features."
      />

      <div className="notice">
        <Activity size={16} />
        <span>
          Training: May 8–August 31, 2022 · Validation: September 2022.
          October remains reserved for final testing.
        </span>
      </div>

      {/* Performance KPI cards */}

      <div className="metrics-grid">
        <MetricCard
          label="XGBoost MAE"
          value="4.001"
          detail="Average absolute error"
          icon={TrendingUp}
        />

        <MetricCard
          label="XGBoost RMSE"
          value="10.865"
          detail="Root mean squared error"
          icon={Activity}
          accent="green"
        />

        <MetricCard
          label="XGBoost WAPE"
          value="57.05%"
          detail="Weighted absolute percentage error"
          icon={BarChart3}
          accent="orange"
        />

        <MetricCard
          label="MAE improvement"
          value={`${maeImprovement}%`}
          detail="Compared with Seasonal Naive"
          icon={TrendingUp}
          accent="purple"
        />
      </div>

      {/* Model comparison */}

      <div className="panel">
        <div className="panel-heading">
          <div>
            <h3>Forecasting model comparison</h3>
            <p>Comparison of all three models on validation data</p>
          </div>
          <span className="tag">September 2022</span>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={modelComparison}
            margin={{ top: 10, right: 20, left: 5, bottom: 5 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              vertical={false}
              stroke="#edf0f5"
            />

            <XAxis
              dataKey="model"
              tickLine={false}
              axisLine={false}
            />

            <YAxis tickLine={false} axisLine={false} />

            <Tooltip />

            <Legend />

            <Bar
              dataKey="mae"
              name="MAE"
              fill="#77a88b"
              radius={[5, 5, 0, 0]}
            />

            <Bar
              dataKey="rmse"
              name="RMSE"
              fill="#c3a76d"
              radius={[5, 5, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>

        <div className="chart-footnote">
          Lower values indicate better forecasting accuracy.
          MAE and RMSE are shown on the same numerical axis.
        </div>
      </div>

      {/* Detailed comparison table */}

      <div className="panel">
        <div className="panel-heading">
          <div>
            <h3>Detailed evaluation metrics</h3>
            <p>Validation results for each forecasting model</p>
          </div>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>WAPE</th>
                <th>Result</th>
              </tr>
            </thead>

            <tbody>
              {modelComparison.map((model) => (
                <tr
                  key={model.model}
                  className={
                    model.model === "XGBoost"
                      ? "highlight-row"
                      : ""
                  }
                >
                  <td><b>{model.model}</b></td>
                  <td>{model.mae.toFixed(3)}</td>
                  <td>{model.rmse.toFixed(3)}</td>
                  <td>{model.wape.toFixed(2)}%</td>
                  <td>
                    {model.model === "XGBoost" ? (
                      <span className="status good">
                        Best performer
                      </span>
                    ) : (
                      <span className="status">
                        Baseline
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* City-wise performance */}

      <div className="panel">
        <div className="panel-heading">
          <div>
            <h3>City-wise XGBoost performance</h3>
            <p>Mean absolute error across the four cities</p>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={280}>
          <BarChart
            data={cityMetrics}
            layout="vertical"
            margin={{ left: 10, right: 25 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              horizontal={false}
              stroke="#edf0f5"
            />

            <XAxis type="number" tickLine={false} axisLine={false} />

            <YAxis
              type="category"
              dataKey="city"
              width={90}
              tickLine={false}
              axisLine={false}
            />

            <Tooltip />

            <Bar
              dataKey="mae"
              name="XGBoost MAE"
              fill="#77a88b"
              radius={[0, 6, 6, 0]}
              barSize={24}
            />
          </BarChart>
        </ResponsiveContainer>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>City</th>
                <th>Regions</th>
                <th>Validation MAE</th>
              </tr>
            </thead>

            <tbody>
              {cityMetrics.map((city) => (
                <tr key={city.city}>
                  <td>{city.city}</td>
                  <td>{city.regions}</td>
                  <td>{city.mae.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature importance */}

      <div className="panel">
        <div className="panel-heading">
          <div>
            <h3>Top 10 feature importance</h3>
            <p>
              Features contributing to XGBoost predictions
            </p>
          </div>
          <span className="tag">Model analysis</span>
        </div>

        <ResponsiveContainer width="100%" height={350}>
          <BarChart
            data={featureImportance}
            layout="vertical"
            margin={{ left: 15, right: 25 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              horizontal={false}
              stroke="#edf0f5"
            />

            <XAxis
              type="number"
              tickLine={false}
              axisLine={false}
            />

            <YAxis
              type="category"
              dataKey="feature"
              width={135}
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 11 }}
            />

            <Tooltip
              formatter={(value) => [
                Number(value).toFixed(5),
                "Importance"
              ]}
            />

            <Bar
              dataKey="importance"
              name="Feature importance"
              fill="#9bb99f"
              radius={[0, 5, 5, 0]}
              barSize={18}
            />
          </BarChart>
        </ResponsiveContainer>

        <div className="chart-footnote">
          The values represent model feature importance, not causal
          effects. lag_24 is the most important feature in this
          analysis.
        </div>
      </div>

      {/* Model interpretation */}

      <div className="two-col">
        <div className="panel">
          <h3>Model interpretation</h3>

          <div className="explain-list">
            <div>
              <b>MAE improvement</b>
              <p>
                XGBoost reduces MAE by {maeImprovement}% compared
                with Seasonal Naive.
              </p>
            </div>

            <div>
              <b>RMSE improvement</b>
              <p>
                XGBoost reduces RMSE by {rmseImprovement}% compared
                with Seasonal Naive.
              </p>
            </div>

            <div>
              <b>Most important feature</b>
              <p>
                lag_24, representing demand from the previous
                24-hour period.
              </p>
            </div>
          </div>
        </div>

        <div className="panel">
          <h3>Model limitations</h3>

          <ul className="limitation-list">
            <li>Extreme demand spikes tend to be underpredicted.</li>
            <li>One-hour-ahead rolling prediction setup.</li>
            <li>Prediction intervals are not currently available.</li>
            <li>October final test evaluation is pending.</li>
          </ul>
        </div>
      </div>
    </>
  );
}





function FutureModule({ type }) {
  const [hotspots, setHotspots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedCity, setSelectedCity] = useState("All cities");

  useEffect(() => {
    let cancelled = false;

    async function loadHotspots() {
      try {
        setLoading(true);
        setError("");

        const response = await fetch(
          `${import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"}/api/hotspots`
        );

        if (!response.ok) {
          throw new Error(`Backend returned ${response.status}`);
        }

        const result = await response.json();

        // Accept either a direct array or a common wrapped API response.
        const records = Array.isArray(result)
          ? result
          : result.data || result.results || result.hotspots || [];

        if (!Array.isArray(records)) {
          throw new Error("Unexpected hotspot API response format.");
        }

        if (!cancelled) {
          setHotspots(records);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Could not load hotspot data.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadHotspots();

    return () => {
      cancelled = true;
    };
  }, []);

  if (type !== "hotspots") return null;

  const cities = [
    "All cities",
    ...new Set(hotspots.map((item) => item.city).filter(Boolean))
  ];

  const filtered = hotspots
    .filter((item) =>
      selectedCity === "All cities" || item.city === selectedCity
    )
    .sort(
      (a, b) =>
        Number(b.hotspot_score || 0) - Number(a.hotspot_score || 0)
    );

  const hotspotCount = filtered.filter(
    (item) => item.is_hotspot === true
  ).length;

  const elevatedCount = filtered.filter(
    (item) => item.hotspot_classification === "Elevated demand"
  ).length;

  const averageDemand = filtered.length
    ? filtered.reduce(
        (sum, item) => sum + Number(item.predicted_demand || 0),
        0
      ) / filtered.length
    : 0;

  return (
    <>
      <SectionTitle
        eyebrow="SPATIAL DEMAND ANALYTICS"
        title="Hotspot Intelligence"
        description="Explore predicted demand intensity and identify high-demand delivery regions."
        right={
          <select
            className="select-control"
            value={selectedCity}
            onChange={(event) => setSelectedCity(event.target.value)}
          >
            {cities.map((city) => (
              <option key={city} value={city}>{city}</option>
            ))}
          </select>
        }
      />

      {loading && (
        <div className="notice">
          <Clock3 size={16} />
          <span>Loading hotspot results from the backend...</span>
        </div>
      )}

      {error && (
        <div className="notice warning">
          <CircleHelp size={16} />
          <span>
            Could not load hotspot data: {error}. Check that FastAPI is
            running and that /api/hotspots is accessible.
          </span>
        </div>
      )}

      {!loading && !error && (
        <>
          <div className="notice">
            <MapPinned size={16} />
            <span>
              Backend-connected hotspot results ·
              {" "}{filtered.length} regional records
            </span>
          </div>

          <div className="metrics-grid">
            <MetricCard
              label="Regions analysed"
              value={filtered.length}
              detail="Regions returned by the API"
              icon={MapPinned}
              accent="green"
            />

            <MetricCard
              label="Detected hotspots"
              value={hotspotCount}
              detail="Records marked is_hotspot = true"
              icon={Activity}
              accent="orange"
            />

            <MetricCard
              label="Elevated demand"
              value={elevatedCount}
              detail="Elevated demand classification"
              icon={TrendingUp}
            />

            <MetricCard
              label="Average predicted demand"
              value={averageDemand.toFixed(3)}
              detail="Selected regions"
              icon={Package}
              accent="purple"
            />
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <h3>Regional demand intensity</h3>
                <p>Regions ranked by backend hotspot score</p>
              </div>
              <span className="tag">Backend data</span>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>City</th>
                    <th>Region</th>
                    <th>Predicted demand</th>
                    <th>Hotspot score</th>
                    <th>Classification</th>
                    <th>Hotspot</th>
                  </tr>
                </thead>

                <tbody>
                  {filtered.map((item, index) => (
                    <tr key={`${item.city}-${item.region_id}-${index}`}>
                      <td>{item.city || "—"}</td>
                      <td>{item.region_id || "—"}</td>
                      <td>
                        {Number(item.predicted_demand || 0).toFixed(3)}
                      </td>
                      <td>
                        {Number(item.hotspot_score || 0).toFixed(4)}
                      </td>
                      <td>{item.hotspot_classification || "—"}</td>
                      <td>
                        <span className={
                          item.is_hotspot === true
                            ? "status good"
                            : "status"
                        }>
                          {item.is_hotspot === true ? "Detected" : "No"}
                        </span>
                      </td>
                    </tr>
                  ))}

                  {filtered.length === 0 && (
                    <tr>
                      <td colSpan="6" style={{ textAlign: "center", padding: "24px" }}>
                        No hotspot records found for this selection.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="panel">
            <h3>Understanding hotspot results</h3>
            <div className="explain-list">
              <div>
                <b>Predicted demand</b>
                <p>Expected parcel demand for the region.</p>
              </div>
              <div>
                <b>Hotspot score</b>
                <p>Backend-computed score used to rank regional demand intensity.</p>
              </div>
              <div>
                <b>Classification</b>
                <p>The demand category returned by the backend.</p>
              </div>
              <div>
                <b>Hotspot flag</b>
                <p>Displays the backend's is_hotspot boolean separately from its classification.</p>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
}



function FleetDashboard() {
  const API = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

  const [fleetData, setFleetData] = useState(null);
  const [routeData, setRouteData] = useState(null);
  const [strategy, setStrategy] = useState("Proactive");
  const [selectedRouteId, setSelectedRouteId] = useState("");
  const [loading, setLoading] = useState(true);
  const [routeLoading, setRouteLoading] = useState(false);
  const [error, setError] = useState("");
  const [routeError, setRouteError] = useState("");

  // Load allocations for all available strategies.
  useEffect(() => {
    let cancelled = false;

    async function loadFleet() {
      try {
        setLoading(true);
        setError("");

        const response = await fetch(`${API}/api/fleet`);

        if (!response.ok) {
          throw new Error(`Fleet API returned ${response.status}`);
        }

        const result = await response.json();

        if (!result.strategies || typeof result.strategies !== "object") {
          throw new Error("Unexpected fleet API response.");
        }

        if (!cancelled) {
          setFleetData(result);

          const names = Object.keys(result.strategies);
          if (names.length && !names.includes("Proactive")) {
            setStrategy(names[0]);
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Could not load fleet allocation.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadFleet();

    return () => {
      cancelled = true;
    };
  }, [API]);

  const strategyNames = Object.keys(fleetData?.strategies || {});
  const activeStrategy = strategyNames.includes(strategy)
    ? strategy
    : strategyNames[0] || "Proactive";

  const allocations = fleetData?.strategies?.[activeStrategy] || [];
  const routes = routeData?.routes || [];

  // Load routes for the selected allocation strategy.
  useEffect(() => {
    let cancelled = false;

    async function loadRoutes() {
      try {
        setRouteLoading(true);
        setRouteError("");
        setRouteData(null);

        const query = new URLSearchParams({
          strategy: activeStrategy,
          fleet_size: "40",
          vehicle_capacity: "20"
        });

        const response = await fetch(`${API}/api/routes?${query}`);

        if (!response.ok) {
          throw new Error(`Routes API returned ${response.status}`);
        }

        const result = await response.json();

        if (!Array.isArray(result.routes)) {
          throw new Error("Unexpected routes API response.");
        }

        if (!cancelled) {
          setRouteData(result);
          setSelectedRouteId(result.routes[0]?.route_id || "");
        }
      } catch (err) {
        if (!cancelled) {
          setRouteError(err.message || "Could not load route results.");
        }
      } finally {
        if (!cancelled) setRouteLoading(false);
      }
    }

    if (fleetData) loadRoutes();

    return () => {
      cancelled = true;
    };
  }, [API, activeStrategy, fleetData]);

  const selectedRoute =
    routes.find((route) => route.route_id === selectedRouteId) ||
    routes[0] ||
    null;

  const sumField = (rows, field) =>
    rows.reduce((sum, row) => sum + (Number(row[field]) || 0), 0);

  const totalVehicles = allocations.length
    ? Math.max(...allocations.map((row) => Number(row.total_fleet_size) || 0))
    : 0;

  const allocatedVehicles = sumField(allocations, "vehicles_allocated");
  const allocatedCapacity = sumField(allocations, "capacity_allocated");
  const servedDemand = sumField(allocations, "planned_served_demand");
  const unservedDemand = sumField(allocations, "planned_unserved_demand");

  const utilization = allocatedCapacity > 0
    ? (servedDemand / allocatedCapacity) * 100
    : 0;

  const routeDistance = routes.reduce(
    (sum, route) => sum + (Number(route.distance_km) || 0),
    0
  );

  const routeMinutes = routes.reduce(
    (sum, route) => sum + (Number(route.estimated_travel_minutes) || 0),
    0
  );

  const formatNumber = (value, digits = 2) =>
    Number(value || 0).toLocaleString("en-US", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    });

  const routePath = (selectedRoute?.path || [])
    .filter((point) =>
      Array.isArray(point) &&
      Number.isFinite(Number(point[0])) &&
      Number.isFinite(Number(point[1]))
    );

  const minLng = routePath.length
    ? Math.min(...routePath.map((point) => Number(point[0])))
    : 0;
  const maxLng = routePath.length
    ? Math.max(...routePath.map((point) => Number(point[0])))
    : 1;
  const minLat = routePath.length
    ? Math.min(...routePath.map((point) => Number(point[1])))
    : 0;
  const maxLat = routePath.length
    ? Math.max(...routePath.map((point) => Number(point[1])))
    : 1;

  const projectPoint = (point) => {
    const width = maxLng - minLng || 0.001;
    const height = maxLat - minLat || 0.001;

    return {
      x: 45 + ((Number(point[0]) - minLng) / width) * 630,
      y: 275 - ((Number(point[1]) - minLat) / height) * 230
    };
  };

  const mapPoints = routePath.map(projectPoint);
  const mapLine = mapPoints.map((point) => `${point.x},${point.y}`).join(" ");

  const comparisonRows = fleetData?.comparison || [];

  if (loading) {
    return (
      <>
        <SectionTitle
          eyebrow="LOGISTICS OPTIMIZATION"
          title="Fleet & Route Optimization"
          description="Vehicle allocation and delivery route results."
        />
        <div className="notice">
          <Clock3 size={16} />
          Loading fleet allocation from the backend...
        </div>
      </>
    );
  }

  if (error) {
    return (
      <>
        <SectionTitle
          eyebrow="LOGISTICS OPTIMIZATION"
          title="Fleet & Route Optimization"
          description="Vehicle allocation and delivery route results."
        />
        <div className="notice warning">
          <CircleHelp size={16} />
          Fleet API error: {error}
        </div>
        <button className="reset-simulation" onClick={() => window.location.reload()}>
          Retry
        </button>
      </>
    );
  }

  return (
    <>
      <SectionTitle
        eyebrow="LOGISTICS OPTIMIZATION"
        title="Fleet & Route Optimization"
        description="Vehicle allocation and delivery route results."
        right={
          <select
            className="select-control"
            value={activeStrategy}
            onChange={(event) => setStrategy(event.target.value)}
          >
            {strategyNames.map((name) => (
              <option key={name} value={name}>{name} strategy</option>
            ))}
          </select>
        }
      />

      <div className="metrics-grid">
        <MetricCard
          label="Total fleet"
          value={totalVehicles}
          detail="Available vehicles"
          icon={Truck}
          accent="green"
        />
        <MetricCard
          label="Vehicles allocated"
          value={allocatedVehicles}
          detail="Assigned across delivery regions"
          icon={MapPinned}
        />
        <MetricCard
          label="Fleet utilization"
          value={`${formatNumber(utilization, 1)}%`}
          detail="Served demand / allocated capacity"
          icon={Activity}
          accent="orange"
        />
        <MetricCard
          label="Unserved demand"
          value={formatNumber(unservedDemand, 3)}
          detail="Planned demand not covered"
          icon={Package}
          accent="purple"
        />
      </div>

      <div className="panel">
        <div className="panel-heading">
          <div>
            <h3>Regional vehicle allocation</h3>
            <p>{activeStrategy} strategy · {allocations.length} allocation records</p>
          </div>
        </div>

        <div className="metrics-grid">
          <MetricCard
            label="Served demand"
            value={formatNumber(servedDemand, 3)}
            detail="Planned served demand"
            icon={Package}
            accent="green"
          />
          <MetricCard
            label="Allocated capacity"
            value={formatNumber(allocatedCapacity, 1)}
            detail="Total assigned vehicle capacity"
            icon={Truck}
          />
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>City</th>
                <th>Region</th>
                <th>Predicted demand</th>
                <th>Vehicles</th>
                <th>Served demand</th>
                <th>Unserved demand</th>
                <th>Hotspot class</th>
              </tr>
            </thead>
            <tbody>
              {allocations.slice(0, 100).map((row, index) => (
                <tr key={`${row.city}-${row.region_id}-${index}`}>
                  <td>{row.city || "—"}</td>
                  <td>{row.region_id || "—"}</td>
                  <td>{formatNumber(row.predicted_demand, 3)}</td>
                  <td>{row.vehicles_allocated ?? "—"}</td>
                  <td>{formatNumber(row.planned_served_demand, 3)}</td>
                  <td>{formatNumber(row.planned_unserved_demand, 3)}</td>
                  <td>{row.hotspot_classification || "—"}</td>
                </tr>
              ))}
              {!allocations.length && (
                <tr>
                  <td colSpan="7" style={{ textAlign: "center", padding: "24px" }}>
                    No allocation records returned for this strategy.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {allocations.length > 100 && (
          <p className="chart-footnote">
            Showing the first 100 of {allocations.length} regional records.
          </p>
        )}
      </div>

      <div className="panel">
        <div className="panel-heading">
          <div>
            <h3>Strategy comparison</h3>
            <p>Backend fleet comparison results</p>
          </div>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {Object.keys(comparisonRows[0] || {}).map((key) => (
                  <th key={key}>{key.replaceAll("_", " ")}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {comparisonRows.map((row, index) => (
                <tr key={index}>
                  {Object.keys(comparisonRows[0] || {}).map((key) => (
                    <td key={key}>
                      {typeof row[key] === "number"
                        ? formatNumber(row[key], 2)
                        : String(row[key] ?? "—")}
                    </td>
                  ))}
                </tr>
              ))}
              {!comparisonRows.length && (
                <tr>
                  <td colSpan="5" style={{ textAlign: "center", padding: "20px" }}>
                    No comparison records returned.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel">
        <div className="panel-heading">
          <div>
            <h3>Delivery route visualization</h3>
            <p>
              {routes.length} routes · {selectedRoute?.city || "No city selected"}
            </p>
          </div>
          <select
            className="select-control"
            value={selectedRoute?.route_id || ""}
            onChange={(event) => setSelectedRouteId(event.target.value)}
            disabled={!routes.length}
          >
            {routes.map((route) => (
              <option key={route.route_id} value={route.route_id}>
                {route.route_id}
              </option>
            ))}
          </select>
        </div>

        {routeLoading && (
          <div className="notice">
            <Clock3 size={16} /> Loading route results...
          </div>
        )}

        {routeError && (
          <div className="notice warning">
            <CircleHelp size={16} /> Route API error: {routeError}
          </div>
        )}

        {!routeLoading && !routeError && selectedRoute && (
          <>
            <div className="metrics-grid">
              <MetricCard
                label="Route distance"
                value={`${formatNumber(selectedRoute.distance_km, 2)} km`}
                detail="Estimated route distance"
                icon={Route}
                accent="green"
              />
              <MetricCard
                label="Estimated travel"
                value={`${formatNumber(selectedRoute.estimated_travel_minutes, 1)} min`}
                detail="Backend estimate"
                icon={Clock3}
              />
              <MetricCard
                label="Delivery stops"
                value={(selectedRoute.stops || []).length}
                detail="Regions on selected route"
                icon={MapPinned}
                accent="orange"
              />
              <MetricCard
                label="Planned load"
                value={formatNumber(selectedRoute.planned_load, 3)}
                detail={`Vehicle capacity: ${selectedRoute.vehicle_capacity ?? "—"}`}
                icon={Package}
              />
            </div>

            <div style={{
              width: "100%",
              overflow: "hidden",
              borderRadius: "12px",
              background: "#14251d",
              border: "1px solid #30483a"
            }}>
              {mapPoints.length > 1 ? (
                <svg
                  viewBox="0 0 720 320"
                  role="img"
                  aria-label={`Route ${selectedRoute.route_id} plotted from backend coordinates`}
                  style={{ width: "100%", display: "block" }}
                >
                  {[70, 150, 230, 310, 390, 470, 550, 630].map((x) => (
                    <line key={`v-${x}`} x1={x} y1="20" x2={x} y2="300"
                      stroke="#355141" strokeDasharray="3 7" />
                  ))}
                  {[60, 120, 180, 240].map((y) => (
                    <line key={`h-${y}`} x1="20" y1={y} x2="700" y2={y}
                      stroke="#355141" strokeDasharray="3 7" />
                  ))}
                  <polyline
                    points={mapLine}
                    fill="none"
                    stroke="#c5a76c"
                    strokeWidth="4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  {mapPoints.map((point, index) => (
                    <circle
                      key={index}
                      cx={point.x}
                      cy={point.y}
                      r={index === 0 || index === mapPoints.length - 1 ? 7 : 5}
                      fill={index === 0 ? "#8dbb98" : "#d0b77e"}
                      stroke="#14251d"
                      strokeWidth="2"
                    />
                  ))}
                </svg>
              ) : (
                <div style={{ padding: "35px", textAlign: "center" }}>
                  No route coordinates are available for this route.
                </div>
              )}
            </div>

            <p className="chart-footnote">
              Route plotted from the longitude and latitude coordinates returned by the API.
              Distance method: {selectedRoute.distance_method || "Not specified"}.
              This is a coordinate plot, not a street map.
            </p>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Stop</th>
                    <th>Region</th>
                    <th>Longitude</th>
                    <th>Latitude</th>
                    <th>Planned delivery</th>
                  </tr>
                </thead>
                <tbody>
                  {(selectedRoute.stops || []).map((stop, index) => (
                    <tr key={`${stop.region_id}-${index}`}>
                      <td>{index + 1}</td>
                      <td>{stop.region_id}</td>
                      <td>{formatNumber(stop.longitude, 5)}</td>
                      <td>{formatNumber(stop.latitude, 5)}</td>
                      <td>{formatNumber(stop.planned_delivery, 4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {!routeLoading && !routeError && !selectedRoute && (
          <div className="notice">
            No routes were returned for this strategy.
          </div>
        )}
      </div>

      <div className="two-col">
        <div className="panel">
          <h3>Route summary</h3>
          <div className="explain-list">
            <div>
              <b>Total routes</b>
              <p>{routes.length}</p>
            </div>
            <div>
              <b>Combined estimated distance</b>
              <p>{formatNumber(routeDistance, 2)} km</p>
            </div>
            <div>
              <b>Combined estimated travel time</b>
              <p>{formatNumber(routeMinutes, 1)} minutes</p>
            </div>
          </div>
        </div>

        <div className="panel">
          <h3>Optimization method</h3>
          <div className="explain-list">
            <div>
              <b>Fleet allocation</b>
              <p>Regional vehicle assignments returned by the selected backend strategy.</p>
            </div>
            <div>
              <b>Route sequencing</b>
              <p>Delivery stops and route paths returned by the route optimizer.</p>
            </div>
            <div>
              <b>Distance calculation</b>
              <p>Current backend output uses the distance method identified above.</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}


export default function App() {
  const [active, setActive] = useState("overview");
  const [selectedCity, setSelectedCity] = useState("All cities");

  const page = useMemo(() => {
    if (active === "forecast") return <Forecast />;
    if (active === "performance") return <Performance />;
if (active === "hotspots") {
  return <FutureModule type="hotspots" />;
}

if (active === "fleet") {
  return <FleetDashboard />;
}

    return (
      <Overview
        selectedCity={selectedCity}
        setSelectedCity={setSelectedCity}
      />
    );
  }, [active, selectedCity]);

  const activeLabel =
    navigation.find((item) => item.id === active)?.label || "Overview";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Boxes size={21} />
          </div>
          <div>
            <b>LaDe-D</b>
            <small>DELIVERY INTELLIGENCE</small>
          </div>
        </div>

        <div className="workspace-label">WORKSPACE</div>

        <nav>
          {navigation.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.id}
                className={`nav-item ${active === item.id ? "active" : ""}`}
                onClick={() => setActive(item.id)}
              >
                <Icon size={18} />
<span>{item.label}</span>
</button>
            );
          })} 
        </nav>

        <div className="sidebar-bottom">
          <div className="dataset-status">
            <span className="status-dot" />
            <div>
              <b>Dataset loaded</b>
              <small>LaDe-D · 2022</small>
            </div>
          </div>

          <div className="sidebar-foot">
            <Settings2 size={16} />
            Project workspace
            <ChevronDown size={14} />
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> <b>{activeLabel}</b>
          </div>

          <div className="top-actions">
          </div>
        </header>

        <div className="page-content">
          {page}

          <footer>
            <span>LaDe-D Delivery Demand Intelligence</span>
            <span>Historical research dashboard · 2022</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
