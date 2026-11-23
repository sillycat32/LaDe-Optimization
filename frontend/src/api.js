
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function getJson(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);

  if (!response.ok) {
    throw new Error(`API error ${response.status}: ${await response.text()}`);
  }

  return response.json();
}

export const api = {
  getHealth: () => getJson("/api/health"),
  getOverview: () => getJson("/api/overview"),
  getForecast: (params = {}) =>
    getJson(`/api/forecast?${new URLSearchParams(params)}`),
  getHourlyDemand: (params = {}) =>
    getJson(`/api/demand?${new URLSearchParams(params)}`),
  getHotspots: (params = {}) =>
    getJson(`/api/hotspots?${new URLSearchParams(params)}`),
  getFleet: (params = {}) =>
    getJson(`/api/fleet?${new URLSearchParams(params)}`),
  getRoutes: (params = {}) =>
    getJson(`/api/routes?${new URLSearchParams(params)}`),
  getCityMetrics: () => getJson("/api/model-performance"),
};
