// DEMO DATA: metrics below are from the current project results.
// Chart series are illustrative placeholders until the frontend is connected
// to the real Parquet/API output. Do not present them as measured results.

export const projectSummary = {
  observations: 713664,
  cities: 4,
  regions: 168,
  meanHourlyDemand: 6.16,
  validationMAE: 4.0009,
  validationRMSE: 10.8646,
  validationWAPE: 57.05,
  validationPeriod: "1–30 September 2022",
  trainingPeriod: "8 May–31 August 2022"
};

export const cityMetrics = [
  { city: "Chongqing", regions: 48, rows: 203904, mae: 2.4350, color: "#4A8BDF" },
  { city: "Hangzhou", regions: 51, rows: 216648, mae: 5.1400, color: "#A0006D" },
  { city: "Shanghai", regions: 52, rows: 220896, mae: 4.9220, color: "#2868AD" },
  { city: "Yantai", regions: 17, rows: 72216, mae: 2.1878, color: "#7AB5E8" }
];

// Illustrative shape only. Replace with actual grouped prediction data from API.
export const hourlyDemo = [
  { hour: "00", actual: 0.1, predicted: 0.2 }, { hour: "01", actual: 0.0, predicted: 0.1 },
  { hour: "02", actual: 0.0, predicted: 0.1 }, { hour: "03", actual: 0.0, predicted: 0.1 },
  { hour: "04", actual: 0.0, predicted: 0.1 }, { hour: "05", actual: 0.0, predicted: 0.1 },
  { hour: "06", actual: 0.15, predicted: 0.21 }, { hour: "07", actual: 10.535, predicted: 10.14 },
  { hour: "08", actual: 38.375, predicted: 37.075 }, { hour: "09", actual: 20.206, predicted: 19.73 },
  { hour: "10", actual: 11.693, predicted: 10.927 }, { hour: "11", actual: 9.307, predicted: 8.709 },
  { hour: "12", actual: 6.729, predicted: 6.622 }, { hour: "13", actual: 9.687, predicted: 9.894 },
  { hour: "14", actual: 16.305, predicted: 16.937 }, { hour: "15", actual: 19.382, predicted: 18.928 },
  { hour: "16", actual: 12.746, predicted: 12.367 }, { hour: "17", actual: 6.15, predicted: 6.712 },
  { hour: "18", actual: 4.377, predicted: 5.034 }, { hour: "19", actual: 1.739, predicted: 2.102 },
  { hour: "20", actual: 0.59, predicted: 0.763 }, { hour: "21", actual: 0.241, predicted: 0.338 },
  { hour: "22", actual: 0.074, predicted: 0.136 }, { hour: "23", actual: 0.02, predicted: 0.044 }
];

export const demandTrendDemo = [
  { day: "Sep 01", demand: 5.8 }, { day: "Sep 04", demand: 6.4 },
  { day: "Sep 07", demand: 6.1 }, { day: "Sep 10", demand: 6.8 },
  { day: "Sep 13", demand: 6.2 }, { day: "Sep 16", demand: 7.1 },
  { day: "Sep 19", demand: 6.7 }, { day: "Sep 22", demand: 6.3 },
  { day: "Sep 25", demand: 6.9 }, { day: "Sep 28", demand: 6.5 },
  { day: "Sep 30", demand: 6.6 }
];