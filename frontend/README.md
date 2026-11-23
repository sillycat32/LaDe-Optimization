# LaDe-D Delivery Intelligence Frontend

A React + Vite frontend starter that can be developed before the backend exists.

## Requirements
- Install Node.js LTS (includes npm)
- Visual Studio Code

## Run in VS Code
1. Extract this ZIP.
2. In VS Code choose **File → Open Folder** and open `lade-dashboard-frontend`.
3. Open **Terminal → New Terminal**.
4. Run:

   ```powershell
   npm install
   npm run dev
   ```

5. Open the local URL printed by Vite (usually http://localhost:5173).

## What's included
- Overview dashboard
- Demand forecast page
- Model performance page
- Hotspot and fleet/route placeholder pages
- Mock-data mode and a central API adapter (`src/api.js`)

## Important data notes
- The metric cards and model comparison numbers are based on the project's current processed dataset and September validation results.
- The hourly chart and daily trend chart are explicitly illustrative demo series; replace them with real API responses before presenting them as measured results.
- The dataset is historical (2022), not live.
- No geographic map is shown because the processed feature dataset currently does not contain region coordinates or boundaries.
- Hotspot detection, fleet allocation and route optimization are not implemented in the backend yet.

## Connecting a future FastAPI backend
Create `.env` in the project root:

```env
VITE_USE_MOCK_DATA=false
VITE_API_BASE_URL=http://localhost:8000
```

The adapter currently expects:
- `GET /api/overview`
- `GET /api/model-performance`
- `GET /api/demand`
- `GET /api/forecast`

The page components currently use local demo data for layout. When your API is ready, replace the local imports in `src/App.jsx` with calls to the functions in `src/api.js`, then map the returned JSON fields into the charts and tables.

If frontend and backend run on different ports, enable CORS in FastAPI for `http://localhost:5173`.
