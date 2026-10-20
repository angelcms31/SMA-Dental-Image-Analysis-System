// Backend base URL. Set VITE_API_URL when deploying (e.g. on Render's static
// site settings); falls back to the local FastAPI server for development.
export const API_BASE: string = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/+$/, '');
