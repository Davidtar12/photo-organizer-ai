Frontend (static) for Face Browser

This is a lightweight static frontend that talks to the running backend API at http://127.0.0.1:5052/api.

How to use

1. Make sure the backend is running. From the project root where your backend `run_backend.py` lives:

```powershell
# activate your photoenv first (if not already active)
# e.g. if you use virtualenv: C:\dscodingpython\photoenv\Scripts\Activate.ps1
python "c:\dscodingpython\File organizers\run_backend.py"
```

2. Serve this folder locally (recommended) so the UI can fetch from the backend without file:// origin issues.

```powershell
# from a Powershell prompt
cd "c:\dscodingpython\File organizers\face_browser\frontend"
python -m http.server 5173
```

3. Open the UI in your browser:

http://127.0.0.1:5173/

Notes

- The static UI expects the backend API to be available at http://127.0.0.1:5052/api. If your backend runs on a different host/port, edit `index.html` and change the `API_BASE` constant.
- This is intentionally a minimal static page to inspect people/clusters and photos. For a full-featured, production-ready frontend we can scaffold a React + Vite app and wire it to the backend; tell me if you'd like me to do that next.

Troubleshooting

- If the page shows an error "Failed to load people", check the browser console for CORS errors. If you see CORS errors, ensure the backend permits requests from the frontend origin or serve the frontend from the same host/port domain (localhost).
- If thumbnails/images 404, verify the backend `media` endpoints are reachable (open http://127.0.0.1:5052/api/persons/ in the browser to verify JSON).