#!/usr/bin/env python
"""Servidor local para abrir el dashboard sin problemas de permisos del navegador."""
from pathlib import Path
import http.server
import socketserver
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
PORT = 8000
dashboard = ROOT / "dashboard" / "dashboard_interactivo_final.html"

if not dashboard.exists():
    raise SystemExit("No existe dashboard/dashboard_interactivo_final.html")

class Handler(http.server.SimpleHTTPRequestHandler):
    pass

print(f"Abriendo dashboard en http://localhost:{PORT}/dashboard/dashboard_interactivo_final.html")
print("Presiona Ctrl+C para detener.")
webbrowser.open(f"http://localhost:{PORT}/dashboard/dashboard_interactivo_final.html")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.RequestHandlerClass.directory = str(ROOT)
    httpd.serve_forever()
