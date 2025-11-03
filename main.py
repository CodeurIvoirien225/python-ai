import asyncio
import aiohttp
import aiohttp.web
import json
import cv2
import numpy as np
from surveillance import BehaviorAnalyzer
import time
import os

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    else:
        return obj

class AISurveillanceServer:
    def __init__(self, max_fps=10):
        print("🔄 Initialisation de l'analyseur de comportement...")
        self.analyzer = BehaviorAnalyzer()
        self.clients = set()
        self.scores_per_client = {}
        self.employee_ids = {}
        self.last_frame_time = {}
        self.max_fps = max_fps
        print("✅ Serveur IA prêt")

    async def handle_ws(self, request):
        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)
        self.clients.add(ws)
        print(f"✅ Nouveau client connecté (WebSocket)")

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data.get('type') == 'init' and 'employee_id' in data:
                        self.employee_ids[ws] = data['employee_id']
                        print(f"🆔 employee_id reçu: {data['employee_id']}")
                except Exception as e:
                    print(f"❌ Erreur JSON: {e}")

            elif msg.type == aiohttp.WSMsgType.BINARY:
                now = time.time()
                last_time = self.last_frame_time.get(ws, 0)
                if now - last_time < 1 / self.max_fps:
                    continue
                self.last_frame_time[ws] = now

                np_arr = np.frombuffer(msg.data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    print("⚠️ Frame vide reçue")
                    continue

                analysis = self.analyzer.analyze_behavior(frame)
                analysis_serializable = make_json_serializable(analysis)
                score = analysis_serializable.get('credibility_score', 100)
                self.scores_per_client.setdefault(ws, []).append(score)

                await ws.send_json(analysis_serializable)

            elif msg.type == aiohttp.WSMsgType.ERROR:
                print(f"❌ WebSocket erreur {ws.exception()}")

        # Cleanup client
        final_score = None
        if ws in self.scores_per_client:
            scores = self.scores_per_client.pop(ws)
            if scores:
                final_score = sum(scores) / len(scores)
                print(f"📊 Score final: {final_score}")

        employee_id = self.employee_ids.pop(ws, None)
        if final_score and employee_id:
            asyncio.create_task(self.send_score_to_backend(final_score, employee_id))

        self.clients.discard(ws)
        self.last_frame_time.pop(ws, None)
        print("🔌 Client déconnecté")
        return ws

    async def send_score_to_backend(self, score, employee_id):
        url = "https://ton-backend-php-url/save_credibility_score.php"
        payload = {"employee_id": employee_id, "score_de_credibilite": round(score)}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        print(f"✅ Score enregistré pour employee_id={employee_id}")
                    else:
                        print(f"❌ Erreur HTTP {resp.status}")
            except Exception as e:
                print(f"❌ Exception lors de l’envoi: {e}")

async def handle_http(request):
    return aiohttp.web.Response(text="✅ Serveur IA actif (Render check OK)")

async def main():
    port = int(os.environ.get("PORT", 10000))
    server = AISurveillanceServer(max_fps=10)

    app = aiohttp.web.Application()
    app.router.add_get("/", handle_http)         # Ping HTTP
    app.router.add_get("/ws", server.handle_ws)  # WebSocket endpoint

    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    print(f"🌍 Serveur HTTP + WebSocket prêt sur le port {port}")
    await asyncio.Future()  # bloque indéfiniment

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Serveur arrêté par l'utilisateur")
