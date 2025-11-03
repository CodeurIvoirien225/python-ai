import asyncio
import websockets
import json
import cv2
import numpy as np
from surveillance import BehaviorAnalyzer
import signal
import sys
import aiohttp
import aiohttp.web
import time
import os

# --- Fonction utilitaire pour rendre JSON serializable ---
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

    async def handle_video_stream(self, websocket):
        self.clients.add(websocket)
        print(f"✅ Nouveau client connecté: {websocket.remote_address}")

        try:
            async for message in websocket:
                try:
                    # --- JSON init ---
                    if isinstance(message, str):
                        data = json.loads(message)
                        if data.get('type') == 'init' and 'employee_id' in data:
                            self.employee_ids[websocket] = data['employee_id']
                            print(f"🆔 employee_id reçu: {data['employee_id']}")
                            continue

                    # --- Frame binaire ---
                    if isinstance(message, (bytes, bytearray)):
                        now = time.time()
                        last_time = self.last_frame_time.get(websocket, 0)
                        if now - last_time < 1 / self.max_fps:
                            continue
                        self.last_frame_time[websocket] = now

                        np_arr = np.frombuffer(message, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if frame is None:
                            print("⚠️ Frame vide reçue (imdecode a échoué)")
                            continue

                        analysis = self.analyzer.analyze_behavior(frame)
                        analysis_serializable = make_json_serializable(analysis)
                        score = analysis_serializable.get('credibility_score', 100)

                        self.scores_per_client.setdefault(websocket, []).append(score)
                        asyncio.create_task(
                            self.safe_send(websocket, json.dumps(analysis_serializable))
                        )

                except Exception as e:
                    print(f"❌ Erreur traitement frame: {e}")
                    continue

        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Client déconnecté: {websocket.remote_address}")
        finally:
            await self.cleanup_client(websocket)

    async def safe_send(self, websocket, message):
        try:
            await websocket.send(message)
        except Exception:
            pass

    async def cleanup_client(self, websocket):
        final_score = None
        if websocket in self.scores_per_client:
            scores = self.scores_per_client.pop(websocket)
            if scores:
                final_score = sum(scores) / len(scores)
                print(f"📊 Score final de crédibilité: {final_score}")

        employee_id = self.employee_ids.pop(websocket, None)
        if final_score and employee_id:
            asyncio.create_task(self.send_score_to_backend(final_score, employee_id))

        self.clients.discard(websocket)
        self.last_frame_time.pop(websocket, None)

    async def send_score_to_backend(self, score, employee_id):
        # ⚠️ Remplace cette URL par celle de ton backend hébergé
        url = "https://ton-backend-php-url/save_credibility_score.php"
        payload = {"employee_id": employee_id, "score_de_credibilite": round(score)}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        print(f"✅ Score enregistré pour employee_id={employee_id}")
                    else:
                        print(f"❌ Erreur HTTP {resp.status} lors de l'envoi")
            except Exception as e:
                print(f"❌ Exception lors de l’envoi: {e}")


# --- Serveur HTTP pour Render ---
async def handle_http(request):
    return aiohttp.web.Response(text="✅ Serveur IA actif (Render check OK)")

async def start_http_server(port):
    app = aiohttp.web.Application()
    app.router.add_get("/", handle_http)
    app.router.add_head("/", handle_http)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"🌍 Serveur HTTP prêt sur le port {port}")


async def main():
    port = int(os.environ.get("PORT", 10000))
    server = AISurveillanceServer(max_fps=10)

    # Lancer HTTP et WebSocket sur le même port
    await start_http_server(port)

    print(f"🚀 Démarrage du serveur WebSocket sur le port {port} (Render)")
    async with websockets.serve(
        server.handle_video_stream,
        host="0.0.0.0",
        port=port,
        ping_interval=30,
        ping_timeout=30,
        max_size=5_000_000,
    ):
        print(f"✅ Serveur WebSocket prêt sur ws://0.0.0.0:{port}")
        await asyncio.Future()  # bloque indéfiniment


def signal_handler(sig, frame):
    print("\n🛑 Arrêt du serveur IA...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Serveur arrêté par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
