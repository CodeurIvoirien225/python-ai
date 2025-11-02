import asyncio
import websockets
import json
import cv2
import numpy as np
from surveillance import BehaviorAnalyzer
import signal
import sys
import aiohttp


class AISurveillanceServer:
    def __init__(self):
        print("🔄 Initialisation de l'analyseur de comportement...")
        self.analyzer = BehaviorAnalyzer()
        self.clients = set()
        self.scores_per_client = {}    # Stocke tous les scores pour chaque client
        self.employee_ids = {}         # websocket -> employee_id

    async def handle_video_stream(self, websocket):
        """Gère la réception vidéo depuis le client (frames binaires JPEG)"""
        self.clients.add(websocket)
        print(f"✅ Nouveau client connecté: {websocket.remote_address}")

        try:
            async for message in websocket:
                try:
                    # --- Message JSON d'initialisation avec employee_id ---
                    if isinstance(message, str):
                        data = json.loads(message)
                        if data.get('type') == 'init' and 'employee_id' in data:
                            self.employee_ids[websocket] = data['employee_id']
                            print(f"🆔 employee_id reçu: {data['employee_id']}")
                            continue

                    # --- Frame binaire ---
                    if isinstance(message, (bytes, bytearray)):
                        print(f"📸 Frame reçue : {len(message)} octets")

                        np_arr = np.frombuffer(message, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if frame is None:
                            print("⚠️ Frame vide reçue (imdecode a échoué)")
                            continue

                        # 🧠 Analyse du comportement
                        analysis = self.analyzer.analyze_behavior(frame)

                        score = analysis.get('credibility_score', 100)

                        # Initialiser la liste des scores pour ce client si besoin
                        if websocket not in self.scores_per_client:
                            self.scores_per_client[websocket] = []

                        # Ajouter le score actuel
                        self.scores_per_client[websocket].append(score)

                        # 📤 Envoi du résultat JSON au client
                        await websocket.send(json.dumps(analysis))

                    else:
                        print("⚠️ Données texte ignorées (attendu: binaire JPEG)")

                except Exception as e:
                    print(f"❌ Erreur traitement frame: {e}")
                    continue

        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 Client déconnecté: {websocket.remote_address}")
        except Exception as e:
            print(f"❌ Erreur connexion: {e}")
        finally:
            # --- calcul du score final ---
            final_score = None
            if websocket in self.scores_per_client:
                scores = self.scores_per_client.pop(websocket)
                if scores:
                    final_score = sum(scores) / len(scores)
                    print(f"📊 Score final de crédibilité: {final_score}")

            # --- Envoi au backend ---
            employee_id = self.employee_ids.pop(websocket, None)
            if final_score is not None and employee_id is not None:
                await self.send_score_to_backend(final_score, employee_id)
            elif final_score is not None:
                print("⚠️ Impossible d’envoyer le score : employee_id manquant")

    async def send_score_to_backend(self, score, employee_id):
        url = "http://localhost/Recrutement/recruitment-app/backend-php/save_credibility_score.php"
        payload = {"employee_id": employee_id, "score_de_credibilite": round(score)}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        print(f"✅ Score enregistré pour employee_id={employee_id}")
                    else:
                        print(f"❌ Erreur enregistrement score: HTTP {resp.status}")
            except Exception as e:
                print(f"❌ Exception lors de l’envoi au backend: {e}")


# 🧹 Gestion propre de l’arrêt avec CTRL+C
def signal_handler(sig, frame):
    print("\n🛑 Arrêt du serveur IA...")
    sys.exit(0)


async def main():
    server = AISurveillanceServer()

    try:
        async with websockets.serve(
            server.handle_video_stream,
            "localhost",
            8765,
            ping_interval=20,
            ping_timeout=10,
            max_size=2_000_000  # ~2MB par message
        ):
            print("🚀 Serveur IA démarré sur ws://localhost:8765")
            print("📡 En attente de connexions clients...")
            await asyncio.Future()  # garde le serveur actif
    except Exception as e:
        print(f"❌ Erreur démarrage serveur: {e}")
    finally:
        print("🔴 Serveur arrêté")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Serveur arrêté par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
