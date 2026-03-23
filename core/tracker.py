import numpy as np
from collections import defaultdict

class PersonTracker:
    """
    Tracker simple par IoU (Intersection over Union).
    Associe les détections frame par frame avec un ID persistant.
    """
    def __init__(self, iou_threshold=0.3, max_lost=30):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost       # Frames avant suppression d'un track perdu
        self.next_id = 1
        self.tracks = {}                # { track_id: { bbox, name, confidence, lost_frames } }

    def _iou(self, box1, box2):
        """Calcule l'IoU entre deux bboxes [x1,y1,x2,y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def update(self, detections, frame=None):
        """
        Met à jour les tracks avec les nouvelles détections.

        Args:
            detections: liste de dict [{ bbox, confidence, name }, ...]
            frame: non utilisé ici mais gardé pour compatibilité

        Returns:
            liste de dict [{ bbox, name, confidence, track_id }, ...]
        """
        if not detections:
            # Incrémenter lost_frames pour tous les tracks
            to_remove = []
            for tid in self.tracks:
                self.tracks[tid]['lost_frames'] += 1
                if self.tracks[tid]['lost_frames'] > self.max_lost:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.tracks[tid]
            return []

        # ── Calcul de la matrice IoU ──
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)))

        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(self.tracks[tid]['bbox'], det['bbox'])

        # ── Association greedy ──
        matched_tracks = set()
        matched_dets = set()
        matches = []

        # Trier par IoU décroissant
        if len(track_ids) > 0 and len(detections) > 0:
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break
                idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                i, j = idx[0], idx[1]

                if i not in matched_tracks and j not in matched_dets:
                    matches.append((track_ids[i], j))
                    matched_tracks.add(i)
                    matched_dets.add(j)

                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0

        # ── Mettre à jour les tracks matchés ──
        results = []

        for tid, det_idx in matches:
            det = detections[det_idx]
            self.tracks[tid]['bbox'] = det['bbox']
            self.tracks[tid]['lost_frames'] = 0

            # Mettre à jour le nom seulement si la nouvelle détection est identifiée
            if det.get('name', 'Inconnu') != 'Inconnu':
                self.tracks[tid]['name'] = det['name']
                self.tracks[tid]['confidence'] = det.get('confidence', 0)
            elif self.tracks[tid]['name'] == 'Inconnu':
                self.tracks[tid]['confidence'] = det.get('confidence', 0)

        # ── Créer de nouveaux tracks pour les détections non matchées ──
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'name': det.get('name', 'Inconnu'),
                    'confidence': det.get('confidence', 0),
                    'lost_frames': 0
                }
                self.next_id += 1

        # ── Supprimer les tracks perdus ──
        unmatched_track_indices = set(range(len(track_ids))) - matched_tracks
        to_remove = []
        for i in unmatched_track_indices:
            tid = track_ids[i]
            self.tracks[tid]['lost_frames'] += 1
            if self.tracks[tid]['lost_frames'] > self.max_lost:
                to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

        # ── Construire le résultat ──
        for tid, track in self.tracks.items():
            if track['lost_frames'] == 0:  # Seulement les tracks actifs
                results.append({
                    'bbox': track['bbox'],
                    'name': track['name'],
                    'confidence': track['confidence'],
                    'track_id': tid
                })

        return results

    def release(self):
        """Nettoie les ressources."""
        self.tracks.clear()
        print("[TRACKER] Ressources libérées")
