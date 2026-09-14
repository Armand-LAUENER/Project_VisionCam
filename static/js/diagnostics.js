// Page Diagnostic : temps par étape, matériel, réglages, mesure des sources d'orientation.
import { api, formatDuration, h, toast } from './common.js';

const STAGES = {
    detection: 'Détection (YOLO)',
    tracking: 'Tracking',
    recognition: 'Reconnaissance',
    pose: 'Orientation',
    encode: 'Encodage vidéo',
    frame: 'Image complète',
};

function pairs(target, rows) {
    target.replaceChildren(...rows.flatMap(([key, value]) => [h('dt', {}, key), h('dd', {}, value ?? '—')]));
}

function renderTimings(timings) {
    const container = document.getElementById('timings');
    const entries = Object.entries(STAGES).filter(([stage]) => timings[stage]);
    if (!entries.length) {
        container.replaceChildren(h('p', { class: 'empty' }, 'Pas encore de mesure : la caméra est-elle connectée ?'));
        return;
    }
    const worst = Math.max(...entries.map(([stage]) => timings[stage].p95_ms), 1);
    container.replaceChildren(...entries.map(([stage, label]) => {
        const t = timings[stage];
        return h('div', { class: 'timing' },
            h('span', {}, label),
            h('div', { class: 'bar', role: 'img', 'aria-label': `${label} : ${t.median_ms} ms` },
                h('span', { style: `width: ${Math.min(100, (t.median_ms / worst) * 100)}%` })),
            h('span', { class: 'mono' }, `${t.median_ms} ms · p95 ${t.p95_ms}`));
    }));
}

async function load() {
    const { ok, data } = await api('/api/diagnostics');
    if (!ok) return;
    document.getElementById('diagSummary').textContent =
        `${data.fps ? data.fps.toFixed(0) : '—'} images/s · en marche depuis ${formatDuration(data.uptime_s)}`
        + ` · ${data.clients.video} flux vidéo et ${data.clients.events} page(s) connectés`;
    renderTimings(data.timings);

    const gpu = data.gpu
        ? `${data.gpu.name} (${data.gpu.memory_used_mb} / ${data.gpu.memory_total_mb} Mo)`
        : 'aucun GPU CUDA';
    pairs(document.getElementById('hardware'), [
        ['GPU', gpu],
        ['Détecteur', data.models.detector],
        ['Embedder d\'apparence', data.models.appearance_embedder],
        ['Tracker', data.models.tracker_backend],
        ['Visages', data.models.face_model],
        ['Orientation', data.models.pose_source],
        ['Caméra', data.camera.source],
        ['Résolution', data.camera.frame_size ? data.camera.frame_size.join(' × ') : null],
    ]);
    pairs(document.getElementById('settings'), [
        ['Accès protégé', data.auth_enabled ? 'oui' : 'non'],
        ['Seuil de reconnaissance', data.recognition.threshold],
        ['Visage minimum', `${data.recognition.min_face_px} px`],
        ['Entrées en base', data.recognition.known_entries],
        ['Distance cosinus max', data.tracking.max_cosine_distance],
        ['Distance IoU max', data.tracking.max_iou_distance],
        ['max_age', `${data.tracking.max_age} images`],
        ['n_init', `${data.tracking.n_init} images`],
    ]);
}

// ── Mesure comparative des sources d'orientation ─────────────────────────

document.getElementById('benchForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const button = document.getElementById('benchBtn');
    const result = document.getElementById('benchResult');
    button.disabled = true;
    button.textContent = 'Mesure en cours…';
    result.replaceChildren();
    try {
        const { data } = await api('/bench/pose', { method: 'POST', body: new FormData(e.target) });
        if (!data.success) {
            toast(data.message, 'error');
            return;
        }
        const rows = [
            ['Observations', data.observations],
            ['Mediapipe', `${data.mediapipe_ms.med} ms (p95 ${data.mediapipe_ms.p95})`],
            ['Keypoints YOLO', `${data.yolo_ms.med} ms (p95 ${data.yolo_ms.p95})`],
            ['Économie', `${data.economie_ms_par_frame} ms par image`],
            ['Accord', data.accord.taux === null ? 'aucune référence'
                : `${data.accord.taux} % (${data.accord.accords}/${data.accord.compares})`],
            ['Abstentions', `Mediapipe ${data.abstentions.mediapipe} · YOLO ${data.abstentions.yolo}`],
        ];
        if (data.ecart_epaules_px) {
            rows.push(['Écart des épaules', `${data.ecart_epaules_px.min} / ${data.ecart_epaules_px.med} / `
                + `${data.ecart_epaules_px.max} px (seuil ${data.seuil_fiabilite_px})`]);
        }
        for (const [verdict, count] of data.confusion) rows.push([verdict, count]);
        pairs(result, rows);
    } finally {
        button.disabled = false;
        button.textContent = 'Lancer la mesure';
    }
});

load();
setInterval(load, 2000);
