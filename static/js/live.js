// Page Live : flux vidéo, boîtes cliquables, capture d'enrôlement, présents et événements.
import { api, formatTime, h, onEvent, setEventCamera, toast } from './common.js';

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

let status = { tracks: [], frame_size: null, currently_present: [] };
let target = null;          // piste choisie pour la capture : { track_id, name }
// Style « Person of Interest » (la Machine) : blanc pour une personne quelconque,
// jaune pour une personne connue, noir à coins jaunes quand le nom est tenu par le
// corps (visage perdu). Bords en pointillés, crochets pleins aux coins.
const POI = { yellow: '#ffd200', white: '#ebebeb', black: '#141414', selected: '#ff4d4d' };

// ── Superposition des boîtes sur le flux ──────────────────────────────────

/** Rectangle réellement occupé par l'image dans le cadre (object-fit: contain). */
function imageRect() {
    const box = canvas.getBoundingClientRect();
    const [fw, fh] = status.frame_size || [16, 9];
    const scale = Math.min(box.width / fw, box.height / fh);
    return { scale, x: (box.width - fw * scale) / 2, y: (box.height - fh * scale) / 2, box };
}

function draw() {
    const { scale, x, y, box } = imageRect();
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(box.width * ratio) || canvas.height !== Math.round(box.height * ratio)) {
        canvas.width = Math.round(box.width * ratio);
        canvas.height = Math.round(box.height * ratio);
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, box.width, box.height);
    if (!status.frame_size) return;

    for (const track of status.tracks) {
        const [x1, y1, x2, y2] = track.bbox;
        const rect = { x: x + x1 * scale, y: y + y1 * scale, w: (x2 - x1) * scale, h: (y2 - y1) * scale };
        drawTrack(track, rect, target && target.track_id === track.track_id, box);
    }
}

/** Apparence d'une piste selon l'origine de son nom (cf. /status : mode). */
function trackStyle(track) {
    if (track.mode === 'face') {
        return { corner: POI.yellow, edge: POI.yellow, tab: 'IDENTIFIED',
                 status: `#${track.track_id}  FACE ${Math.round(track.confidence * 100)}%` };
    }
    if (track.mode === 'body') {
        const since = track.face_age_s == null ? '' : `  LAST FACE ${Math.round(track.face_age_s)}S`;
        return { corner: POI.yellow, edge: POI.black, tab: 'TRACKING',
                 status: `#${track.track_id}  BODY${since}` };
    }
    return { corner: POI.white, edge: POI.white, tab: 'UNKNOWN', status: `#${track.track_id}` };
}

function drawTrack(track, r, selected, box) {
    const style = trackStyle(track);
    const corner = selected ? POI.selected : style.corner;

    // Bords fins en pointillés, crochets épais aux quatre coins.
    ctx.setLineDash([6, 6]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = style.edge;
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    ctx.setLineDash([]);
    ctx.lineWidth = selected ? 4 : 3;
    ctx.strokeStyle = corner;
    const c = Math.max(8, 0.2 * Math.min(r.w, r.h));
    ctx.beginPath();
    for (const [cx, cy, dx, dy] of [[r.x, r.y, 1, 1], [r.x + r.w, r.y, -1, 1],
                                     [r.x + r.w, r.y + r.h, -1, -1], [r.x, r.y + r.h, 1, -1]]) {
        ctx.moveTo(cx + dx * c, cy);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx, cy + dy * c);
    }
    ctx.stroke();

    // Panneau : onglet plein, puis nom et état sur fond sombre translucide.
    const name = track.name === 'Inconnu' ? 'UNIDENTIFIED' : track.name.toUpperCase();
    const lines = [name, style.status];
    if (track.pose) lines.push(`POSE: ${String(track.pose).toUpperCase()}`);
    ctx.font = '600 11px ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace';
    const pad = 5, lineH = 15;
    const tabW = ctx.measureText(style.tab).width + 2 * pad;
    const width = Math.max(tabW, ...lines.map((t) => ctx.measureText(t).width + 2 * pad));
    const height = lineH + lines.length * lineH + pad;
    let px = r.x + r.w + 6;
    if (px + width > box.width) px = r.x - width - 6;
    px = Math.max(0, px);
    const py = Math.min(Math.max(0, r.y), Math.max(0, box.height - height));

    ctx.fillStyle = corner;
    ctx.fillRect(px, py, tabW, lineH);
    ctx.fillStyle = '#000000';
    ctx.fillText(style.tab, px + pad, py + lineH - 4);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(px, py + lineH, width, height - lineH);
    ctx.fillStyle = corner;
    lines.forEach((text, i) => ctx.fillText(text, px + pad, py + lineH * (i + 2) - 4));
}

function trackAt(clientX, clientY) {
    const { scale, x, y, box } = imageRect();
    const fx = (clientX - box.left - x) / scale;
    const fy = (clientY - box.top - y) / scale;
    // La plus petite boîte contenant le point : une personne devant une autre gagne.
    return status.tracks
        .filter(({ bbox: [x1, y1, x2, y2] }) => fx >= x1 && fx <= x2 && fy >= y1 && fy <= y2)
        .sort((a, b) => (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1])
                      - (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]))[0] || null;
}

canvas.addEventListener('click', (e) => {
    const track = trackAt(e.clientX, e.clientY);
    if (!track) {
        toast('Cliquez à l\'intérieur de la boîte d\'une personne.');
        return;
    }
    openCapture(track);
});
new ResizeObserver(draw).observe(canvas);
video.addEventListener('error', () => toast('Flux vidéo interrompu : nouvelle tentative…', 'error'));

// ── Statistiques, présents ────────────────────────────────────────────────

function renderStatus() {
    document.getElementById('statFps').textContent = status.fps ? status.fps.toFixed(0) : '—';
    document.getElementById('statVisible').textContent = status.tracks.length;
    document.getElementById('statNamed').textContent = status.tracks.filter((t) => t.name !== 'Inconnu').length;
    document.getElementById('statKnown').textContent = status.total_known ?? 0;

    const list = document.getElementById('presentList');
    const present = status.currently_present || [];
    list.replaceChildren(...present.map((p) => h('li', { class: 'person-row' },
        h('span', {},
            h('span', { class: 'name' }, p.name),
            ' ', h('span', { class: 'muted' }, `#${p.track_id}`)),
        h('span', {},
            p.pose ? h('span', { class: 'badge' }, p.pose) : null, ' ',
            p.name === 'Inconnu'
                ? h('span', { class: 'badge' }, 'inconnu')
                : h('span', { class: 'badge ok' }, `${Math.round((p.confidence || 0) * 100)} %`)),
    )));
    document.getElementById('presentEmpty').hidden = present.length > 0;

    if (target) {
        const live = status.tracks.find((t) => t.track_id === target.track_id);
        document.getElementById('livePose').textContent = live ? (live.pose || '—') : 'hors champ';
    }
    draw();
}

// ── Choix de la caméra (affiché s'il y en a plusieurs) ────────────────────

const cameraPicker = document.getElementById('cameraPicker');
const cameraSelect = document.getElementById('cameraSelect');
let camera = null;

function renderCameras(cameras) {
    cameraPicker.hidden = cameras.length < 2;
    const current = [...cameraSelect.options].map((o) => o.value).join();
    if (current === cameras.join()) return;
    cameraSelect.replaceChildren(...cameras.map((id) => h('option', { value: id }, id)));
    cameraSelect.value = camera || cameras[0];
}

cameraSelect.addEventListener('change', () => {
    camera = cameraSelect.value;
    video.src = `/video?camera=${encodeURIComponent(camera)}`;
    status = { tracks: [], frame_size: null, currently_present: status.currently_present };
    setEventCamera(camera);
});

onEvent('status', (event) => {
    status = event;
    renderCameras(event.cameras || []);
    renderStatus();
});

// ── Fil d'événements ──────────────────────────────────────────────────────

const EVENT_TEXT = {
    arrival: (e) => ['ok', `${e.name} est arrivé(e)`],
    departure: (e) => ['', `${e.name} est parti(e)`],
    unknown: (e) => ['warn', `Personne inconnue à l'écran (#${e.track_id}${e.camera ? `, ${e.camera}` : ''})`],
    enrolled: (e) => ['ok', `${e.name} enrôlé(e)`],
    renamed: (e) => ['', `${e.old_name} renommé(e) en ${e.name}`],
    deleted: (e) => ['danger', `${e.name} supprimé(e)`],
    rebuilt: (e) => ['', `Base reconstruite : ${e.total_known} entrée(s)`],
};

onEvent('*', (event) => {
    const describe = EVENT_TEXT[event.type];
    if (!describe) return;
    const [kind, text] = describe(event);
    const feed = document.getElementById('eventFeed');
    feed.prepend(h('li', {}, h('time', { datetime: event.time }, formatTime(event.time)),
        kind ? h('span', { class: `badge ${kind}` }, text) : text));
    while (feed.children.length > 50) feed.lastChild.remove();
    document.getElementById('eventEmpty').hidden = true;
});

// ── Capture ───────────────────────────────────────────────────────────────

const dialog = document.getElementById('captureDialog');
const form = document.getElementById('captureForm');
const nameInput = document.getElementById('captureName');
const guided = document.getElementById('guided');
const submit = document.getElementById('captureSubmit');

async function openCapture(track) {
    target = { track_id: track.track_id, name: track.name };
    document.getElementById('captureTarget').textContent =
        `Piste #${track.track_id} — actuellement « ${track.name} »`;
    nameInput.value = track.name === 'Inconnu' ? '' : track.name;
    guided.querySelectorAll('button').forEach((b) => b.classList.remove('done'));
    form.mode.value = 'photo';
    guided.hidden = true;
    submit.hidden = false;
    dialog.showModal();
    nameInput.focus();
    draw();

    const { ok, data } = await api('/api/people');
    if (ok) {
        document.getElementById('knownNames').replaceChildren(
            ...data.people.map((p) => h('option', { value: p.name })));
    }
}

function closeCapture() {
    dialog.close();
    target = null;
    draw();
}

async function capture(label) {
    const name = nameInput.value.trim();
    if (!name) {
        nameInput.reportValidity();
        return false;
    }
    const { ok, data } = await api('/api/capture', {
        method: 'POST',
        json: { name, track_id: target.track_id, ...(camera ? { camera } : {}), ...(label ? { label } : {}) },
    });
    toast(data.message || (ok ? 'Enrôlé.' : 'Échec de la capture.'), ok ? 'ok' : 'error');
    return ok;
}

form.addEventListener('change', (e) => {
    if (e.target.name !== 'mode') return;
    guided.hidden = form.mode.value !== 'guided';
    submit.hidden = form.mode.value === 'guided';
});
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    submit.disabled = true;
    try {
        if (await capture(null)) closeCapture();
    } finally {
        submit.disabled = false;
    }
});
guided.addEventListener('click', async (e) => {
    const button = e.target.closest('button[data-label]');
    if (!button) return;
    button.disabled = true;
    try {
        if (await capture(button.dataset.label)) button.classList.add('done');
    } finally {
        button.disabled = false;
    }
});
document.getElementById('captureCancel').addEventListener('click', closeCapture);
dialog.addEventListener('close', () => {
    target = null;
    draw();
});
