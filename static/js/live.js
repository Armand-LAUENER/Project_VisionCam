// Page Live : flux vidéo, boîtes cliquables, capture d'enrôlement, présents et événements.
import { api, formatTime, h, onEvent, toast } from './common.js';

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

let status = { tracks: [], frame_size: null, currently_present: [] };
let target = null;          // piste choisie pour la capture : { track_id, name }
const COLORS = { known: '#34d399', unknown: '#60a5fa', selected: '#ffffff' };

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

    ctx.font = '600 13px system-ui, sans-serif';
    for (const track of status.tracks) {
        const [x1, y1, x2, y2] = track.bbox;
        const selected = target && target.track_id === track.track_id;
        const color = selected ? COLORS.selected : track.name === 'Inconnu' ? COLORS.unknown : COLORS.known;
        const rx = x + x1 * scale, ry = y + y1 * scale, rw = (x2 - x1) * scale, rh = (y2 - y1) * scale;
        ctx.lineWidth = selected ? 3 : 2;
        ctx.strokeStyle = color;
        ctx.strokeRect(rx, ry, rw, rh);
        const label = track.pose ? `${track.name} · ${track.pose}` : track.name;
        const width = ctx.measureText(label).width + 10;
        ctx.fillStyle = color;
        ctx.fillRect(rx, Math.max(0, ry - 20), width, 20);
        ctx.fillStyle = '#04111c';
        ctx.fillText(label, rx + 5, Math.max(14, ry - 6));
    }
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

onEvent('status', (event) => {
    status = event;
    renderStatus();
});

// ── Fil d'événements ──────────────────────────────────────────────────────

const EVENT_TEXT = {
    arrival: (e) => ['ok', `${e.name} est arrivé(e)`],
    departure: (e) => ['', `${e.name} est parti(e)`],
    unknown: (e) => ['warn', `Personne inconnue à l'écran (#${e.track_id})`],
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
        method: 'POST', json: { name, track_id: target.track_id, ...(label ? { label } : {}) },
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
