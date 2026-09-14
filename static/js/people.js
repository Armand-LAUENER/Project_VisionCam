// Page Personnes : liste, renommage, suppression, ajout de photos, reconstruction.
import { api, h, onEvent, toast } from './common.js';

const cards = document.getElementById('cards');
let current = null;   // personne visée par un dialogue

function card(person) {
    const initials = person.name.split(/\s+/).map((part) => part[0]).join('').slice(0, 2).toUpperCase();
    const thumb = person.thumbnail_url
        // Paramètre de version : la miniature change après un ajout de photos.
        ? h('img', { src: `${person.thumbnail_url}?v=${person.photos}`, alt: '', loading: 'lazy' })
        : initials;
    const mode = person.templates.length
        ? `${person.templates.length} angle(s) : ${person.templates.join(', ')}`
        : 'photos moyennées';
    return h('article', { class: 'card' },
        h('div', { class: 'thumb', 'aria-hidden': 'true' }, thumb),
        h('div', { class: 'body' },
            h('h2', {}, person.name),
            h('div', { class: 'muted' }, `${person.photos} photo(s) · ${mode}`),
            person.enrolled
                ? h('span', { class: 'badge ok' }, 'reconnue')
                : h('span', { class: 'badge warn', title: 'Aucun visage exploitable sur ses photos' }, 'pas de visage exploitable'),
            h('div', { class: 'actions' },
                h('button', { class: 'small', type: 'button', onClick: () => openPhotos(person.name) }, 'Photos +'),
                h('button', { class: 'small', type: 'button', onClick: () => openRename(person) }, 'Renommer'),
                h('button', { class: 'small danger', type: 'button', onClick: () => openDelete(person) }, 'Supprimer'),
            ),
        ),
    );
}

async function load() {
    const { ok, data } = await api('/api/people');
    if (!ok) {
        toast('Impossible de charger les personnes.', 'error');
        return;
    }
    cards.replaceChildren(...data.people.map(card));
    document.getElementById('peopleEmpty').hidden = data.people.length > 0;
    document.getElementById('peopleCount').textContent =
        `${data.people.length} personne(s) enrôlée(s)`;
}

// ── Dialogues ─────────────────────────────────────────────────────────────

for (const dialog of document.querySelectorAll('dialog')) {
    dialog.querySelector('[data-close]').addEventListener('click', () => dialog.close());
}

function openRename(person) {
    current = person;
    const input = document.getElementById('renameInput');
    input.value = person.name;
    document.getElementById('renameDialog').showModal();
    input.select();
}

document.getElementById('renameForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const newName = document.getElementById('renameInput').value.trim();
    const { ok, data } = await api(`/api/people/${encodeURIComponent(current.name)}/rename`, {
        method: 'POST', json: { new_name: newName },
    });
    toast(data.message, ok ? 'ok' : 'error');
    if (ok) {
        document.getElementById('renameDialog').close();
        load();
    }
});

function openDelete(person) {
    current = person;
    document.getElementById('deleteName').textContent = person.name;
    document.getElementById('deleteDialog').showModal();
}

document.getElementById('deleteForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const { ok, data } = await api(`/api/people/${encodeURIComponent(current.name)}`, { method: 'DELETE' });
    toast(data.message, ok ? 'ok' : 'error');
    document.getElementById('deleteDialog').close();
    if (ok) load();
});

function openPhotos(name = '') {
    const nameInput = document.getElementById('photosName');
    nameInput.value = name;
    nameInput.readOnly = Boolean(name);
    document.getElementById('photosFiles').value = '';
    document.getElementById('photosTitle').textContent = name ? `Ajouter des photos à ${name}` : 'Nouvelle personne';
    document.getElementById('photosDialog').showModal();
    (name ? document.getElementById('photosFiles') : nameInput).focus();
}

document.getElementById('addBtn').addEventListener('click', () => openPhotos());

document.getElementById('photosForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('photosName').value.trim();
    const files = document.getElementById('photosFiles').files;
    const body = new FormData();
    for (const file of files) body.append('images', file);
    const button = e.submitter;
    button.disabled = true;
    button.textContent = 'Analyse…';
    try {
        const { ok, data } = await api(`/api/people/${encodeURIComponent(name)}/photos`, { method: 'POST', body });
        toast(data.message, ok ? 'ok' : 'error');
        if (ok) {
            document.getElementById('photosDialog').close();
            load();
        }
    } finally {
        button.disabled = false;
        button.textContent = 'Envoyer';
    }
});

// ── Reconstruction ────────────────────────────────────────────────────────

const rebuildBtn = document.getElementById('rebuildBtn');
rebuildBtn.addEventListener('click', async () => {
    const { data } = await api('/rebuild', { method: 'POST' });
    toast(data.message, data.started ? '' : 'error');
    if (data.started) {
        rebuildBtn.disabled = true;
        rebuildBtn.textContent = 'Reconstruction…';
    }
});

onEvent('rebuilt', (e) => {
    rebuildBtn.disabled = false;
    rebuildBtn.textContent = 'Reconstruire la base';
    toast(`Base reconstruite : ${e.total_known} entrée(s).`, 'ok');
    load();
});

// Une modification faite depuis un autre onglet ou la page Live se voit ici.
for (const type of ['enrolled', 'renamed', 'deleted']) onEvent(type, load);

load();
