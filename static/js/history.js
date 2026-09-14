// Page Historique : sessions de présence filtrées, export CSV, rafraîchissement en direct.
import { api, formatDateTime, formatDuration, h, onEvent, toast } from './common.js';

const form = document.getElementById('filters');
const select = document.getElementById('filterName');

function query() {
    const params = new URLSearchParams();
    for (const [key, value] of new FormData(form)) if (value) params.set(key, value);
    return params;
}

async function load() {
    const params = query();
    const { ok, data } = await api(`/api/history?${params}`);
    if (!ok) {
        toast(data.message || 'Impossible de charger l\'historique.', 'error');
        return;
    }

    // Garde la personne choisie si la liste des noms change.
    const chosen = select.value;
    select.replaceChildren(h('option', { value: '' }, 'Toutes'),
        ...data.names.map((name) => h('option', { value: name, selected: name === chosen }, name)));

    document.getElementById('sessions').replaceChildren(...data.sessions.map((s) => h('tr', {},
        h('td', {}, s.name),
        h('td', {}, formatDateTime(s.arrived)),
        h('td', {}, s.ongoing ? h('span', { class: 'badge ok' }, 'présent(e)') : formatDateTime(s.departed)),
        h('td', { class: 'mono' }, formatDuration(s.duration_s)),
    )));
    document.getElementById('historyEmpty').hidden = data.sessions.length > 0;

    const total = data.sessions.reduce((sum, s) => sum + s.duration_s, 0);
    document.getElementById('historySummary').textContent =
        `${data.sessions.length} session(s) · ${formatDuration(total)} de présence cumulée`;
    document.getElementById('csvLink').href = `/api/history.csv?${params}`;
}

form.addEventListener('submit', (e) => {
    e.preventDefault();
    load();
});
form.addEventListener('reset', () => setTimeout(load));

// Une arrivée ou un départ ajoute ou ferme une ligne : on recharge.
for (const type of ['arrival', 'departure', 'renamed', 'deleted']) onEvent(type, load);

load();
